"""
Reference Model Management for SPO Training.

In DPO/SPO training, we need to maintain a frozen copy of the original model
as a reference. This module provides utilities for managing the reference model.

Key features:
- Deep copy of model with frozen parameters
- Memory-efficient storage options
- Utilities for comparing current vs reference model
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager


class ReferenceModelManager:
    """
    Manages a frozen reference copy of the PMDM model for DPO training.

    The reference model is used to compute the KL divergence constraint
    in DPO loss, preventing the fine-tuned model from diverging too far
    from the original model.

    Usage:
        ref_manager = ReferenceModelManager(model)
        ref_manager.to(device)

        # During training
        with torch.no_grad():
            ref_output = ref_manager.forward(*args, **kwargs)

        # Optionally update reference periodically
        ref_manager.update_reference(model, ema_decay=0.999)

    Attributes:
        ref_model: The frozen reference model
        device: Device the model is on
    """

    def __init__(
        self,
        model: nn.Module,
        copy_to_device: Optional[torch.device] = None,
        use_ema: bool = False,
        ema_decay: float = 0.999
    ):
        """
        Initialize reference model manager.

        Args:
            model: The model to create a reference copy of
            copy_to_device: Optional device to move the reference to
            use_ema: Whether to use EMA for reference updates
            ema_decay: EMA decay rate if use_ema=True
        """
        # Deep copy the model
        self.ref_model = copy.deepcopy(model)

        # Freeze all parameters
        self._freeze_model()

        # Move to device if specified
        if copy_to_device is not None:
            self.ref_model.to(copy_to_device)

        self.device = copy_to_device
        self.use_ema = use_ema
        self.ema_decay = ema_decay

    def _freeze_model(self):
        """Freeze all parameters and set to eval mode."""
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    def to(self, device: torch.device) -> 'ReferenceModelManager':
        """Move reference model to device."""
        self.ref_model.to(device)
        self.device = device
        return self

    def forward(self, *args, **kwargs):
        """
        Forward pass through reference model (no gradients).

        This is a convenience method that ensures no gradients are computed.
        """
        with torch.no_grad():
            return self.ref_model(*args, **kwargs)

    def net(self, *args, **kwargs):
        """
        Call the net method of reference model (no gradients).

        This matches PMDM's interface where the main computation is in .net()
        """
        with torch.no_grad():
            return self.ref_model.net(*args, **kwargs)

    def update_reference(
        self,
        model: nn.Module,
        ema_decay: Optional[float] = None
    ):
        """
        Update reference model parameters (optional, for EMA updates).

        This can be used to periodically update the reference model
        using exponential moving average of the current model.

        Args:
            model: Current model to update from
            ema_decay: EMA decay rate (uses self.ema_decay if not specified)
        """
        if not self.use_ema:
            raise ValueError("Reference updates require use_ema=True")

        decay = ema_decay if ema_decay is not None else self.ema_decay

        with torch.no_grad():
            for ref_param, model_param in zip(
                self.ref_model.parameters(),
                model.parameters()
            ):
                ref_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

    def hard_update(self, model: nn.Module):
        """
        Hard update: completely replace reference with current model.

        This is useful for periodic full updates of the reference.
        """
        with torch.no_grad():
            for ref_param, model_param in zip(
                self.ref_model.parameters(),
                model.parameters()
            ):
                ref_param.data.copy_(model_param.data)

        # Re-freeze
        self._freeze_model()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return {
            'ref_model': self.ref_model.state_dict(),
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.ref_model.load_state_dict(state_dict['ref_model'])
        self.use_ema = state_dict.get('use_ema', False)
        self.ema_decay = state_dict.get('ema_decay', 0.999)
        self._freeze_model()

    @property
    def alphas(self) -> torch.Tensor:
        """Access alphas from reference model (for log prob computation)."""
        return self.ref_model.alphas

    @property
    def betas(self) -> torch.Tensor:
        """Access betas from reference model."""
        return self.ref_model.betas

    @property
    def num_timesteps(self) -> int:
        """Access num_timesteps from reference model."""
        return self.ref_model.num_timesteps


class MemoryEfficientReferenceManager:
    """
    Memory-efficient reference model manager using parameter checkpointing.

    Instead of keeping a full copy of the model in memory, this manager
    stores parameters on CPU and loads them on-demand.

    This is useful when GPU memory is limited.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None
    ):
        """
        Args:
            model: Model to create reference from
            checkpoint_path: Optional path to save reference checkpoint
        """
        # Store reference parameters on CPU
        self.ref_params = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
        }

        # Store reference buffers on CPU
        self.ref_buffers = {
            name: buf.clone().cpu()
            for name, buf in model.named_buffers()
        }

        # Store model class and config for reconstruction
        self.model_class = type(model)
        self.model_config = getattr(model, 'config', None)

        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path)

    def _save_checkpoint(self, path: str):
        """Save reference to checkpoint."""
        torch.save({
            'params': self.ref_params,
            'buffers': self.ref_buffers,
        }, path)

    @contextmanager
    def load_to_device(
        self,
        model: nn.Module,
        device: torch.device
    ):
        """
        Context manager that temporarily loads reference params into model.

        Usage:
            with ref_manager.load_to_device(model, device) as ref_model:
                ref_output = ref_model.net(*args)

        This swaps the model parameters with reference parameters,
        then swaps back when exiting the context.
        """
        # Save current parameters
        current_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        current_buffers = {
            name: buf.clone()
            for name, buf in model.named_buffers()
        }

        # Load reference parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.ref_params:
                    param.data.copy_(self.ref_params[name].to(device))

            for name, buf in model.named_buffers():
                if name in self.ref_buffers:
                    buf.copy_(self.ref_buffers[name].to(device))

        # Set to eval mode
        training = model.training
        model.eval()

        try:
            yield model
        finally:
            # Restore original parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in current_params:
                        param.data.copy_(current_params[name])

                for name, buf in model.named_buffers():
                    if name in current_buffers:
                        buf.copy_(current_buffers[name])

            # Restore training mode
            model.train(training)


def create_reference_model(
    model: nn.Module,
    device: torch.device,
    memory_efficient: bool = False,
    checkpoint_path: Optional[str] = None
) -> Union[ReferenceModelManager, MemoryEfficientReferenceManager]:
    """
    Factory function to create appropriate reference model manager.

    Args:
        model: Model to create reference from
        device: Device to use
        memory_efficient: Whether to use memory-efficient version
        checkpoint_path: Optional path for checkpointing

    Returns:
        Reference model manager instance
    """
    if memory_efficient:
        return MemoryEfficientReferenceManager(model, checkpoint_path)
    else:
        return ReferenceModelManager(model, copy_to_device=device)
