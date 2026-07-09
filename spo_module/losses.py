"""
SPO (Step-aware Preference Optimization) Loss Functions.

This module implements the DPO (Direct Preference Optimization) loss function
used for fine-tuning PMDM with preference feedback from SPM.

The DPO loss encourages the model to prefer generating "win" samples over "lose" samples:
    L = -log sigmoid(beta * (log(pi(y_win|x)/pi_ref(y_win|x)) - log(pi(y_lose|x)/pi_ref(y_lose|x))))

Key concepts:
- pi(y|x): Probability of generating y under current model
- pi_ref(y|x): Probability under reference (frozen) model
- beta: Temperature parameter controlling preference strength
- eps: Clipping range for probability ratios to stabilize training
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SPOLoss(nn.Module):
    """
    Step-aware Preference Optimization Loss.

    This loss function implements DPO for diffusion models, adapted for PMDM.
    It computes the preference loss between win/lose sample pairs at each timestep.

    The loss formula:
        L = -log(sigmoid(beta * (log(ratio_win) - log(ratio_lose))))

    where:
        ratio_win = clip(exp(log_prob_win - log_ref_win), 1-eps, 1+eps)
        ratio_lose = clip(exp(log_prob_lose - log_ref_lose), 1-eps, 1+eps)

    Attributes:
        beta: Temperature parameter (higher = stronger preference)
        eps: Clipping range for probability ratios
        reduction: How to reduce the loss ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        beta: float = 10.0,
        eps: float = 0.1,
        reduction: str = 'mean',
        use_reference_model: bool = True
    ):
        """
        Initialize SPO Loss.

        Args:
            beta: Temperature parameter controlling preference strength.
                  Higher values make the model more confident in preferences.
            eps: Clipping range [1-eps, 1+eps] for probability ratios.
                 This prevents extreme updates and stabilizes training.
            reduction: Loss reduction method ('mean', 'sum', 'none')
            use_reference_model: Whether to use reference model for KL constraint.
                                 If False, treats log_ref as 0 (no KL constraint).
        """
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.use_reference_model = use_reference_model

    def forward(
        self,
        log_prob_win: torch.Tensor,
        log_prob_lose: torch.Tensor,
        log_ref_win: Optional[torch.Tensor] = None,
        log_ref_lose: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SPO/DPO loss.

        The DPO loss formula (from the DPO paper):
            L = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))

        Which simplifies to:
            L = -log(sigmoid(beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))))

        Args:
            log_prob_win: Log probability of win samples under current model [B]
            log_prob_lose: Log probability of lose samples under current model [B]
            log_ref_win: Log probability of win samples under reference model [B]
            log_ref_lose: Log probability of lose samples under reference model [B]
            weights: Optional per-sample weights [B]

        Returns:
            loss: Scalar loss value
            metrics: Dictionary of metrics for logging
        """
        # Handle reference model usage
        if self.use_reference_model and log_ref_win is not None and log_ref_lose is not None:
            # Compute log ratios: log(pi/pi_ref) = log_pi - log_pi_ref
            log_ratio_win = log_prob_win - log_ref_win
            log_ratio_lose = log_prob_lose - log_ref_lose
        else:
            # Without reference model, use absolute log probabilities
            # This is less stable but can work for simple cases
            log_ratio_win = log_prob_win
            log_ratio_lose = log_prob_lose

        # Compute ratios: pi/pi_ref = exp(log_ratio)
        # Clamp log_ratio before exp to prevent overflow
        ratio_win = torch.exp(log_ratio_win.clamp(max=20))
        ratio_lose = torch.exp(log_ratio_lose.clamp(max=20))

        # Clip ratios to [1-eps, 1+eps] following the reference implementation
        # This is the correct order: exp first, then clamp, then log
        # Reference: train_spo.py line 584-586
        ratio_win_clipped = torch.clamp(ratio_win, 1 - self.eps, 1 + self.eps)
        ratio_lose_clipped = torch.clamp(ratio_lose, 1 - self.eps, 1 + self.eps)

        # Compute DPO logits: beta * (log(ratio_win) - log(ratio_lose))
        # Following the reference implementation exactly
        logits = self.beta * (torch.log(ratio_win_clipped) - torch.log(ratio_lose_clipped))

        # DPO loss: -log(sigmoid(logits))
        loss = -F.logsigmoid(logits)

        # Apply weights if provided
        if weights is not None:
            loss = loss * weights

        # Compute metrics before reduction
        metrics = self._compute_metrics(
            ratio_win, ratio_lose, ratio_win_clipped, ratio_lose_clipped,
            logits, loss, log_prob_win, log_prob_lose
        )

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # 'none' keeps per-sample losses

        return loss, metrics

    def _compute_metrics(
        self,
        ratio_win: torch.Tensor,
        ratio_lose: torch.Tensor,
        ratio_win_clipped: torch.Tensor,
        ratio_lose_clipped: torch.Tensor,
        logits: torch.Tensor,
        loss: torch.Tensor,
        log_prob_win: torch.Tensor,
        log_prob_lose: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for logging."""
        with torch.no_grad():
            metrics = {
                # Loss value
                'loss': loss.mean().item(),

                # Probability ratios (should stay close to 1)
                'ratio_win_mean': ratio_win.mean().item(),
                'ratio_win_std': ratio_win.std().item(),
                'ratio_lose_mean': ratio_lose.mean().item(),
                'ratio_lose_std': ratio_lose.std().item(),

                # Clipping rate (how often we hit the clip bounds)
                'ratio_win_clip_rate': (
                    (ratio_win < 1 - self.eps) | (ratio_win > 1 + self.eps)
                ).float().mean().item(),
                'ratio_lose_clip_rate': (
                    (ratio_lose < 1 - self.eps) | (ratio_lose > 1 + self.eps)
                ).float().mean().item(),

                # Logits (positive = correct preference)
                'logits_mean': logits.mean().item(),
                'logits_std': logits.std().item(),

                # Accuracy (how often win has higher prob ratio than lose)
                'accuracy': (logits > 0).float().mean().item(),

                # Raw log probabilities
                'log_prob_win_mean': log_prob_win.mean().item(),
                'log_prob_lose_mean': log_prob_lose.mean().item(),
                'log_prob_diff_mean': (log_prob_win - log_prob_lose).mean().item(),
            }

        return metrics


class SPOLossWithKL(SPOLoss):
    """
    SPO Loss with explicit KL divergence regularization.

    This variant adds an explicit KL term to prevent the model from
    diverging too far from the reference model.

    Loss = L_DPO + kl_weight * KL(pi || pi_ref)
    """

    def __init__(
        self,
        beta: float = 10.0,
        eps: float = 0.1,
        kl_weight: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Args:
            beta: DPO temperature
            eps: Ratio clipping range
            kl_weight: Weight for KL regularization term
            reduction: Loss reduction method
        """
        super().__init__(beta=beta, eps=eps, reduction=reduction)
        self.kl_weight = kl_weight

    def forward(
        self,
        log_prob_win: torch.Tensor,
        log_prob_lose: torch.Tensor,
        log_ref_win: torch.Tensor,
        log_ref_lose: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SPO loss with KL regularization."""
        # Get base DPO loss
        dpo_loss, metrics = super().forward(
            log_prob_win, log_prob_lose, log_ref_win, log_ref_lose, weights
        )

        # Compute KL divergence approximation
        # KL(pi || pi_ref) ≈ E[log(pi/pi_ref)] = E[log_prob - log_ref]
        kl_win = (log_prob_win - log_ref_win).mean()
        kl_lose = (log_prob_lose - log_ref_lose).mean()
        kl_loss = (kl_win + kl_lose) / 2

        # Total loss
        total_loss = dpo_loss + self.kl_weight * kl_loss

        # Update metrics
        metrics.update({
            'kl_loss': kl_loss.item(),
            'kl_win': kl_win.item(),
            'kl_lose': kl_lose.item(),
            'dpo_loss': dpo_loss.item() if isinstance(dpo_loss, torch.Tensor) else dpo_loss,
        })

        return total_loss, metrics


class MultiStepSPOLoss(nn.Module):
    """
    Multi-step SPO Loss for aggregating losses across multiple timesteps.

    In SPO training, we collect win/lose pairs at multiple timesteps during
    sampling. This loss aggregates them with optional timestep weighting.
    """

    def __init__(
        self,
        beta: float = 10.0,
        eps: float = 0.1,
        timestep_weighting: str = 'uniform',
        reduction: str = 'mean'
    ):
        """
        Args:
            beta: DPO temperature
            eps: Ratio clipping range
            timestep_weighting: How to weight different timesteps
                - 'uniform': Equal weight for all timesteps
                - 'linear_decay': Higher weight for earlier timesteps
                - 'sqrt_decay': sqrt-based decay
            reduction: Loss reduction method
        """
        super().__init__()
        self.base_loss = SPOLoss(beta=beta, eps=eps, reduction='none')
        self.timestep_weighting = timestep_weighting
        self.reduction = reduction

    def forward(
        self,
        pairs: list,
        compute_log_prob_fn
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute aggregated loss over multiple timestep pairs.

        Args:
            pairs: List of dictionaries containing:
                - 'timestep': int
                - 'current_state': (pos_t, atom_t)
                - 'win_state': (pos_win, atom_win)
                - 'lose_state': (pos_lose, atom_lose)
                - 'batch': batch indices
            compute_log_prob_fn: Function to compute log probabilities
                Should take (state, target_state, timestep) and return log_prob

        Returns:
            loss: Aggregated loss
            metrics: Aggregated metrics
        """
        if len(pairs) == 0:
            return torch.tensor(0.0), {'loss': 0.0, 'num_pairs': 0}

        all_losses = []
        all_metrics = []
        timesteps = []

        for pair in pairs:
            t = pair['timestep']
            timesteps.append(t)

            # Compute log probabilities
            log_prob_win = compute_log_prob_fn(
                pair['current_state'], pair['win_state'], t
            )
            log_prob_lose = compute_log_prob_fn(
                pair['current_state'], pair['lose_state'], t
            )
            log_ref_win = pair.get('log_ref_win', None)
            log_ref_lose = pair.get('log_ref_lose', None)

            # Compute loss for this pair
            loss, metrics = self.base_loss(
                log_prob_win, log_prob_lose, log_ref_win, log_ref_lose
            )
            all_losses.append(loss)
            all_metrics.append(metrics)

        # Stack losses
        losses = torch.stack(all_losses)

        # Compute timestep weights
        weights = self._compute_timestep_weights(timesteps, losses.device)

        # Weighted aggregation
        weighted_loss = (losses * weights).sum() / weights.sum()

        # Aggregate metrics
        agg_metrics = self._aggregate_metrics(all_metrics, weights.cpu().numpy())
        agg_metrics['num_pairs'] = len(pairs)

        return weighted_loss, agg_metrics

    def _compute_timestep_weights(
        self,
        timesteps: list,
        device: torch.device
    ) -> torch.Tensor:
        """Compute weights for each timestep."""
        t = torch.tensor(timesteps, dtype=torch.float32, device=device)

        if self.timestep_weighting == 'uniform':
            weights = torch.ones_like(t)
        elif self.timestep_weighting == 'linear_decay':
            # Higher weight for larger timesteps (earlier in diffusion)
            weights = t / t.max().clamp(min=1)
        elif self.timestep_weighting == 'sqrt_decay':
            weights = torch.sqrt(t / t.max().clamp(min=1))
        else:
            weights = torch.ones_like(t)

        return weights

    def _aggregate_metrics(
        self,
        metrics_list: list,
        weights: 'np.ndarray'
    ) -> Dict[str, float]:
        """Aggregate metrics with weighting."""
        import numpy as np

        if len(metrics_list) == 0:
            return {}

        weights = weights / weights.sum()
        agg_metrics = {}

        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            agg_metrics[key] = float(np.sum(np.array(values) * weights))

        return agg_metrics
