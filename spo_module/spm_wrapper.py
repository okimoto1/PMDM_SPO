"""
SPM (Step-aware Preference Model) Wrapper for PMDM.

This module provides a wrapper around the Ligand SPM model to ensure
correct input formatting and seamless integration with SPO training.

Key features:
- Input format validation and conversion
- Timestep normalization
- Batch handling for variable-size molecules
- Caching for efficiency
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import sys
import os

# Add SPO path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SPO'))


class SPMWrapper:
    """
    Wrapper for Ligand SPM to ensure correct input format and integration.

    This wrapper handles:
    - Converting PMDM outputs to SPM input format
    - Timestep formatting (integer vs normalized)
    - Batch index handling
    - Score extraction and normalization

    Usage:
        spm = LigandPreferenceModel(...)
        wrapper = SPMWrapper(spm, num_timesteps=1000)

        # Evaluate single candidate
        score = wrapper.evaluate(atom_features, positions, batch, timestep)

        # Compare two candidates
        win_prob = wrapper.compare(cand1, cand2, timestep)
    """

    def __init__(
        self,
        spm_model: nn.Module,
        num_timesteps: int = 1000,
        normalize_timestep: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SPM wrapper.

        Args:
            spm_model: Trained LigandPreferenceModel instance
            num_timesteps: Total number of diffusion timesteps
            normalize_timestep: Whether to normalize timestep to [0, 1]
            device: Device to run on
        """
        self.spm = spm_model
        self.num_timesteps = num_timesteps
        self.normalize_timestep = normalize_timestep
        self.device = device

        # Set to eval mode and freeze
        self.spm.eval()
        for param in self.spm.parameters():
            param.requires_grad = False

    def to(self, device: torch.device) -> 'SPMWrapper':
        """Move SPM to device."""
        self.spm.to(device)
        self.device = device
        return self

    def _prepare_timestep(
        self,
        timestep: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Prepare timestep tensor for SPM.

        Args:
            timestep: Integer timestep value
            batch_size: Number of samples in batch
            device: Device for tensor

        Returns:
            time_cond: [B] tensor of timesteps
        """
        if self.normalize_timestep:
            # Normalize to [0, 1]
            t = timestep / self.num_timesteps
            time_cond = torch.full((batch_size,), t, dtype=torch.float32, device=device)
        else:
            # Keep as integer
            time_cond = torch.full((batch_size,), timestep, dtype=torch.long, device=device)

        return time_cond

    def evaluate(
        self,
        atom_features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Evaluate quality score for ligand candidates.

        Args:
            atom_features: [N, D] atom features
            positions: [N, 3] atom coordinates
            batch: [N] batch indices
            timestep: Current diffusion timestep

        Returns:
            scores: [B] quality scores for each molecule
        """
        device = atom_features.device
        num_graphs = batch.max().item() + 1

        # Prepare timestep
        time_cond = self._prepare_timestep(timestep, num_graphs, device)

        # Get quality scores (using same ligand for both inputs since we just want scores)
        with torch.no_grad():
            _, _, quality_0, quality_1 = self.spm(
                atom_features, positions, batch,
                atom_features, positions, batch,
                time_cond,
                return_scores=True
            )

            # Average the two quality predictions (should be same)
            scores = (quality_0 + quality_1) / 2.0

        return scores.squeeze(-1)  # [B]

    def evaluate_batch(
        self,
        candidates: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        timestep: int
    ) -> torch.Tensor:
        """
        Evaluate multiple candidates efficiently.

        Args:
            candidates: List of (atom_features, positions, batch) tuples
            timestep: Current diffusion timestep

        Returns:
            scores: [num_candidates, B] quality scores
        """
        all_scores = []

        for atom_features, positions, batch in candidates:
            scores = self.evaluate(atom_features, positions, batch, timestep)
            all_scores.append(scores)

        return torch.stack(all_scores, dim=0)  # [num_candidates, B]

    def compare(
        self,
        atom_features_0: torch.Tensor,
        positions_0: torch.Tensor,
        batch_0: torch.Tensor,
        atom_features_1: torch.Tensor,
        positions_1: torch.Tensor,
        batch_1: torch.Tensor,
        timestep: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compare two ligand candidates.

        Args:
            atom_features_0/1: [N, D] atom features for candidate 0/1
            positions_0/1: [N, 3] positions for candidate 0/1
            batch_0/1: [N] batch indices for candidate 0/1
            timestep: Current diffusion timestep

        Returns:
            prob_0_wins: [B] probability that candidate 0 is better
            score_0: [B] quality score for candidate 0
            score_1: [B] quality score for candidate 1
        """
        device = atom_features_0.device
        num_graphs = batch_0.max().item() + 1

        # Prepare timestep
        time_cond = self._prepare_timestep(timestep, num_graphs, device)

        with torch.no_grad():
            _, _, quality_0, quality_1 = self.spm(
                atom_features_0, positions_0, batch_0,
                atom_features_1, positions_1, batch_1,
                time_cond,
                return_scores=True
            )

            score_0 = quality_0.squeeze(-1)  # [B]
            score_1 = quality_1.squeeze(-1)  # [B]

            # Probability that 0 is better (using logit scale from SPM)
            logit_scale = self.spm.logit_scale.exp()
            diff = logit_scale * (score_0 - score_1)
            prob_0_wins = torch.sigmoid(diff)

        return prob_0_wins, score_0, score_1

    def get_features(
        self,
        atom_features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Get ligand feature embeddings.

        Args:
            atom_features: [N, D] atom features
            positions: [N, 3] atom coordinates
            batch: [N] batch indices
            timestep: Current diffusion timestep

        Returns:
            features: [B, projection_dim] ligand features
        """
        device = atom_features.device
        num_graphs = batch.max().item() + 1
        time_cond = self._prepare_timestep(timestep, num_graphs, device)

        with torch.no_grad():
            features = self.spm.get_ligand_features(
                atom_features, positions, batch, time_cond
            )

        return features

    def select_win_lose(
        self,
        candidates: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        timestep: int,
        threshold: float = 0.1
    ) -> Tuple[List[int], List[int], torch.Tensor]:
        """
        Select win/lose pairs from candidates based on SPM scores.

        Args:
            candidates: List of (atom_features, positions, batch) tuples
            timestep: Current diffusion timestep
            threshold: Minimum score difference to consider valid

        Returns:
            win_indices: [B] indices of winning candidates per sample
            lose_indices: [B] indices of losing candidates per sample
            valid_mask: [B] which samples have valid win/lose pairs
        """
        # Evaluate all candidates
        scores = self.evaluate_batch(candidates, timestep)  # [num_candidates, B]

        # Sort by score (descending)
        sorted_scores, sorted_indices = torch.sort(scores, dim=0, descending=True)

        # Win is highest score, lose is lowest
        win_indices = sorted_indices[0]  # [B]
        lose_indices = sorted_indices[-1]  # [B]

        # Convert to probabilities for threshold check
        probs = torch.softmax(sorted_scores, dim=0)

        # Valid if difference between best and worst is above threshold
        valid_mask = (probs[0] - probs[-1]) > threshold

        return win_indices.tolist(), lose_indices.tolist(), valid_mask


def load_spm_model(
    checkpoint_path: str,
    device: torch.device,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> SPMWrapper:
    """
    Load SPM model from checkpoint and wrap it.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        model_kwargs: Optional model configuration overrides

    Returns:
        Wrapped SPM model ready for use
    """
    # Import LigandPreferenceModel
    try:
        from spo.ligand_spm.models.ligand_preference_model import LigandPreferenceModel
    except ImportError:
        # Fallback path
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spo', 'ligand_spm'))
        from models.ligand_preference_model import LigandPreferenceModel

    # Default model config
    default_kwargs = {
        'hidden_channels': 128,
        'num_filters': 128,
        'num_interactions': 2,
        'edge_channels': 128,
        'cutoff': 6.0,
        'input_dim': 10,
        'projection_dim': 128,
    }

    if model_kwargs:
        default_kwargs.update(model_kwargs)

    # Create model
    model = LigandPreferenceModel(**default_kwargs)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # Move to device
    model.to(device)

    # Wrap and return
    return SPMWrapper(model, device=device)


class DummySPMWrapper:
    """
    Dummy SPM wrapper for testing without a trained model.

    This can be used for:
    - Unit testing the SPO training pipeline
    - Debugging without GPU requirements
    - Comparing against random baseline
    """

    def __init__(self, scoring_method: str = 'random'):
        """
        Args:
            scoring_method: How to generate scores
                - 'random': Random uniform scores
                - 'size': Score based on molecule size
                - 'constant': Constant score
        """
        self.scoring_method = scoring_method

    def to(self, device: torch.device) -> 'DummySPMWrapper':
        return self

    def evaluate(
        self,
        atom_features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """Generate dummy scores."""
        num_graphs = batch.max().item() + 1
        device = atom_features.device

        if self.scoring_method == 'random':
            scores = torch.rand(num_graphs, device=device)
        elif self.scoring_method == 'size':
            # Larger molecules get higher scores
            from torch_scatter import scatter_add
            counts = scatter_add(torch.ones_like(batch, dtype=torch.float), batch)
            scores = counts / counts.max()
        elif self.scoring_method == 'constant':
            scores = torch.ones(num_graphs, device=device)
        else:
            scores = torch.rand(num_graphs, device=device)

        return scores

    def evaluate_batch(
        self,
        candidates: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        timestep: int
    ) -> torch.Tensor:
        """Evaluate multiple candidates."""
        all_scores = []
        for atom_features, positions, batch in candidates:
            scores = self.evaluate(atom_features, positions, batch, timestep)
            all_scores.append(scores)
        return torch.stack(all_scores, dim=0)

    def select_win_lose(
        self,
        candidates: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        timestep: int,
        threshold: float = 0.1
    ) -> Tuple[List[int], List[int], torch.Tensor]:
        """Select win/lose pairs."""
        scores = self.evaluate_batch(candidates, timestep)
        sorted_scores, sorted_indices = torch.sort(scores, dim=0, descending=True)

        win_indices = sorted_indices[0]
        lose_indices = sorted_indices[-1]

        probs = torch.softmax(sorted_scores, dim=0)
        valid_mask = (probs[0] - probs[-1]) > threshold

        return win_indices.tolist(), lose_indices.tolist(), valid_mask
