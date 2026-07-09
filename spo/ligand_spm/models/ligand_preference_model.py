"""
Ligand Step-Aware Preference Model

This model learns to predict preference between two ligand molecules at different
denoising timesteps, analogous to the image-based step-aware preference model.
"""

import torch
import torch.nn as nn
from .time_conditioned_ligand_encoder import TimeConditionedLigandEncoder


class LigandPreferenceModel(nn.Module):
    """
    Step-aware preference model for ligand molecules.

    Unlike the image-based SPM which uses CLIP (text-image contrastive learning),
    this model only compares two ligands without conditioning on text/protein.

    Architecture:
        - TimeConditionedLigandEncoder: Encodes ligands with timestep conditioning
        - Preference scoring: Uses learned quality scoring head

    Args:
        hidden_channels: Hidden dimension (default: 128)
        num_filters: Conv filters (default: 128)
        num_interactions: Number of interaction blocks (default: 2)
        edge_channels: Edge feature dimension (default: 128)
        cutoff: Radius graph cutoff (default: 6.0)
        input_dim: Atom feature dimension (default: 10)
        projection_dim: Projection dimension for embeddings (default: 128)
        use_projection: Whether to project embeddings (default: True)
    """

    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=2,
        edge_channels=128,
        cutoff=6.0,
        input_dim=10,
        projection_dim=128,
        use_projection=True,
    ):
        super().__init__()

        self.encoder = TimeConditionedLigandEncoder(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            edge_channels=edge_channels,
            cutoff=cutoff,
            input_dim=input_dim,
        )

        self.use_projection = use_projection
        self.projection_dim = projection_dim

        if use_projection:
            self.projection = nn.Linear(hidden_channels, projection_dim, bias=False)

        # Quality prediction head: predicts a scalar quality score for each ligand
        # This allows the model to learn absolute quality, not just relative similarity
        self.quality_head = nn.Sequential(
            nn.Linear(projection_dim if use_projection else hidden_channels, projection_dim // 2),
            nn.ReLU(),
            nn.Linear(projection_dim // 2, 1)
        )

        # Learnable temperature parameter (similar to CLIP's logit_scale)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07)))

    def get_ligand_features(self, node_attr, pos, batch, time_cond):
        """
        Encode ligand molecules into feature vectors.

        Args:
            node_attr: [N, input_dim] atom features
            pos: [N, 3] atom coordinates
            batch: [N] batch indices
            time_cond: [B] timesteps

        Returns:
            ligand_features: [B, projection_dim] molecule-level features
        """
        # Encode atoms: [N, hidden_channels]
        atom_features = self.encoder(node_attr, pos, batch, time_cond)

        # Global pooling: aggregate atom features to molecule level
        # Use scatter_mean for efficient batched pooling
        from torch_scatter import scatter_mean
        ligand_features = scatter_mean(atom_features, batch, dim=0)  # [B, hidden_channels]

        # Project if needed
        if self.use_projection:
            ligand_features = self.projection(ligand_features)  # [B, projection_dim]

        return ligand_features

    def forward(self, node_attr_0, pos_0, batch_0, node_attr_1, pos_1, batch_1, time_cond,
                return_scores=False):
        """
        Forward pass: encode both ligands and optionally return quality scores.

        Args:
            node_attr_0: [N0, input_dim] atom features for ligand 0
            pos_0: [N0, 3] coordinates for ligand 0
            batch_0: [N0] batch indices for ligand 0
            node_attr_1: [N1, input_dim] atom features for ligand 1
            pos_1: [N1, 3] coordinates for ligand 1
            batch_1: [N1] batch indices for ligand 1
            time_cond: [B] timesteps (same for both ligands in a pair)
            return_scores: Whether to return quality scores

        Returns:
            If return_scores=False:
                ligand_0_features: [B, projection_dim]
                ligand_1_features: [B, projection_dim]
            If return_scores=True:
                ligand_0_features: [B, projection_dim]
                ligand_1_features: [B, projection_dim]
                quality_0: [B, 1] quality scores for ligand 0
                quality_1: [B, 1] quality scores for ligand 1
        """
        ligand_0_features = self.get_ligand_features(node_attr_0, pos_0, batch_0, time_cond)
        ligand_1_features = self.get_ligand_features(node_attr_1, pos_1, batch_1, time_cond)

        if return_scores:
            quality_0 = self.quality_head(ligand_0_features)  # [B, 1]
            quality_1 = self.quality_head(ligand_1_features)  # [B, 1]
            return ligand_0_features, ligand_1_features, quality_0, quality_1
        else:
            return ligand_0_features, ligand_1_features

    def init_adaln_paras(self):
        """Initialize AdaLN parameters (called after model creation)."""
        self.encoder.init_adaln_paras()

    def save_pretrained(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        """Load model from checkpoint."""
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        return model
