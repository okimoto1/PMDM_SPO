"""
Time-Conditioned Ligand Encoder for Step-Aware Preference Model

Adapts the LigandEncoder (SchNet-based) to accept timestep conditioning,
similar to how CLIP vision encoder is adapted in the image-based SPM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear
from torch_geometric.nn import MessagePassing, radius_graph


def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    # x: [N, D], shift/scale: [N, D]
    return x * (1 + scale) + shift


class GaussianSmearing(Module):
    """Gaussian smearing for distance encoding."""

    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(Module):
    """Shifted softplus activation function."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """Continuous-filter convolution layer from SchNet."""

    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0, smooth=False):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.cutoff = cutoff
        self.smooth = smooth

    def forward(self, x, edge_index, edge_length, edge_attr):
        W = self.nn(edge_attr)

        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * math.pi / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
        else:
            C = (edge_length <= self.cutoff).float()
        W = W * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class TimeConditionedInteractionBlock(Module):
    """
    Time-conditioned SchNet interaction block.

    Similar to TimeConditionedCLIPEncoderLayer, this adds AdaLN modulation
    based on timestep embeddings.
    """

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth=False):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        # AdaLN modulation: timestep -> shift, scale, gate
        self.adaLN_modulation = Sequential(
            nn.SiLU(),
            Linear(hidden_channels, 3 * hidden_channels, bias=True),
        )

    def forward(self, x, edge_index, edge_length, edge_attr, time_cond):
        """
        Args:
            x: [N, hidden_channels] node features
            edge_index: [2, E] edge indices
            edge_length: [E] edge lengths
            edge_attr: [E, edge_channels] edge attributes
            time_cond: [N, hidden_channels] timestep conditioning per node
        """
        # Get modulation parameters
        shift, scale, gate = self.adaLN_modulation(time_cond).chunk(3, dim=1)

        # Apply modulation before convolution
        x_mod = modulate(x, shift, scale)

        # Convolution
        x_conv = self.conv(x_mod, edge_index, edge_length, edge_attr)
        x_conv = self.act(x_conv)
        x_conv = self.lin(x_conv)

        # Apply gate and residual
        x = x + x_conv * gate

        return x


class TimestepEmbedder(Module):
    """
    Embeds scalar timesteps into vector representations.
    Uses sinusoidal positional encoding similar to Transformer and DiT.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = Sequential(
            Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: [N] tensor of timestep indices (0 to num_timesteps-1)
            dim: dimension of output
            max_period: controls minimum frequency

        Returns:
            embedding: [N, dim] tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        Args:
            t: [N] or [B] tensor of timesteps

        Returns:
            t_emb: [N, hidden_size] or [B, hidden_size] embeddings
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeConditionedLigandEncoder(Module):
    """
    Time-conditioned ligand encoder for step-aware preference learning.

    This is the ligand equivalent of TimeConditionedCLIPVisionTransformer.
    It adds timestep conditioning to the SchNet-based ligand encoder using
    adaptive layer normalization (AdaLN).

    Args:
        hidden_channels: Hidden dimension size (default: 128)
        num_filters: Number of filters in convolution (default: 128)
        num_interactions: Number of interaction blocks (default: 2)
        edge_channels: Edge feature dimension (default: 128)
        cutoff: Cutoff distance for radius graph (default: 6.0)
        input_dim: Input atom feature dimension (default: 10)
        frequency_embedding_size: Timestep embedding frequency dimension (default: 256)

    Input:
        node_attr: [N, input_dim] - Atom type features (one-hot encoded)
        pos: [N, 3] - 3D coordinates of atoms
        batch: [N] - Batch indices for each atom
        time_cond: [B] - Timestep for each molecule in batch (0 to num_timesteps-1)

    Output:
        h: [N, hidden_channels] - Time-conditioned encoded atom features
    """

    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=2,
        edge_channels=128,
        cutoff=6.0,
        input_dim=10,
        frequency_embedding_size=256,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.input_dim = input_dim
        self.cutoff = cutoff

        # Distance expansion for edge features
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)

        # Input embedding layer
        self.emblin = Linear(self.input_dim, hidden_channels)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_channels, frequency_embedding_size)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Time-conditioned interaction blocks
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = TimeConditionedInteractionBlock(
                hidden_channels, edge_channels, num_filters, cutoff, smooth=True
            )
            self.interactions.append(block)

        # Final AdaLN modulation for global pooling
        self.final_adaLN_modulation = Sequential(
            nn.SiLU(),
            Linear(hidden_channels, 2 * hidden_channels, bias=True)
        )
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch, time_cond):
        """
        Forward pass of the time-conditioned ligand encoder.

        Args:
            node_attr: [N, input_dim] - Atom features
            pos: [N, 3] - Atom coordinates
            batch: [N] - Batch assignment for each atom
            time_cond: [B] - Timestep for each molecule in batch

        Returns:
            h: [N, hidden_channels] - Encoded features
        """
        # Build radius graph
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)

        # Compute edge lengths
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)

        # Expand distances to edge features
        edge_attr = self.distance_expansion(edge_length)

        # Embed timesteps: [B, hidden_channels]
        t_emb = self.t_embedder(time_cond)

        # Broadcast timestep embedding to all atoms in each molecule: [N, hidden_channels]
        time_cond_per_atom = t_emb[batch]

        # Initial embedding
        h = self.emblin(node_attr)

        # Apply time-conditioned interaction blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_length, edge_attr, time_cond_per_atom)

        # Apply final modulation (similar to CLIP's pooler output modulation)
        shift, scale = self.final_adaLN_modulation(time_cond_per_atom).chunk(2, dim=1)
        h = h * (1 + scale) + shift

        return h

    def init_adaln_paras(self):
        """Initialize AdaLN parameters (called after loading pretrained weights)."""
        for interaction in self.interactions:
            # Init adaLN_modulation in each interaction block
            nn.init.constant_(interaction.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(interaction.adaLN_modulation[-1].bias, 0)
            # Set initial bias so scale starts at 1 and shift at 0
            bias = torch.zeros(3 * interaction.hidden_channels,
                             dtype=interaction.adaLN_modulation[-1].bias.dtype)
            with torch.no_grad():
                # Scale starts at 1 (index 1 of the 3 chunks)
                bias[interaction.hidden_channels: 2 * interaction.hidden_channels] = 1
                # Gate starts at 1 (index 2 of the 3 chunks)
                bias[2 * interaction.hidden_channels:] = 1
                interaction.adaLN_modulation[-1].bias = nn.Parameter(bias)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize final modulation
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)
