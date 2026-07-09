"""
Multi-Candidate Sampling for PMDM.

This module provides utilities for generating multiple candidate trajectories
during PMDM sampling. This is essential for SPO training where we need to
compare different denoising paths and select win/lose pairs.

Key features:
- Generate multiple candidates at each timestep
- Support for different sampling strategies
- Efficient batch processing
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from torch_scatter import scatter_mean


@dataclass
class SamplingCandidate:
    """Container for a sampling candidate."""
    ligand_pos: torch.Tensor
    ligand_atom: torch.Tensor
    protein_pos: torch.Tensor
    timestep: int
    candidate_idx: int
    metadata: Optional[Dict] = None


class MultiSamplePMDM:
    """
    Multi-candidate sampling wrapper for PMDM.

    This class wraps a PMDM model to support generating multiple candidate
    trajectories at each denoising step. Used during SPO training to
    create win/lose pairs.

    Usage:
        sampler = MultiSamplePMDM(model, num_candidates=4)

        # Generate candidates for one step
        candidates = sampler.sample_step_multi(
            ligand_pos, ligand_atom, t, protein_ctx, protein_pos, batch
        )

        # Score and select best
        scores = [score_fn(c) for c in candidates]
        best_idx = torch.argmax(torch.stack(scores))
    """

    def __init__(
        self,
        model: nn.Module,
        num_candidates: int = 4,
        sampling_type: str = 'generalized',
        eta: float = 1.0
    ):
        """
        Initialize multi-candidate sampler.

        Args:
            model: PMDM model instance
            num_candidates: Number of candidates to generate per step
            sampling_type: Sampling method ('generalized', 'ddpm_noisy', 'ld')
            eta: DDIM eta parameter (controls stochasticity)
        """
        self.model = model
        self.num_candidates = num_candidates
        self.sampling_type = sampling_type
        self.eta = eta

    def sample_step_multi(
        self,
        ligand_pos: torch.Tensor,
        ligand_atom: torch.Tensor,
        protein_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_ctx: torch.Tensor,
        protein_atom_type: torch.Tensor,
        protein_backbone_mask: Optional[torch.Tensor],
        protein_batch: torch.Tensor,
        timestep_i: int,
        timestep_j: int,
        num_graphs: int,
        ligand_bond_index: Optional[torch.Tensor] = None,
        ligand_bond_type: Optional[torch.Tensor] = None,
        num_node_ctx: Optional[torch.Tensor] = None,
        extend_order: bool = False,
        extend_radius: bool = True,
        is_sidechain: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        w_global_pos: float = 1.0,
        w_local_pos: float = 1.0,
        w_global_node: float = 1.0,
        w_local_node: float = 1.0,
        clip: float = 1000.0,
        clip_local: Optional[float] = None,
        step_lr: float = 1e-6,
        num_candidates: Optional[int] = None,
        **kwargs
    ) -> List[SamplingCandidate]:
        """
        Generate multiple candidates for one denoising step.

        Args:
            ligand_pos: Current ligand positions [N, 3]
            ligand_atom: Current ligand atom features [N, D]
            protein_pos: Protein positions [M, 3]
            ligand_batch: Ligand batch indices [N]
            protein_ctx: Protein context embeddings
            protein_atom_type: Protein atom features
            protein_backbone_mask: Protein backbone mask
            protein_batch: Protein batch indices [M]
            timestep_i: Current timestep
            timestep_j: Next timestep
            num_graphs: Number of graphs in batch
            num_candidates: Override number of candidates (uses self.num_candidates if None)
            ... other arguments passed to _denoise_one_step

        Returns:
            candidates: List of SamplingCandidate objects
        """
        candidates = []
        device = ligand_pos.device

        # Use provided num_candidates or fall back to self.num_candidates
        n_candidates = num_candidates if num_candidates is not None else self.num_candidates

        # Get model parameters
        sigmas = (1.0 - self.model.alphas).sqrt() / self.model.alphas.sqrt()

        for cand_idx in range(n_candidates):
            # Clone current state (each candidate starts from same point)
            pos_clone = ligand_pos.clone()
            atom_clone = ligand_atom.clone()
            prot_pos_clone = protein_pos.clone()

            # Create timestep tensor
            t = torch.full(
                size=(num_graphs,),
                fill_value=timestep_i,
                dtype=torch.long,
                device=device
            )

            # Execute one denoising step with new random noise
            pos_next, atom_next, prot_next = self._denoise_step(
                pos_clone, atom_clone, prot_pos_clone,
                ligand_batch, protein_ctx, protein_atom_type,
                protein_backbone_mask, protein_batch,
                t, timestep_i, timestep_j, num_graphs,
                ligand_bond_index, ligand_bond_type,
                num_node_ctx, extend_order, extend_radius,
                is_sidechain, context,
                w_global_pos, w_local_pos, w_global_node, w_local_node,
                clip, clip_local, step_lr, sigmas,
                **kwargs
            )

            candidates.append(SamplingCandidate(
                ligand_pos=pos_next,
                ligand_atom=atom_next,
                protein_pos=prot_next,
                timestep=timestep_i,
                candidate_idx=cand_idx,
                metadata={'timestep_j': timestep_j}
            ))

        return candidates

    def _denoise_step(
        self,
        ligand_pos: torch.Tensor,
        ligand_atom: torch.Tensor,
        protein_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_ctx: torch.Tensor,
        protein_atom_type: torch.Tensor,
        protein_backbone_mask: Optional[torch.Tensor],
        protein_batch: torch.Tensor,
        t: torch.Tensor,
        timestep_i: int,
        timestep_j: int,
        num_graphs: int,
        ligand_bond_index: Optional[torch.Tensor],
        ligand_bond_type: Optional[torch.Tensor],
        num_node_ctx: Optional[torch.Tensor],
        extend_order: bool,
        extend_radius: bool,
        is_sidechain: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        w_global_pos: float,
        w_local_pos: float,
        w_global_node: float,
        w_local_node: float,
        clip: float,
        clip_local: Optional[float],
        step_lr: float,
        sigmas: torch.Tensor,
        local_start_sigma: float = float('inf'),
        global_start_sigma: float = float('inf'),
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one denoising step.

        This is adapted from PMDM's _denoise_one_step method.
        """
        device = ligand_pos.device

        # Helper function
        def compute_alpha(beta, t_val):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t_val + 1)
            return a

        # NOTE: protein_ctx is already encoded by protein_encoder in sample_and_collect_pairs
        # It has shape [N_protein, hidden_dim=128], not raw features
        # Do NOT call protein_encoder again here

        # Get model prediction
        net_out = self.model.net(
            ligand_atom_type=ligand_atom,
            ligand_pos=ligand_pos,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            ligand_batch=ligand_batch,
            protein_embeddings=protein_ctx,  # Already encoded embeddings
            time_step=t,
            num_node_ctx=num_node_ctx,
            protein_atom_feature=protein_atom_type,
            protein_pos=protein_pos,
            protein_backbone_mask=protein_backbone_mask,
            protein_batch=protein_batch,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
            is_sidechain=is_sidechain,
            property_context=context,
            vae_noise=None,
        )

        # Extract predictions
        if self.model.vae_context:
            (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
             edge_index, edge_type, edge_length, local_edge_mask) = net_out[:-1]
        else:
            (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
             edge_index, edge_type, edge_length, local_edge_mask) = net_out

        # Apply sigma-dependent weighting
        if sigmas[timestep_i] < local_start_sigma:
            node_eq_local = pos_eq_local
            if clip_local is not None:
                node_eq_local = self._clip_norm(node_eq_local, limit=clip_local)
        else:
            node_eq_local = 0
            node_score_local = 0

        if sigmas[timestep_i] < global_start_sigma:
            node_eq_global = pos_eq_global
            node_eq_global = self._clip_norm(node_eq_global, limit=clip)
        else:
            node_eq_global = 0
            node_score_global = 0

        # Combine predictions
        eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global
        eps_node = w_local_node * node_score_local + w_global_node * node_score_global

        # Generate random noise for this candidate
        noise = torch.randn_like(ligand_pos)
        noise_node = torch.randn_like(ligand_atom)

        # Get alpha values
        b = self.model.betas
        t_val = t[0]
        next_t = torch.tensor([timestep_j], device=device)
        at = compute_alpha(b, t_val.long())
        at_next = compute_alpha(b, next_t.long())

        # Apply sampling step based on sampling type
        if self.sampling_type == "generalized":
            pos_next, atom_next = self._generalized_step(
                ligand_pos, ligand_atom, eps_pos, eps_node,
                noise, noise_node, at, at_next, sigmas[timestep_i], step_lr
            )
        elif self.sampling_type == "ddpm_noisy":
            pos_next, atom_next = self._ddpm_step(
                ligand_pos, ligand_atom, eps_pos, eps_node,
                noise, noise_node, at, at_next, t_val
            )
        elif self.sampling_type == "ld":
            pos_next, atom_next = self._ld_step(
                ligand_pos, ligand_atom, eps_pos, eps_node,
                noise, noise_node, at, sigmas[timestep_i], step_lr
            )
        else:
            raise ValueError(f"Unknown sampling type: {self.sampling_type}")

        # Center positions
        pos_next, protein_pos = self._center_pos(
            pos_next, protein_pos, ligand_batch, protein_batch
        )

        return pos_next, atom_next, protein_pos

    def _generalized_step(
        self,
        ligand_pos, ligand_atom, eps_pos, eps_node,
        noise, noise_node, at, at_next, sigma_i, step_lr
    ):
        """Generalized (DDIM-style) sampling step."""
        eta = self.eta
        et = -eps_pos
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1**2).sqrt()

        step_size_pos_ld = step_lr * (sigma_i / 0.01) ** 2 / sigma_i
        step_size_pos_generalized = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
        step_size_pos = min(step_size_pos_ld, step_size_pos_generalized)

        step_size_noise_ld = torch.sqrt((step_lr * (sigma_i / 0.01) ** 2) * 2)
        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
        step_size_noise = min(step_size_noise_ld, step_size_noise_generalized)

        eps_node = eps_node / (1 - at).sqrt()
        pos_next = ligand_pos - et * step_size_pos + noise * step_size_noise
        atom_next = ligand_atom - eps_node * step_size_pos + noise_node * step_size_noise

        return pos_next, atom_next

    def _ddpm_step(
        self,
        ligand_pos, ligand_atom, eps_pos, eps_node,
        noise, noise_node, at, at_next, t_val
    ):
        """DDPM-style sampling step."""
        atm1 = at_next
        beta_t = 1 - at / atm1
        e = -eps_pos

        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
        mask = 1 - (t_val == 0).float()
        logvar = beta_t.log()
        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

        e = eps_node
        node0_from_e = (1.0 / at).sqrt() * ligand_atom - (1.0 / at - 1).sqrt() * e
        mean_eps = ((atm1.sqrt() * beta_t) * node0_from_e +
                    ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom) / (1.0 - at)
        atom_next = mean_eps + mask * torch.exp(0.5 * logvar) * noise_node

        return pos_next, atom_next

    def _ld_step(
        self,
        ligand_pos, ligand_atom, eps_pos, eps_node,
        noise, noise_node, at, sigma_i, step_lr
    ):
        """Langevin dynamics sampling step."""
        step_size = step_lr * (sigma_i / 0.01) ** 2
        pos_next = ligand_pos + step_size * eps_pos / sigma_i + noise * torch.sqrt(step_size * 2)
        eps_node = eps_node / (1 - at).sqrt()
        atom_next = ligand_atom - step_size * eps_node / sigma_i + noise_node * torch.sqrt(step_size * 2)

        return pos_next, atom_next

    def _clip_norm(self, vec: torch.Tensor, limit: float) -> torch.Tensor:
        """Clip vector norm."""
        norm = torch.norm(vec, dim=-1, keepdim=True)
        scale = torch.clamp(limit / (norm + 1e-8), max=1.0)
        return vec * scale

    def _center_pos(
        self,
        ligand_pos: torch.Tensor,
        protein_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        protein_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Center positions around ligand center of mass.

        This follows PMDM's center_pos_pl function:
        - Ligand is centered to its own center of mass
        - Protein is shifted by the same ligand center of mass

        This ensures both ligand and protein remain aligned relative to each other.
        """
        # Compute center of mass for ligand (per graph)
        ligand_com = scatter_mean(ligand_pos, ligand_batch, dim=0)  # [num_graphs, 3]

        # Center ligand around its own COM
        ligand_pos_centered = ligand_pos - ligand_com[ligand_batch]

        # Shift protein by the same ligand COM
        # This maintains the relative positioning between ligand and protein
        protein_pos_centered = protein_pos - ligand_com[protein_batch]

        return ligand_pos_centered, protein_pos_centered


def sample_trajectory_multi(
    model: nn.Module,
    initial_pos: torch.Tensor,
    initial_atom: torch.Tensor,
    protein_data: Dict,
    num_candidates: int,
    divert_start_step: int,
    score_fn: Callable,
    n_steps: int = 0,  # 0 means use model's num_timesteps (default: 1000)
    **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Sample a complete trajectory with multi-candidate selection.

    This function runs the full denoising process, generating multiple
    candidates at each step after divert_start_step and selecting
    based on score_fn.

    Args:
        model: PMDM model
        initial_pos: Initial noisy positions
        initial_atom: Initial noisy atom features
        protein_data: Dictionary with protein information
        num_candidates: Number of candidates per step
        divert_start_step: Timestep to start generating multiple candidates
        score_fn: Function to score candidates
        n_steps: Number of denoising steps (0 = use model.num_timesteps)
        **sampling_kwargs: Additional sampling arguments

    Returns:
        final_pos: Final generated positions
        final_atom: Final generated atom features
        decisions: List of selection decisions at each step
    """
    sampler = MultiSamplePMDM(
        model=model,
        num_candidates=num_candidates,
        **sampling_kwargs
    )

    decisions = []

    # Handle n_steps=0 (use model default)
    if n_steps == 0:
        n_steps = model.num_timesteps

    # Setup
    skip = model.num_timesteps // n_steps
    seq = list(range(0, model.num_timesteps, skip))
    seq_next = [-1] + seq[:-1]
    seq_pairs = list(zip(reversed(seq), reversed(seq_next)))

    ligand_pos = initial_pos
    ligand_atom = initial_atom

    for idx, (t_i, t_j) in enumerate(seq_pairs):
        if t_i >= divert_start_step:
            # Generate multiple candidates
            candidates = sampler.sample_step_multi(
                ligand_pos=ligand_pos,
                ligand_atom=ligand_atom,
                timestep_i=t_i,
                timestep_j=t_j,
                **protein_data
            )

            # Score candidates
            scores = [score_fn(c) for c in candidates]
            scores_tensor = torch.stack(scores)

            # Select best candidate
            best_idx = scores_tensor.argmax()
            best_candidate = candidates[best_idx]

            # Record decision
            decisions.append({
                'timestep': t_i,
                'scores': scores_tensor.detach().cpu(),
                'selected_idx': best_idx.item(),
                'win_idx': best_idx.item(),
                'lose_idx': scores_tensor.argmin().item()
            })

            ligand_pos = best_candidate.ligand_pos
            ligand_atom = best_candidate.ligand_atom

        else:
            # Single sample step
            candidates = sampler.sample_step_multi(
                ligand_pos=ligand_pos,
                ligand_atom=ligand_atom,
                timestep_i=t_i,
                timestep_j=t_j,
                num_candidates=1,  # Only one candidate
                **protein_data
            )
            ligand_pos = candidates[0].ligand_pos
            ligand_atom = candidates[0].ligand_atom

    return ligand_pos, ligand_atom, decisions
