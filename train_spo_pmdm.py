#!/usr/bin/env python
"""
SPO-PMDM Training Script

Train PMDM using Step-aware Preference Optimization (SPO) with a trained Ligand SPM.

This script implements the SPO training pipeline:
1. Load pre-trained PMDM and frozen reference model
2. Load trained Ligand SPM for candidate evaluation
3. For each batch:
   - Sample trajectories with multiple candidates at each timestep
   - Use SPM to score candidates and select win/lose pairs
   - Compute DPO loss and update PMDM

Usage:
    python train_spo_pmdm.py \
        --pmdm_ckpt 500.pt \
        --spm_ckpt SPO/ligand_spm/checkpoints/best_model.pt \
        --config configs/crossdock_epoch.yml \
        --output_dir logs/spo_pmdm
"""

import os
import sys
import argparse
import shutil
import copy
import warnings
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
import yaml
from easydict import EasyDict

# Suppress RDKit and OpenBabel warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Suppress OpenBabel warnings by redirecting stderr temporarily during molecule building
import openbabel
openbabel.obErrorLog.SetOutputLevel(openbabel.obError)  # Only show errors, not warnings

# Local imports
from models.epsnet import get_model
from utils.datasets import get_dataset
from utils.transforms import (
    FeaturizeProteinAtom, FeaturizeLigandAtom, FeaturizeLigandBond,
    CountNodesPerGraph, GetAdj, Compose
)
from utils.misc import seed_all, get_new_log_dir, get_logger

# SPO modules
from spo_module import (
    SPOLoss,
    ReferenceModelManager,
    MultiSamplePMDM,
    SPMWrapper,
    compute_log_prob_pmdm,
    SPOTensorBoardLogger,
    TrainingHealthMonitor,
)

# For combined scoring (QED, SA, clash)
from rdkit import Chem
from rdkit.Chem import QED as RDKitQED
from evaluation.sascorer import compute_sa_score
from configs.dataset_config import get_dataset_info
from utils.reconstruct_mdm import make_mol_openbabel


# ============================================================
# GPU Memory Monitoring Utilities
# ============================================================
def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated,
        }
    return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}


def log_gpu_memory(tag, logger=None):
    """Log GPU memory usage with a tag."""
    mem = get_gpu_memory_mb()
    msg = f"[GPU MEM] {tag}: allocated={mem['allocated_mb']:.1f}MB, reserved={mem['reserved_mb']:.1f}MB, max={mem['max_allocated_mb']:.1f}MB"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return mem


def reset_gpu_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class GPUMemoryTracker:
    """Track GPU memory at different stages."""

    def __init__(self, logger=None, enabled=True):
        self.logger = logger
        self.enabled = enabled
        self.checkpoints = []

    def checkpoint(self, tag):
        """Record a memory checkpoint."""
        if not self.enabled:
            return
        mem = get_gpu_memory_mb()
        self.checkpoints.append({
            'tag': tag,
            **mem
        })
        msg = f"[GPU] {tag}: {mem['allocated_mb']:.1f}MB allocated, {mem['reserved_mb']:.1f}MB reserved"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return mem

    def report(self):
        """Print a summary report of all checkpoints."""
        if not self.checkpoints:
            return
        print("\n" + "="*60)
        print("GPU Memory Report")
        print("="*60)
        for cp in self.checkpoints:
            print(f"  {cp['tag']:40s}: {cp['allocated_mb']:8.1f} MB")
        print("="*60 + "\n")

    def clear(self):
        """Clear checkpoints."""
        self.checkpoints = []


class CombinedScorer:
    """
    Combined scoring function for SPO training.

    Computes: w_spm * SPM + w_qed * QED + w_sa * SA + w_clash * clash_penalty

    Args:
        spm_wrapper: SPM model wrapper
        weight_spm: Weight for SPM score
        weight_qed: Weight for QED score
        weight_sa: Weight for SA score
        weight_clash: Weight for clash penalty
        clash_threshold: Distance threshold for clash (Angstrom)
        dataset_info: Dataset info for molecule reconstruction
        atomic_numbers: Atomic numbers for element mapping
    """

    def __init__(
        self,
        spm_wrapper,
        weight_spm: float = 1.0,
        weight_qed: float = 0.0,
        weight_sa: float = 0.0,
        weight_clash: float = 0.0,
        clash_threshold: float = 3.0,
        dataset_info=None,
        atomic_numbers=None,
    ):
        self.spm_wrapper = spm_wrapper
        self.weight_spm = weight_spm
        self.weight_qed = weight_qed
        self.weight_sa = weight_sa
        self.weight_clash = weight_clash
        self.clash_threshold = clash_threshold
        self.dataset_info = dataset_info
        self.atomic_numbers = atomic_numbers

        # Check if we need molecular metrics
        self.use_mol_metrics = (weight_qed > 0 or weight_sa > 0)

    def compute_qed(self, mol):
        """Compute QED score for a molecule."""
        try:
            if mol is None:
                return 0.0
            return RDKitQED.qed(mol)
        except:
            return 0.0

    def compute_sa(self, mol):
        """Compute normalized SA score (higher = better)."""
        try:
            if mol is None:
                return 0.0
            # compute_sa_score returns (sa_raw, sa_norm)
            # sa_norm = (10 - sa_raw) / 9, already normalized to [0, 1]
            _, sa_norm = compute_sa_score(mol)
            return sa_norm
        except:
            return 0.0

    def compute_clash(self, ligand_pos, protein_pos, ligand_batch, protein_batch):
        """
        Compute clash penalty based on ligand-protein distances.

        Formula: clash = -max(threshold - min_dist, 0)
        where min_dist is the minimum distance from any ligand atom to any protein atom.

        Returns negative penalty (closer = more negative = worse).
        """
        import torch

        clash_scores = []
        num_graphs = ligand_batch.max().item() + 1

        for g in range(num_graphs):
            lig_mask = (ligand_batch == g)
            prot_mask = (protein_batch == g)

            lig_pos = ligand_pos[lig_mask]
            prot_pos = protein_pos[prot_mask]

            if len(lig_pos) == 0 or len(prot_pos) == 0:
                clash_scores.append(0.0)
                continue

            # Step 1: Compute pairwise distances
            # lig_pos: [N_lig, 3], prot_pos: [N_prot, 3]
            dist_matrix = torch.cdist(lig_pos, prot_pos)  # [N_lig, N_prot]

            # Step 2: For each ligand atom, find min distance to any protein atom
            min_dists_per_ligand, _ = dist_matrix.min(dim=1)  # [N_lig]

            # Step 3: Find the overall minimum distance
            min_dist = min_dists_per_ligand.min().item()

            # Step 4: Compute penalty: -max(threshold - min_dist, 0)
            # If min_dist >= threshold: penalty = 0 (no clash)
            # If min_dist < threshold: penalty = -(threshold - min_dist) (negative)
            clash_penalty = -max(self.clash_threshold - min_dist, 0.0)
            clash_scores.append(clash_penalty)

        return torch.tensor(clash_scores, device=ligand_pos.device)

    def build_molecules(self, atom_features, positions, batch):
        """
        Try to build molecules from atom features and positions.

        Returns list of (mol, success) tuples per graph.
        """
        import torch

        num_graphs = batch.max().item() + 1
        molecules = []

        if self.dataset_info is None or self.atomic_numbers is None:
            return [(None, False) for _ in range(num_graphs)]

        for g in range(num_graphs):
            mask = (batch == g)
            pos = positions[mask]
            atom = atom_features[mask]

            try:
                # Get element indices
                num_atom_type = len(self.atomic_numbers)
                new_element = torch.argmax(atom[:, :num_atom_type], dim=1)

                # Build molecule
                mol, _ = make_mol_openbabel(pos.cpu(), new_element.cpu(), self.dataset_info)

                if mol is not None:
                    smile = Chem.MolToSmiles(mol)
                    if smile is not None and len(smile) >= 4 and "." not in smile:
                        molecules.append((mol, True))
                    else:
                        molecules.append((None, False))
                else:
                    molecules.append((None, False))

            except Exception as e:
                molecules.append((None, False))

        return molecules

    def score_candidates(self, candidates, timestep, protein_pos, protein_batch):
        """
        Score multiple candidates with combined metrics.

        Uses fallback logic similar to compare_spo_v2.py:
        - If all candidates successfully build molecules: use full combined score
        - If any candidate fails: fallback to SPM + clash only (no QED/SA)

        Args:
            candidates: List of (atom_features, positions, batch) tuples
            timestep: Current diffusion timestep
            protein_pos: Protein positions
            protein_batch: Protein batch indices

        Returns:
            scores: [num_candidates, num_graphs] combined scores
            mol_success_rate: float, fraction of successful molecule builds
        """
        import torch

        num_candidates = len(candidates)
        if num_candidates == 0:
            return torch.zeros(0), 0.0

        num_graphs = candidates[0][2].max().item() + 1
        device = candidates[0][0].device

        # First pass: compute SPM and clash for all candidates (always available)
        spm_scores_all = torch.zeros(num_candidates, num_graphs, device=device)
        clash_scores_all = torch.zeros(num_candidates, num_graphs, device=device)

        for c_idx, (atom_features, positions, batch) in enumerate(candidates):
            # SPM score
            if self.weight_spm > 0 and self.spm_wrapper is not None:
                spm_scores = self.spm_wrapper.evaluate(
                    atom_features, positions, batch, timestep
                )
                spm_scores_all[c_idx] = spm_scores

            # Clash penalty
            if self.weight_clash > 0:
                clash_scores = self.compute_clash(
                    positions, protein_pos, batch, protein_batch
                )
                clash_scores_all[c_idx] = clash_scores

        # Second pass: try to build molecules and compute QED/SA
        qed_scores_all = torch.zeros(num_candidates, num_graphs, device=device)
        sa_scores_all = torch.zeros(num_candidates, num_graphs, device=device)
        mol_success = torch.ones(num_candidates, num_graphs, dtype=torch.bool, device=device)

        if self.use_mol_metrics:
            for c_idx, (atom_features, positions, batch) in enumerate(candidates):
                molecules = self.build_molecules(atom_features, positions, batch)

                for g_idx, (mol, success) in enumerate(molecules):
                    if success and mol is not None:
                        if self.weight_qed > 0:
                            qed_scores_all[c_idx, g_idx] = self.compute_qed(mol)
                        if self.weight_sa > 0:
                            sa_scores_all[c_idx, g_idx] = self.compute_sa(mol)
                    else:
                        mol_success[c_idx, g_idx] = False

        # Per-graph fallback logic: check if ALL candidates succeeded for each graph
        all_scores = torch.zeros(num_candidates, num_graphs, device=device)
        total_success = 0
        total_attempts = 0

        for g_idx in range(num_graphs):
            all_candidates_success = mol_success[:, g_idx].all().item()

            if all_candidates_success and self.use_mol_metrics:
                # Full combined score: w_spm*SPM + w_qed*QED + w_sa*SA + w_clash*clash
                for c_idx in range(num_candidates):
                    all_scores[c_idx, g_idx] = (
                        self.weight_spm * spm_scores_all[c_idx, g_idx] +
                        self.weight_qed * qed_scores_all[c_idx, g_idx] +
                        self.weight_sa * sa_scores_all[c_idx, g_idx] +
                        self.weight_clash * clash_scores_all[c_idx, g_idx]
                    )
                total_success += num_candidates
            else:
                # Fallback: only SPM + clash (no QED/SA)
                for c_idx in range(num_candidates):
                    all_scores[c_idx, g_idx] = (
                        self.weight_spm * spm_scores_all[c_idx, g_idx] +
                        self.weight_clash * clash_scores_all[c_idx, g_idx]
                    )
            total_attempts += num_candidates

        mol_success_rate = total_success / max(total_attempts, 1)
        return all_scores, mol_success_rate

    def select_win_lose(self, candidates, timestep, threshold, protein_pos, protein_batch):
        """
        Select win/lose pairs from candidates based on combined scores.

        Args:
            candidates: List of (atom_features, positions, batch) tuples
            timestep: Current diffusion timestep
            threshold: Minimum score difference for valid pair
            protein_pos: Protein positions
            protein_batch: Protein batch indices

        Returns:
            win_indices: List of winning candidate indices per graph
            lose_indices: List of losing candidate indices per graph
            valid_mask: Boolean tensor indicating valid pairs
            mol_success_rate: float, fraction of successful molecule builds
        """
        import torch

        scores, mol_success_rate = self.score_candidates(
            candidates, timestep, protein_pos, protein_batch
        )
        # scores: [num_candidates, num_graphs]

        # Sort by score (descending)
        sorted_scores, sorted_indices = torch.sort(scores, dim=0, descending=True)

        # Win is highest, lose is lowest
        win_indices = sorted_indices[0]  # [num_graphs]
        lose_indices = sorted_indices[-1]  # [num_graphs]

        # Valid if score difference is above threshold
        score_diff = sorted_scores[0] - sorted_scores[-1]
        valid_mask = score_diff > threshold

        return win_indices.tolist(), lose_indices.tolist(), valid_mask, mol_success_rate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SPO Training for PMDM')

    # Model paths
    parser.add_argument('--pmdm_ckpt', type=str, required=True,
                        help='Path to pre-trained PMDM checkpoint')
    parser.add_argument('--spm_ckpt', type=str, default=None,
                        help='Path to trained SPM checkpoint (optional if using dummy)')
    parser.add_argument('--config', type=str, default='configs/crossdock_epoch.yml',
                        help='PMDM configuration file')

    # SPO parameters
    parser.add_argument('--num_candidates', type=int, default=5,
                        help='Number of candidates per denoising step')
    parser.add_argument('--divert_start_step', type=int, default=900,
                        help='Timestep to start generating multiple candidates')
    parser.add_argument('--spo_interval', type=int, default=5,
                        help='Generate candidates every N steps')
    parser.add_argument('--score_threshold', type=float, default=0.05,
                        help='Minimum score difference for valid win/lose pair')

    # DPO parameters
    parser.add_argument('--beta', type=float, default=10.0,
                        help='DPO temperature parameter')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Probability ratio clipping range')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='DDIM eta parameter')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')

    # Sampling parameters
    parser.add_argument('--n_steps', type=int, default=0,
                        help='Number of denoising steps during sampling (0 = use model default, typically 1000)')
    parser.add_argument('--sampling_type', type=str, default='generalized',
                        choices=['generalized', 'ddpm_noisy', 'ld'],
                        help='Sampling method')

    # Logging and checkpoints
    parser.add_argument('--output_dir', type=str, default='logs/spo_pmdm',
                        help='Output directory')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N batches')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--use_dummy_spm', action='store_true',
                        help='Use dummy SPM for testing')
    parser.add_argument('--profile_memory', action='store_true',
                        help='Enable GPU memory profiling for the first batch')

    # ========== Combined scoring weights ==========
    parser.add_argument('--score_weight_spm', type=float, default=1.0,
                        help='Weight for SPM score in combined scoring')
    parser.add_argument('--score_weight_qed', type=float, default=0.0,
                        help='Weight for QED score in combined scoring')
    parser.add_argument('--score_weight_sa', type=float, default=0.0,
                        help='Weight for SA score in combined scoring')
    parser.add_argument('--score_weight_clash', type=float, default=0.0,
                        help='Weight for clash penalty in combined scoring')
    parser.add_argument('--clash_threshold', type=float, default=3.0,
                        help='Distance threshold for clash penalty (Angstrom)')

    return parser.parse_args()


def load_pmdm_model(ckpt_path, config, device):
    """Load pre-trained PMDM model."""
    print(f"Loading PMDM from {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Get config from checkpoint if available
    if 'config' in ckpt:
        model_config = ckpt['config'].model
    else:
        model_config = config.model

    # Create model
    model = get_model(model_config)

    # Load weights
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    return model


def load_spm_model(ckpt_path, device, use_dummy=False):
    """Load trained SPM model."""
    if use_dummy or ckpt_path is None:
        print("Using dummy SPM (random scoring)")
        from spo.spm_wrapper import DummySPMWrapper
        return DummySPMWrapper(scoring_method='random')

    print(f"Loading SPM from {ckpt_path}")

    # Import LigandPreferenceModel
    try:
        from spo.ligand_spm.models.ligand_preference_model import LigandPreferenceModel
    except ImportError:
        print("Warning: Could not import LigandPreferenceModel, using dummy SPM")
        from spo.spm_wrapper import DummySPMWrapper
        return DummySPMWrapper(scoring_method='random')

    # Create model
    model = LigandPreferenceModel(
        hidden_channels=128,
        num_filters=128,
        num_interactions=2,
        edge_channels=128,
        cutoff=6.0,
        input_dim=10,
        projection_dim=128,
    )

    # Load weights
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)

    # Wrap
    return SPMWrapper(model, device=device)


def create_dataloader(config, args):
    """Create training data loader."""
    # Feature transforms
    pocket = True
    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=pocket)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=pocket)

    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        CountNodesPerGraph(),
        GetAdj(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(config=config.dataset, transform=transform)
    train_set = subsets['train']

    # Create loader
    follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        follow_batch=follow_batch,
        exclude_keys=collate_exclude_keys,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader


def sample_and_collect_pairs(
    model,
    ref_manager,
    combined_scorer,
    multi_sampler,
    batch,
    args,
    device,
    mem_tracker=None  # Optional GPU memory tracker
):
    """
    Sample trajectories and collect win/lose pairs for SPO training.

    Args:
        model: PMDM model
        ref_manager: Reference model manager
        combined_scorer: CombinedScorer for scoring candidates
        multi_sampler: MultiSamplePMDM for generating candidates
        batch: Training batch
        args: Training arguments
        device: Device
        mem_tracker: Optional GPUMemoryTracker for debugging

    Returns:
        pairs: List of dictionaries with win/lose pair information
        stats: Dictionary with sampling statistics
    """
    from torch_scatter import scatter_mean

    if mem_tracker:
        mem_tracker.checkpoint("sample_start")

    pairs = []
    stats = {
        'num_candidates_generated': 0,
        'num_valid_pairs': 0,
        'avg_score_diff': 0.0,
        'mol_success_rate': 0.0,  # Fraction using full scoring (QED+SA available)
        'num_scoring_events': 0,
    }

    num_graphs = batch.ligand_element_batch.max().item() + 1

    # Initialize noise with proper dimensions
    # ligand_atom should match batch.ligand_atom_feature dimensions
    ligand_pos = torch.randn_like(batch.ligand_pos)
    ligand_atom = torch.randn_like(batch.ligand_atom_feature.float())

    # Center ligand noise around protein center of mass (following PMDM's initialization)
    # This aligns the initial noisy ligand with the protein pocket
    protein_com = scatter_mean(batch.protein_pos, batch.protein_element_batch, dim=0)
    ligand_com = scatter_mean(ligand_pos, batch.ligand_element_batch, dim=0)
    # Shift ligand to protein center
    ligand_pos = ligand_pos - ligand_com[batch.ligand_element_batch] + protein_com[batch.ligand_element_batch]

    # Keep a copy of protein positions (we'll update this during centering)
    protein_pos = batch.protein_pos.clone()

    if mem_tracker:
        mem_tracker.checkpoint("after_init_noise")

    # Get protein embeddings (frozen during sampling)
    with torch.no_grad():
        protein_ctx = model.protein_encoder(
            node_attr=batch.protein_atom_feature_full.float(),
            pos=protein_pos,
            batch=batch.protein_element_batch,
        )

    if mem_tracker:
        mem_tracker.checkpoint("after_protein_encoder")

    # Denoising schedule
    skip = model.num_timesteps // args.n_steps
    seq = list(range(0, model.num_timesteps, skip))
    seq_next = [-1] + seq[:-1]
    seq_pairs = list(zip(reversed(seq), reversed(seq_next)))

    sigmas = (1.0 - model.alphas).sqrt() / model.alphas.sqrt()
    score_diffs = []

    # Get bond index if available
    ligand_bond_index = batch.ligand_bond_index if hasattr(batch, 'ligand_bond_index') else None
    ligand_bond_type = batch.ligand_bond_type if hasattr(batch, 'ligand_bond_type') else None

    # Create sampling progress bar
    sampling_pbar = tqdm(
        enumerate(seq_pairs),
        total=len(seq_pairs),
        desc="  Sampling",
        leave=False,
        ncols=80,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [t={postfix}]'
    )

    for idx, (t_i, t_j) in sampling_pbar:
        # Update progress bar with current timestep
        sampling_pbar.set_postfix_str(f"{t_i}")
        # Check if we should generate multiple candidates
        should_branch = (
            t_i <= args.divert_start_step and
            idx % args.spo_interval == 0 and
            idx + args.spo_interval < len(seq_pairs)
        )

        if should_branch:
            # Update progress bar to show branching
            sampling_pbar.set_description(f"  Sampling (branch)")

            # Generate multiple candidates
            candidates = multi_sampler.sample_step_multi(
                ligand_pos=ligand_pos,
                ligand_atom=ligand_atom,
                protein_pos=protein_pos,  # Use local copy that gets updated
                ligand_batch=batch.ligand_element_batch,
                protein_ctx=protein_ctx,
                protein_atom_type=batch.protein_atom_feature.float(),
                protein_backbone_mask=None,
                protein_batch=batch.protein_element_batch,
                timestep_i=t_i,
                timestep_j=t_j,
                num_graphs=num_graphs,
                ligand_bond_index=ligand_bond_index,
                ligand_bond_type=ligand_bond_type,
                extend_order=False,
                extend_radius=True,
            )

            stats['num_candidates_generated'] += len(candidates) * num_graphs

            # Update progress bar to show scoring
            sampling_pbar.set_description(f"  Scoring")

            # Score candidates with combined scorer
            candidate_data = [
                (c.ligand_atom, c.ligand_pos, batch.ligand_element_batch)
                for c in candidates
            ]
            win_indices, lose_indices, valid_mask, mol_success_rate = combined_scorer.select_win_lose(
                candidate_data, t_i, args.score_threshold,
                protein_pos, batch.protein_element_batch
            )

            # Track molecule building success rate
            stats['mol_success_rate'] = (
                (stats['mol_success_rate'] * stats['num_scoring_events'] + mol_success_rate) /
                (stats['num_scoring_events'] + 1)
            )
            stats['num_scoring_events'] += 1

            # Store valid pairs (move to CPU to save GPU memory)
            for graph_idx in range(num_graphs):
                if valid_mask[graph_idx]:
                    win_cand = candidates[win_indices[graph_idx]]
                    lose_cand = candidates[lose_indices[graph_idx]]

                    # Only store ligand-specific data per pair (on CPU)
                    # Protein data is shared via shared_protein_data
                    pairs.append({
                        'timestep': t_i,
                        'prev_timestep': t_j,
                        'current_pos': ligand_pos.clone().cpu(),
                        'current_atom': ligand_atom.clone().cpu(),
                        'win_pos': win_cand.ligand_pos.clone().cpu(),
                        'win_atom': win_cand.ligand_atom.clone().cpu(),
                        'lose_pos': lose_cand.ligand_pos.clone().cpu(),
                        'lose_atom': lose_cand.ligand_atom.clone().cpu(),
                        'graph_idx': graph_idx,
                        'batch': batch.ligand_element_batch.clone().cpu(),
                        'ligand_bond_index': ligand_bond_index.clone().cpu() if ligand_bond_index is not None else None,
                        'ligand_bond_type': ligand_bond_type.clone().cpu() if ligand_bond_type is not None else None,
                    })

                    stats['num_valid_pairs'] += 1

            # Update progress bar with pairs count and mol success rate
            sampling_pbar.set_description(f"  Sampling")
            mol_pct = int(stats['mol_success_rate'] * 100)
            sampling_pbar.set_postfix_str(f"{t_i}, p={stats['num_valid_pairs']}, mol={mol_pct}%")

            # Continue with random candidate (also update protein_pos from candidate)
            rand_idx = torch.randint(0, len(candidates), (1,)).item()
            ligand_pos = candidates[rand_idx].ligand_pos
            ligand_atom = candidates[rand_idx].ligand_atom
            protein_pos = candidates[rand_idx].protein_pos

        else:
            # Single denoising step
            t = torch.full((num_graphs,), t_i, dtype=torch.long, device=device)

            # Use model's denoise step
            with torch.no_grad():
                candidates = multi_sampler.sample_step_multi(
                    ligand_pos=ligand_pos,
                    ligand_atom=ligand_atom,
                    protein_pos=protein_pos,  # Use local copy
                    ligand_batch=batch.ligand_element_batch,
                    protein_ctx=protein_ctx,
                    protein_atom_type=batch.protein_atom_feature.float(),
                    protein_backbone_mask=None,
                    protein_batch=batch.protein_element_batch,
                    timestep_i=t_i,
                    timestep_j=t_j,
                    num_graphs=num_graphs,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    extend_order=False,
                    extend_radius=True,
                )
                # Update state from the single candidate
                ligand_pos = candidates[0].ligand_pos
                ligand_atom = candidates[0].ligand_atom
                protein_pos = candidates[0].protein_pos

    # Close sampling progress bar
    sampling_pbar.close()

    if mem_tracker:
        mem_tracker.checkpoint("after_sampling_loop")

    # Create shared protein data (store on CPU to save GPU memory)
    shared_protein_data = {
        'protein_ctx': protein_ctx.cpu(),
        'protein_pos': protein_pos.cpu(),
        'protein_atom_type': batch.protein_atom_feature.float().cpu(),
        'protein_batch': batch.protein_element_batch.cpu(),
    }

    if mem_tracker:
        mem_tracker.checkpoint("after_move_to_cpu")

    return pairs, stats, shared_protein_data


def compute_spo_loss_with_accumulation(
    model,
    ref_manager,
    pairs,
    shared_protein_data,
    spo_loss_fn,
    args,
    device,
    mem_tracker=None  # Optional GPU memory tracker
):
    """
    Compute SPO loss with gradient accumulation - backward after each pair.

    This is memory-efficient: each pair's forward+backward is computed independently,
    gradients are accumulated, and the computation graph is released after each pair.

    Args:
        model: PMDM model (trainable)
        ref_manager: Reference model manager
        pairs: List of win/lose pairs (stored on CPU)
        shared_protein_data: Dict with protein data shared across pairs (stored on CPU)
        spo_loss_fn: SPO loss function
        args: Training arguments
        device: Device
        mem_tracker: Optional GPUMemoryTracker for debugging

    Returns:
        total_loss_value: float, total loss value (for logging)
        metrics: Loss metrics dictionary
    """
    if len(pairs) == 0:
        return 0.0, {'loss': 0.0, 'num_pairs': 0}

    if mem_tracker:
        mem_tracker.checkpoint("loss_start")

    total_loss_value = 0.0
    all_metrics = []
    num_pairs = len(pairs)

    # Move shared protein data to GPU once
    protein_ctx = shared_protein_data['protein_ctx'].to(device)
    protein_pos = shared_protein_data['protein_pos'].to(device)
    protein_atom_type = shared_protein_data['protein_atom_type'].to(device)
    protein_batch = shared_protein_data['protein_batch'].to(device)

    if mem_tracker:
        mem_tracker.checkpoint("loss_after_protein_to_gpu")

    # Create progress bar for pairs processing
    pairs_pbar = tqdm(
        enumerate(pairs),
        total=num_pairs,
        desc="    Pairs",
        leave=False,
        ncols=60,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}'
    )

    for pair_idx, pair in pairs_pbar:
        # Only log first pair to avoid too much output
        log_this_pair = (pair_idx == 0) and mem_tracker

        t = pair['timestep']
        t_prev = pair['prev_timestep']
        graph_idx = pair['graph_idx']

        # Move pair data from CPU to GPU
        batch = pair['batch'].to(device)
        current_pos = pair['current_pos'].to(device)
        current_atom = pair['current_atom'].to(device)
        win_pos = pair['win_pos'].to(device)
        win_atom = pair['win_atom'].to(device)
        lose_pos = pair['lose_pos'].to(device)
        lose_atom = pair['lose_atom'].to(device)

        if log_this_pair:
            mem_tracker.checkpoint(f"loss_pair0_after_load_data")

        # Create timestep tensor
        num_graphs = batch.max().item() + 1
        t_tensor = torch.full((num_graphs,), t, dtype=torch.long, device=device)

        # Get bond index from pair (if stored during sampling)
        ligand_bond_index = pair.get('ligand_bond_index', None)
        ligand_bond_type = pair.get('ligand_bond_type', None)
        if ligand_bond_index is not None:
            ligand_bond_index = ligand_bond_index.to(device)
        if ligand_bond_type is not None:
            ligand_bond_type = ligand_bond_type.to(device)

        if log_this_pair:
            mem_tracker.checkpoint(f"loss_pair0_before_model_forward")
            # Log tensor sizes for debugging
            print(f"  [DEBUG] current_pos: {current_pos.shape}, current_atom: {current_atom.shape}")
            print(f"  [DEBUG] protein_ctx: {protein_ctx.shape}, protein_pos: {protein_pos.shape}")
            print(f"  [DEBUG] batch: {batch.shape}, protein_batch: {protein_batch.shape}")

        # === Current model forward pass ===
        # Get epsilon predictions for win sample
        net_out_win = model.net(
            ligand_atom_type=current_atom,
            ligand_pos=current_pos,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            ligand_batch=batch,
            protein_embeddings=protein_ctx,
            time_step=t_tensor,
            num_node_ctx=None,
            protein_atom_feature=protein_atom_type,
            protein_pos=protein_pos,
            protein_backbone_mask=None,
            protein_batch=protein_batch,
            return_edges=True,
            extend_order=False,
            extend_radius=True,
        )

        if model.vae_context:
            (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
             edge_index, edge_type, edge_length, local_edge_mask) = net_out_win[:-1]
        else:
            (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
             edge_index, edge_type, edge_length, local_edge_mask) = net_out_win

        eps_pos = pos_eq_global + pos_eq_local
        eps_atom = node_score_global + node_score_local

        if log_this_pair:
            mem_tracker.checkpoint(f"loss_pair0_after_model_forward")

        # === Reference model forward pass ===
        with torch.no_grad():
            ref_out = ref_manager.net(
                ligand_atom_type=current_atom,
                ligand_pos=current_pos,
                ligand_bond_index=ligand_bond_index,
                ligand_bond_type=ligand_bond_type,
                ligand_batch=batch,
                protein_embeddings=protein_ctx,
                time_step=t_tensor,
                num_node_ctx=None,
                protein_atom_feature=protein_atom_type,
                protein_pos=protein_pos,
                protein_backbone_mask=None,
                protein_batch=protein_batch,
                return_edges=True,
                extend_order=False,
                extend_radius=True,
            )

            if ref_manager.ref_model.vae_context:
                (ref_pos_global, ref_pos_local, ref_node_global, ref_node_local,
                 _, _, _, _) = ref_out[:-1]
            else:
                (ref_pos_global, ref_pos_local, ref_node_global, ref_node_local,
                 _, _, _, _) = ref_out

            ref_eps_pos = ref_pos_global + ref_pos_local
            ref_eps_atom = ref_node_global + ref_node_local

        if log_this_pair:
            mem_tracker.checkpoint(f"loss_pair0_after_ref_forward")

        # Compute sigmas for log prob calculation
        sigmas = (1.0 - model.alphas).sqrt() / model.alphas.sqrt()

        # === Compute log probabilities ===
        # Log prob for win sample (using GPU tensors)
        log_prob_win = compute_log_prob_pmdm(
            model_output=(eps_pos, eps_atom),
            current_state=(current_pos, current_atom),
            next_state=(win_pos, win_atom),
            timestep=t,
            prev_timestep=t_prev,
            alphas_cumprod=model.alphas,
            batch=batch,
            eta=args.eta,
            sampling_type=args.sampling_type,
            sigmas=sigmas,
        )

        # Log prob for lose sample
        log_prob_lose = compute_log_prob_pmdm(
            model_output=(eps_pos, eps_atom),
            current_state=(current_pos, current_atom),
            next_state=(lose_pos, lose_atom),
            timestep=t,
            prev_timestep=t_prev,
            alphas_cumprod=model.alphas,
            batch=batch,
            eta=args.eta,
            sampling_type=args.sampling_type,
            sigmas=sigmas,
        )

        # Reference log probs
        ref_sigmas = (1.0 - ref_manager.alphas).sqrt() / ref_manager.alphas.sqrt()
        with torch.no_grad():
            log_ref_win = compute_log_prob_pmdm(
                model_output=(ref_eps_pos, ref_eps_atom),
                current_state=(current_pos, current_atom),
                next_state=(win_pos, win_atom),
                timestep=t,
                prev_timestep=t_prev,
                alphas_cumprod=ref_manager.alphas,
                batch=batch,
                eta=args.eta,
                sampling_type=args.sampling_type,
                sigmas=ref_sigmas,
            )

            log_ref_lose = compute_log_prob_pmdm(
                model_output=(ref_eps_pos, ref_eps_atom),
                current_state=(current_pos, current_atom),
                next_state=(lose_pos, lose_atom),
                timestep=t,
                prev_timestep=t_prev,
                alphas_cumprod=ref_manager.alphas,
                batch=batch,
                eta=args.eta,
                sampling_type=args.sampling_type,
                sigmas=ref_sigmas,
            )

        # === Compute SPO loss ===
        # Extract log prob for specific graph
        loss, metrics = spo_loss_fn(
            log_prob_win[graph_idx:graph_idx+1],
            log_prob_lose[graph_idx:graph_idx+1],
            log_ref_win[graph_idx:graph_idx+1],
            log_ref_lose[graph_idx:graph_idx+1],
        )

        # Scale loss for gradient accumulation (average over pairs)
        scaled_loss = loss / num_pairs

        # === IMMEDIATE BACKWARD - releases computation graph ===
        if scaled_loss.requires_grad and not torch.isnan(scaled_loss):
            scaled_loss.backward()

        # Track loss value (detached)
        total_loss_value += loss.detach().item()
        all_metrics.append(metrics)

        if log_this_pair:
            mem_tracker.checkpoint(f"loss_pair0_after_backward")

        # Cleanup intermediate tensors and computation graph
        del net_out_win, ref_out, eps_pos, eps_atom, ref_eps_pos, ref_eps_atom
        del log_prob_win, log_prob_lose, log_ref_win, log_ref_lose
        del current_pos, current_atom, win_pos, win_atom, lose_pos, lose_atom
        del loss, scaled_loss
        del pos_eq_global, pos_eq_local, node_score_global, node_score_local
        del edge_index, edge_type, edge_length, local_edge_mask

        # Clear cache every pair to prevent memory buildup
        torch.cuda.empty_cache()

        if log_this_pair:
            mem_tracker.checkpoint(f"loss_pair0_after_cleanup")

    # Average loss value for logging
    avg_loss_value = total_loss_value / num_pairs if num_pairs > 0 else 0.0

    # Aggregate metrics
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    avg_metrics['num_pairs'] = len(pairs)

    # Cleanup shared protein data
    del protein_ctx, protein_pos, protein_atom_type, protein_batch
    torch.cuda.empty_cache()

    if mem_tracker:
        mem_tracker.checkpoint("loss_end")

    return avg_loss_value, avg_metrics


def train_epoch(
    model,
    ref_manager,
    combined_scorer,
    multi_sampler,
    train_loader,
    optimizer,
    spo_loss_fn,
    args,
    epoch,
    tb_logger,
    health_monitor,
    logger,
    device,
    ckpt_dir=None,  # Checkpoint directory for per-batch saving
    profile_memory=False,  # Enable memory profiling for debugging
    start_batch=0  # Skip batches before this index (for resume)
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_pairs = 0
    total_batches = 0
    epoch_metrics = []

    # Track SPM scores for this epoch
    all_win_scores = []
    all_lose_scores = []

    # Create memory tracker for first batch if profiling is enabled
    mem_tracker = GPUMemoryTracker(logger=logger, enabled=profile_memory) if profile_memory else None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Skip batches before start_batch (for resume)
        if batch_idx < start_batch:
            continue

        batch = batch.to(device)
        global_step = epoch * len(train_loader) + batch_idx

        # Enable memory tracking only for first batch
        use_tracker = mem_tracker if (batch_idx == 0 and profile_memory) else None

        if use_tracker:
            reset_gpu_memory_stats()
            use_tracker.clear()
            use_tracker.checkpoint("batch_start")
            logger.info(f"[Memory Profile] Starting batch {batch_idx}")

        # === Sampling phase (no gradients) ===
        with torch.no_grad():
            pairs, sample_stats, shared_protein_data = sample_and_collect_pairs(
                model, ref_manager, combined_scorer, multi_sampler,
                batch, args, device,
                mem_tracker=use_tracker
            )

        # Log sampling statistics
        tb_logger.log_custom_scalar('sampling/candidates_generated',
                                     sample_stats['num_candidates_generated'], global_step)
        tb_logger.log_custom_scalar('sampling/valid_pairs',
                                     sample_stats['num_valid_pairs'], global_step)
        tb_logger.log_custom_scalar('sampling/mol_success_rate',
                                     sample_stats['mol_success_rate'], global_step)

        if use_tracker:
            use_tracker.checkpoint("after_sampling")

        # === Training phase ===
        if len(pairs) > 0:
            optimizer.zero_grad()

            if use_tracker:
                use_tracker.checkpoint("before_compute_loss")

            # Compute SPO loss with gradient accumulation
            # Backward is called inside for each pair, gradients are accumulated
            loss_value, metrics = compute_spo_loss_with_accumulation(
                model, ref_manager, pairs, shared_protein_data, spo_loss_fn, args, device,
                mem_tracker=use_tracker
            )

            if use_tracker:
                use_tracker.checkpoint("after_compute_loss_with_backward")

            # Check if gradients were accumulated (loss_value > 0 means pairs were processed)
            has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)

            if has_grads:
                # Log gradients before clipping
                grad_stats = tb_logger.log_gradients(model, global_step)

                # Gradient clipping
                grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                tb_logger.log_custom_scalar('gradient/clipped_norm', grad_norm, global_step)

                optimizer.step()

                if use_tracker:
                    use_tracker.checkpoint("after_optimizer_step")

                # Log loss metrics
                tb_logger.log_loss_metrics(metrics, global_step)

                # Log learning rate
                tb_logger.log_learning_rate(optimizer, global_step)

                # Log parameter changes
                tb_logger.log_parameter_changes(model, global_step)

                # Health check
                issues = health_monitor.maybe_check(global_step)
                if issues:
                    logger.warning(f"[Step {global_step}] Training issues: {issues}")

                total_loss += loss_value
                total_pairs += len(pairs)
                total_batches += 1
                epoch_metrics.append(metrics)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'pairs': len(pairs),
                    'acc': f"{metrics.get('accuracy', 0):.2f}",
                    'clip': f"{metrics.get('ratio_win_clip_rate', 0):.2f}"
                })

            else:
                # No valid gradients (e.g., all NaN losses)
                logger.warning(f"[Step {global_step}] No valid gradients accumulated")
                tb_logger.log_custom_scalar('train/no_grad_count', 1, global_step)

        # Explicit GPU memory cleanup after each batch
        del pairs, shared_protein_data
        torch.cuda.empty_cache()

        # Save checkpoint every N batches
        if ckpt_dir is not None and (batch_idx + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_e{epoch}_b{batch_idx}.pt')
            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / max(total_batches, 1),
                'args': vars(args),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # Print memory report after first batch
        if use_tracker:
            use_tracker.checkpoint("batch_end_after_cleanup")
            use_tracker.report()
            logger.info("[Memory Profile] First batch complete. Report printed above.")

    # Epoch summary
    avg_loss = total_loss / max(total_batches, 1)
    avg_pairs = total_pairs / max(total_batches, 1)

    # Aggregate epoch metrics
    if epoch_metrics:
        avg_epoch_metrics = {}
        for key in epoch_metrics[0].keys():
            values = [m[key] for m in epoch_metrics if key in m]
            if values:
                avg_epoch_metrics[key] = sum(values) / len(values)
        avg_epoch_metrics['avg_pairs_per_batch'] = avg_pairs
        avg_epoch_metrics['total_pairs'] = total_pairs

        # Log epoch summary
        tb_logger.log_epoch_summary(epoch, avg_epoch_metrics)

    logger.info(
        f"[Epoch {epoch}] Loss: {avg_loss:.4f} | "
        f"Avg pairs/batch: {avg_pairs:.1f} | "
        f"Total pairs: {total_pairs} | "
        f"Accuracy: {avg_epoch_metrics.get('accuracy', 0):.3f}"
    )

    # Health check at end of epoch
    is_healthy, issues = tb_logger.check_training_health()
    if not is_healthy:
        logger.warning(f"[Epoch {epoch}] Health check failed:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return avg_loss


def main():
    """Main training loop."""
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Setup logging
    logger = get_logger('spo_train', args.output_dir)

    # Setup TensorBoard logger with health monitoring
    tb_logger = SPOTensorBoardLogger(
        log_dir=args.output_dir,
        experiment_name=None,  # Already in output_dir
        log_histograms=True,
        log_param_changes=True,
        history_window=100,
    )

    # Create health monitor with alert callback
    def on_health_issue(step, issues):
        logger.warning(f"[Health Monitor] Step {step}: {issues}")

    health_monitor = TrainingHealthMonitor(
        logger=tb_logger,
        check_interval=50,  # Check every 50 steps
        alert_callback=on_health_issue,
    )

    logger.info(f"Arguments: {args}")
    logger.info(f"TensorBoard logs: {tb_logger.log_dir}")

    # Load config
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    # Save config to output
    config_save_path = os.path.join(args.output_dir, 'config.yml')
    with open(config_save_path, 'w') as f:
        yaml.dump(dict(config), f)

    logger.info(f"Config: {config}")

    # === Load models ===
    logger.info("Loading models...")

    # Load PMDM
    model = load_pmdm_model(args.pmdm_ckpt, config, device)
    logger.info(f"PMDM model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    # Handle n_steps=0 (use model default)
    if args.n_steps == 0:
        args.n_steps = model.num_timesteps
        logger.info(f"Using model default n_steps: {args.n_steps}")

    # Create reference model (frozen copy)
    logger.info("Creating reference model...")
    ref_manager = ReferenceModelManager(model, copy_to_device=device)

    # Load SPM
    spm_wrapper = load_spm_model(args.spm_ckpt, device, args.use_dummy_spm)
    logger.info("SPM model loaded")

    # Get dataset info for molecule reconstruction (needed for QED/SA scoring)
    dataset_info = get_dataset_info(config.dataset.name, False)
    atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
    logger.info(f"Dataset info loaded: {config.dataset.name}")

    # Create combined scorer
    combined_scorer = CombinedScorer(
        spm_wrapper=spm_wrapper,
        weight_spm=args.score_weight_spm,
        weight_qed=args.score_weight_qed,
        weight_sa=args.score_weight_sa,
        weight_clash=args.score_weight_clash,
        clash_threshold=args.clash_threshold,
        dataset_info=dataset_info,
        atomic_numbers=atomic_numbers,
    )
    logger.info(f"Combined scorer created with weights: "
                f"SPM={args.score_weight_spm}, QED={args.score_weight_qed}, "
                f"SA={args.score_weight_sa}, Clash={args.score_weight_clash}")

    # Create multi-sampler
    multi_sampler = MultiSamplePMDM(
        model=model,
        num_candidates=args.num_candidates,
        sampling_type=args.sampling_type,
        eta=args.eta,
    )

    # === Create data loader ===
    logger.info("Creating data loader...")
    train_loader = create_dataloader(config, args)
    logger.info(f"Training on {len(train_loader)} batches")

    # === Setup optimizer and loss ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    spo_loss_fn = SPOLoss(beta=args.beta, eps=args.eps)

    # === Resume from checkpoint if specified ===
    start_epoch = 0
    start_batch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)

            # Load model state
            model.load_state_dict(resume_ckpt['model_state_dict'])
            logger.info("Model state loaded")

            # Load optimizer state
            if 'optimizer_state_dict' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
                logger.info("Optimizer state loaded")

            # Get epoch and batch to resume from
            start_epoch = resume_ckpt.get('epoch', 0)
            start_batch = resume_ckpt.get('batch_idx', -1) + 1  # Start from next batch

            # If we finished all batches in that epoch, move to next epoch
            if start_batch >= len(train_loader):
                start_epoch += 1
                start_batch = 0

            logger.info(f"Resuming from epoch {start_epoch}, batch {start_batch}")

            # Update reference model to match resumed model
            logger.info("Updating reference model to match resumed model...")
            ref_manager = ReferenceModelManager(model, copy_to_device=device)
        else:
            logger.warning(f"Resume checkpoint not found: {args.resume}, starting from scratch")

    # Log hyperparameters
    hparams = {
        'beta': args.beta,
        'eps': args.eps,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_candidates': args.num_candidates,
        'divert_start_step': args.divert_start_step,
        'spo_interval': args.spo_interval,
        'score_threshold': args.score_threshold,
        'batch_size': args.batch_size,
        'n_steps': args.n_steps,
        'sampling_type': args.sampling_type,
        'eta': args.eta,
        'max_grad_norm': args.max_grad_norm,
        # Combined scoring weights
        'score_weight_spm': args.score_weight_spm,
        'score_weight_qed': args.score_weight_qed,
        'score_weight_sa': args.score_weight_sa,
        'score_weight_clash': args.score_weight_clash,
        'clash_threshold': args.clash_threshold,
    }
    tb_logger.log_hparams(hparams, {})
    logger.info(f"Hyperparameters logged to TensorBoard")

    # === Training loop ===
    logger.info("Starting SPO training...")
    best_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        # Only profile memory on first epoch
        profile_this_epoch = args.profile_memory and (epoch == start_epoch)

        # Determine start_batch for this epoch (only applies to first resumed epoch)
        epoch_start_batch = start_batch if epoch == start_epoch else 0

        loss = train_epoch(
            model, ref_manager, combined_scorer, multi_sampler,
            train_loader, optimizer, spo_loss_fn,
            args, epoch, tb_logger, health_monitor, logger, device,
            ckpt_dir=ckpt_dir,  # Pass ckpt_dir for per-batch saving
            profile_memory=profile_this_epoch,
            start_batch=epoch_start_batch  # Skip batches when resuming
        )

        # Save end-of-epoch checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'args': vars(args),
        }, ckpt_path)
        logger.info(f"Saved epoch checkpoint to {ckpt_path}")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(ckpt_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
            logger.info(f"Saved best model (loss={loss:.4f})")

    # Save final model
    final_path = os.path.join(ckpt_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")

    # Get training summary
    summary = tb_logger.get_summary()
    logger.info(f"Training Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    # Log final metrics with hyperparameters
    final_metrics = {
        'final_loss': summary.get('loss_final', best_loss),
        'final_accuracy': summary.get('accuracy_final', 0),
        'best_loss': best_loss,
    }
    tb_logger.log_hparams(hparams, final_metrics)

    # Close TensorBoard logger
    tb_logger.close()
    logger.info(f"TensorBoard logs saved to: {tb_logger.log_dir}")


if __name__ == '__main__':
    main()
