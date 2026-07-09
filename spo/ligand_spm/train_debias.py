"""
Debiased Training script for ligand step-aware preference model.

This script implements:
1. N-atom masking: randomly replace N atoms with "other" to prevent N-ratio bias
2. Large ring pair reweighting (to improve ring structure awareness)

The trained model is fully compatible with the original LigandPreferenceModel.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ligand_preference_model import LigandPreferenceModel
from dataset import LigandPairDataset, collate_ligand_pairs
from utils.diffusion_utils import DiffusionNoiseScheduler


# =============================================================================
# N-Atom Masking (Data Augmentation for Debiasing)
# =============================================================================

def mask_nitrogen_atoms(node_attr, batch, mask_prob=0.5):
    """
    Randomly mask ALL N atoms in a molecule (molecule-level masking).

    For each molecule in the batch, with probability mask_prob,
    replace ALL its N atoms with "other" category.

    Args:
        node_attr: [N, 10] one-hot atom features (index 2 = N, index 9 = other)
        batch: [N] batch indices for each atom
        mask_prob: probability of masking a molecule's N atoms (all or nothing)

    Returns:
        node_attr_masked: [N, 10] with N atoms replaced by "other" for selected molecules
    """
    node_attr_masked = node_attr.clone()

    # Get number of molecules in batch
    num_molecules = batch.max().item() + 1

    # Decide which molecules to mask (molecule-level decision)
    mask_molecules = torch.rand(num_molecules, device=node_attr.device) < mask_prob

    # For each molecule that should be masked, replace all its N atoms
    for mol_idx in range(num_molecules):
        if not mask_molecules[mol_idx]:
            continue

        # Find atoms belonging to this molecule
        mol_mask = (batch == mol_idx)

        # Find N atoms in this molecule (index 2)
        is_nitrogen = node_attr_masked[:, 2] > 0.5
        n_in_mol = mol_mask & is_nitrogen

        if n_in_mol.sum() > 0:
            # Replace all N atoms with "other"
            node_attr_masked[n_in_mol, 2] = 0.0   # Remove N
            node_attr_masked[n_in_mol, 9] = 1.0   # Set as "other"

    return node_attr_masked


# =============================================================================
# Utility Functions
# =============================================================================

def compute_pair_weights(ring_label_0, ring_label_1, ring_pair_weight=5.0):
    """
    Compute sample weights based on ring labels.

    Pairs where one ligand has large ring and the other doesn't
    are weighted higher to help the model learn ring structure awareness.

    Args:
        ring_label_0: [B] 1.0 if ligand 0 has >=9 ring, else 0.0
        ring_label_1: [B] 1.0 if ligand 1 has >=9 ring, else 0.0
        ring_pair_weight: weight for pairs with ring difference

    Returns:
        weights: [B] sample weights
    """
    # One has ring, one doesn't -> high weight
    has_ring_diff = (ring_label_0 != ring_label_1).float()

    # Weight = 1.0 for normal pairs, ring_pair_weight for ring-different pairs
    weights = 1.0 + (ring_pair_weight - 1.0) * has_ring_diff

    return weights


# =============================================================================
# Loss Functions
# =============================================================================

class WeightedPreferenceLoss(nn.Module):
    """
    Preference learning loss with sample weights.
    """

    def __init__(self):
        super().__init__()

    def forward(self, quality_0, quality_1, label_0, label_1, weights=None):
        """
        Compute weighted preference loss.

        Args:
            quality_0: [B, 1] quality scores for ligand 0
            quality_1: [B, 1] quality scores for ligand 1
            label_0: [B] preference labels for ligand 0
            label_1: [B] preference labels for ligand 1
            weights: [B] sample weights (optional)

        Returns:
            loss: scalar
        """
        device = quality_0.device

        score_0 = quality_0.squeeze(-1)  # [B]
        score_1 = quality_1.squeeze(-1)  # [B]

        # Stack into logits [B, 2]
        logits = torch.stack([score_0, score_1], dim=-1)

        # Create targets
        target_0 = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        target_1 = target_0 + 1

        # Compute cross-entropy for each option
        loss_0 = F.cross_entropy(logits, target_0, reduction='none')
        loss_1 = F.cross_entropy(logits, target_1, reduction='none')

        # Weight by labels
        loss = label_0 * loss_0 + label_1 * loss_1

        # Apply sample weights
        if weights is not None:
            loss = loss * weights

        return loss.mean()


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    noise_scheduler,
    device,
    num_timesteps=1000,
    # N-atom masking
    n_mask_prob=0.5,
    # Ring pair reweighting
    ring_pair_weight=5.0,
):
    """
    Train for one epoch with N-atom masking for debiasing.

    Args:
        model: LigandPreferenceModel
        dataloader: training data loader
        optimizer: optimizer for model
        noise_scheduler: diffusion noise scheduler
        device: torch device
        num_timesteps: number of diffusion timesteps
        n_mask_prob: probability of masking each N atom (0.0 = no masking)
        ring_pair_weight: weight for ring-different pairs

    Returns:
        avg_loss: average training loss
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    criterion = WeightedPreferenceLoss()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        node_attr_0 = batch['node_attr_0'].to(device)
        pos_0 = batch['pos_0'].to(device)
        batch_0 = batch['batch_0'].to(device)
        node_attr_1 = batch['node_attr_1'].to(device)
        pos_1 = batch['pos_1'].to(device)
        batch_1 = batch['batch_1'].to(device)
        label_0 = batch['label_0'].to(device)
        label_1 = batch['label_1'].to(device)
        ring_label_0 = batch['ring_label_0'].to(device)
        ring_label_1 = batch['ring_label_1'].to(device)

        batch_size = label_0.shape[0]

        # Apply N-atom masking for debiasing (molecule-level: all or nothing)
        if n_mask_prob > 0:
            node_attr_0 = mask_nitrogen_atoms(node_attr_0, batch_0, n_mask_prob)
            node_attr_1 = mask_nitrogen_atoms(node_attr_1, batch_1, n_mask_prob)

        # Compute pair weights based on ring labels
        pair_weights = compute_pair_weights(ring_label_0, ring_label_1, ring_pair_weight)

        # Sample random timesteps
        timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device)

        # Add noise to ligands
        pos_0_noisy, node_attr_0_noisy = noise_scheduler.add_noise(
            pos_0, node_attr_0, timesteps, batch_0, center=True
        )
        pos_1_noisy, node_attr_1_noisy = noise_scheduler.add_noise(
            pos_1, node_attr_1, timesteps, batch_1, center=True
        )

        # Forward pass
        _, _, quality_0, quality_1 = model(
            node_attr_0_noisy, pos_0_noisy, batch_0,
            node_attr_1_noisy, pos_1_noisy, batch_1,
            timesteps,
            return_scores=True
        )

        # Preference loss
        loss = criterion(quality_0, quality_1, label_0, label_1, pair_weights)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, noise_scheduler, device, num_timesteps=1000):
    """Validate the model (without N-atom masking)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    criterion = WeightedPreferenceLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            node_attr_0 = batch['node_attr_0'].to(device)
            pos_0 = batch['pos_0'].to(device)
            batch_0 = batch['batch_0'].to(device)
            node_attr_1 = batch['node_attr_1'].to(device)
            pos_1 = batch['pos_1'].to(device)
            batch_1 = batch['batch_1'].to(device)
            label_0 = batch['label_0'].to(device)
            label_1 = batch['label_1'].to(device)

            batch_size = label_0.shape[0]

            # Sample random timesteps
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device)

            # Add noise (no N-atom masking during validation)
            pos_0_noisy, node_attr_0_noisy = noise_scheduler.add_noise(
                pos_0, node_attr_0, timesteps, batch_0, center=True
            )
            pos_1_noisy, node_attr_1_noisy = noise_scheduler.add_noise(
                pos_1, node_attr_1, timesteps, batch_1, center=True
            )

            # Forward
            _, _, quality_0, quality_1 = model(
                node_attr_0_noisy, pos_0_noisy, batch_0,
                node_attr_1_noisy, pos_1_noisy, batch_1,
                timesteps,
                return_scores=True
            )

            # Loss (without weights for validation)
            loss = criterion(quality_0, quality_1, label_0, label_1)

            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            scores_0 = quality_0.squeeze(-1)
            scores_1 = quality_1.squeeze(-1)

            predictions = (scores_0 > scores_1).float()
            targets = (label_0 > label_1).float()

            non_tie_mask = (label_0 != 0.5)
            if non_tie_mask.any():
                correct += (predictions[non_tie_mask] == targets[non_tie_mask]).sum().item()
                total += non_tie_mask.sum().item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ligand SPM with N-atom masking debiasing")

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ligand data directory')
    parser.add_argument('--db_path', type=str, default=None, help='Path to SQLite database')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_debias', help='Output directory')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers for dataloader')

    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_interactions', type=int, default=2, help='Number of interaction blocks')

    # N-atom masking arguments
    parser.add_argument('--n_mask_prob', type=float, default=0.5,
                        help='Probability of masking each N atom (0.0 = disabled, default: 0.5)')

    # Ring pair reweighting arguments
    parser.add_argument('--ring_pair_weight', type=float, default=5.0,
                        help='Weight for pairs where one has large ring (default: 5.0)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = LigandPairDataset(args.data_dir, args.db_path, split='train')
    val_dataset = LigandPairDataset(args.data_dir, args.db_path, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ligand_pairs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ligand_pairs,
    )

    # Create model
    print("Creating model...")
    model = LigandPreferenceModel(
        hidden_channels=args.hidden_channels,
        num_filters=args.hidden_channels,
        num_interactions=args.num_interactions,
        edge_channels=args.hidden_channels,
        cutoff=6.0,
        input_dim=10,
        projection_dim=args.hidden_channels,
    )
    model.init_adaln_paras()
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print debiasing settings
    if args.n_mask_prob > 0:
        print(f"\nN-atom masking ENABLED")
        print(f"  - Mask probability: {args.n_mask_prob}")
    else:
        print(f"\nN-atom masking DISABLED")

    print(f"\nRing pair reweighting: {args.ring_pair_weight}x for ring-different pairs")

    # Create noise scheduler
    noise_scheduler = DiffusionNoiseScheduler(
        num_diffusion_timesteps=args.num_timesteps,
        beta_schedule='sigmoid',
        beta_start=1e-7,
        beta_end=2e-3,
    )
    noise_scheduler.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, noise_scheduler, device, args.num_timesteps,
            n_mask_prob=args.n_mask_prob,
            ring_pair_weight=args.ring_pair_weight,
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, noise_scheduler, device, args.num_timesteps)
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
            model.save_pretrained(ckpt_path)
            print(f"Saved best model to {ckpt_path}")

        # Save latest checkpoint
        ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        model.save_pretrained(ckpt_path)

    print("\nTraining completed!")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
