#!/usr/bin/env python3
"""
Convert SPO training checkpoint to PMDM-compatible format.

SPO training saves checkpoints with keys:
    - model_state_dict
    - optimizer_state_dict
    - epoch, batch_idx, global_step, loss, args

PMDM scripts expect:
    - model
    - optimizer
    - config
    - iteration

This script converts SPO checkpoints to be compatible with existing PMDM scripts
like sample_batch.py, evaluate.py, etc.

Usage:
    python convert_spo_checkpoint.py <spo_checkpoint.pt> [--output <output.pt>] [--original <500.pt>]

Examples:
    # Basic usage (auto-generates output name)
    python convert_spo_checkpoint.py logs/spo_combined_scoring/checkpoints/checkpoint_e0_b29.pt

    # Specify output path
    python convert_spo_checkpoint.py checkpoint_e0_b29.pt --output checkpoint_e0_b29_compatible.pt

    # Use different original model for config
    python convert_spo_checkpoint.py checkpoint_e0_b29.pt --original 500.pt
"""

import argparse
import os
import torch


def convert_checkpoint(spo_ckpt_path: str, output_path: str = None, original_path: str = "500.pt"):
    """
    Convert SPO checkpoint to PMDM-compatible format.

    Args:
        spo_ckpt_path: Path to SPO training checkpoint
        output_path: Output path (default: adds '_compatible' suffix)
        original_path: Path to original PMDM model (for config)

    Returns:
        output_path: Path where converted checkpoint was saved
    """
    # Load SPO checkpoint
    print(f"Loading SPO checkpoint: {spo_ckpt_path}")
    spo_ckpt = torch.load(spo_ckpt_path, map_location='cpu', weights_only=False)

    # Verify it's an SPO checkpoint
    expected_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'batch_idx', 'global_step']
    if not all(k in spo_ckpt for k in expected_keys):
        # Check if already in compatible format
        if 'model' in spo_ckpt and 'config' in spo_ckpt:
            print("Checkpoint is already in PMDM-compatible format!")
            return spo_ckpt_path
        else:
            raise ValueError(f"Unexpected checkpoint format. Keys: {list(spo_ckpt.keys())}")

    # Load original model for config
    print(f"Loading original model for config: {original_path}")
    original = torch.load(original_path, map_location='cpu', weights_only=False)

    if 'config' not in original:
        raise ValueError(f"Original model missing 'config'. Keys: {list(original.keys())}")

    # Create compatible checkpoint
    compatible_ckpt = {
        # From original PMDM
        'config': original['config'],

        # Renamed from SPO checkpoint
        'model': spo_ckpt['model_state_dict'],
        'optimizer': spo_ckpt['optimizer_state_dict'],
        'iteration': spo_ckpt['global_step'],

        # Preserve SPO-specific info
        'epoch': spo_ckpt['epoch'],
        'batch_idx': spo_ckpt['batch_idx'],
        'loss': spo_ckpt.get('loss', None),
        'args': spo_ckpt.get('args', None),
    }

    # Generate output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(spo_ckpt_path)
        output_path = f"{base}_compatible{ext}"

    # Save
    print(f"Saving compatible checkpoint: {output_path}")
    torch.save(compatible_ckpt, output_path)

    # Verify
    loaded = torch.load(output_path, map_location='cpu', weights_only=False)
    print(f"\nVerification:")
    print(f"  - model keys: {len(loaded['model'])}")
    print(f"  - config exists: {'config' in loaded}")
    print(f"  - iteration: {loaded['iteration']}")
    print(f"  - epoch: {loaded['epoch']}, batch: {loaded['batch_idx']}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert SPO checkpoint to PMDM-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "checkpoint",
        help="Path to SPO training checkpoint"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path (default: adds '_compatible' suffix)"
    )
    parser.add_argument(
        "--original",
        default="500.pt",
        help="Path to original PMDM model for config (default: 500.pt)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    if not os.path.exists(args.original):
        print(f"Error: Original model not found: {args.original}")
        return 1

    try:
        output_path = convert_checkpoint(
            args.checkpoint,
            args.output,
            args.original
        )
        print(f"\nSuccess! Compatible checkpoint saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
