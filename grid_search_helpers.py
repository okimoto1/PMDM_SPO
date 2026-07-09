#!/usr/bin/env python3
"""
Grid Search Helper Script
Features:
1. sample_train_subset: randomly sample a subset of proteins from the training set
2. parse_eval_results: parse evaluation results and write to CSV
3. generate_summary: generate a summary report
"""

import argparse
import os
import sys
import random
import re
import csv
from pathlib import Path


def sample_train_subset(args):
    """Randomly sample a specified number of proteins from the training set"""
    import torch

    # Load the split file
    split_file = args.split_file if args.split_file else "./data/split_by_name.pt"

    print(f"Loading split file: {split_file}")
    split_data = torch.load(split_file)

    train_data = split_data['train']
    print(f"Total training samples: {len(train_data)}")

    # Group by protein file (the same pocket may have multiple ligands)
    protein_to_indices = {}
    for idx, (protein_file, ligand_file) in enumerate(train_data):
        # Extract the protein identifier (directory + pocket file)
        protein_key = protein_file
        if protein_key not in protein_to_indices:
            protein_to_indices[protein_key] = []
        protein_to_indices[protein_key].append(idx)

    unique_proteins = list(protein_to_indices.keys())
    print(f"Unique proteins in training set: {len(unique_proteins)}")

    # Random sampling
    random.seed(args.seed)
    num_to_sample = min(args.num_proteins, len(unique_proteins))
    sampled_proteins = random.sample(unique_proteins, num_to_sample)

    print(f"Sampled {num_to_sample} proteins")

    # Collect all selected indices (only take the first sample's index for each protein)
    sampled_indices = []
    for protein in sampled_proteins:
        # Only take the first index for each protein
        sampled_indices.append(protein_to_indices[protein][0])

    # Save indices to file
    output_file = args.output_file
    with open(output_file, 'w') as f:
        for idx in sampled_indices:
            protein_file, ligand_file = train_data[idx]
            f.write(f"{idx}\t{protein_file}\t{ligand_file}\n")

    print(f"Saved {len(sampled_indices)} sample indices to {output_file}")

    # Print some examples
    print("\nSample entries:")
    for i in range(min(3, len(sampled_indices))):
        idx = sampled_indices[i]
        protein_file, ligand_file = train_data[idx]
        print(f"  [{idx}] {protein_file}")


def parse_eval_results(args):
    """Parse evaluation results and append them to the CSV file"""
    eval_dir = args.eval_dir

    # Initialize the results dictionary
    results = {
        'task_id': args.task_id,
        'use_lipinski': args.use_lipinski,
        'w_spm': args.w_spm,
        'w_qed': args.w_qed,
        'w_sa': args.w_sa,
        'w_clash': args.w_clash,
        'mean_qed': '',
        'mean_sa': '',
        'mean_logP': '',
        'mean_lipinski': '',
        'mean_mol_weight': '',
        'diversity': '',
        'collision_rate': '',
        'brenk_rate': '',
        'ring3_rate': '',
        'ring4_rate': '',
        'ring9plus_rate': '',
        'cccc_range_rate': '',
        'num_valid_samples': '',
        'status': 'OK'
    }

    # Parse 01_evaluate.txt
    eval_file = os.path.join(eval_dir, '01_evaluate.txt')
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            content = f.read()

            # Extract each metric
            patterns = {
                'mean_qed': r'mean qed:\s*([\d.]+)',
                'mean_sa': r'mean sa:\s*([\d.]+)',
                'mean_logP': r'mean logP:\s*([-\d.]+)',
                'mean_lipinski': r'mean Lipinski:\s*([\d.]+)',
                'mean_mol_weight': r'mean molecular weight:\s*([\d.]+)',
                'diversity': r'diversity:\s*([\d.]+)',
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    results[key] = match.group(1)

            # Extract the number of valid samples
            match = re.search(r'Final validity:\s*([\d.]+)', content)
            if match:
                # validity is a ratio, needs to be multiplied by the total count
                validity = float(match.group(1))
                match2 = re.search(r'ligands summary\s*(\d+)', content)
                if match2:
                    total = int(match2.group(1))
                    results['num_valid_samples'] = str(int(validity * total))

    # Parse 02_evaluate_collisions.txt
    collision_file = os.path.join(eval_dir, '02_evaluate_collisions.txt')
    if os.path.exists(collision_file):
        with open(collision_file, 'r') as f:
            content = f.read()
            # Format: Samples with collisions: 237 (47.4%)
            match = re.search(r'Samples with collisions:.*?\(([\d.]+)%\)', content)
            if match:
                rate = float(match.group(1)) / 100.0
                results['collision_rate'] = f"{rate:.4f}"

    # Parse 03_filter_rdBrenk.txt
    brenk_file = os.path.join(eval_dir, '03_filter_rdBrenk.txt')
    if os.path.exists(brenk_file):
        with open(brenk_file, 'r') as f:
            content = f.read()
            # Format: Removal rate: 0.1234
            match = re.search(r'Removal rate:\s*([\d.]+)', content)
            if match:
                results['brenk_rate'] = match.group(1)

    # Parse 04_check_ring_brenk.txt
    ring_file = os.path.join(eval_dir, '04_check_ring_brenk.txt')
    if os.path.exists(ring_file):
        with open(ring_file, 'r') as f:
            content = f.read()

            # ring size 3: 0.031 (3.1%)
            match = re.search(r'ring size 3:.*?\(([\d.]+)%\)', content)
            if match:
                rate = float(match.group(1)) / 100.0
                results['ring3_rate'] = f"{rate:.4f}"

            # ring size 4: ...
            match = re.search(r'ring size 4:.*?\(([\d.]+)%\)', content)
            if match:
                rate = float(match.group(1)) / 100.0
                results['ring4_rate'] = f"{rate:.4f}"

            # Percentage of ligands with ring size >= 9: 2.21%
            match = re.search(r'ring size >= 9:\s*([\d.]+)%', content)
            if match:
                rate = float(match.group(1)) / 100.0
                results['ring9plus_rate'] = f"{rate:.4f}"

    # Parse 05_calc_dihedral_kldiv.txt
    dihedral_file = os.path.join(eval_dir, '05_calc_dihedral_kldiv.txt')
    if os.path.exists(dihedral_file):
        with open(dihedral_file, 'r') as f:
            content = f.read()
            # cccc Range ... Generated Set 1: 0.0763 (7.63%)
            # Find the "cccc Range" section
            match = re.search(r'cccc Range.*?Generated Set 1:.*?\(([\d.]+)%\)', content, re.DOTALL)
            if match:
                rate = float(match.group(1)) / 100.0
                results['cccc_range_rate'] = f"{rate:.4f}"

    # Write to CSV
    output_csv = args.output_csv

    # Read existing content to check for duplicates
    existing_tasks = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_tasks.add(row.get('task_id', ''))

    # Skip if the task already exists
    if args.task_id in existing_tasks:
        print(f"Task {args.task_id} already exists in CSV, skipping")
        return

    # Append the result
    fieldnames = ['task_id', 'use_lipinski', 'w_spm', 'w_qed', 'w_sa', 'w_clash',
                  'mean_qed', 'mean_sa', 'mean_logP', 'mean_lipinski', 'mean_mol_weight',
                  'diversity', 'collision_rate', 'brenk_rate', 'ring3_rate', 'ring4_rate',
                  'ring9plus_rate', 'cccc_range_rate', 'num_valid_samples', 'status']

    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(results)

    print(f"Results for {args.task_id} written to {output_csv}")


def generate_summary(args):
    """Generate a summary report and identify the best configuration"""
    import csv

    results_csv = args.results_csv
    output_dir = args.output_dir

    if not os.path.exists(results_csv):
        print(f"Results CSV not found: {results_csv}")
        return

    # Read all results
    results = []
    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'OK':
                results.append(row)

    if not results:
        print("No valid results found")
        return

    print(f"Loaded {len(results)} valid results")

    # Define evaluation metrics (higher is better vs. lower is better)
    # To maximize: mean_qed, mean_sa, diversity
    # To minimize: collision_rate, brenk_rate, ring3_rate, ring4_rate, ring9plus_rate, cccc_range_rate

    # Compute the composite score
    def compute_score(row):
        """Compute the composite score"""
        score = 0.0
        weights = {
            'mean_qed': 2.0,       # higher is better
            'mean_sa': 1.5,        # higher is better
            'diversity': 1.0,      # higher is better
            'collision_rate': -3.0,   # lower is better
            'brenk_rate': -2.0,       # lower is better
            'ring3_rate': -1.0,       # lower is better
            'ring4_rate': -0.5,       # lower is better
            'ring9plus_rate': -1.0,   # lower is better
            'cccc_range_rate': -1.0,  # lower is better
        }

        for key, weight in weights.items():
            val = row.get(key, '')
            if val and val != '':
                try:
                    score += weight * float(val)
                except ValueError:
                    pass

        return score

    for row in results:
        row['composite_score'] = compute_score(row)

    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)

    # Generate the summary report
    summary_file = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Grid Search Summary Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total configurations tested: {len(results)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("TOP 10 CONFIGURATIONS (by composite score)\n")
        f.write("-" * 70 + "\n\n")

        for i, row in enumerate(results[:10], 1):
            f.write(f"Rank {i}: {row['task_id']}\n")
            f.write(f"  Lipinski: {row['use_lipinski']}, "
                   f"SPM: {row['w_spm']}, QED: {row['w_qed']}, "
                   f"SA: {row['w_sa']}, clash: {row['w_clash']}\n")
            f.write(f"  Composite Score: {row['composite_score']:.4f}\n")
            f.write(f"  Metrics:\n")
            f.write(f"    QED: {row.get('mean_qed', 'N/A')}, SA: {row.get('mean_sa', 'N/A')}\n")
            f.write(f"    Collision: {row.get('collision_rate', 'N/A')}, Brenk: {row.get('brenk_rate', 'N/A')}\n")
            f.write(f"    Ring3: {row.get('ring3_rate', 'N/A')}, Ring4: {row.get('ring4_rate', 'N/A')}\n")
            f.write(f"    Ring9+: {row.get('ring9plus_rate', 'N/A')}, cccc: {row.get('cccc_range_rate', 'N/A')}\n")
            f.write(f"    Diversity: {row.get('diversity', 'N/A')}\n")
            f.write("\n")

        # Find the best by each metric
        f.write("-" * 70 + "\n")
        f.write("BEST BY INDIVIDUAL METRICS\n")
        f.write("-" * 70 + "\n\n")

        # Metrics to maximize
        for metric in ['mean_qed', 'mean_sa', 'diversity']:
            valid_results = [r for r in results if r.get(metric) and r.get(metric) != '']
            if valid_results:
                best = max(valid_results, key=lambda x: float(x[metric]))
                f.write(f"Best {metric}: {best[metric]} ({best['task_id']})\n")

        f.write("\n")

        # Metrics to minimize
        for metric in ['collision_rate', 'brenk_rate', 'ring3_rate', 'ring4_rate',
                       'ring9plus_rate', 'cccc_range_rate']:
            valid_results = [r for r in results if r.get(metric) and r.get(metric) != '']
            if valid_results:
                best = min(valid_results, key=lambda x: float(x[metric]))
                f.write(f"Best {metric}: {best[metric]} ({best['task_id']})\n")

        f.write("\n")
        f.write("=" * 70 + "\n")

    print(f"Summary report saved to {summary_file}")

    # Generate the best configurations file
    best_file = os.path.join(output_dir, 'best_configs.txt')
    with open(best_file, 'w') as f:
        f.write("# Best configurations for different objectives\n\n")

        # Top 1 overall best (composite score)
        best = results[0]
        f.write("# OVERALL BEST (composite score)\n")
        f.write(f"use_lipinski={best['use_lipinski']}\n")
        f.write(f"w_spm={best['w_spm']}\n")
        f.write(f"w_qed={best['w_qed']}\n")
        f.write(f"w_sa={best['w_sa']}\n")
        f.write(f"w_clash={best['w_clash']}\n")
        f.write(f"# Score: {best['composite_score']:.4f}\n")
        f.write("\n")

        # Best by collision_rate
        valid = [r for r in results if r.get('collision_rate') and r.get('collision_rate') != '']
        if valid:
            best_collision = min(valid, key=lambda x: float(x['collision_rate']))
            f.write("# BEST COLLISION RATE\n")
            f.write(f"use_lipinski={best_collision['use_lipinski']}\n")
            f.write(f"w_spm={best_collision['w_spm']}\n")
            f.write(f"w_qed={best_collision['w_qed']}\n")
            f.write(f"w_sa={best_collision['w_sa']}\n")
            f.write(f"w_clash={best_collision['w_clash']}\n")
            f.write(f"# Collision rate: {best_collision['collision_rate']}\n")
            f.write("\n")

        # Best by QED
        valid = [r for r in results if r.get('mean_qed') and r.get('mean_qed') != '']
        if valid:
            best_qed = max(valid, key=lambda x: float(x['mean_qed']))
            f.write("# BEST QED\n")
            f.write(f"use_lipinski={best_qed['use_lipinski']}\n")
            f.write(f"w_spm={best_qed['w_spm']}\n")
            f.write(f"w_qed={best_qed['w_qed']}\n")
            f.write(f"w_sa={best_qed['w_sa']}\n")
            f.write(f"w_clash={best_qed['w_clash']}\n")
            f.write(f"# QED: {best_qed['mean_qed']}\n")

    print(f"Best configurations saved to {best_file}")


def generate_weight_combinations(args):
    """Generate linearly independent weight combinations

    Principle: two weight vectors (w1, w2, w3, w4) are linearly dependent if they
    are identical after normalization.
    For example, (0, 0.25, 0.5, 0.25) and (0, 0.5, 1.0, 0.5) are identical after normalization.

    We deduplicate by recording the normalized form and keeping only the first
    occurrence of each combination.

    Weight ranges:
    - w_spm: [0, 0.25, 0.5, 0.75, 1.0] (5 values)
    - w_qed, w_sa, w_clash: [0.25, 0.5, 0.75, 1.0] (4 values)
    """
    from fractions import Fraction

    weight_spm_values = [0, 0.25, 0.5, 0.75, 1.0]
    weight_other_values = [0.25, 0.5, 0.75, 1.0]
    output_file = args.output_file

    # Used to record normalized forms already seen
    seen_normalized = set()
    valid_combinations = []

    def normalize_weights(weights):
        """Normalize a weight vector into a tuple of simplest-fraction form"""
        # Skip all-zero
        if all(w == 0 for w in weights):
            return None

        # Convert to fractions to obtain exact ratios
        fracs = [Fraction(w).limit_denominator(1000) for w in weights]

        # Find the greatest common divisor
        from math import gcd
        from functools import reduce

        # Find the least common multiple of the denominators of all non-zero fractions
        non_zero_fracs = [f for f in fracs if f != 0]
        if not non_zero_fracs:
            return None

        # Put all fractions over a common denominator, then divide by their GCD
        # to get the simplest integer ratio. First find the common denominator.
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        common_denom = reduce(lcm, [f.denominator for f in fracs])
        # Convert to integers
        int_vals = [int(f * common_denom) for f in fracs]
        # Divide by the GCD
        common_gcd = reduce(gcd, [v for v in int_vals if v != 0])
        normalized = tuple(v // common_gcd for v in int_vals)

        return normalized

    total_checked = 0

    for w_spm in weight_spm_values:
        for w_qed in weight_other_values:
            for w_sa in weight_other_values:
                for w_clash in weight_other_values:
                    total_checked += 1
                    weights = (w_spm, w_qed, w_sa, w_clash)

                    normalized = normalize_weights(weights)
                    if normalized is None:
                        continue

                    if normalized not in seen_normalized:
                        seen_normalized.add(normalized)
                        valid_combinations.append(weights)

    print(f"Total combinations checked: {total_checked}")
    print(f"Linearly independent combinations: {len(valid_combinations)}")

    # Write to file
    with open(output_file, 'w') as f:
        for w_spm, w_qed, w_sa, w_clash in valid_combinations:
            f.write(f"{w_spm}|{w_qed}|{w_sa}|{w_clash}\n")

    print(f"Saved to {output_file}")

    # Print a few examples
    print("\nFirst 10 combinations:")
    for i, (w_spm, w_qed, w_sa, w_clash) in enumerate(valid_combinations[:10]):
        print(f"  {i+1}. SPM={w_spm}, QED={w_qed}, SA={w_sa}, clash={w_clash}")

    return len(valid_combinations)


def main():
    parser = argparse.ArgumentParser(description="Grid Search Helper Functions")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # generate_weight_combinations subcommand
    parser_weights = subparsers.add_parser('generate_weight_combinations',
                                           help='Generate linearly independent weight combinations')
    parser_weights.add_argument('--output_file', type=str, required=True,
                               help='Output file for weight combinations')

    # sample_train_subset subcommand
    parser_sample = subparsers.add_parser('sample_train_subset',
                                          help='Sample proteins from training set')
    parser_sample.add_argument('--num_proteins', type=int, required=True,
                              help='Number of proteins to sample')
    parser_sample.add_argument('--output_file', type=str, required=True,
                              help='Output file for sampled indices')
    parser_sample.add_argument('--split_file', type=str, default=None,
                              help='Path to split file (default: ./data/split_by_name.pt)')
    parser_sample.add_argument('--seed', type=int, default=42,
                              help='Random seed')

    # parse_eval_results subcommand
    parser_parse = subparsers.add_parser('parse_eval_results',
                                         help='Parse evaluation results and write to CSV')
    parser_parse.add_argument('--task_id', type=str, required=True)
    parser_parse.add_argument('--use_lipinski', type=str, required=True)
    parser_parse.add_argument('--w_spm', type=str, required=True)
    parser_parse.add_argument('--w_qed', type=str, required=True)
    parser_parse.add_argument('--w_sa', type=str, required=True)
    parser_parse.add_argument('--w_clash', type=str, required=True)
    parser_parse.add_argument('--eval_dir', type=str, required=True,
                             help='Directory containing evaluation results')
    parser_parse.add_argument('--output_csv', type=str, required=True,
                             help='Output CSV file')

    # generate_summary subcommand
    parser_summary = subparsers.add_parser('generate_summary',
                                           help='Generate summary report from results')
    parser_summary.add_argument('--results_csv', type=str, required=True,
                               help='Path to results CSV file')
    parser_summary.add_argument('--output_dir', type=str, required=True,
                               help='Output directory for summary files')

    args = parser.parse_args()

    if args.command == 'generate_weight_combinations':
        generate_weight_combinations(args)
    elif args.command == 'sample_train_subset':
        sample_train_subset(args)
    elif args.command == 'parse_eval_results':
        parse_eval_results(args)
    elif args.command == 'generate_summary':
        generate_summary(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
