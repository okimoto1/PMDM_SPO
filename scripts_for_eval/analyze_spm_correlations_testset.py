"""
Analyze SPM correlations with MolSkill, QED, SA, N/O ratio, and ring size.

This script:
1. Loads ligands from db.sqlite (uses metrics table for QED, SA, molskill, ring info)
2. Splits into test set using the same logic as dataset.py (80/10/10, seed=42)
3. Computes correlations between SPM scores and various metrics
4. Generates scatter plots and violin plots

Usage:
    docker compose exec pmdm_spo bash -c "source /opt/mol/bin/activate && python scripts_for_eval/analyze_spm_correlations_testset.py"

    # Custom paths:
    docker compose exec pmdm_spo bash -c "source /opt/mol/bin/activate && python scripts_for_eval/analyze_spm_correlations_testset.py \\
        --db_path /workspace/SPO/db.sqlite \\
        --spm_scores /workspace/SPO/ligand_spm/analysis_v4_dbsqlite/spm_molskill_scores.csv \\
        --output_dir /workspace/SPO/ligand_spm/analysis_v4_dbsqlite"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import sqlite3
from rdkit import Chem
from tqdm import tqdm
import sys
import os
import argparse
import random


# Color and label settings (Namiki and PMDM only)
COLORS = {"ns": "#FF6B6B", "non_ns": "#4ECDC4"}
LABELS_MAP = {"ns": "Namiki", "non_ns": "PMDM"}


def get_test_ligand_ids(db_path, train_ratio=0.8, val_ratio=0.1, random_seed=42):
    """
    Get ligand IDs that appear in test set pairs.
    Uses the same split logic as dataset.py.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, ligand_a_id, ligand_b_id FROM pair ORDER BY id")
    all_pairs = cursor.fetchall()
    conn.close()

    indices = list(range(len(all_pairs)))
    rng = random.Random(random_seed)
    rng.shuffle(indices)

    n_train = int(len(all_pairs) * train_ratio)
    n_val = int(len(all_pairs) * val_ratio)
    test_indices = indices[n_train + n_val:]

    print(f"Total pairs: {len(all_pairs)}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {len(test_indices)}")

    test_ligand_ids = set()
    for idx in test_indices:
        pair = all_pairs[idx]
        test_ligand_ids.add(pair[1])
        test_ligand_ids.add(pair[2])

    return test_ligand_ids


def load_ligands_from_db(db_path, ligand_ids):
    """
    Load ligand data with metrics from db.sqlite.
    Reads QED, SA, molskill, ring info directly from the metrics table.
    Only computes N/O ratio from SMILES.
    """
    conn = sqlite3.connect(db_path)

    ligand_ids_str = ",".join(map(str, ligand_ids))
    query = f"""
    SELECT l.id, l.name, l.smiles,
           m.qed, m.sa, m.molskill_score,
           m.ring_9, m.ring_10, m.ring_11, m.ring_larger
    FROM ligand l
    JOIN metrics m ON l.id = m.ligand_id
    WHERE l.id IN ({ligand_ids_str})
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Determine source_type from name prefix
    df["source_type"] = df["name"].apply(lambda x: "ns" if x.startswith("NS") else "non_ns")

    # Has large ring (>= 9) from metrics table
    df["has_large_ring"] = (
        df["ring_9"].fillna(0) + df["ring_10"].fillna(0) +
        df["ring_11"].fillna(0) + df["ring_larger"].fillna(0)
    ) > 0

    # N/O ratio (only thing that needs RDKit)
    print("Computing N/O ratios...")
    n_o_ratios = []
    for smiles in tqdm(df["smiles"].values, desc="N/O ratio"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
                o_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
                n_o_ratios.append(n_count / o_count if o_count > 0 else np.nan)
            else:
                n_o_ratios.append(np.nan)
        except:
            n_o_ratios.append(np.nan)
    df["n_o_ratio"] = n_o_ratios

    return df


def compute_correlations(df_subset, score_col):
    """Compute per-source Spearman correlations."""
    results = {}
    for source in ["ns", "non_ns"]:
        mask = df_subset["source_type"] == source
        if mask.sum() > 1:
            r, p = spearmanr(df_subset.loc[mask, "spm_score"], df_subset.loc[mask, score_col])
            results[source] = (r, mask.sum())
    return results


def plot_correlation(df_data, score_col, xlabel, title, output_path):
    """Create scatter plot with per-source breakdown."""
    fig, ax = plt.subplots(figsize=(12, 9))

    spm = df_data["spm_score"].values
    scores = df_data[score_col].values
    spearman_r, _ = spearmanr(spm, scores)
    correlations = compute_correlations(df_data, score_col)

    for source in ["non_ns", "ns"]:
        mask = df_data["source_type"] == source
        if mask.sum() > 0 and source in correlations:
            r = correlations[source][0]
            ax.scatter(df_data.loc[mask, score_col], df_data.loc[mask, "spm_score"],
                       alpha=0.3, s=12,
                       label=f"{LABELS_MAP[source]} (n={mask.sum():,}, r={r:.3f})",
                       c=COLORS[source])

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("SPM Score (higher = better)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)

    textstr = f"Total n = {len(df_data):,}\nSpearman r = {spearman_r:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment="bottom", bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    return spearman_r, correlations


def plot_ring_size_violin(df_data, output_path):
    """Create violin plot comparing SPM scores by ring size."""
    fig, ax = plt.subplots(figsize=(10, 8))

    no_large_ring = df_data[df_data["has_large_ring"] == False]["spm_score"].values
    has_large_ring = df_data[df_data["has_large_ring"] == True]["spm_score"].values

    print(f"No large ring (< 9): {len(no_large_ring)}")
    print(f"Has large ring (>= 9): {len(has_large_ring)}")

    # Violin plot
    parts = ax.violinplot([no_large_ring, has_large_ring], positions=[1, 2],
                          showmeans=True, showmedians=True)

    # Set colors
    for pc in parts["bodies"]:
        pc.set_facecolor("#4ECDC4")
        pc.set_alpha(0.7)

    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("black")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"No Large Ring\n(ring size < 9)\nn={len(no_large_ring):,}",
                        f"Has Large Ring\n(ring size >= 9)\nn={len(has_large_ring):,}"], fontsize=12)
    ax.set_ylabel("SPM Score (higher = better)", fontsize=14)
    ax.set_title("SPM Score Distribution by Ring Size", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics
    mean_no = np.mean(no_large_ring)
    mean_yes = np.mean(has_large_ring) if len(has_large_ring) > 0 else 0
    textstr = f"Mean (no large ring): {mean_no:.4f}\nMean (has large ring): {mean_yes:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    return mean_no, mean_yes


def main():
    parser = argparse.ArgumentParser(description="Analyze SPM correlations on test set")
    parser.add_argument("--db_path", type=str, default="/workspace/SPO/db.sqlite",
                        help="Path to db.sqlite (with metrics table)")
    parser.add_argument("--spm_scores", type=str,
                        default="/workspace/SPO/ligand_spm/analysis_v4_dbsqlite/spm_molskill_scores.csv",
                        help="Path to SPM scores CSV (from calculate_spm_molskill_correlation.py)")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/SPO/ligand_spm/analysis_v4_dbsqlite",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get test set ligand IDs
    print("Extracting test set ligands...")
    test_ligand_ids = get_test_ligand_ids(args.db_path)
    print(f"Unique test ligands: {len(test_ligand_ids)}")

    # Load ligand data with metrics from database
    print("\nLoading ligand data from database...")
    df = load_ligands_from_db(args.db_path, test_ligand_ids)
    print(f"Loaded {len(df)} ligands with metrics")
    print("Source type distribution:")
    print(df["source_type"].value_counts())

    # Merge with SPM scores
    spm_df = pd.read_csv(args.spm_scores)
    df = df.merge(spm_df[["ligand_name", "spm_score"]],
                  left_on="name", right_on="ligand_name", how="inner")
    print(f"\nMerged with SPM scores: {len(df)} ligands")

    if len(df) < 50:
        print("Not enough samples for meaningful analysis!")
        return

    # Prepare subsets
    df_molskill = df.dropna(subset=["spm_score", "molskill_score", "source_type"])
    df_qed = df.dropna(subset=["spm_score", "qed", "source_type"])
    df_sa = df.dropna(subset=["spm_score", "sa", "source_type"])
    df_ring = df.dropna(subset=["spm_score", "has_large_ring"])
    df_ratio = df.dropna(subset=["spm_score", "n_o_ratio", "source_type"])

    print(f"\nValid: MolSkill={len(df_molskill)}, QED={len(df_qed)}, "
          f"SA={len(df_sa)}, Ring={len(df_ring)}, N/O={len(df_ratio)}")

    # ========== Plot 1: SPM vs MolSkill ==========
    print("\n" + "=" * 60)
    print("SPM vs MolSkill:")
    r, corrs = plot_correlation(
        df_molskill, "molskill_score",
        "MolSkill Score (lower = better)",
        "SPM vs MolSkill Correlation",
        os.path.join(args.output_dir, "spm_vs_molskill.png")
    )
    print(f"  Total: r = {r:.4f}")
    for src, (r_src, n) in corrs.items():
        print(f"  {LABELS_MAP[src]}: r = {r_src:.4f}, n = {n}")

    # ========== Plot 2: SPM vs QED ==========
    print("\n" + "=" * 60)
    print("SPM vs QED:")
    r, corrs = plot_correlation(
        df_qed, "qed",
        "QED Score (higher = more drug-like)",
        "SPM vs QED Correlation",
        os.path.join(args.output_dir, "spm_vs_qed.png")
    )
    print(f"  Total: r = {r:.4f}")
    for src, (r_src, n) in corrs.items():
        print(f"  {LABELS_MAP[src]}: r = {r_src:.4f}, n = {n}")

    # ========== Plot 3: SPM vs SA ==========
    print("\n" + "=" * 60)
    print("SPM vs SA:")
    r, corrs = plot_correlation(
        df_sa, "sa",
        "SA Score (higher = easier to synthesize)",
        "SPM vs SA Correlation",
        os.path.join(args.output_dir, "spm_vs_sa.png")
    )
    print(f"  Total: r = {r:.4f}")
    for src, (r_src, n) in corrs.items():
        print(f"  {LABELS_MAP[src]}: r = {r_src:.4f}, n = {n}")

    # ========== Plot 4: Ring Size Violin ==========
    print("\n" + "=" * 60)
    print("SPM by Ring Size:")
    mean_no, mean_yes = plot_ring_size_violin(
        df_ring,
        os.path.join(args.output_dir, "spm_violin_by_ring_size.png")
    )
    print(f"  No large ring: mean = {mean_no:.4f}")
    print(f"  Has large ring: mean = {mean_yes:.4f}")

    # ========== Plot 5: SPM vs N/O Ratio ==========
    print("\n" + "=" * 60)
    print("SPM vs N/O Ratio:")
    r, corrs = plot_correlation(
        df_ratio, "n_o_ratio",
        "N/O Ratio (N atoms / O atoms)",
        "SPM vs N/O Ratio Correlation",
        os.path.join(args.output_dir, "spm_vs_no_ratio.png")
    )
    print(f"  Total: r = {r:.4f}")
    for src, (r_src, n) in corrs.items():
        print(f"  {LABELS_MAP[src]}: r = {r_src:.4f}, n = {n}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
