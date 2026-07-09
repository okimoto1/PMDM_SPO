#!/usr/bin/env python
"""
File: calc_dihedral_kldiv_all.py
Description:
    Calculate KL divergence between dihedral angle distributions of test set and generated
    molecule set for all predefined dihedral angle SMARTS patterns. Input is a .pkl file
    containing:
      - Test set molecules given by sample["ligand_file"] (path string, load first molecule from path)
      - Generated molecules provided directly by sample["mol"] (RDKit molecule object)
    Supported dihedral angle SMARTS patterns:
      "CCCC"  -> [C][C][C][C]
      "cccc"  -> c:c:c:c
      "CCCO"  -> [C][C][C][O]
      "OCCO"  -> [O][C][C][O]
      "Cccc"  -> [C]c:c:c
      "CC=CC" -> [C][C]=[C][C]
Usage:
    python calc_dihedral_kldiv_all.py --ligand_files samples_all.pkl samples_all2.pkl --root_dir /path/to/data 
          [--output_prefix dihedral_histogram] [--n_cores 8] [--set_names "Generated Set 1,Generated Set 2"]
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
from scipy.stats import entropy
from multiprocessing import Pool, cpu_count
from functools import partial

# Predefined dihedral angle SMARTS pattern dictionary
dihedral_patterns = {
    "CCCC": "[C][C][C][C]",
    "cccc": "c:c:c:c",
    "CCCO": "[C][C][C][O]",
    "OCCO": "[O][C][C][O]",
    "Cccc": "[C]c:c:c",
    "CC=CC": "[C][C]=[C][C]"
}

def process_dihedral_angles(mol, pattern_smarts):
    """
    For a single molecule, extract all dihedral angles (formed by 4 atoms) matching the given
    SMARTS pattern and return list of angles (in degrees).
    """
    angles = []
    if mol is None:
        return angles
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return angles
    if mol.GetNumConformers() == 0:
        return angles
    conf = mol.GetConformer()
    pattern = Chem.MolFromSmarts(pattern_smarts)
    if pattern is None:
        return angles
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        if len(match) == 4:
            try:
                angle = GetDihedralDeg(conf, match[0], match[1], match[2], match[3])
                angles.append(angle)
            except Exception as e:
                print("Warning: Failed to compute dihedral angle for match:", match, e)
    return angles

def extract_angles_from_mols(mols, process_func, pattern_smarts, n_cores):
    """
    Use multiprocessing to extract angles from molecule list (works for dihedral angles or bond angles).
    """
    with Pool(n_cores) as pool:
        func = partial(process_func, pattern_smarts=pattern_smarts)
        results = pool.map(func, mols)
    # Flatten list
    angles = [angle for sublist in results for angle in sublist]
    return angles

def create_histogram(angles, bins=30, angle_range=(-180, 180)):
    """
    Create normalized histogram from angle list.
    """
    if len(angles) == 0:
        return None, None
    hist, edges = np.histogram(angles, bins=bins, range=angle_range, density=True)
    return hist, edges

def kl_divergence(p, q, epsilon=1e-10):
    """
    Calculate KL divergence between two distributions.
    """
    p = p / np.sum(p)
    q = q / np.sum(q)
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return entropy(p, q)

def calculate_cccc_range_ratio(angles):
    """
    Calculate the ratio of cccc dihedral angles within the range:
    24° to 156° or -156° to -24°
    
    Args:
        angles: List of dihedral angles in degrees
    
    Returns:
        ratio: Proportion of angles within the specified range
    """
    if len(angles) == 0:
        return 0.0
    
    angles_array = np.array(angles)
    # Count angles in the range: (24 <= angle <= 156) or (-156 <= angle <= -24)
    in_range = np.logical_or(
        np.logical_and(angles_array >= 24, angles_array <= 156),
        np.logical_and(angles_array >= -156, angles_array <= -24)
    )
    ratio = np.sum(in_range) / len(angles_array)
    return ratio

def plot_histogram(hist_test, hist_gen_list, gen_names, edges, mode, output_file, kl_div_values):
    """
    Plot comparison histogram of dihedral angles between test set and multiple generated sets,
    annotate with KL divergence, and save to output file.
    """
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(10, 6))
    
    # Define distinct colors for better visualization
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot test set with solid line
    plt.plot(bin_centers, hist_test, label="Test Set", color=colors[0], linewidth=2)
    
    # Plot each generated set with solid lines of different colors
    for i, (hist_gen, name) in enumerate(zip(hist_gen_list, gen_names)):
        plt.plot(bin_centers, hist_gen, label=name, color=colors[(i+1) % len(colors)], linewidth=2)
    
    plt.xlabel("Dihedral Angle (degrees)")
    plt.ylabel("Density")
    plt.title(f"Dihedral Angle Histogram Comparison ({mode})")
    
    # Add legend first to determine its position
    legend = plt.legend(loc='upper right')
    
    # Get legend bounding box to avoid overlap
    legend_bbox = legend.get_window_extent()
    ax = plt.gca()
    ax_bbox = ax.get_window_extent()
    
    # Calculate relative position of legend in axes coordinates
    legend_left = (legend_bbox.x0 - ax_bbox.x0) / (ax_bbox.x1 - ax_bbox.x0)
    legend_bottom = (legend_bbox.y0 - ax_bbox.y0) / (ax_bbox.y1 - ax_bbox.y0)
    
    # Position annotation to avoid legend overlap
    # If legend is in upper right, put annotation in upper left
    if legend_left > 0.5:
        annotation_x = 0.05
    else:
        annotation_x = 0.65
    
    # Add KL divergence annotation with aligned text
    max_name_len = max(len(name) for name in gen_names)
    annotation_text = ""
    for name, kl_div in zip(gen_names, kl_div_values):
        annotation_text += f"{name:<{max_name_len}} KL Div: {kl_div:.4f}\n"

    plt.text(annotation_x, 0.95, annotation_text.strip(),
             transform=plt.gca().transAxes, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8))
    
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Calculate KL divergence of dihedral angle distributions between test set and generated molecules"
    )
    parser.add_argument("--ligand_files", type=str, nargs='+', required=True, 
                        help="Paths to sample pkl files (can specify multiple)")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing test set molecule files")
    parser.add_argument("--output_prefix", type=str, default="dihedral_histogram",
                        help="Output image prefix, final filename will be output_prefix_mode.png")
    parser.add_argument("--n_cores", type=int, default=cpu_count()-1, help="Number of CPU cores to use")
    parser.add_argument("--set_names", type=str, default="",
                        help="Comma-separated names for generated sets (default: 'Generated Set 1, Generated Set 2, ...')")
    args = parser.parse_args()

    # Initialize set names
    if args.set_names:
        gen_set_names = args.set_names.split(',')
    else:
        gen_set_names = [f"Generated Set {i+1}" for i in range(len(args.ligand_files))]
    
    # Ensure we have enough names for all ligand files
    if len(gen_set_names) < len(args.ligand_files):
        for i in range(len(gen_set_names), len(args.ligand_files)):
            gen_set_names.append(f"Generated Set {i+1}")
    
    print(f"Processing {len(args.ligand_files)} ligand files")
    
    # Load all sample pkl files
    all_gen_mols = []
    test_mols = []
    first_file = True
    
    for i, ligand_file in enumerate(args.ligand_files):
        with open(ligand_file, "rb") as f:
            samples = pickle.load(f)
        print(f"Loaded {len(samples)} samples from {ligand_file}")
        
        # Extract generated molecules
        gen_mols = [sample["mol"] for sample in samples if sample.get("mol") is not None]
        all_gen_mols.append(gen_mols)
        
        # Only extract test molecules from the first ligand file to avoid duplicates
        if first_file:
            for sample in samples:
                ligand_path = sample.get("ligand_file")
                if ligand_path:
                    full_path = os.path.join(args.root_dir, ligand_path)
                    suppl = Chem.SDMolSupplier(full_path, sanitize=False)
                    if len(suppl) > 0 and suppl[0] is not None:
                        test_mols.append(suppl[0])
            first_file = False

    print(f"Number of test set molecules: {len(test_mols)}")
    for i, gen_mols in enumerate(all_gen_mols):
        print(f"Number of {gen_set_names[i]} molecules: {len(gen_mols)}")
    
    if len(test_mols) == 0 or any(len(gen_mols) == 0 for gen_mols in all_gen_mols):
        print("Error: Test set or at least one generated set has 0 molecules")
        return

    for mode, pattern_smarts in dihedral_patterns.items():
        print("\n==============================")
        print(f"Calculating dihedral angle pattern: {mode}, SMARTS: {pattern_smarts}")
        start_time = time.time()
        
        # Extract angles from test set
        test_angles = extract_angles_from_mols(test_mols, process_dihedral_angles, pattern_smarts, args.n_cores)
        print(f"Extracted {len(test_angles)} dihedral angles from test set")
        
        # Extract angles from all generated sets
        all_gen_angles = []
        for i, gen_mols in enumerate(all_gen_mols):
            gen_angles = extract_angles_from_mols(gen_mols, process_dihedral_angles, pattern_smarts, args.n_cores)
            all_gen_angles.append(gen_angles)
            print(f"Extracted {len(gen_angles)} dihedral angles from {gen_set_names[i]}")
        
        end_time = time.time()
        print(f"Time taken for angle extraction: {end_time - start_time:.2f} seconds")

        # Create histograms
        hist_test, edges = create_histogram(test_angles, bins=30, angle_range=(-180, 180))
        if hist_test is None:
            print("Error: Cannot build histogram for test set")
            continue
            
        all_hist_gen = []
        kl_divergences = []
        
        for i, gen_angles in enumerate(all_gen_angles):
            hist_gen, _ = create_histogram(gen_angles, bins=30, angle_range=(-180, 180))
            if hist_gen is None:
                print(f"Error: Cannot build histogram for {gen_set_names[i]}")
                continue
                
            all_hist_gen.append(hist_gen)
            divergence = kl_divergence(hist_test, hist_gen)
            kl_divergences.append(divergence)
            print(f"Dihedral angle mode [{mode}] {gen_set_names[i]} KL Divergence: {divergence}")
        
        # Calculate cccc range ratio if mode is cccc
        if mode == "cccc":
            print("\n--- cccc Range (24°~156° or -156°~-24°) Ratio ---")
            test_ratio = calculate_cccc_range_ratio(test_angles)
            print(f"Test Set: {test_ratio:.4f} ({test_ratio*100:.2f}%)")
            
            for i, gen_angles in enumerate(all_gen_angles):
                gen_ratio = calculate_cccc_range_ratio(gen_angles)
                print(f"{gen_set_names[i]}: {gen_ratio:.4f} ({gen_ratio*100:.2f}%)")
        
        # Plot histograms
        output_file = f"{args.output_prefix}_{mode}.png"
        plot_histogram(hist_test, all_hist_gen, gen_set_names, edges, mode, output_file, kl_divergences)
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    main()