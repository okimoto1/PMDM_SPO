import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import shutil
import glob

# RDKit imports
from rdkit import Chem

# BioPython for protein parsing
from Bio.PDB import PDBParser


def get_atom_3d_position(mol, atom_idx):
    """
    Retrieve the 3D coordinate of an atom from an RDKit Mol object.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit Mol with 3D coordinates.
    atom_idx : int
        The index of the atom.

    Returns
    -------
    np.ndarray
        A numpy array of shape (3, ) with the x, y, z coordinates.
    """
    conf = mol.GetConformer()
    pos = conf.GetAtomPosition(atom_idx)
    return np.array([pos.x, pos.y, pos.z], dtype=float)


def check_protein_ligand_collision(mol, protein_file, protein_root, dist_thresh=2.5):
    """
    Check for collisions between protein and ligand molecules.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The ligand molecule with 3D coordinates.
    protein_file : str
        Path to the protein PDB file relative to protein_root, or absolute path.
    protein_root : str
        Root directory for protein files (ignored if protein_file is absolute path).
    dist_thresh : float
        Distance threshold for collision detection in Angstroms.
        
    Returns
    -------
    dict
        Dictionary containing collision information.
    """
    # Construct full protein file path
    if os.path.isabs(protein_file):
        # protein_file is already an absolute path (directory mode)
        full_protein_path = protein_file
    else:
        # protein_file is relative to protein_root (original mode)
        full_protein_path = os.path.join(protein_root, protein_file)
    
    if not os.path.exists(full_protein_path):
        return {"has_collision": False, "error": f"Protein file not found: {full_protein_path}"}
    
    # Parse protein structure
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", full_protein_path)
    except Exception as e:
        return {"has_collision": False, "error": f"Error parsing protein: {str(e)}"}
    
    collision_details = []
    all_distances = []
    has_collision = False
    
    # Check each ligand atom against protein atoms
    for atom_idx in range(mol.GetNumAtoms()):
        # Skip hydrogen atoms in ligand
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 1:  # H atom
            continue
            
        try:
            atom_pos = get_atom_3d_position(mol, atom_idx)
        except Exception:
            continue
            
        ligand_atom = mol.GetAtomWithIdx(atom_idx)
        ligand_symbol = ligand_atom.GetSymbol()
        
        # Check against all protein atoms
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip water and other hetero atoms
                    if residue.get_resname() in ["HOH", "WAT"]:
                        continue
                        
                    for protein_atom in residue:
                        # Skip hydrogen atoms in protein
                        if protein_atom.element == "H":
                            continue
                            
                        dist = np.linalg.norm(protein_atom.coord - atom_pos)
                        
                        # Record all distances
                        all_distances.append({
                            "ligand_atom_idx": atom_idx,
                            "ligand_atom_symbol": ligand_symbol,
                            "protein_residue": residue.get_resname(),
                            "protein_atom": protein_atom.get_name(),
                            "distance": dist
                        })
                        
                        # Only count as collision if below threshold
                        if dist < dist_thresh:
                            collision_details.append({
                                "ligand_atom_idx": atom_idx,
                                "ligand_atom_symbol": ligand_symbol,
                                "protein_residue": residue.get_resname(),
                                "protein_atom": protein_atom.get_name(),
                                "distance": dist
                            })
                            has_collision = True
    
    return {
        "has_collision": has_collision,
        "collision_count": len(collision_details),
        "collision_details": collision_details,
        "all_distances": all_distances,
        "error": None
    }


def load_actual_test_data(dataset, split_file=None):
    """
    Load actual ligand molecules from test dataset split.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('crossdock', 'pdbind', 'crossdock_pdbind').
    split_file : str, optional
        Path to split file. If None, uses default for dataset.
        
    Returns
    -------
    list
        List of dictionaries with 'mol', 'protein_file', 'ligand_file', 'smile' keys.
    """
    # Determine protein root directory
    if dataset == 'crossdock':
        protein_root = './data/crossdocked_pocket10'
        if split_file is None:
            split_file = './data/split_by_name.pt'
    elif dataset == 'pdbind':
        protein_root = './data/pdbind'
        if split_file is None:
            split_file = './data/pdbind_split_by_name.pt'
    elif dataset == 'crossdock_pdbind':
        protein_root = './data/crossdock_pdbind'
        if split_file is None:
            split_file = './data/crossdock_pdbind_split_by_name.pt'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load split data
    if split_file.endswith('.pkl'):
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
    elif split_file.endswith('.pt'):
        import torch
        split_data = torch.load(split_file)
    else:
        raise ValueError(f"Unsupported split file format: {split_file}")
    
    if 'test' not in split_data:
        raise ValueError(f"No 'test' key found in split data. Available keys: {list(split_data.keys())}")
    
    test_data = split_data['test']
    print(f"Found {len(test_data)} test samples in split file")
    
    # Load actual ligand molecules
    ligand_set = set()
    actual_data = []
    
    for protein_filename, ligand_filename in tqdm(test_data, desc="Loading actual ligands"):
        # Skip duplicates (same as evaluate.py logic)
        if ligand_filename in ligand_set:
            continue
        ligand_set.add(ligand_filename)
        
        ligand_path = os.path.join(protein_root, ligand_filename)
        
        # Load actual ligand molecule from SDF file
        try:
            actual_mol = next(Chem.SDMolSupplier(ligand_path))
            if actual_mol is not None:
                # Create entry with actual molecule
                actual_entry = {
                    'mol': actual_mol,
                    'protein_file': protein_filename,
                    'ligand_file': ligand_filename,
                    'smile': Chem.MolToSmiles(actual_mol)
                }
                actual_data.append(actual_entry)
        except Exception as e:
            print(f"Error loading ligand {ligand_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(actual_data)} unique actual ligand molecules")
    return actual_data


def load_directory_pairs(input_dir):
    """
    Scan directory and subdirectories to find paired xxx_ligand.sdf and xxx_protein.pdb files.
    
    Parameters
    ----------
    input_dir : str
        Root directory to scan for file pairs.
        
    Returns
    -------
    list
        List of dictionaries with 'mol', 'protein_file', 'ligand_file', 'smile' keys.
    """
    print(f"Scanning directory: {input_dir}")
    
    # Find all ligand.sdf files
    ligand_pattern = os.path.join(input_dir, "**", "*_ligand.sdf")
    ligand_files = glob.glob(ligand_pattern, recursive=True)
    
    print(f"Found {len(ligand_files)} ligand files")
    
    paired_data = []
    missing_proteins = []
    invalid_ligands = []
    
    for ligand_file in tqdm(ligand_files, desc="Processing ligand-protein pairs"):
        # Extract the base name (remove _ligand.sdf)
        ligand_path = Path(ligand_file)
        base_name = ligand_path.stem.replace("_ligand", "")
        
        # Look for corresponding protein file in the same directory
        protein_file = ligand_path.parent / f"{base_name}_protein.pdb"
        
        if not protein_file.exists():
            missing_proteins.append(ligand_file)
            continue
        
        # Load ligand molecule from SDF file
        try:
            mol = next(Chem.SDMolSupplier(str(ligand_file)))
            if mol is not None:
                # Create entry with molecule and file paths
                entry = {
                    'mol': mol,
                    'protein_file': str(protein_file),  # Use absolute path for directory mode
                    'ligand_file': str(ligand_file),
                    'smile': Chem.MolToSmiles(mol),
                    'base_name': base_name
                }
                paired_data.append(entry)
            else:
                invalid_ligands.append(ligand_file)
        except Exception as e:
            print(f"Error loading ligand {ligand_file}: {e}")
            invalid_ligands.append(ligand_file)
    
    print(f"Successfully loaded {len(paired_data)} ligand-protein pairs")
    if missing_proteins:
        print(f"Warning: {len(missing_proteins)} ligand files missing corresponding protein files")
    if invalid_ligands:
        print(f"Warning: {len(invalid_ligands)} ligand files could not be loaded")
    
    return paired_data


def evaluate_sample_collisions(sample_file, dataset, dist_thresh=2.5, eval_actual=False, split_file=None, input_dir=None):
    """
    Evaluate collision frequency for all molecules in a sample file, actual test data, or directory pairs.
    
    Parameters
    ----------
    sample_file : str
        Path to the sample pickle file (ignored if eval_actual=True or input_dir is provided).
    dataset : str
        Dataset name to determine protein root directory (ignored if input_dir is provided).
    dist_thresh : float
        Distance threshold for collision detection.
    eval_actual : bool
        If True, evaluate actual ligands from test dataset instead of generated samples.
    split_file : str, optional
        Path to split file for actual data evaluation.
    input_dir : str, optional
        If provided, scan this directory for paired ligand.sdf and protein.pdb files.
        
    Returns
    -------
    dict
        Dictionary containing collision statistics.
    """
    # Load data based on mode
    if input_dir:
        print("Loading ligand-protein pairs from directory...")
        data = load_directory_pairs(input_dir)
        eval_mode = "directory"
    elif eval_actual:
        print("Loading actual test ligands...")
        data = load_actual_test_data(dataset, split_file)
        eval_mode = "actual"
    else:
        print("Loading generated samples...")
        # Load sample data
        with open(sample_file, 'rb') as f:
            data = pickle.load(f)
        eval_mode = "generated"
    
    # Determine protein root directory
    if input_dir:
        # In directory mode, protein files have absolute paths
        protein_root = ""
    else:
        if dataset == 'crossdock':
            protein_root = './data/crossdocked_pocket10'
        elif dataset == 'pdbind':
            protein_root = './data/pdbind'
        elif dataset == 'crossdock_pdbind':
            protein_root = './data/crossdock_pdbind'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    # Initialize statistics
    collision_stats = {
        "total_samples": 0,
        "samples_with_collisions": 0,
        "total_collisions": 0,
        "collision_details": [],
        "distance_distribution": [],
        "all_distance_distribution": [],
        "samples_with_errors": 0,
        "error_details": [],
        "eval_mode": eval_mode
    }
    
    print(f"Evaluating collisions for {len(data)} samples...")
    
    # Process each sample
    for i, sample in enumerate(tqdm(data)):
        try:
            mol = sample['mol']
            protein_file = sample['protein_file']
            
            # Check collisions
            collision_result = check_protein_ligand_collision(
                mol, protein_file, protein_root, dist_thresh
            )
            
            if collision_result["error"]:
                collision_stats["samples_with_errors"] += 1
                collision_stats["error_details"].append({
                    "sample_idx": i,
                    "protein_file": protein_file,
                    "error": collision_result["error"]
                })
                continue
            
            # Update statistics
            collision_stats["total_samples"] += 1
            
            if collision_result["has_collision"]:
                collision_stats["samples_with_collisions"] += 1
                collision_stats["total_collisions"] += collision_result["collision_count"]
                
                # Store collision details
                sample_collision = {
                    "sample_idx": i,
                    "protein_file": protein_file,
                    "ligand_file": sample.get('ligand_file', ''),
                    "smile": sample.get('smile', ''),
                    "collision_count": collision_result["collision_count"],
                    "collisions": collision_result["collision_details"]
                }
                collision_stats["collision_details"].append(sample_collision)
                
                # Collect distance distribution
                for collision in collision_result["collision_details"]:
                    collision_stats["distance_distribution"].append(collision["distance"])
                
            # Collect all distance distribution
            for distance_info in collision_result["all_distances"]:
                collision_stats["all_distance_distribution"].append(distance_info["distance"])
        
        except Exception as e:
            collision_stats["samples_with_errors"] += 1
            collision_stats["error_details"].append({
                "sample_idx": i,
                "protein_file": sample.get('protein_file', 'unknown'),
                "error": f"Processing error: {str(e)}"
            })
    
    return collision_stats


def plot_collision_statistics(collision_stats, output_dir):
    """
    Generate visualizations of collision statistics.
    
    Parameters
    ----------
    collision_stats : dict
        Dictionary containing collision statistics.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Collision frequency pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of samples with/without collisions
    total_valid = collision_stats["total_samples"]
    samples_with_collisions = collision_stats["samples_with_collisions"]
    samples_without_collisions = total_valid - samples_with_collisions
    
    if total_valid > 0:
        labels = ['With Collisions', 'Without Collisions']
        sizes = [samples_with_collisions, samples_without_collisions]
        colors = ['lightcoral', 'lightgreen']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax1.pie(
            sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors
        )
        ax1.set_title(f'Collision Frequency\n(Total valid samples: {total_valid})')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
    
    # 2. Distance distribution histogram
    if collision_stats["all_distance_distribution"]:
        distances = collision_stats["all_distance_distribution"]
        ax2.hist(distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Distance (Angstrom)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'All Distance Distribution\n(Total distances: {len(distances)})')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add threshold line at 2.5Å
        ax2.axvline(2.5, color='red', linestyle='-', linewidth=2, 
                   label='Collision threshold (2.5Å)')
        
        # Add statistics text
        mean_dist = np.mean(distances)
        ax2.axvline(mean_dist, color='orange', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_dist:.2f}Å')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No distances recorded', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('All Distance Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Collision count distribution
    if collision_stats["collision_details"]:
        collision_counts = [detail["collision_count"] for detail in collision_stats["collision_details"]]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Bar plot of collision count distribution
        max_count = max(collision_counts)
        count_bins = range(max_count + 2)
        count_freq = [collision_counts.count(i) for i in count_bins[:-1]]
        
        bars = ax.bar(count_bins[:-1], count_freq, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Number of Collisions per Sample')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Distribution of Collision Counts per Sample')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, freq in zip(bars, count_freq):
            if freq > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{freq}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'collision_count_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def save_collision_details(collision_stats, output_dir):
    """
    Save detailed collision information to files.
    
    Parameters
    ----------
    collision_stats : dict
        Dictionary containing collision statistics.
    output_dir : str
        Directory to save files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    eval_mode = collision_stats.get("eval_mode", "unknown")
    summary_file = os.path.join(output_dir, f'collision_summary_{eval_mode}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Collision Analysis Summary ({eval_mode.upper()} mode)\n")
        f.write("=" * 40 + "\n\n")
        
        total_samples = collision_stats["total_samples"]
        samples_with_collisions = collision_stats["samples_with_collisions"]
        total_collisions = collision_stats["total_collisions"]
        
        f.write(f"Evaluation mode: {eval_mode}\n")
        f.write(f"Total valid samples: {total_samples}\n")
        f.write(f"Samples with collisions: {samples_with_collisions}\n")
        f.write(f"Collision rate: {samples_with_collisions/total_samples*100:.2f}%\n")
        f.write(f"Total collision count: {total_collisions}\n")
        
        if samples_with_collisions > 0:
            avg_collisions = total_collisions / samples_with_collisions
            f.write(f"Average collisions per sample (with collisions): {avg_collisions:.2f}\n")
        
        if collision_stats["distance_distribution"]:
            distances = collision_stats["distance_distribution"]
            f.write(f"Average collision distance: {np.mean(distances):.3f} Å\n")
            f.write(f"Min collision distance: {np.min(distances):.3f} Å\n")
            f.write(f"Max collision distance: {np.max(distances):.3f} Å\n")
        
        f.write(f"\nSamples with errors: {collision_stats['samples_with_errors']}\n")
    
    # Save detailed collision information
    eval_mode = collision_stats.get("eval_mode", "unknown")
    details_file = os.path.join(output_dir, f'collision_details_{eval_mode}.pkl')
    with open(details_file, 'wb') as f:
        pickle.dump(collision_stats, f)
    
    # Save collision samples for further analysis
    if collision_stats["collision_details"]:
        collision_samples_dir = os.path.join(output_dir, f'collision_samples_{eval_mode}')
        os.makedirs(collision_samples_dir, exist_ok=True)
        
        for sample_info in collision_stats["collision_details"]:
            sample_idx = sample_info["sample_idx"]
            collision_count = sample_info["collision_count"]
            
            sample_file = os.path.join(collision_samples_dir, f'sample_{sample_idx}_collisions_{collision_count}.txt')
            with open(sample_file, 'w') as f:
                f.write(f"Sample {sample_idx} Collision Details ({eval_mode.upper()} mode)\n")
                f.write("=" * 50 + "\n")
                f.write(f"Protein file: {sample_info['protein_file']}\n")
                if sample_info.get('ligand_file'):
                    f.write(f"Ligand file: {sample_info['ligand_file']}\n")
                f.write(f"SMILES: {sample_info['smile']}\n")
                f.write(f"Total collisions: {collision_count}\n\n")
                
                for i, collision in enumerate(sample_info["collisions"], 1):
                    f.write(f"Collision {i}:\n")
                    f.write(f"  Ligand atom: {collision['ligand_atom_idx']} ({collision['ligand_atom_symbol']})\n")
                    f.write(f"  Protein: {collision['protein_residue']} {collision['protein_atom']}\n")
                    f.write(f"  Distance: {collision['distance']:.3f} Å\n\n")


def check_max_distance_outliers(mol, protein_file, protein_root, max_dist_thresh=80.0):
    """
    Check for ligand-protein pairs where the maximum distance between atoms exceeds threshold.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The ligand molecule with 3D coordinates.
    protein_file : str
        Path to the protein PDB file relative to protein_root, or absolute path.
    protein_root : str
        Root directory for protein files (ignored if protein_file is absolute path).
    max_dist_thresh : float
        Maximum distance threshold in Angstroms.
        
    Returns
    -------
    dict
        Dictionary containing distance outlier information.
    """
    # Construct full protein file path
    if os.path.isabs(protein_file):
        # protein_file is already an absolute path (directory mode)
        full_protein_path = protein_file
    else:
        # protein_file is relative to protein_root (original mode)
        full_protein_path = os.path.join(protein_root, protein_file)
    
    if not os.path.exists(full_protein_path):
        return {"is_outlier": False, "error": f"Protein file not found: {full_protein_path}"}
    
    # Parse protein structure
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", full_protein_path)
    except Exception as e:
        return {"is_outlier": False, "error": f"Error parsing protein: {str(e)}"}
    
    max_distance = 0.0
    max_distance_info = None
    all_distances = []
    
    # Check each ligand atom against protein atoms
    for atom_idx in range(mol.GetNumAtoms()):
        try:
            atom_pos = get_atom_3d_position(mol, atom_idx)
        except Exception:
            continue
            
        ligand_atom = mol.GetAtomWithIdx(atom_idx)
        ligand_symbol = ligand_atom.GetSymbol()
        
        # Check against all protein atoms
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip water and other hetero atoms
                    if residue.get_resname() in ["HOH", "WAT"]:
                        continue
                        
                    for protein_atom in residue:
                        dist = np.linalg.norm(protein_atom.coord - atom_pos)
                        all_distances.append(dist)
                        
                        # Track maximum distance
                        if dist > max_distance:
                            max_distance = dist
                            max_distance_info = {
                                "ligand_atom_idx": atom_idx,
                                "ligand_atom_symbol": ligand_symbol,
                                "protein_residue": residue.get_resname(),
                                "protein_atom": protein_atom.get_name(),
                                "distance": dist
                            }
    
    is_outlier = max_distance > max_dist_thresh
    
    return {
        "is_outlier": is_outlier,
        "max_distance": max_distance,
        "max_distance_info": max_distance_info,
        "all_distances": all_distances,
        "error": None
    }


def save_outlier_structures(sample, sample_idx, protein_root, outlier_dir, max_distance_info):
    """
    Save PDB and SDF files for distance outlier pairs.
    
    Parameters
    ----------
    sample : dict
        Sample data containing mol, protein_file, ligand_file info.
    sample_idx : int
        Index of the sample.
    protein_root : str
        Root directory for protein files.
    outlier_dir : str
        Directory to save outlier structures.
    max_distance_info : dict
        Information about the maximum distance.
    """
    os.makedirs(outlier_dir, exist_ok=True)
    
    # Create subdirectory for this outlier pair
    pair_dir = os.path.join(outlier_dir, f"sample_{sample_idx}_maxdist_{max_distance_info['distance']:.1f}A")
    os.makedirs(pair_dir, exist_ok=True)
    
    # Save protein PDB file
    protein_file = sample['protein_file']
    if os.path.isabs(protein_file):
        # Directory mode: protein_file is already absolute path
        full_protein_path = protein_file
    else:
        # Original mode: construct path using protein_root
        full_protein_path = os.path.join(protein_root, protein_file)
    
    if os.path.exists(full_protein_path):
        protein_out_path = os.path.join(pair_dir, f"protein_{sample_idx}.pdb")
        shutil.copy2(full_protein_path, protein_out_path)
    
    # Save ligand SDF file
    mol = sample['mol']
    ligand_out_path = os.path.join(pair_dir, f"ligand_{sample_idx}.sdf")
    writer = Chem.SDWriter(ligand_out_path)
    writer.write(mol)
    writer.close()
    
    # Save information file
    info_file = os.path.join(pair_dir, f"outlier_info_{sample_idx}.txt")
    with open(info_file, 'w') as f:
        f.write(f"Distance Outlier Information - Sample {sample_idx}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Protein file: {protein_file}\n")
        if sample.get('ligand_file'):
            f.write(f"Ligand file: {sample['ligand_file']}\n")
        f.write(f"SMILES: {sample.get('smile', '')}\n")
        f.write(f"Maximum distance: {max_distance_info['distance']:.3f} Å\n\n")
        
        f.write("Maximum distance details:\n")
        f.write(f"  Ligand atom: {max_distance_info['ligand_atom_idx']} ({max_distance_info['ligand_atom_symbol']})\n")
        f.write(f"  Protein: {max_distance_info['protein_residue']} {max_distance_info['protein_atom']}\n")
        f.write(f"  Distance: {max_distance_info['distance']:.3f} Å\n")


def evaluate_distance_outliers(sample_file, dataset, max_dist_thresh=80.0, eval_actual=False, split_file=None, input_dir=None):
    """
    Evaluate and save ligand-protein pairs with maximum distances exceeding threshold.
    
    Parameters
    ----------
    sample_file : str
        Path to the sample pickle file (ignored if eval_actual=True or input_dir is provided).
    dataset : str
        Dataset name to determine protein root directory (ignored if input_dir is provided).
    max_dist_thresh : float
        Maximum distance threshold for outlier detection.
    eval_actual : bool
        If True, evaluate actual ligands from test dataset instead of generated samples.
    split_file : str, optional
        Path to split file for actual data evaluation.
    input_dir : str, optional
        If provided, scan this directory for paired ligand.sdf and protein.pdb files.
        
    Returns
    -------
    dict
        Dictionary containing outlier statistics.
    """
    # Load data based on mode
    if input_dir:
        print("Loading ligand-protein pairs from directory for distance outlier analysis...")
        data = load_directory_pairs(input_dir)
        eval_mode = "directory"
    elif eval_actual:
        print("Loading actual test ligands for distance outlier analysis...")
        data = load_actual_test_data(dataset, split_file)
        eval_mode = "actual"
    else:
        print("Loading generated samples for distance outlier analysis...")
        # Load sample data
        with open(sample_file, 'rb') as f:
            data = pickle.load(f)
        eval_mode = "generated"
    
    # Determine protein root directory
    if input_dir:
        # In directory mode, protein files have absolute paths
        protein_root = ""
    else:
        if dataset == 'crossdock':
            protein_root = './data/crossdocked_pocket10'
        elif dataset == 'pdbind':
            protein_root = './data/pdbind'
        elif dataset == 'crossdock_pdbind':
            protein_root = './data/crossdock_pdbind'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    # Initialize statistics
    outlier_stats = {
        "total_samples": 0,
        "outlier_samples": 0,
        "outlier_details": [],
        "max_distances": [],
        "samples_with_errors": 0,
        "error_details": [],
        "eval_mode": eval_mode,
        "max_dist_thresh": max_dist_thresh
    }
    
    # Create outlier output directory
    if input_dir:
        input_dir_name = os.path.basename(os.path.abspath(input_dir))
        outlier_dir = f'./distance_outliers_{input_dir_name}_{eval_mode}_thresh{max_dist_thresh}A'
    elif eval_actual:
        outlier_dir = f'./distance_outliers_{dataset}_{eval_mode}_thresh{max_dist_thresh}A'
    else:
        sample_dir = os.path.dirname(sample_file)
        sample_name = os.path.splitext(os.path.basename(sample_file))[0]
        outlier_dir = os.path.join(sample_dir, f'{sample_name}_distance_outliers_thresh{max_dist_thresh}A')
    
    print(f"Evaluating distance outliers for {len(data)} samples...")
    print(f"Outlier structures will be saved to: {outlier_dir}")
    
    # Process each sample
    for i, sample in enumerate(tqdm(data)):
        try:
            mol = sample['mol']
            protein_file = sample['protein_file']
            
            # Check for distance outliers
            outlier_result = check_max_distance_outliers(
                mol, protein_file, protein_root, max_dist_thresh
            )
            
            if outlier_result["error"]:
                outlier_stats["samples_with_errors"] += 1
                outlier_stats["error_details"].append({
                    "sample_idx": i,
                    "protein_file": protein_file,
                    "error": outlier_result["error"]
                })
                continue
            
            # Update statistics
            outlier_stats["total_samples"] += 1
            outlier_stats["max_distances"].append(outlier_result["max_distance"])
            
            if outlier_result["is_outlier"]:
                outlier_stats["outlier_samples"] += 1
                
                # Store outlier details
                sample_outlier = {
                    "sample_idx": i,
                    "protein_file": protein_file,
                    "ligand_file": sample.get('ligand_file', ''),
                    "smile": sample.get('smile', ''),
                    "max_distance": outlier_result["max_distance"],
                    "max_distance_info": outlier_result["max_distance_info"]
                }
                outlier_stats["outlier_details"].append(sample_outlier)
                
                # Save outlier structures
                save_outlier_structures(
                    sample, i, protein_root, outlier_dir, 
                    outlier_result["max_distance_info"]
                )
        
        except Exception as e:
            outlier_stats["samples_with_errors"] += 1
            outlier_stats["error_details"].append({
                "sample_idx": i,
                "protein_file": sample.get('protein_file', 'unknown'),
                "error": f"Processing error: {str(e)}"
            })
    
    return outlier_stats, outlier_dir


def plot_distance_outlier_statistics(outlier_stats, output_dir):
    """
    Generate visualizations of distance outlier statistics.
    
    Parameters
    ----------
    outlier_stats : dict
        Dictionary containing outlier statistics.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Outlier frequency and max distance distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of samples with/without outliers
    total_valid = outlier_stats["total_samples"]
    outlier_samples = outlier_stats["outlier_samples"]
    normal_samples = total_valid - outlier_samples
    
    if total_valid > 0:
        labels = [f'Distance Outliers\n(>{outlier_stats["max_dist_thresh"]}Å)', 'Normal Samples']
        sizes = [outlier_samples, normal_samples]
        colors = ['lightcoral', 'lightgreen']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax1.pie(
            sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors
        )
        ax1.set_title(f'Distance Outlier Frequency\n(Total valid samples: {total_valid})')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
    
    # 2. Maximum distance distribution histogram
    if outlier_stats["max_distances"]:
        max_distances = outlier_stats["max_distances"]
        ax2.hist(max_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Maximum Distance (Angstrom)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Maximum Distance Distribution\n(Total samples: {len(max_distances)})')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add threshold line
        ax2.axvline(outlier_stats["max_dist_thresh"], color='red', linestyle='-', linewidth=2, 
                   label=f'Outlier threshold ({outlier_stats["max_dist_thresh"]}Å)')
        
        # Add statistics text
        mean_dist = np.mean(max_distances)
        ax2.axvline(mean_dist, color='orange', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_dist:.1f}Å')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No distances recorded', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Maximum Distance Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_outlier_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_distance_outlier_summary(outlier_stats, output_dir):
    """
    Save summary of distance outlier analysis.
    
    Parameters
    ----------
    outlier_stats : dict
        Dictionary containing outlier statistics.
    output_dir : str
        Directory to save files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    eval_mode = outlier_stats.get("eval_mode", "unknown")
    summary_file = os.path.join(output_dir, f'distance_outlier_summary_{eval_mode}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Distance Outlier Analysis Summary ({eval_mode.upper()} mode)\n")
        f.write("=" * 50 + "\n\n")
        
        total_samples = outlier_stats["total_samples"]
        outlier_samples = outlier_stats["outlier_samples"]
        max_dist_thresh = outlier_stats["max_dist_thresh"]
        
        f.write(f"Evaluation mode: {eval_mode}\n")
        f.write(f"Distance threshold: {max_dist_thresh} Å\n")
        f.write(f"Total valid samples: {total_samples}\n")
        f.write(f"Distance outlier samples: {outlier_samples}\n")
        if total_samples > 0:
            f.write(f"Outlier rate: {outlier_samples/total_samples*100:.2f}%\n")
        
        if outlier_stats["max_distances"]:
            distances = outlier_stats["max_distances"]
            f.write(f"\nDistance statistics:\n")
            f.write(f"Average maximum distance: {np.mean(distances):.3f} Å\n")
            f.write(f"Min maximum distance: {np.min(distances):.3f} Å\n")
            f.write(f"Max maximum distance: {np.max(distances):.3f} Å\n")
        
        f.write(f"\nSamples with errors: {outlier_stats['samples_with_errors']}\n")
        
        if outlier_stats["outlier_details"]:
            f.write(f"\nOutlier samples:\n")
            for detail in outlier_stats["outlier_details"]:
                f.write(f"Sample {detail['sample_idx']}: {detail['max_distance']:.1f} Å - {detail['protein_file']}\n")
    
    # Save detailed outlier information
    details_file = os.path.join(output_dir, f'distance_outlier_details_{eval_mode}.pkl')
    with open(details_file, 'wb') as f:
        pickle.dump(outlier_stats, f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate collision frequency in sample files, actual test data, or directory pairs")
    parser.add_argument(
        '--sample_file', 
        type=str, 
        help='Path to the sample pickle file (required if not using --eval_actual or --input_dir)'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='crossdock',
        choices=['crossdock', 'pdbind', 'crossdock_pdbind'],
        help='Dataset name to determine protein root directory (ignored if using --input_dir)'
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        help='Directory to scan for paired xxx_ligand.sdf and xxx_protein.pdb files'
    )
    parser.add_argument(
        '--dist_thresh', 
        type=float, 
        default=2.5,
        help='Distance threshold for collision detection (Angstroms)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory for results (default: based on dataset/sample file)'
    )
    parser.add_argument(
        '--eval_actual', 
        action='store_true',
        help='Evaluate actual ligands from test dataset instead of generated samples'
    )
    parser.add_argument(
        '--split_file', 
        type=str,
        help='Path to split file for actual data evaluation (optional, uses default based on dataset)'
    )
    parser.add_argument(
        '--eval_outliers', 
        action='store_true',
        help='Evaluate distance outliers and save PDB/SDF files for pairs with max distance > threshold'
    )
    parser.add_argument(
        '--max_dist_thresh', 
        type=float, 
        default=80.0,
        help='Maximum distance threshold for outlier detection (Angstroms)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.eval_actual and not args.sample_file and not args.input_dir:
        parser.error("--sample_file is required when not using --eval_actual or --input_dir")
    
    # Validate mutually exclusive options
    mode_count = sum([bool(args.eval_actual), bool(args.sample_file), bool(args.input_dir)])
    if mode_count > 1:
        parser.error("Only one of --eval_actual, --sample_file, or --input_dir can be used at a time")
    
    # Determine output directory
    if args.output_dir is None:
        if args.input_dir:
            input_dir_name = os.path.basename(os.path.abspath(args.input_dir))
            if args.eval_outliers:
                args.output_dir = f'./outlier_analysis_{input_dir_name}_directory'
            else:
                args.output_dir = f'./collision_analysis_{input_dir_name}_directory'
        elif args.eval_actual:
            if args.eval_outliers:
                args.output_dir = f'./outlier_analysis_{args.dataset}_actual'
            else:
                args.output_dir = f'./collision_analysis_{args.dataset}_actual'
        else:
            sample_dir = os.path.dirname(args.sample_file)
            sample_name = os.path.splitext(os.path.basename(args.sample_file))[0]
            if args.eval_outliers:
                args.output_dir = os.path.join(sample_dir, f'{sample_name}_outlier_analysis')
            else:
                args.output_dir = os.path.join(sample_dir, f'{sample_name}_collision_analysis')
    
    if args.input_dir:
        print(f"Evaluation mode: Directory pairs")
        print(f"Input directory: {args.input_dir}")
    elif args.eval_actual:
        print(f"Evaluation mode: Actual test data")
        print(f"Dataset: {args.dataset}")
        if args.split_file:
            print(f"Split file: {args.split_file}")
    else:
        print(f"Evaluation mode: Generated samples")
        print(f"Sample file: {args.sample_file}")
        print(f"Dataset: {args.dataset}")
    
    if args.eval_outliers:
        print(f"Analysis type: Distance outlier detection")
        print(f"Maximum distance threshold: {args.max_dist_thresh} Å")
        print(f"Output directory: {args.output_dir}")
        
        # Evaluate distance outliers
        print("\nStarting distance outlier evaluation...")
        outlier_stats, outlier_structures_dir = evaluate_distance_outliers(
            args.sample_file, 
            args.dataset, 
            args.max_dist_thresh,
            args.eval_actual,
            args.split_file,
            args.input_dir
        )
        
        # Generate visualizations
        print("\nGenerating outlier visualizations...")
        plot_distance_outlier_statistics(outlier_stats, args.output_dir)
        
        # Save detailed results
        print("Saving detailed outlier results...")
        save_distance_outlier_summary(outlier_stats, args.output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("DISTANCE OUTLIER ANALYSIS SUMMARY")
        print("="*50)
        
        total_samples = outlier_stats["total_samples"]
        outlier_samples = outlier_stats["outlier_samples"]
        max_dist_thresh = outlier_stats["max_dist_thresh"]
        
        if total_samples > 0:
            print(f"Total valid samples: {total_samples}")
            print(f"Distance outlier samples (>{max_dist_thresh}Å): {outlier_samples} ({outlier_samples/total_samples*100:.1f}%)")
            
            if outlier_stats["max_distances"]:
                distances = outlier_stats["max_distances"]
                print(f"Average maximum distance: {np.mean(distances):.3f} Å")
                print(f"Distance range: {np.min(distances):.3f} - {np.max(distances):.3f} Å")
        
        if outlier_stats["samples_with_errors"] > 0:
            print(f"\nWarning: {outlier_stats['samples_with_errors']} samples had errors")
        
        eval_mode = outlier_stats.get("eval_mode", "unknown")
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Outlier structures saved to: {outlier_structures_dir}")
        print("Files generated:")
        print("- distance_outlier_statistics.png")
        print(f"- distance_outlier_summary_{eval_mode}.txt")
        print(f"- distance_outlier_details_{eval_mode}.pkl")
        print(f"- Individual outlier PDB/SDF files in {outlier_structures_dir}")
        
    else:
        print(f"Analysis type: Collision detection")
        print(f"Distance threshold: {args.dist_thresh} Å")
        print(f"Output directory: {args.output_dir}")
        
        # Evaluate collisions
        print("\nStarting collision evaluation...")
        collision_stats = evaluate_sample_collisions(
            args.sample_file, 
            args.dataset, 
            args.dist_thresh,
            args.eval_actual,
            args.split_file,
            args.input_dir
        )
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_collision_statistics(collision_stats, args.output_dir)
        
        # Save detailed results
        print("Saving detailed results...")
        save_collision_details(collision_stats, args.output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("COLLISION ANALYSIS SUMMARY")
        print("="*50)
        
        total_samples = collision_stats["total_samples"]
        samples_with_collisions = collision_stats["samples_with_collisions"]
        total_collisions = collision_stats["total_collisions"]
        
        if total_samples > 0:
            print(f"Total valid samples: {total_samples}")
            print(f"Samples with collisions: {samples_with_collisions} ({samples_with_collisions/total_samples*100:.1f}%)")
            print(f"Total collision count: {total_collisions}")
            
            if samples_with_collisions > 0:
                avg_collisions = total_collisions / samples_with_collisions
                print(f"Average collisions per sample (with collisions): {avg_collisions:.2f}")
            
            if collision_stats["distance_distribution"]:
                distances = collision_stats["distance_distribution"]
                print(f"Average collision distance: {np.mean(distances):.3f} Å")
                print(f"Distance range: {np.min(distances):.3f} - {np.max(distances):.3f} Å")
        
        if collision_stats["samples_with_errors"] > 0:
            print(f"\nWarning: {collision_stats['samples_with_errors']} samples had errors")
        
        eval_mode = collision_stats.get("eval_mode", "unknown")
        print(f"\nResults saved to: {args.output_dir}")
        print("Files generated:")
        print("- collision_statistics.png")
        print("- collision_count_distribution.png") 
        print(f"- collision_summary_{eval_mode}.txt")
        print(f"- collision_details_{eval_mode}.pkl")
        print(f"- collision_samples_{eval_mode}/ (individual collision details)")


if __name__ == "__main__":
    main() 