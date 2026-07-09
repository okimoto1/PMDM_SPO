import sys, os, time
import argparse
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, SDMolSupplier, SDWriter
from rdkit.Chem import FilterCatalog
import glob
from collections import defaultdict


def parse_mol_and_ignore_valence_tags(mol):
    """
    PMDM-generated SDFs carry explicit-H / valence tags from reconstruction
    that make RDKit's default sanitize fail or mis-count Hs. Clear those tags,
    let RDKit recompute implicit Hs, then sanitize and reassign stereochemistry
    so property calculation runs on a chemically consistent molecule.
    """
    mol_fixed = Chem.Mol(mol)
    for atom in mol_fixed.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    Chem.SanitizeMol(mol_fixed)
    Chem.AssignStereochemistry(mol_fixed, cleanIt=True)
    return mol_fixed


def center_molecule(mol):
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    centroid = np.zeros(3)
    
    for atom_idx in range(num_atoms):
        pos = conf.GetAtomPosition(atom_idx)
        centroid += np.array([pos.x, pos.y, pos.z])
    
    centroid /= num_atoms
    
    for atom_idx in range(num_atoms):
        pos = conf.GetAtomPosition(atom_idx)
        new_pos = pos - centroid
        conf.SetAtomPosition(atom_idx, new_pos)

def process_mol_with_smiles(mol, use_smiles_conversion=False):
    """Process molecule with fixval base and optional SMILES conversion.

    Mirrors evaluate.py: always apply fixval (parse_mol_and_ignore_valence_tags),
    then optionally do a SMILES round-trip on top when use_smiles_conversion is set.
    """
    if mol is None:
        return None

    try:
        processed_mol = parse_mol_and_ignore_valence_tags(mol)
    except Exception as e:
        print(f"Warning: fixval failed, skipping molecule: {e}")
        return None

    if use_smiles_conversion:
        try:
            # Convert to SMILES then back to mol on top of fixval
            smiles = Chem.MolToSmiles(processed_mol)
            smiles_mol = Chem.MolFromSmiles(smiles)
            if smiles_mol is not None:
                # Ensure ring info is computed (exactly like mod version)
                try:
                    Chem.GetSymmSSSR(smiles_mol)
                except Exception as ring_e:
                    print(f"Warning: Failed to compute ring info for molecule: {ring_e}")
                    return processed_mol  # fallback to fixval mol if ring computation fails
                return smiles_mol
            else:
                return processed_mol  # fallback to fixval mol if conversion fails
        except Exception as e:
            print(f"Warning: SMILES conversion failed, using fixval mol: {e}")
            return processed_mol

    return processed_mol

def filter_molecules_single_file(inp_sdf, use_smiles_conversion=False):
    """Process a single SDF file and return statistics"""
    catalog = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)

    try:
        supplier = SDMolSupplier(inp_sdf, sanitize=False, removeHs=False)
    except:
        print(f"Warning: Could not read {inp_sdf}")
        return None

    total_molecules = 0
    skipped_molecules = 0
    removed_count = 0
    filtered_count = 0

    for mol in supplier:
        total_molecules += 1
        if mol is None:
            skipped_molecules += 1
            continue  # invalid molecule

        # Process molecule with optional SMILES conversion
        processed_mol = process_mol_with_smiles(mol, use_smiles_conversion)
        if processed_mol is None:
            skipped_molecules += 1
            continue

        matched = False
        for entry_id in range(catalog.GetNumEntries()):
            entry = catalog.GetEntryWithIdx(entry_id)
            if entry.HasFilterMatch(processed_mol):
                removed_count += 1
                matched = True
                break  # once matched

        if not matched:
            filtered_count += 1

    # Calculate removal rate
    valid_molecules = total_molecules - skipped_molecules
    removal_rate = float(removed_count / valid_molecules) if valid_molecules > 0 else 0.0

    return {
        'file': inp_sdf,
        'total_molecules': total_molecules,
        'skipped_molecules': skipped_molecules,
        'removed_count': removed_count,
        'filtered_count': filtered_count,
        'removal_rate': removal_rate
    }

def filter_molecules(inp_sdf, out_removed_sdf, out_filtered_sdf, use_smiles_conversion=False):
    """Original function for single file processing with output files"""
    catalog = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)

    supplier = SDMolSupplier(inp_sdf, sanitize=False, removeHs=False)
    #supplier = SDMolSupplier(inp_sdf, sanitize=False)
    # outputfiles
    removed_writer = SDWriter(out_removed_sdf)
    filtered_writer = SDWriter(out_filtered_sdf)

    total_molecules = 0
    skipped_molecules = 0
    removed_count = 0
    filtered_count = 0

    for mol in supplier:
        total_molecules += 1
        if mol is None:
            skipped_molecules += 1
            continue  # invalid molecule

        # Process molecule with optional SMILES conversion
        processed_mol = process_mol_with_smiles(mol, use_smiles_conversion)
        if processed_mol is None:
            skipped_molecules += 1
            continue

        matched = False
        for entry_id in range(catalog.GetNumEntries()):
            entry = catalog.GetEntryWithIdx(entry_id)
            if entry.HasFilterMatch(processed_mol):
                description = entry.GetDescription()
                mol.SetProp("MatchedFilter", description)  # add tag to original mol
                center_molecule(mol)
                removed_writer.write(mol)
                removed_count += 1
                matched = True
                break  # once matched

        if not matched:
            filtered_writer.write(mol)
            filtered_count += 1

    # close writer
    removed_writer.close()
    filtered_writer.close()

    # output
    print(f"Total molecules: {total_molecules}")
    print(f"Skipped molecules: {skipped_molecules}")
    print(f"Removed molecules: {removed_count}")
    print(f"Filtered molecules: {filtered_count}")
    print(f"Rate(Removed/(Total-Skip)):", float(removed_count / (total_molecules - skipped_molecules)))

def find_sdf_files(directory):
    """Recursively find all SDF files in directory and subdirectories"""
    sdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.sdf'):
                sdf_files.append(os.path.join(root, file))
    return sdf_files

def process_pkl_file(pkl_file, eval_actual=False, dataset="crossdock", use_smiles_conversion=False):
    """Process a PKL file and calculate Brenk filter statistics"""
    print(f"Processing PKL file: {pkl_file}")
    
    catalog = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading PKL file: {e}")
        return

    # Handle eval_actual mode
    if eval_actual:
        # Determine protein root directory based on dataset
        if dataset == "crossdock":
            protein_root = "./data/crossdocked_pocket10"
        elif dataset == "pdbind":
            protein_root = "./data/pdbind"
        elif dataset == "crossdock_pdbind":
            protein_root = "./data/crossdock_pdbind"
        else:
            print(f"Unknown dataset: {dataset}, using default crossdock")
            protein_root = "./data/crossdocked_pocket10"

        print(f"Loading actual ligands from {protein_root}")
        ligand_set = set()
        actual_data = []
        
        # Load actual ligands instead of generated ones
        for d in data:
            protein_filename = d["protein_file"]
            ligand_filename = d["ligand_file"]
            
            if ligand_filename in ligand_set:
                continue
            ligand_set.add(ligand_filename)
            
            ligand_path = os.path.join(protein_root, ligand_filename)
            
            try:
                actual_mol = next(Chem.SDMolSupplier(ligand_path))
                if actual_mol is not None:
                    # Create a new entry with the actual molecule
                    actual_entry = d.copy()
                    actual_entry["mol"] = actual_mol
                    actual_data.append(actual_entry)
            except Exception as e:
                print(f"Error loading actual ligand {ligand_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(actual_data)} actual ligand molecules")
        data = actual_data
    
    total_molecules = 0
    skipped_molecules = 0
    removed_count = 0
    filtered_count = 0
    
    for d in data:
        total_molecules += 1
        mol = d["mol"]
        
        if mol is None:
            skipped_molecules += 1
            continue
        
        # fixval 
        try:
            processed_mol = parse_mol_and_ignore_valence_tags(mol)
        except Exception as e:
            print(f"Warning: Failed to fixval molecule {total_molecules}: {e}")
            skipped_molecules += 1
            continue

        if use_smiles_conversion:
            # Convert mol->smiles->mol on top of fixval
            smiles = Chem.MolToSmiles(processed_mol)
            processed_mol = Chem.MolFromSmiles(smiles)
            try:
                Chem.GetSymmSSSR(processed_mol)
            except Exception as e:
                print(f"Warning: Failed to process molecule {total_molecules}: {e}")
                skipped_molecules += 1
                continue
        
        matched = False
        for entry_id in range(catalog.GetNumEntries()):
            entry = catalog.GetEntryWithIdx(entry_id)
            if entry.HasFilterMatch(processed_mol):
                removed_count += 1
                matched = True
                break
        
        if not matched:
            filtered_count += 1
    
    # Calculate removal rate
    valid_molecules = total_molecules - skipped_molecules
    removal_rate = float(removed_count / valid_molecules) if valid_molecules > 0 else 0.0
    
    # Print results
    print("\n" + "="*60)
    print("PKL FILE BRENK FILTER STATISTICS")
    print("="*60)
    print(f"Total molecules: {total_molecules}")
    print(f"Skipped molecules: {skipped_molecules}")
    print(f"Removed molecules: {removed_count}")
    print(f"Filtered molecules: {filtered_count}")
    print(f"Removal rate: {removal_rate:.4f}")

def process_directory(directory, use_smiles_conversion=False):
    """Process all SDF files in directory and calculate mean statistics"""
    print(f"Searching for SDF files in: {directory}")
    sdf_files = find_sdf_files(directory)
    
    if not sdf_files:
        print("No SDF files found in the directory and subdirectories.")
        return
    
    print(f"Found {len(sdf_files)} SDF files")
    
    all_stats = []
    failed_files = []
    
    for sdf_file in sdf_files:
        print(f"Processing: {sdf_file}")
        stats = filter_molecules_single_file(sdf_file, use_smiles_conversion)
        if stats is not None:
            all_stats.append(stats)
        else:
            failed_files.append(sdf_file)
    
    if not all_stats:
        print("No valid SDF files could be processed.")
        return
    
    # Calculate mean statistics
    total_files = len(all_stats)
    mean_total_molecules = np.mean([s['total_molecules'] for s in all_stats])
    mean_skipped_molecules = np.mean([s['skipped_molecules'] for s in all_stats])
    mean_removed_count = np.mean([s['removed_count'] for s in all_stats])
    mean_filtered_count = np.mean([s['filtered_count'] for s in all_stats])
    mean_removal_rate = np.mean([s['removal_rate'] for s in all_stats])
    
    # Calculate standard deviations
    std_total_molecules = np.std([s['total_molecules'] for s in all_stats])
    std_skipped_molecules = np.std([s['skipped_molecules'] for s in all_stats])
    std_removed_count = np.std([s['removed_count'] for s in all_stats])
    std_filtered_count = np.std([s['filtered_count'] for s in all_stats])
    std_removal_rate = np.std([s['removal_rate'] for s in all_stats])
    
    # Print results
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total SDF files processed: {total_files}")
    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nMean statistics across all files:")
    print(f"  Total molecules per file: {mean_total_molecules:.2f} ± {std_total_molecules:.2f}")
    print(f"  Skipped molecules per file: {mean_skipped_molecules:.2f} ± {std_skipped_molecules:.2f}")
    print(f"  Removed molecules per file: {mean_removed_count:.2f} ± {std_removed_count:.2f}")
    print(f"  Filtered molecules per file: {mean_filtered_count:.2f} ± {std_filtered_count:.2f}")
    print(f"  Removal rate per file: {mean_removal_rate:.4f} ± {std_removal_rate:.4f}")
    
    # Calculate total across all files
    total_all_molecules = sum([s['total_molecules'] for s in all_stats])
    total_all_skipped = sum([s['skipped_molecules'] for s in all_stats])
    total_all_removed = sum([s['removed_count'] for s in all_stats])
    total_all_filtered = sum([s['filtered_count'] for s in all_stats])
    overall_removal_rate = total_all_removed / (total_all_molecules - total_all_skipped) if (total_all_molecules - total_all_skipped) > 0 else 0.0
    
    print(f"\nTotal across all files:")
    print(f"  Total molecules: {total_all_molecules}")
    print(f"  Total skipped molecules: {total_all_skipped}")
    print(f"  Total removed molecules: {total_all_removed}")
    print(f"  Total filtered molecules: {total_all_filtered}")
    print(f"  Overall removal rate: {overall_removal_rate:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter molecules by Brenk PAINS filter and calculate statistics."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input file or directory (.sdf, .pkl file, or directory containing SDF files)",
    )
    parser.add_argument(
        "--eval_actual",
        action="store_true",
        help="Evaluate actual ligands instead of generated ones (only for PKL input)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="crossdock",
        help="Dataset name for protein root directory (default: crossdock)",
    )
    parser.add_argument(
        "--use_smiles_conversion",
        action="store_true",
        help="Use SMILES conversion method: mol -> SMILES -> mol + GetSymmSSSR (without sanitization)",
    )
    args = parser.parse_args()
    
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"No such file or directory: '{input_path}'")
        sys.exit(1)
    
    start_time = time.time()
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.pkl'):
            # PKL file processing with optional eval_actual mode
            process_pkl_file(input_path, args.eval_actual, args.dataset, args.use_smiles_conversion)
        elif input_path.lower().endswith('.sdf'):
            # Original functionality for single SDF file
            if args.eval_actual:
                print("Warning: --eval_actual is ignored for SDF input files")
            out_removed_sdf = "removed.sdf"
            out_filtered_sdf = "filtered.sdf"
            
            filter_molecules(input_path, out_removed_sdf, out_filtered_sdf, args.use_smiles_conversion)
        else:
            print("Input file must be an SDF or PKL file")
            sys.exit(1)
        
    elif os.path.isdir(input_path):
        # Directory processing
        if args.eval_actual:
            print("Warning: --eval_actual is ignored for directory input")
        process_directory(input_path, args.use_smiles_conversion)
    
    else:
        print(f"'{input_path}' is neither a file nor a directory")
        sys.exit(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")

