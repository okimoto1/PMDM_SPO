import sys
import argparse
from collections import defaultdict
import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem, SDMolSupplier, SDWriter
from rdkit.Chem import QED
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors
import sys
import os

# Add the evaluation directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
evaluation_dir = os.path.join(parent_dir, "evaluation")
sys.path.insert(0, evaluation_dir)
import sascorer


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


def calc_qed(mol):
    """
    QED(0-1)
    """
    try:
        return QED.qed(mol)
    except Exception:
        return -1


def calc_sa(mol):
    """
    SA0 (1-10)
    SA1 (0-1)
    """
    try:
        sa_score = sascorer.calculateScore(mol)
    except Exception:
        sa_score = -1
    if sa_score is None:
        sa_score = -1
    if 1 <= sa_score <= 10:
        sa_score_norm = (10 - sa_score) / 9
    else:
        sa_score_norm = -1
    return sa_score, sa_score_norm


def check_brenk(mol_calc, mol_output, smarts_mols, filt_lines):
    """
    Brenk Filter (105)
    mol_calc: molecule for calculation (mol1)
    mol_output: molecule for output (mol)
    """
    matched_indices = []
    matched_smarts = []
    match_counts = []
    for idx, unwanted in enumerate(smarts_mols):
        if unwanted is not None:
            matches = mol_calc.GetSubstructMatches(unwanted)
            count = len(matches)
            match_counts.append(str(count))
            if count > 0:
                matched_indices.append(str(idx + 1))
                matched_smarts.append(filt_lines[idx])
        else:
            match_counts.append("0")
    # mol.SetProp("Brenk_SMARTS_Index", ",".join(matched_indices))
    # mol.SetProp("Brenk_SMARTS_Patterns", " | ".join(matched_smarts))
    # mol.SetProp("Brenk_SMARTS_Count", str(len(matched_indices)))
    mol_output.SetProp(
        "Brenk_1hotvecs", ",".join(match_counts)
    )  # each SMARTS count (filt.txt order)
    # total count is also added as a tag
    total_count = sum(int(x) for x in match_counts)
    mol_output.SetProp("Brenk_SMARTS_TotalCount", str(total_count))
    return total_count


def check_ring_3_20_each(mol):
    """
    Check if there is a ring of each size from 3 to 20.
    Return a list of 0s and 1s, where 1 indicates the presence of a ring of that size.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    ring_sizes_in_mol = set(len(ring) for ring in atom_rings)
    return [1 if size in ring_sizes_in_mol else 0 for size in range(3, 21)]


def calc_lipinski(mol):
    """
    Lipinski's Rule of Five
    Return a list of 0s and 1s, where 1 indicates the presence of a ring of that size.
    """
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    nrot = Lipinski.NumRotatableBonds(mol)
    is_lipinski = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
    return mw, logp, hbd, hba, nrot, is_lipinski


def custom_bridgehead_atoms(mol):
    # RingInfo object
    ri = mol.GetRingInfo()
    # Dictionary to store the set of unique ring IDs for each atom
    # Key is atom index, value is the set of ring IDs that the atom belongs to
    atom_to_ring_ids = {i: set() for i in range(mol.GetNumAtoms())}
    # Get the list of atom indices for each ring
    # ri.AtomRings() returns a list of tuples of atom indices that make up each ring
    # print(ri.AtomRings())
    # Get the list of ring indices for each atom
    # ring_list is a list of tuples of atom indices that make up each ring
    for ring_idx, ring_atoms_tuple in enumerate(ri.AtomRings()):
        for atom_idx in ring_atoms_tuple:
            atom_to_ring_ids[atom_idx].add(ring_idx)  # Use ring index as ID
    # print(atom_to_ring_ids)
    multi_ring_atoms = {}
    for atom_idx, ring_ids_set in atom_to_ring_ids.items():
        if len(ring_ids_set) > 1:
            multi_ring_atoms[atom_idx] = len(ring_ids_set)
    # print(multi_ring_atoms)
    return multi_ring_atoms


def process_pkl_file(pkl_file, output_sdf, smarts_mols, filt_lines, eval_actual=False, dataset="crossdock"):
    """Process PKL file and calculate ring/Brenk filter statistics"""
    print(f"Processing PKL file: {pkl_file}")

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

    writer = SDWriter(output_sdf)

    total_molecules = 0
    skipped_count = 0
    skipped_invalid_mol = 0
    skipped_smiles_error = 0

    # Set data for average values
    all_brenk_data = []
    all_brenk_count_data = []
    all_ringsize_data = []
    all_nring_data = []
    all_aromatic_rings_data = []
    all_9plus_ring_data = []

    for d in data:
        total_molecules += 1
        mol = d["mol"]

        try:
            if mol is None:
                raise ValueError("Invalid molecule (None)")

            # fixval (mirror evaluate.py --readmol fixval): clear valence tags then sanitize
            mol = parse_mol_and_ignore_valence_tags(mol)
            mol1 = mol
        except Exception as e:
            print(f"Warning: Failed to fixval molecule {total_molecules}: {e}")
            skipped_count += 1
            skipped_invalid_mol += 1
            continue

        # Brenk Filter
        brenk_total_count = check_brenk(mol1, mol, smarts_mols, filt_lines)
        brenk_binary = 1 if brenk_total_count > 0 else 0
        all_brenk_count_data.append(brenk_total_count)
        all_brenk_data.append(brenk_binary)

        # RING: Check if there is a ring of each size from 3 to 20.
        ring_3_20_flags = check_ring_3_20_each(mol1)
        all_ringsize_data.append(ring_3_20_flags)

        for i, flag in enumerate(ring_3_20_flags, start=3):
            mol.SetProp(f"Ring{i}", str(flag))

        # RING: Check if there is a 3-4 ring
        has_3_4_ring = int(any(ring_3_20_flags[0:2]))
        mol.SetProp("Has_3_4_Ring", str(has_3_4_ring))

        # RING: Check if there is a 8-20 ring
        has_8_20_ring = int(any(ring_3_20_flags[5:]))
        mol.SetProp("Has_8_20_Ring", str(has_8_20_ring))

        # RING: Check if there is a 9+ ring (ring_3_20_flags[6:] corresponds to rings 9-20)
        has_9plus_ring = int(any(ring_3_20_flags[6:]))
        mol.SetProp("Has_9plus_Ring", str(has_9plus_ring))
        all_9plus_ring_data.append(has_9plus_ring)

        # OTHERS(aromatic ring count, total ring count, bridgehead, spiro atoms, TPSA)
        aromatic_rings = Lipinski.NumAromaticRings(mol1)
        num_rings = mol1.GetRingInfo().NumRings()
        all_nring_data.append(num_rings)
        all_aromatic_rings_data.append(aromatic_rings)

        writer.write(mol)

    writer.close()

    # Print average values and summary
    print_statistics(
        all_ringsize_data,
        all_nring_data,
        all_aromatic_rings_data,
        all_brenk_data,
        all_brenk_count_data,
        all_9plus_ring_data,
        output_sdf,
        total_molecules,
        skipped_count,
        skipped_invalid_mol,
        skipped_smiles_error,
    )


def print_statistics(
    all_ringsize_data,
    all_nring_data,
    all_aromatic_rings_data,
    all_brenk_data,
    all_brenk_count_data,
    all_9plus_ring_data,
    output_sdf,
    total_molecules,
    skipped_count,
    skipped_invalid_mol,
    skipped_smiles_error,
):
    """Print statistics for both SDF and PKL processing"""
    # Average values
    if all_ringsize_data:
        all_ringsize_array = np.array(all_ringsize_data)
        avg_ringsize_presence = np.mean(all_ringsize_array, axis=0)
        print(f"---Rate of ring size 3-20---")
        for i, avg in enumerate(avg_ringsize_presence, start=3):
            print(f"ring size {i}: {avg:.3f} ({avg*100:.1f}%)")

    if all_nring_data:
        all_nring_array = np.array(all_nring_data)
        avg_nring = np.mean(all_nring_array)
        print(f"---Average number of rings---")
        print(f"average number of rings: {avg_nring:.3f}")

    if all_aromatic_rings_data:
        all_aromatic_rings_array = np.array(all_aromatic_rings_data)
        avg_aromatic_rings = np.mean(all_aromatic_rings_array)
        print(f"---Average number of aromatic rings---")
        print(f"average number of aromatic rings: {avg_aromatic_rings:.3f}")

    if all_brenk_data:
        all_brenk_array = np.array(all_brenk_data)
        avg_brenk = np.mean(all_brenk_array)
        all_brenk_count_array = np.array(all_brenk_count_data)
        avg_brenk_count = np.mean(all_brenk_count_array)
        print(f"---Average number of brenk---")
        print(f"average number of brenk: {avg_brenk:.3f}")
        print(f"average number of brenk count: {avg_brenk_count:.3f}")

    if all_9plus_ring_data:
        all_9plus_ring_array = np.array(all_9plus_ring_data)
        percentage_9plus = np.mean(all_9plus_ring_array) * 100
        count_9plus = np.sum(all_9plus_ring_array)
        print(f"---Ligands with ring size >= 9---")
        print(f"Percentage of ligands with ring size >= 9: {percentage_9plus:.2f}% ({int(count_9plus)}/{len(all_9plus_ring_data)})")

    print(f"---SUMMARY---")
    print(f"Output SDF saved: {output_sdf}")
    print(f"Total molecules: {total_molecules}")
    print(f"Skipped molecules: {skipped_count}")
    print(f"  - Invalid molecules: {skipped_invalid_mol}")
    print(f"  - SMILES conversion errors: {skipped_smiles_error}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter SDF by SMARTS patterns and annotate with match info."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="all.sdf",
        help="Input SDF or PKL file (default: all.sdf)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.sdf",
        help="Output SDF file (default: out.sdf)",
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
    args = parser.parse_args()

    filt_lines = []  #
    smarts_mols = []  #

    # Brenk Filter File(filt.txt) is a list of SMARTS patterns.
    brenk_file = "./brenkfilter.txt"
    if not os.path.exists(brenk_file):
        brenk_file = "./brenk_filter/brenkfilter.txt"
    if not os.path.exists(brenk_file):
        print(f"Error: Could not find brenkfilter.txt file")
        sys.exit(1)

    with open(brenk_file, "r") as f:
        for line in f:
            filt_lines.append(line.rstrip("\n"))  #
            if line.startswith("#") or not line.strip():
                smarts_mols.append(None)
            else:
                smarts = line.strip().split()[0]
                mol = Chem.MolFromSmarts(smarts)
                if mol:
                    smarts_mols.append(mol)
                else:
                    smarts_mols.append(None)
                    print(f"Invalid SMARTS: {smarts}")

    print(
        f"Loaded {sum([m is not None for m in smarts_mols])} unwanted substructures (valid SMARTS)."
    )

    input_file = args.input
    output_sdf = args.output

    # Check input file type and process accordingly
    if input_file.lower().endswith(".pkl"):
        # Process PKL file
        process_pkl_file(input_file, output_sdf, smarts_mols, filt_lines, args.eval_actual, args.dataset)
    elif input_file.lower().endswith(".sdf"):
        # Process SDF file (original functionality)
        if args.eval_actual:
            print("Warning: --eval_actual is ignored for SDF input files")
        process_sdf_file(input_file, output_sdf, smarts_mols, filt_lines)
    else:
        print("Error: Input file must be either .sdf or .pkl file")
        sys.exit(1)


def process_sdf_file(input_sdf, output_sdf, smarts_mols, filt_lines):
    """Process SDF file (original functionality)"""
    # Read SDF file
    suppl = SDMolSupplier(input_sdf, removeHs=False, sanitize=False)
    writer = SDWriter(output_sdf)

    total_molecules = 0
    skipped_count = 0
    skipped_invalid_mol = 0
    skipped_smiles_error = 0

    # Set data for average values
    all_brenk_data = []
    all_brenk_count_data = []
    all_ringsize_data = []
    all_nring_data = []
    all_aromatic_rings_data = []
    all_9plus_ring_data = []

    for mol in suppl:
        total_molecules += 1
        try:
            if mol is None:
                raise ValueError("Invalid molecule (None)")
            # fixval (mirror evaluate.py --readmol fixval): clear valence tags then sanitize
            mol = parse_mol_and_ignore_valence_tags(mol)
            mol1 = mol
        except Exception as e:
            print(f"Warning: Failed to fixval molecule {total_molecules}: {e}")
            skipped_count += 1
            skipped_invalid_mol += 1
            continue

        # Brenk Filter
        brenk_total_count = check_brenk(mol1, mol, smarts_mols, filt_lines)
        brenk_binary = 1 if brenk_total_count > 0 else 0
        all_brenk_count_data.append(brenk_total_count)
        all_brenk_data.append(brenk_binary)

        # RING: Check if there is a ring of each size from 3 to 20.
        ring_3_20_flags = check_ring_3_20_each(mol1)
        all_ringsize_data.append(ring_3_20_flags)

        for i, flag in enumerate(ring_3_20_flags, start=3):
            mol.SetProp(f"Ring{i}", str(flag))

        # RING: Check if there is a 3-4 ring
        has_3_4_ring = int(any(ring_3_20_flags[0:2]))
        mol.SetProp("Has_3_4_Ring", str(has_3_4_ring))

        # RING: Check if there is a 8-20 ring
        has_8_20_ring = int(any(ring_3_20_flags[5:]))
        mol.SetProp("Has_8_20_Ring", str(has_8_20_ring))

        # RING: Check if there is a 9+ ring (ring_3_20_flags[6:] corresponds to rings 9-20)
        has_9plus_ring = int(any(ring_3_20_flags[6:]))
        mol.SetProp("Has_9plus_Ring", str(has_9plus_ring))
        all_9plus_ring_data.append(has_9plus_ring)

        # OTHERS(aromatic ring count, total ring count, bridgehead, spiro atoms, TPSA)
        aromatic_rings = Lipinski.NumAromaticRings(mol1)
        num_rings = mol1.GetRingInfo().NumRings()
        all_nring_data.append(num_rings)
        all_aromatic_rings_data.append(aromatic_rings)

        writer.write(mol)

    writer.close()

    # Print average values and summary
    print_statistics(
        all_ringsize_data,
        all_nring_data,
        all_aromatic_rings_data,
        all_brenk_data,
        all_brenk_count_data,
        all_9plus_ring_data,
        output_sdf,
        total_molecules,
        skipped_count,
        skipped_invalid_mol,
        skipped_smiles_error,
    )


if __name__ == "__main__":
    main()
