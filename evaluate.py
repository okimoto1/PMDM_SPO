import argparse
import pickle
from statistics import mean
from pathlib import Path
import os
import glob
from tqdm import tqdm
import numpy as np
import torch

from joblib import Parallel, delayed
from rdkit.Chem.Descriptors import MolLogP, qed, ExactMolWt  # , MolLogP
from rdkit.Chem.rdMolTransforms import GetBondLength
from rdkit import Chem

from configs.dataset_config import get_dataset_info
from evaluation import *
from evaluation.docking import *
from evaluation.docking_2 import *
from evaluation.sascorer import *
from evaluation.score_func import *

# from rdkit.Chem import Draw
from evaluation.similarity import calculate_diversity
from utils.reconstruct import *
from utils.transforms import *


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


def find_file_pairs(directory):
    """
    Traverse the directory and subdirectories to find file pairs of xxx_ligand.sdf and xxx_protein.pdb
    Return format: [{'protein_file': path, 'ligand_file': path, 'base_name': xxx}, ...]
    """
    file_pairs = []

    # Use glob to recursively search all files
    ligand_files = glob.glob(os.path.join(directory, "**/*_ligand.sdf"), recursive=True)
    protein_files = glob.glob(
        os.path.join(directory, "**/*_protein.pdb"), recursive=True
    )

    # Create a mapping from base name to protein file
    protein_map = {}
    for protein_file in protein_files:
        basename = os.path.basename(protein_file)
        if basename.endswith("_protein.pdb"):
            base_name = basename[:-12]  # Remove '_protein.pdb'
            protein_map[base_name] = protein_file

    # Find the corresponding protein file for each ligand file
    for ligand_file in ligand_files:
        basename = os.path.basename(ligand_file)
        if basename.endswith("_ligand.sdf"):
            base_name = basename[:-11]  # Remove '_ligand.sdf'
            if base_name in protein_map:
                file_pairs.append(
                    {
                        "protein_file": protein_map[base_name],
                        "ligand_file": ligand_file,
                        "base_name": base_name,
                    }
                )
            else:
                print(f"Warning: No matching protein file found for {ligand_file}")

    print(f"Found {len(file_pairs)} file pairs in directory {directory}")
    return file_pairs


def load_data_from_directory(directory):
    """
    Load file pairs from the directory and create the data format, compatible with the original pickle format
    """
    file_pairs = find_file_pairs(directory)
    data = []

    for pair in file_pairs:
        # Load the molecule from the SDF file
        try:
            # sanitize=False so PMDM SDFs with valence-tag problems are not
            # silently dropped before the fixval correction can be applied.
            mol_supplier = Chem.SDMolSupplier(
                pair["ligand_file"], sanitize=False, removeHs=False
            )
            mol = next(mol_supplier)
            if mol is not None:
                try:
                    smile = Chem.MolToSmiles(mol)
                except Exception:
                    smile = ""
                data_entry = {
                    "protein_file": pair["protein_file"],
                    "ligand_file": pair["ligand_file"],
                    "mol": mol,
                    "smile": smile,
                }
                data.append(data_entry)
            else:
                print(f"Warning: Could not load molecule from {pair['ligand_file']}")
        except Exception as e:
            print(f"Error loading {pair['ligand_file']}: {e}")

    print(f"Successfully loaded {len(data)} molecules from directory")
    return data


def calculate_avg_bond_length(mol):
    """Calculate average bond length for a molecule"""
    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    total_length = 0.0
    count = 0

    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        length = GetBondLength(conf, atom1_idx, atom2_idx)
        total_length += length
        count += 1

    if count == 0:
        return None
    return total_length / count


def evaluate(
    m,
    n,
    test_vina_score_list,
    compute_vina=True,
    dir_mode=False,
    use_smiles_conversion=False,
):
    smile = m["smile"]
    protein_filename = m["protein_file"]
    ligand_filename = m["ligand_file"]

    # mol_id = hash(smile+protein_filename)

    # if mol_id in results:
    #     print(f'Skipping {smile}, already computed.')
    #     return results[mol_id]
    mol_id = n

    mol = m["mol"]

    try:
        # Keep the raw molecule (with its 3D conformer) for docking.
        original_mol = copy.deepcopy(mol)

        # Reflect calc_sdfprop_latest4.py --readmol fixval: fix PMDM SDF
        # valence tags before computing any property.
        mol = parse_mol_and_ignore_valence_tags(mol)

        if use_smiles_conversion:
            gsmile = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(gsmile)

        _, g_sa = compute_sa_score(mol)
        print("Generate SA score:", g_sa)

        g_qed = qed(mol)
        print("Generate QED score:", g_qed)

        g_logP = MolLogP(mol)
        print("Generate logP:", g_logP)

        g_mol_weight = ExactMolWt(mol)
        print("Generate molecular weight:", g_mol_weight)

        g_Lipinski = obey_lipinski(mol)
        print("Generate Lipinski:", g_Lipinski)

        # Calculate average bond length
        g_bond_length = calculate_avg_bond_length(mol)
        print("Average bond length:", g_bond_length)
    except Exception as e:
        print("mol error", e)
        return None

    # Default vina score if not computing
    g_vina_score = 0

    if compute_vina:
        if dir_mode:
            # Directory mode: use the protein file path directly, but it needs to be converted to pdbqt format
            receptor_file = protein_filename.replace(".pdb", ".pdbqt")
            if not os.path.exists(receptor_file):
                # If the pdbqt file does not exist, use the original pdb file path
                receptor_file = protein_filename
        else:
            # Original mode
            receptor_file = (
                os.path.basename(protein_filename).replace(".pdb", "") + ".pdbqt"
            )
            receptor_file = receptor_file.replace("pocketW3", "pocket10")
            receptor_file = Path(os.path.join(protein_root, receptor_file))

        index = n % 100
        try:
            g_vina_score = calculate_qvina2_score(
                receptor_file, original_mol, out_dir, return_rdmol=False, index=index
            )[0]
        except Exception as e:
            print(f"Error calculating vina score: {e}")
            g_vina_score = 0

        print("Generate vina score:", g_vina_score)

    # NO modify this
    # if test_vina_score_list:
    #     rd_vina_score = 0 # test_vina_score_list[protein_filename]
    #     g_high_affinity = 0
    #     if float(g_vina_score) < float(rd_vina_score):
    #         g_high_affinity = 1
    #         high_affinity.append(1)
    # NO modify this
    g_high_affinity = 0

    metrics = {
        "SA": g_sa,
        "QED": g_qed,
        "logP": g_logP,
        "Lipinski": g_Lipinski,
        "vina": g_vina_score,
        "high_affinity": g_high_affinity,
        "mol_weight": g_mol_weight,
        "bond_length": g_bond_length,
    }
    result = {
        "smile": smile,
        "protein_file": protein_filename,
        "ligand_file": ligand_filename,
        "mol": mol,
        "metrics": metrics,
    }
    # results_dict[mol_id] = result
    # with open(save_mol_result_path, 'wb') as f:
    #     pickle.dump(results, f)

    return result


def save_sdf(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="crossdock")
    parser.add_argument("--path", type=str, default="")
    parser.add_argument(
        "--dir_mode",
        action="store_true",
        help="Directory mode: scan directory for xxx_ligand.sdf and xxx_protein.pdb pairs",
    )
    parser.add_argument(
        "--eval_actual",
        action="store_true",
        help="Evaluate actual ligands instead of generated ones",
    )
    parser.add_argument(
        "--skip_vina", action="store_true", help="Skip vina calculation (saves time)"
    )
    parser.add_argument(
        "--use_smiles_conversion",
        action="store_true",
        help="Execute SMILES conversion (default: skip conversion)",
    )
    args = parser.parse_args()

    dataset_info = get_dataset_info(args.dataset, False)
    path = args.path

    # Check argument compatibility
    if args.dir_mode and args.eval_actual:
        print(
            "Warning: --eval_actual is ignored in directory mode as molecules are loaded directly from SDF files"
        )

    # Data loading: supports directory mode and the original pickle file mode
    if args.dir_mode:
        # Directory mode: scan the directory to find file pairs
        if not os.path.isdir(path):
            raise ValueError(
                f"In directory mode, --path must be a directory, got: {path}"
            )
        print(f"Directory mode: scanning {path}")
        data = load_data_from_directory(path)
        save_dir = path
    else:
        # Original mode: load from a pickle file
        print(os.path.dirname(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
        save_dir = os.path.dirname(path)

    save_mol_result_path = os.path.join(save_dir, "mol_results.pkl")
    if os.path.exists(save_mol_result_path):
        with open(save_mol_result_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        print(f"Created directory: {tmp_dir}")

    if args.dataset == "crossdock":
        # protein_root = './data/test_data_1k/test_pdbqt'
        protein_root = "./data/crossdocked_pocket10"
    elif args.dataset == "pdbind":
        protein_root = "./data/pdbind"
    elif args.dataset == "crossdock_pdbind":
        protein_root = "./data/crossdock_pdbind"

    out_dir = os.path.join(save_dir, "ligand")
    os.makedirs(out_dir, exist_ok=True)
    sdf_dir = save_dir
    results_mol = []
    high_affinity = []
    stable = 0
    valid = 0
    smile_list = []
    num_samples = 0
    position_list = []
    atom_type_list = []
    sa_list = []
    qed_list = []
    logP_list = []
    Lipinski_list = []
    vina_score_list = []
    diversity_list = []
    mol_weight_list = []
    bond_length_list = []
    mol_dict = {}
    idx = 0
    t_vina_dict = {}

    # NO add hiaff
    hiaff = 1
    # with open('test_vina_{}_dict.pkl'.format(args.dataset), 'rb') as f:
    test_vina_score_list = []  # pickle.load(f)

    if args.eval_actual and not args.dir_mode:
        # eval_actual mode is only valid in non-directory mode, since directory mode already loads directly from SDF
        ligand_set = set()
        actual_data = []
        # For actual ligands evaluation
        for d in tqdm(data):
            protein_filename = d["protein_file"]
            ligand_filename = d["ligand_file"]
            ligand_path = os.path.join(protein_root, ligand_filename)

            # Load actual ligand molecule from SDF file
            try:
                if ligand_filename in ligand_set:
                    continue
                ligand_set.add(ligand_filename)
                actual_mol = next(Chem.SDMolSupplier(ligand_path, removeHs=False))
                if actual_mol is not None:
                    # Create a new entry with the actual molecule
                    actual_entry = d.copy()
                    actual_entry["mol"] = actual_mol
                    actual_entry["smile"] = Chem.MolToSmiles(actual_mol)
                    actual_data.append(actual_entry)

                    if protein_filename not in mol_dict.keys():
                        mol_dict[protein_filename] = []
                    mol_dict[protein_filename].append(actual_mol)
            except Exception as e:
                raise e
        # Replace data with actual_data for evaluation
        data = actual_data
    else:
        # Original code for generated molecules or directory mode
        for d in tqdm(data):
            mol = d["mol"]
            protein_filename = d["protein_file"]
            if protein_filename not in mol_dict.keys():
                mol_dict[protein_filename] = []
            mol_dict[protein_filename].append(mol)

    # if args.dir_mode:
    #     # In directory mode, compute the overall diversity of all molecules
    #     all_mols = []
    #     for d in data:
    #         all_mols.append(d["mol"])
    #     overall_diversity = calculate_diversity(all_mols)
    #     diversity_list = [overall_diversity]
    #     print(f"Directory mode - Overall diversity: {overall_diversity:.3f}")
    # else:
    # Original mode: compute diversity grouped by protein
    for n, key in enumerate(tqdm(mol_dict)):
        if len(mol_dict[key]) != 100:
            print(key + "  %d" % (len(mol_dict[key])))
        diversity_list.append(calculate_diversity(mol_dict[key]))

    diversity_list = np.array(diversity_list)
    print(mean(diversity_list))

    # NO modify
    if hiaff == 1:
        # Cannot pickle Boost.Python.function objects in parallel processing
        # Use serial processing instead
        print("Running evaluation in serial mode...")
        results = []
        for n, m in enumerate(tqdm(data)):
            results.append(
                evaluate(
                    m,
                    n,
                    test_vina_score_list,
                    not args.skip_vina,
                    args.dir_mode,
                    args.use_smiles_conversion,
                )
            )
        # Old parallel code:
        # results = Parallel(n_jobs=-1)(delayed(evaluate)(m, n, test_vina_score_list, not args.skip_vina) for n, m in enumerate(tqdm(data)))

    # results = [evaluate(m,n) for n, m in enumerate(tqdm(data))]
    # results = []
    # for m in data:
    #     results.append(evaluate(m))
    for result in tqdm(results):
        if result is not None:
            results_mol.append(result)
            metrics = result["metrics"]
            (
                g_sa,
                g_qed,
                g_logP,
                g_Lipinski,
                g_vina,
                g_h_a,
                g_mol_weight,
                g_bond_length,
            ) = (
                metrics["SA"],
                metrics["QED"],
                metrics["logP"],
                metrics["Lipinski"],
                metrics["vina"],
                metrics["high_affinity"],
                metrics["mol_weight"],
                metrics["bond_length"],
            )
            # if g_vina<-6.5:
            #     save_sdf(result['mol'],sdf_dir,str(g_vina)+'.sdf')
            sa_list.append(g_sa)
            qed_list.append(g_qed)
            logP_list.append(g_logP)
            Lipinski_list.append(g_Lipinski)
            high_affinity.append(g_h_a)
            mol_weight_list.append(g_mol_weight)
            if g_bond_length is not None:
                bond_length_list.append(g_bond_length)
            valid += 1
            if g_vina < 0:
                vina_score_list.append(g_vina)

    num_samples = len(results)  # this is user setting parameter.
    # validity_dict = analyze_stability_for_molecules(position_list, atom_type_list, dataset_info)
    # print(validity_dict)

    if args.dir_mode:
        prefix = "Directory"
    else:
        prefix = "Actual" if args.eval_actual else "Generated"
    print(f"{prefix} ligands summary", num_samples)
    print(f"Final validity:", valid / num_samples)
    print(f"Final stable:", stable / num_samples)  # stable is calculated.
    # print(f"Time per pocket: {times_arr.mean():.3f} \pm "
    #         f"{times_arr.std(unbiased=False):.2f}")
    print("mean sa:%f" % mean(sa_list))
    print("mean qed:%f" % mean(qed_list))
    print("mean logP:%f" % mean(logP_list))
    print("mean Lipinski:%f" % np.mean(Lipinski_list))

    print("mean molecular weight:%f" % mean(mol_weight_list))
    print("high affinity:%d" % np.sum(high_affinity))
    print(
        "high affinity rate:%f" % (np.sum(high_affinity) / len(high_affinity))
    )  # NO ADD

    if bond_length_list:
        print("mean bond length:%f" % mean(bond_length_list))

    if not args.dir_mode:
        diversity_list = diversity_list.tolist()
    print("diversity:%f" % mean(diversity_list))

    if vina_score_list:
        print("mean vina:%f" % mean(vina_score_list))

    # print(vina_score_list)

    sa_list = torch.tensor(sa_list)
    qed_list = torch.tensor(qed_list)
    logP_list = torch.tensor(logP_list)
    Lipinski_list = torch.tensor(Lipinski_list)
    vina_score_list = torch.tensor(vina_score_list)
    mol_weight_list = torch.tensor(mol_weight_list)
    bond_length_list = torch.tensor(bond_length_list) if bond_length_list else []
    metrics_list = {
        "diversity": diversity_list,
        "sa": sa_list,
        "qed": qed_list,
        "logP": logP_list,
        "Lipinski": Lipinski_list,
        "vina": vina_score_list,
        "high_affinity": high_affinity,
        "mol_weight": mol_weight_list,
        "bond_length": bond_length_list,
    }

    suffix = "_actual" if args.eval_actual else ""
    suffix += "_dir" if args.dir_mode else ""
    save_mol_result_path = os.path.join(save_dir, f"mol_results{suffix}.pkl")
    with open(save_mol_result_path, "wb") as f:
        pickle.dump(results_mol, f)
        f.close()

    save_metric_result_path = os.path.join(save_dir, f"metric_results{suffix}.pkl")
    with open(save_metric_result_path, "wb") as f:
        pickle.dump(metrics_list, f)
        f.close()
