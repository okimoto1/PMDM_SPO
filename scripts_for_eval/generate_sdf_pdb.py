import argparse
import os
import pickle
import shutil
import uuid

from rdkit import Chem
from tqdm import tqdm


def generate_samples_file(samples, output_dir="protein_water_samples", generate_water=True):

    os.makedirs(output_dir, exist_ok=True)
    uid = 0

    for sample_idx, sample in tqdm(enumerate(samples)):
        try:
            protein_file = os.path.join("data/crossdock_pdbind", sample["protein_file"])

            # Construct the W3.pdb filename (containing original waters)
            protein_file_water = (
                protein_file.replace("10.pdb", "W3.pdb")
                if os.path.exists(protein_file.replace("10.pdb", "W3.pdb"))
                else protein_file
            )

            # Get number of generated waters
            water_pos = (
                sample["water_pos"].numpy()
                if hasattr(sample["water_pos"], "numpy")
                else sample["water_pos"]
            )
            generated_water_count = water_pos.shape[0]

            sample_name = sample["protein_file"].split("/")[-1].split(".")[0]

            # Check if original file had waters
            original_water_count = 0
            if os.path.isfile(protein_file_water):
                with open(protein_file_water, "r") as f:
                    for line in f:
                        if line.startswith("HETATM") and "HOH" in line:
                            original_water_count += 1

            uid += 1
            
            # Create sample-specific directory
            sample_dir = os.path.join(output_dir, f"sample_{sample_name}")
            os.makedirs(sample_dir, exist_ok=True)

            # 1. Copy original protein file (only once per sample directory)
            protein_output = os.path.join(sample_dir, f"protein_{sample_name}.pdb")
            if not os.path.exists(protein_output):
                shutil.copy2(protein_file_water, protein_output)

            # 2. Save ligand as SDF
            ligand_output = os.path.join(sample_dir, f"ligand_{sample_name}_{uid}.sdf")
            writer = Chem.SDWriter(ligand_output)
            writer.write(sample["mol"])
            writer.close()

            with open(ligand_output, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f"{sample_name}" + content)

            # 3. Save waters as PDB file (only if generate_water is True)
            if generate_water:
                water_output = os.path.join(sample_dir, f"waters_{sample_name}_{uid}.pdb")
                with open(water_output, "w") as f:
                    f.write(
                        "TITLE     Water molecules from sample {}\n".format(sample_name)
                    )

                    # Write each water molecule as a HETATM record
                    for idx, pos in enumerate(water_pos):
                        f.write(
                            f"HETATM{idx+1:5d}  O   HOH A{idx+1:4d}    "
                            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                            f"  1.00  0.00           O \n"
                        )
                    f.write("END\n")

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            continue


if __name__ == "__main__":
    # "logs/crossdock_pdbind_exp_2025_01_25__23_34_24/generalizedbuild_0_10000_result_2025_01_30__02_13_55/samples_all.pkl"
    parser = argparse.ArgumentParser(description="Visualize generated samples")
    parser.add_argument(
        "--samples_file",
        type=str,
        required=True,
        help="Path to the samples file (.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the output files (default: same directory as samples_file)",
    )
    parser.add_argument(
        "--generate_water",
        action="store_true",
        default=False,
        help="Generate water PDB files (default: False)",
    )
    args = parser.parse_args()

    samples_file = args.samples_file
    output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(samples_file), "ligands")
    with open(samples_file, "rb") as f:
        samples = pickle.load(f)
    generate_samples_file(samples, output_dir, args.generate_water)
