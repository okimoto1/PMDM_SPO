import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.dataset_config import get_dataset_info
from evaluation import *
from evaluation.sascorer import *
from models.epsnet import get_model
from utils.datasets import get_dataset
from utils.misc import *
from utils.reconstruct import *
from utils.reconstruct_mdm import make_mol_openbabel
from utils.sample import DistributionNodes
from utils.sample import construct_dataset_pocket
from utils.transforms import *
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.data import torchify_dict
from torch.utils.data import Subset
from torch_geometric.data import Batch
import torch.nn as nn
import sys

STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
FOLLOW_BATCH = ["ligand_atom_feature", "protein_atom_feature_full"]

atomic_numbers_crossdock = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])
atomic_numbers_pocket = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 34])
atomic_numbers_pdbind = torch.LongTensor(
    [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 27, 29, 33, 34, 35, 44, 51, 53, 78]
)
P_ligand_element_100 = torch.LongTensor(
    [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 29, 33, 34, 35, 44, 51, 53, 78]
)
# P_ligand_element_filter = torch.LongTensor([1, 35, 5, 6, 7, 8, 9, 15, 16, 17, 53])
P_ligand_element_filter = torch.LongTensor([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


def RMSD(probe, ref):
    rmsd = 0.0
    # print(amap)
    assert len(probe) == len(ref)
    atomNum = len(probe)
    for i in range(len(probe)):
        posp = probe[i]
        posf = ref[i]
        rmsd += dist_2(posp, posf)
    rmsd = math.sqrt(rmsd / atomNum)
    return rmsd


def dist_2(atoma_xyz, atomb_xyz):
    dis2 = 0.0
    for i, j in zip(atoma_xyz, atomb_xyz):
        dis2 += (i - j) ** 2
    return dis2


def get_adj_matrix(n_particles):
    rows, cols = [], []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    # print(n_particles)
    rows = torch.LongTensor(rows).unsqueeze(0)
    cols = torch.LongTensor(cols).unsqueeze(0)
    # print(rows.size())
    adj = torch.cat([rows, cols], dim=0)
    return adj


def save_sdf(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def try_build_molecule(pos, atom_type, build_method, atomic_numbers, atomic_numbers_crossdock, dataset_info, largest_mol_flag=False):
    """
    Attempt to build a molecule, returns (mol, water_pos, success_flag)
    For fair comparison: ensures baseline and SPO samples are paired
    """
    try:
        new_ligand = torch.zeros([pos.size(0), len(ATOM_FAMILIES)], dtype=np.long)
        num_atom_type = len(atomic_numbers)
        
        if build_method == "reconstruct":
            new_element = torch.tensor([
                atomic_numbers_crossdock[n]
                for n in torch.argmax(atom_type[:, :8], dim=1)
            ])
            indicators_elements = torch.argmax(atom_type[:, 8:], dim=1)
            indicators = torch.zeros([pos.size(0), len(ATOM_FAMILIES)], dtype=np.long)
            for i, n in enumerate(indicators_elements):
                indicators[i, n] = 1
            gmol = reconstruct_from_generated(pos, new_element, indicators)
            water_pos = None
            
        elif build_method == "build":
            new_element = torch.argmax(atom_type[:, :num_atom_type], dim=1)
            gmol, water_pos = make_mol_openbabel(pos, new_element, dataset_info)
        else:
            return None, None, False
        
        g_smile = Chem.MolToSmiles(gmol)
        
        if g_smile is None:
            return None, None, False
        
        # Check molecular fragments
        if "." in g_smile and largest_mol_flag:
            mol_frags = Chem.rdmolops.GetMolFrags(gmol, asMols=True, sanitizeFrags=False)
            gmol = max(mol_frags, default=gmol, key=lambda m: m.GetNumAtoms())
            g_smile = Chem.MolToSmiles(gmol)
            if g_smile is None:
                return None, None, False
        
        # Check validity
        if g_smile.count(".") > 0:  # still has fragments
            return None, None, False
        if len(g_smile) < 4:
            return None, None, False
        
        return gmol, water_pos, True
        
    except (RuntimeError, MolReconsError, TypeError, IndexError, OverflowError) as e:
        return None, None, False


def visualize_spm_comparison(baseline_results, spo_results, output_dir):
    """
    Visualize SPM score comparison
    Creates a separate comparison plot for each sample pair for direct comparison
    """
    # Group by global_id to extract SPM scores and SMILES
    def extract_trajectories_with_info(results):
        """Extract each sample's trajectory and info: {global_id: {'traj': [...], 'smile': ..., ...}}"""
        trajectories = {}
        for r in results:
            # Prefer global_id, fall back to sample_idx if unavailable (for backward compatibility with old data)
            global_id = r.get('global_id', r.get('sample_idx', 0))
            
            if global_id not in trajectories:
                trajectories[global_id] = {
                    'traj': [],
                    'smile': r.get('smile', 'N/A'),
                    'protein_file': r.get('protein_file', 'N/A'),
                    'sample_idx': r.get('sample_idx', 0),
                    'pocket_idx': r.get('pocket_idx', 0),
                }
            
        if 'spm_scores' in r and r['spm_scores']:
                for score_dict in r['spm_scores']:
                    if 'step' in score_dict and 'score' in score_dict:
                        trajectories[global_id]['traj'].append(
                            (score_dict['step'], score_dict['score'])
                        )
        
        # Sort each trajectory
        for idx in trajectories:
            trajectories[idx]['traj'].sort(key=lambda x: x[0])  # sort by step
        
        return trajectories
    
    baseline_info = extract_trajectories_with_info(baseline_results)
    spo_info = extract_trajectories_with_info(spo_results)
    
    if not baseline_info and not spo_info:
        print("Warning: No SPM scores to visualize")
        return

    print(f"[Visualization] Found {len(baseline_info)} baseline trajectories")
    print(f"[Visualization] Found {len(spo_info)} SPO trajectories")
    
    # Create paired_comparisons directory
    paired_dir = os.path.join(output_dir, 'paired_comparisons')
    os.makedirs(paired_dir, exist_ok=True)
    
    # Find common sample_idx
    common_indices = sorted(set(baseline_info.keys()) & set(spo_info.keys()))
    
    if not common_indices:
        print("[Visualization] Warning: No common sample indices found for pairing!")
        return
    
    print(f"[Visualization] Creating {len(common_indices)} paired comparison plots...")
    
    # ========== Create a separate plot for each pair ==========
    for sample_idx in common_indices:
        baseline_traj = baseline_info[sample_idx]['traj']
        spo_traj = spo_info[sample_idx]['traj']
        
        if not baseline_traj or not spo_traj:
            print(f"[Visualization] Skipping sample {sample_idx}: empty trajectory")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Baseline trajectory
        b_steps, b_scores = zip(*baseline_traj)
        ax.plot(b_steps, b_scores, 'o-', linewidth=2, markersize=4, 
                alpha=0.8, color='blue', label='Baseline (no SPO)')
        
        # SPO trajectory
        s_steps, s_scores = zip(*spo_traj)
        ax.plot(s_steps, s_scores, 's-', linewidth=2, markersize=4, 
                alpha=0.8, color='red', label='SPO Guided')
        
        # Mark start and end points
        ax.scatter([b_steps[0]], [b_scores[0]], s=150, c='blue', marker='o', 
                  edgecolors='black', linewidths=2, zorder=5, label='Start')
        ax.scatter([b_steps[-1]], [b_scores[-1]], s=150, c='blue', marker='*', 
                  edgecolors='black', linewidths=2, zorder=5)
        ax.scatter([s_steps[-1]], [s_scores[-1]], s=150, c='red', marker='*', 
                  edgecolors='black', linewidths=2, zorder=5, label='End')
        
        # Set labels and title
        ax.set_xlabel('Sampling Step', fontsize=13)
        ax.set_ylabel('SPM Score', fontsize=13)
        
        # Extract protein name and index info
        protein_name = os.path.basename(baseline_info[sample_idx]['protein_file']).split('.')[0]
        local_idx = baseline_info[sample_idx]['sample_idx']
        pocket_idx = baseline_info[sample_idx]['pocket_idx']
        
        title = f'Pair #{sample_idx:02d} - SPM Score Trajectory Comparison\n'
        title += f'Protein: {protein_name} | Pocket #{pocket_idx} | Sample #{local_idx}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        final_baseline = b_scores[-1]
        final_spo = s_scores[-1]
        improvement = final_spo - final_baseline
        improvement_pct = (improvement / abs(final_baseline)) * 100 if final_baseline != 0 else 0

        stats_text = f'Final Scores:\n'
        stats_text += f'Baseline: {final_baseline:.4f}\n'
        stats_text += f'SPO: {final_spo:.4f}\n'
        stats_text += f'Δ: {improvement:+.4f} ({improvement_pct:+.1f}%)\n\n'
        stats_text += f'Trajectories:\n'
        stats_text += f'Baseline: {len(b_steps)} evals\n'
        stats_text += f'SPO: {len(s_steps)} evals'

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Add SMILES info
        smiles_text = f'Baseline SMILES:\n{baseline_info[sample_idx]["smile"][:60]}...\n\n'
        smiles_text += f'SPO SMILES:\n{spo_info[sample_idx]["smile"][:60]}...'

        ax.text(0.98, 0.02, smiles_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # Save
        save_path = os.path.join(paired_dir, f'pair_{sample_idx:02d}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Visualization] Sample {sample_idx}: "
              f"Baseline {len(b_steps)} points [{min(b_steps)}-{max(b_steps)}], "
              f"SPO {len(s_steps)} points [{min(s_steps)}-{max(s_steps)}], "
              f"Final Δ={improvement:+.4f}")
    
    print(f"[Visualization] Saved {len(common_indices)} paired comparisons to {paired_dir}/")
    
    # ========== Create summary plot: overview of all samples ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Baseline overview
    for sample_idx in common_indices:
        traj = baseline_info[sample_idx]['traj']
        if not traj:
            continue
        steps, scores = zip(*traj)
        ax1.plot(steps, scores, 'o-', linewidth=1.5, markersize=2, 
                alpha=0.6, label=f'Sample {sample_idx}')
    
    ax1.set_xlabel('Sampling Step', fontsize=12)
    ax1.set_ylabel('SPM Score', fontsize=12)
    ax1.set_title(f'Baseline Trajectories Overview\n{len(common_indices)} samples', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # SPO overview
    for sample_idx in common_indices:
        traj = spo_info[sample_idx]['traj']
        if not traj:
            continue
        steps, scores = zip(*traj)
        ax2.plot(steps, scores, 's-', linewidth=1.5, markersize=2, 
                alpha=0.6, label=f'Sample {sample_idx}')
    
    ax2.set_xlabel('Sampling Step', fontsize=12)
    ax2.set_ylabel('SPM Score', fontsize=12)
    ax2.set_title(f'SPO Guided Trajectories Overview\n{len(common_indices)} samples', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    overview_path = os.path.join(output_dir, 'spm_score_comparison_overview.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved overview to {overview_path}")

    # ========== Figure 2: Final Score Distribution ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract final scores (the last score for each sample)
    def get_final_scores(results):
        final_scores = []
        for r in results:
            if 'spm_scores' in r and r['spm_scores']:
                # Get all scores for this sample
                sample_scores = r['spm_scores']
                if sample_scores:
                    # Find the score with the maximum step
                    max_step_score = max(sample_scores, key=lambda x: x.get('step', -1))
                    if 'score' in max_step_score:
                        final_scores.append(max_step_score['score'])
        return final_scores

    baseline_final = get_final_scores(baseline_results)
    spo_final = get_final_scores(spo_results)

    print(f"[Visualization] Extracted {len(baseline_final)} baseline final scores")
    print(f"[Visualization] Extracted {len(spo_final)} SPO final scores")

    if baseline_final or spo_final:
        # Set bins to cover both datasets
        all_scores = baseline_final + spo_final
        if all_scores:
            bins = np.linspace(min(all_scores), max(all_scores), 30)
            
            if baseline_final:
                ax.hist(baseline_final, bins=bins, alpha=0.5, label='Baseline', 
                       color='blue', edgecolor='black', linewidth=0.5)
            
            if spo_final:
                ax.hist(spo_final, bins=bins, alpha=0.5, label='SPO Guided', 
                       color='red', edgecolor='black', linewidth=0.5)

            ax.set_xlabel('Final SPM Score', fontsize=12)
            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.set_title('Distribution of Final SPM Scores', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistics text
            stats_text = ""
            if baseline_final:
                stats_text += f"Baseline:\n  Mean: {np.mean(baseline_final):.4f}\n  Std: {np.std(baseline_final):.4f}\n  Median: {np.median(baseline_final):.4f}\n\n"
            if spo_final:
                stats_text += f"SPO Guided:\n  Mean: {np.mean(spo_final):.4f}\n  Std: {np.std(spo_final):.4f}\n  Median: {np.median(spo_final):.4f}"
            
            if baseline_final and spo_final:
                improvement = ((np.mean(spo_final) - np.mean(baseline_final)) / abs(np.mean(baseline_final))) * 100
                stats_text += f"\n\nMean Improvement: {improvement:+.2f}%"

            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            plt.tight_layout()
            save_path = os.path.join(output_dir, 'final_score_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Saved final score distribution to {save_path}")
    else:
        print("[Visualization] Warning: No final scores to plot distribution")


class LearnableUnconditionalProtein(nn.Module):
    """Learnable unconditional protein representation - consistent with the definition in train_modify1.py"""
    def __init__(self, protein_feature_dim, hidden_dim, num_atoms=10):
        super().__init__()
        self.num_atoms = num_atoms
        self.protein_feature_dim = protein_feature_dim
        self.hidden_dim = hidden_dim

        self.uncond_protein_features = nn.Parameter(
            torch.randn(num_atoms, protein_feature_dim) * 0.02
        )
        self.uncond_protein_pos = nn.Parameter(
            torch.randn(num_atoms, 3) * 0.5
        )
        
    def forward(self, batch_size, device):
        protein_features = self.uncond_protein_features.unsqueeze(0).expand(
            batch_size, -1, -1
        ).reshape(batch_size * self.num_atoms, -1).to(device)
        
        protein_pos = self.uncond_protein_pos.unsqueeze(0).expand(
            batch_size, -1, -1
        ).reshape(batch_size * self.num_atoms, -1).to(device)
        
        protein_batch = torch.arange(batch_size, device=device).repeat_interleave(
            self.num_atoms
        )
        
        protein_backbone_mask = torch.ones(
            batch_size * self.num_atoms, dtype=torch.bool, device=device
        )
        
        return protein_features, protein_pos, protein_batch, protein_backbone_mask


if __name__ == "__main__":
    # sys.path.append('/.')
    # os.chdir('/.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default=True)
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--ckpt", type=str, help="path for loading the checkpoint")
    parser.add_argument(
        "--save_traj",
        action="store_true",
        help="whether store the whole trajectory for sampling",
    )
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--save_sdf", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "-build_method", type=str, default="build", help="build or reconstruct"
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--test_set", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=25000)
    parser.add_argument("--clip", type=float, default=1000.0)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=0,
        help="sampling num steps; for DSM framework, this means num steps for each noise scale",
    )
    parser.add_argument(
        "--global_start_sigma",
        type=float,
        default=float("inf"),
        help="enable global gradients only when noise is low",
    )
    parser.add_argument(
        "--local_start_sigma",
        type=float,
        default=float("inf"),
        help="enable local gradients only when noise is low",
    )
    parser.add_argument(
        "--w_global_pos", type=float, default=1.0, help="weight for global gradients"
    )
    parser.add_argument(
        "--w_local_pos", type=float, default=1.0, help="weight for local gradients"
    )
    parser.add_argument(
        "--w_global_node", type=float, default=1.0, help="weight for global gradients"
    )
    parser.add_argument(
        "--w_local_node", type=float, default=1.0, help="weight for local gradients"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="classifier-free guidance scale (1.0 disables guidance)",
    )
    # Parameters for DDPM
    parser.add_argument(
        "--sampling_type",
        type=str,
        default="ld",
        help="generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="weight for DDIM and DDPM: 0->DDIM, 1->DDPM",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=1000,
        help="max timesteps for sampling(will change n_step simutaneously)",
    )

    parser.add_argument("--dataset", type=str, default="crossdock")
    parser.add_argument("--w_clash", type=float, default=0.0, help="weight for clash loss")
    
    # SPM scoring parameters
    parser.add_argument("--use_spm", action="store_true", help="whether to use SPM for scoring during sampling")
    parser.add_argument("--spm_checkpoint", type=str, default="SPO/ligand_spm/checkpoints/best_model.pt", help="path to SPM checkpoint")
    parser.add_argument("--spm_eval_interval", type=int, default=100, help="evaluate SPM every N sampling steps")
    
    # SPO (Step-aware Preference Optimization) resampling parameters
    parser.add_argument("--use_spo", action="store_true", help="whether to use SPO-guided resampling during sampling")
    parser.add_argument("--spo_resample_interval", type=int, default=5, help="resample every N steps for SPO")
    parser.add_argument("--spo_num_candidates", type=int, default=5, help="number of candidate trajectories to generate at each resampling point")
    parser.add_argument("--spo_start_step", type=int, default=0, help="step to start applying SPO resampling (0=from beginning)")

    # Fixed number of atoms
    parser.add_argument("--fixed_num_atoms", type=int, default=None, help="fix the number of atoms in generated ligands (if None, use reference ligand's atom count)")

    # ========== Run mode parameters ==========
    parser.add_argument("--run_mode", type=str, default="comparison",
                       choices=["comparison", "spo_only", "baseline_only"],
                       help="Run mode: comparison (both baseline and SPO), spo_only (only SPO), baseline_only (only baseline)")
    # Keep --comparison_mode as a shorthand for --run_mode comparison
    parser.add_argument("--comparison_mode", action="store_true",
                       help="[Deprecated] Use --run_mode comparison instead. Enable comparison mode.")

    # ========== SPO scoring weight parameters ==========
    parser.add_argument("--score_weight_spm", type=float, default=1.0,
                       help="Weight for SPM score in combined scoring (default: 1.0)")
    parser.add_argument("--score_weight_qed", type=float, default=2.0,
                       help="Weight for QED score in combined scoring (default: 2.0)")
    parser.add_argument("--score_weight_sa", type=float, default=1.0,
                       help="Weight for SA score in combined scoring (default: 1.0)")
    parser.add_argument("--score_weight_clash", type=float, default=1.0,
                       help="Weight for clash penalty in combined scoring (default: 1.0)")
    parser.add_argument("--clash_threshold", type=float, default=3.0,
                       help="Distance threshold for clash penalty (default: 3.0 Angstrom)")
    parser.add_argument("--spo_use_spm", action="store_true", default=True,
                       help="Whether to use SPM score in SPO combined scoring (default: True)")
    parser.add_argument("--spo_no_spm", action="store_true", default=False,
                       help="Disable SPM score in SPO combined scoring (use QED+SA+clash only)")
    parser.add_argument("--use_lipinski", action="store_true", default=False,
                       help="Use Lipinski score as multiplicative factor: Lipinski * (weighted_sum). Lipinski is normalized to [0,1] based on rules satisfied.")

    # ========== Grid search support parameters ==========
    parser.add_argument("--train_subset_file", type=str, default=None,
                       help="File containing training subset indices (for grid search). Format: idx\\tprotein_file\\tligand_file per line")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (overrides default log_dir based naming)")

    args = parser.parse_args()

    # Handle the --spo_no_spm flag
    if args.spo_no_spm:
        args.spo_use_spm = False

    # Handle run mode: --comparison_mode is equivalent to --run_mode comparison
    if args.comparison_mode:
        args.run_mode = "comparison"
    
    # Validate SPO parameters
    # use_spm is only required when spo_use_spm=True
    if args.use_spo and args.spo_use_spm and not args.use_spm:
        print("Error: --use_spo with SPM scoring requires --use_spm to be enabled")
        print("SPO resampling needs SPM model to evaluate candidates")
        print("Or use --spo_no_spm to disable SPM in SPO scoring")
        sys.exit(1)

    # Print run mode and scoring weight configuration
    print("=" * 60)
    print(f"RUN MODE: {args.run_mode.upper()}")
    print("=" * 60)

    if args.run_mode == "comparison":
        print("Each pocket will be sampled twice:")
        print("  1. Baseline (no SPO) - to establish baseline")
        print("  2. SPO guided - using same initial noise")
        print("This ensures fair comparison by using identical starting points.")
    elif args.run_mode == "spo_only":
        print("Only SPO-guided sampling will be performed.")
        args.use_spo = True  # Force enable SPO
    elif args.run_mode == "baseline_only":
        print("Only baseline sampling will be performed (no SPO).")
        args.use_spo = False  # Force disable SPO

    print("")
    print("SPO Scoring Weights:")
    print(f"  SPO use SPM:  {args.spo_use_spm}")
    if args.spo_use_spm:
        print(f"  SPM weight:   {args.score_weight_spm}")
    print(f"  QED weight:   {args.score_weight_qed}")
    print(f"  SA weight:    {args.score_weight_sa}")
    print(f"  Clash weight: {args.score_weight_clash}")
    print(f"  Clash threshold: {args.clash_threshold} Angstrom")
    print(f"  Use Lipinski: {args.use_lipinski}")
    if args.spo_use_spm:
        base_formula = f"{args.score_weight_spm}*SPM + {args.score_weight_qed}*QED + {args.score_weight_sa}*SA + {args.score_weight_clash}*clash"
    else:
        base_formula = f"{args.score_weight_qed}*QED + {args.score_weight_sa}*SA + {args.score_weight_clash}*clash"
    if args.use_lipinski:
        print(f"  Formula: Lipinski * ({base_formula})")
        print(f"  (Lipinski normalized to [0,1]: satisfied_rules / 5)")
    else:
        print(f"  Formula: {base_formula}")
    print(f"  (clash = -max({args.clash_threshold} - min_dist, 0), penalizes ligand-protein distance < {args.clash_threshold}A)")
    print("=" * 60)

    if args.run_mode in ["comparison", "spo_only"] and not args.use_spm and args.spo_use_spm:
        print("\nWarning: --use_spm not enabled but spo_use_spm=True.")
        print("SPM scores will not be recorded. Consider adding --use_spm or --spo_no_spm")

    # Force use_spo=False in comparison mode; we control it manually
    if args.run_mode == "comparison":
        args.use_spo = False  # Set to False first, controlled manually later

    # Load configs
    device = torch.device("cuda:" + args.gpu if args.cuda else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]

    args.cuda = args.cuda and torch.cuda.is_available()


    num_samples = args.num_samples
    batch_size = args.batch_size

    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    if args.n_steps == 0:
        args.n_steps = ckpt["config"].model.num_diffusion_timesteps

    # Logging

    # logger = get_logger('sample', log_dir)
    # Set tag based on run mode
    if args.run_mode == "comparison":
        tag = "comparison"
    elif args.run_mode == "spo_only":
        tag = "spo_only"
    elif args.run_mode == "baseline_only":
        tag = "baseline_only"
    else:
        tag = "result"

    # Support custom output directory (for grid search)
    if args.output_dir is not None:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = get_new_log_dir(
            log_dir,
            args.sampling_type
            + args.build_method
            + "_"
            + str(args.start_idx)
            + "_"
            + str(args.end_idx)
            + "_"
            + tag,
            tag=args.tag,
        )
    print("output_dir: ", output_dir)
    logger = get_logger("test", output_dir)

    # # for 1k sample
    # config.dataset.split='/om/user/layne_h/project/PMDM_raw/data/split_by_name.pt'

    logger.info(args)
    logger.info(config)

    dataset_info = get_dataset_info("crossdock", False)
    histogram = dataset_info["n_nodes"]
    nodes_dist = DistributionNodes(histogram)

    # Data
    # rewrite dataset info
    if args.dataset == "crossdock":
        config.dataset.name = args.dataset
        config.dataset.path = "./data/crossdocked_pocket10"
        config.dataset.split = "./data/split_by_name.pt"
    elif args.dataset == "pdbind":
        config.dataset.name = args.dataset
        config.dataset.path = "./data/pdbind"
        config.dataset.split = "./data/pdbind_split_by_name_valid.pt"
    elif args.dataset == "crossdock_pdbind":
        config.dataset.name = args.dataset
        config.dataset.path = "./data/crossdocked_pdbind"
        if os.environ.get("WATER") == "protein":
            config.dataset.split = (
                "./data/crossdock_pdbind_split_by_name_valid_proteinW3.pt"
            )
        else:
            config.dataset.split = "./data/crossdock_pdbind_split_by_name_valid.pt"
    else:
        raise NotImplementedError

    logger.info("Loading {} data...".format(config.dataset.name))
    if config.dataset.name == "crossdock":
        if "pocket" or "sa" in args.ckpt:
            atomic_numbers = atomic_numbers_pocket
            dataset_info = get_dataset_info("crossdock_pocket", False)
            pocket = True
        else:
            # atomic_numbers = atomic_numbers_pocket
            # pocket=True
            atomic_numbers = atomic_numbers_crossdock
            dataset_info = get_dataset_info("crossdock", False)
        # protein_root = "/om/user/layne_h/project/PMDM_raw/data/crossdocked_pocket10"
    elif config.dataset.name == "pdbind":
        atomic_numbers = atomic_numbers_pdbind
        # protein_root = "./data/protein_ligand/pdbind"
        dataset_info = get_dataset_info("pdbind", False)
        pocket = True
    elif config.dataset.name == "crossdock_pdbind":
        atomic_numbers = atomic_numbers_pdbind
        # protein_root = "./data/protein_ligand/pdbind"
        dataset_info = get_dataset_info("crossdock_pdbind", False)
        pocket = True
    else:
        if "filter" in config.dataset.split:
            atomic_numbers = P_ligand_element_filter
        elif "100" in config.dataset.split:
            atomic_numbers = P_ligand_element_100
        else:
            atomic_numbers = atomic_numbers_pdbind
        protein_root = "./data/protein_ligand/pdbind"
        pocket = True

    logger.info("dataset: " + config.dataset.name)
    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=pocket)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=pocket)

    transform = Compose(
        [
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            FeaturizeLigandBond(),
            CountNodesPerGraph(),
            GetAdj(),
        ]
    )

    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform,
    )
    testset = subsets["test"]
    trainset = subsets["train"]
    print(len(trainset))
    print(len(testset))

    if config.dataset.name == "crossdock_pdbind":
        # filter crossdock
        pdbind_testset = set()
        for key, value in dataset.name2id.items():
            if len(key[0].split("/")[0]) == 4:
                pdbind_testset.add(value)

        new_testset_idx = [idx for idx in testset.indices if idx in pdbind_testset]
        testset = Subset(testset.dataset, new_testset_idx)

    test_set_selected = []
    # FOLLOW_BATCH = ['ligand_atom_type','protein_atom_feature_full']

    # ========== Support sampling from a training subset (for grid search) ==========
    if args.train_subset_file is not None:
        logger.info(f"Loading training subset from: {args.train_subset_file}")
        train_subset_indices = []
        with open(args.train_subset_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    try:
                        idx = int(parts[0])
                        train_subset_indices.append(idx)
                    except ValueError:
                        continue

        logger.info(f"Loaded {len(train_subset_indices)} training subset indices")

        # Extract data at the specified indices from the training set
        for idx in train_subset_indices:
            if idx < len(trainset):
                data = trainset[idx]
                test_set_selected.append(data)
            else:
                logger.warning(f"Index {idx} out of range for trainset (size={len(trainset)})")

        logger.info(f"Selected {len(test_set_selected)} samples from training subset")
    else:
        # Original logic: select from the test set
        for i, data in enumerate(testset):
            if not (args.start_idx <= i < args.end_idx):
                continue
            test_set_selected.append(data)
            # break

    print("test_set_selected: ", len(test_set_selected))

    # Record pocket info (only write when there is data)
    if test_set_selected:
        with open(os.path.join(output_dir, "pocket_info.txt"), "a") as f:
            for data in test_set_selected:
                f.write(data.protein_filename + "\n")

    logger.info("Building model...")
    logger.info(config.model["network"])
    print(config.model)
    model = get_model(config.model).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Load SPM model (if enabled)
    spm_model = None
    if args.use_spm:
        logger.info(f"Loading SPM model from: {args.spm_checkpoint}")
        
        # Import directly from the full path
        from spo.ligand_spm.models.ligand_preference_model import LigandPreferenceModel
        
        logger.info("Successfully imported LigandPreferenceModel")
        
        spm_model = LigandPreferenceModel(
            hidden_channels=128,
            num_filters=128,
            num_interactions=2,
            edge_channels=128,
            cutoff=6.0,
            input_dim=10,
            projection_dim=128,
        ).to(device)
        
        try:
            spm_state_dict = torch.load(args.spm_checkpoint, map_location=device)
            spm_model.load_state_dict(spm_state_dict)
            spm_model.eval()
            logger.info("SPM model loaded successfully")
            logger.info(f"SPM will evaluate every {args.spm_eval_interval} sampling steps")
        except Exception as e:
            logger.warning(f"Failed to load SPM model: {e}")
            logger.warning("Continuing without SPM scoring")
            args.use_spm = False
            spm_model = None
    
    # Load the learnable unconditional protein generator (for CFG sampling)
    uncond_protein_gen = None
    if 'uncond_protein_gen' in ckpt and args.guidance_scale != 1.0:
        protein_feature_dim = 31  # default value
        if hasattr(config, 'protein_feature_dim'):
            protein_feature_dim = config.protein_feature_dim
        
        uncond_protein_gen = LearnableUnconditionalProtein(
            protein_feature_dim=protein_feature_dim,
            hidden_dim=config.model.hidden_dim,
            num_atoms=10
        ).to(device)
        
        try:
            uncond_protein_gen.load_state_dict(ckpt['uncond_protein_gen'])
            uncond_protein_gen.eval()
            logger.info(f'Loaded unconditional protein generator for CFG sampling (guidance_scale={args.guidance_scale})')
        except Exception as e:
            logger.warning(f'Failed to load unconditional protein generator: {e}')
            uncond_protein_gen = None
    elif args.guidance_scale != 1.0:
        logger.warning(f'CFG requested (guidance_scale={args.guidance_scale}) but no uncond_protein_gen in checkpoint!')
        logger.warning('Falling back to standard sampling (guidance_scale will be ignored)')
        args.guidance_scale = 1.0

    clip_local = None
    print(device)
    time_list = []
    sa_list = []
    r_sa_list = []
    rd_sa_list = []

    qed_list = []
    r_qed_list = []
    rd_qed_list = []

    plogp_list = []
    r_plogo_list = []

    valid = 0
    stable = 0
    sum_rms = 0
    sum_rmsd = 0
    high_affinity = 0
    rmsd_list = []

    outliers = []

    smile_list = []
    results = []

    # Comparison mode: needs two result lists
    if args.run_mode == "comparison":
        baseline_results = []
        spo_results = []

    protein_files = []

    logP_list = []
    Lipinski_list = []
    vina_score_list = []

    rd_vina_score_list = []

    save_results = args.save_results
    save_sdf_flag = args.save_sdf
    if save_results:
        file_save_dir = "./data/test_data/"
        if not os.path.exists(file_save_dir):
            os.mkdir(file_save_dir)
    if save_sdf_flag:
        sdf_dir = "./results/crossdocked/MDM/protein_context_Schent_build/"
        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)

    nodes_dist = DistributionNodes(dataset_info["n_nodes"])

    # with open('test_vina_{}.pkl'.format(config.dataset.name), 'rb') as f:
    #     test_vina_score_list = pickle.load(f)

    trial = 0

    for n, data in enumerate(tqdm(test_set_selected)):
        num_samples = args.num_samples
        try:
            rmol = reconstruct_from_generated(
                data.ligand_pos, data.ligand_element, data.ligand_atom_feature
            )
        except Exception as e:
            print(e)
            continue
        r_smile = Chem.MolToSmiles(rmol)
        print("reference smile:", r_smile)
        try_num = 20
        FINISHED = False

        element = data.ligand_element.tolist()
        protein_files.append(data.protein_filename)
        f_dir, f_name = os.path.split(data.protein_filename)
        # print(f_dir)

        gen_file_name = f_name.split(".")[0] + "_gen.sdf"
        print(gen_file_name)
        # sdf_dir =  os.path.join(file_save_dir, f_dir)

        pdb_name = f_name.split("_")[0]

        protein_atom_feature = data.protein_atom_feature.float()
        protein_atom_feature_full = data.protein_atom_feature_full.float()

        with torch.no_grad():
            # Use fixed_num_atoms parameter or the reference ligand's atom count
            num_points = data.ligand_element.size(0)
            
            # num_points_fix is used to fix the number of atoms for all samples
            if args.fixed_num_atoms is not None:
                num_points_fix = args.fixed_num_atoms
                logger.info(f"Using fixed number of atoms for all samples: {num_points_fix}")
            else:
                num_points_fix = None
                
            context = None
            t_pocket_start = time.time()
            while num_samples > 0 and try_num > 0:
                largest_mol_flag = False

                if num_samples < 1:
                    print(num_samples)

                if try_num < 10 and args.fixed_num_atoms is None:
                    # Only allow variation when fixed_num_atoms is not specified
                    num_points_fix = None
                    
                data_list, _ = construct_dataset_pocket(
                    num_samples,
                    batch_size,
                    dataset_info,
                    num_points,
                    num_points_fix,
                    None,
                    None,
                    None,
                    protein_atom_feature,
                    protein_atom_feature_full,
                    data.protein_pos,
                    data.protein_bond_index,
                )

                batch = Batch.from_data_list(
                    data_list[0], follow_batch=FOLLOW_BATCH
                ).to(device)
                try:
                    try_num -= 1
                    model.num_timesteps = args.max_timesteps
                    args.n_steps = min(args.n_steps, args.max_timesteps)
                    trial += 1

                    # CFG sampling: if guidance_scale != 1.0 and uncond_protein_gen exists
                    if args.guidance_scale != 1.0 and uncond_protein_gen is not None:
                        logger.info(f"Using CFG sampling with guidance_scale={args.guidance_scale}")

                        # Generate unconditional protein
                        uncond_features, uncond_pos, uncond_batch_idx, uncond_backbone = \
                            uncond_protein_gen(batch.num_graphs, device)

                        # Use CFG sampling method (combines predictions at each denoising step)
                        pos_gen, pos_gen_traj, atom_type, atom_traj, spm_scores = (
                            model.langevin_dynamics_sample_cfg(
                                ligand_atom_type=batch.ligand_atom_feature,
                                ligand_pos_init=batch.ligand_pos,
                                ligand_bond_index=batch.ligand_bond_index,
                                ligand_bond_type=None,
                                ligand_num_node=batch.ligand_num_node,
                                ligand_batch=batch.ligand_atom_feature_batch,
                                # Conditional protein (real)
                                protein_atom_type_cond=batch.protein_atom_feature.float(),
                                protein_atom_feature_full_cond=batch.protein_atom_feature_full.float(),
                                protein_pos_cond=batch.protein_pos,
                                protein_backbone_mask_cond=None,
                                protein_batch_cond=batch.protein_atom_feature_full_batch,
                                # Unconditional protein (learnable)
                                protein_atom_type_uncond=uncond_features[:, :10].float(),  # element features only
                                protein_atom_feature_full_uncond=uncond_features.float(),
                                protein_pos_uncond=uncond_pos,
                                protein_backbone_mask_uncond=uncond_backbone,
                                protein_batch_uncond=uncond_batch_idx,
                                num_graphs=batch.num_graphs,
                                context=context,
                                guidance_scale=args.guidance_scale,
                                extend_order=False,
                                n_steps=args.n_steps,
                                step_lr=1e-6,
                                w_global_pos=args.w_global_pos,
                                w_global_node=args.w_global_node,
                                w_local_pos=args.w_local_pos,
                                w_local_node=args.w_local_node,
                                global_start_sigma=args.global_start_sigma,
                                clip=args.clip,
                                clip_local=clip_local,
                                sampling_type=args.sampling_type,
                                eta=args.eta,
                                w_clash=args.w_clash,
                                clash_start_sigma=10,
                            )
                        )
                    else:
                        # Standard sampling (without CFG)
                        if args.run_mode == "comparison":
                            # ========== COMPARISON MODE ==========
                            # Save initial noise so both samplings use the same starting point
                            initial_noise = batch.ligand_pos.clone()
                            logger.info(f"[COMPARISON] Saved initial noise for {pdb_name}")

                            # ========== 1. Baseline sampling (without SPO) ==========
                            print("\n" + "="*80)
                            print("BASELINE SAMPLING (NO SPO)")
                            print("="*80)
                            logger.info(f"[COMPARISON] Running BASELINE sampling (no SPO)...")
                            pos_gen_baseline, pos_gen_traj_baseline, atom_type_baseline, atom_traj_baseline, spm_scores_baseline, spo_decisions_baseline = (
                                model.langevin_dynamics_sample(
                                    ligand_atom_type=batch.ligand_atom_feature,
                                    ligand_pos_init=batch.ligand_pos,
                                    ligand_bond_index=batch.ligand_bond_index,
                                    ligand_bond_type=None,
                                    ligand_num_node=batch.ligand_num_node,
                                    ligand_batch=batch.ligand_atom_feature_batch,
                                    protein_atom_type=batch.protein_atom_feature.float(),
                                    protein_atom_feature_full=batch.protein_atom_feature_full.float(),
                                    protein_pos=batch.protein_pos,
                                    protein_bond_index=batch.protein_bond_index,
                                    protein_backbone_mask=None,
                                    protein_batch=batch.protein_atom_feature_full_batch,
                                    num_graphs=batch.num_graphs,
                                    extend_order=False,
                                    n_steps=args.n_steps,
                                    step_lr=1e-6,
                                    w_global_pos=args.w_global_pos,
                                    w_global_node=args.w_global_node,
                                    w_local_pos=args.w_local_pos,
                                    w_local_node=args.w_local_node,
                                    global_start_sigma=args.global_start_sigma,
                                    clip=args.clip,
                                    clip_local=clip_local,
                                    sampling_type=args.sampling_type,
                                    eta=args.eta,
                                    context=context,
                                    w_clash=args.w_clash,
                                    clash_start_sigma=10,
                                    # SPM-related parameters (for recording scores)
                                    spm_model=spm_model,
                                    spm_eval_interval=args.spm_eval_interval,
                                    # Force disable SPO
                                    use_spo=False,
                                    spo_resample_interval=args.spo_resample_interval,
                                    spo_num_candidates=args.spo_num_candidates,
                                    spo_start_step=args.spo_start_step,
                                    # QED and SA calculation parameters
                                    dataset_info=dataset_info,
                                    atomic_numbers=atomic_numbers,
                                    # Scoring weight parameters
                                    score_weight_spm=args.score_weight_spm,
                                    score_weight_qed=args.score_weight_qed,
                                    score_weight_sa=args.score_weight_sa,
                                    score_weight_clash=args.score_weight_clash,
                                    clash_threshold=args.clash_threshold,
                                    spo_use_spm=args.spo_use_spm,
                                    use_lipinski=args.use_lipinski,
                                )
                            )
                            logger.info(f"[COMPARISON] Baseline sampling completed")

                            # ========== 2. Rebuild batch using the same initial noise ==========
                            batch = Batch.from_data_list(data_list[0], follow_batch=FOLLOW_BATCH).to(device)
                            batch.ligand_pos = initial_noise.clone()  # Key: use the same initial noise!
                            logger.info(f"[COMPARISON] Reset batch with same initial noise")

                            # ========== 3. SPO sampling (with SPO guidance) ==========
                            print("\n" + "="*80)
                            print("SPO SAMPLING (WITH SPO GUIDANCE)")
                            print("="*80)
                            logger.info(f"[COMPARISON] Running SPO sampling (with SPO guidance)...")
                            pos_gen, pos_gen_traj, atom_type, atom_traj, spm_scores, spo_decisions = (
                                model.langevin_dynamics_sample(
                                    ligand_atom_type=batch.ligand_atom_feature,
                                    ligand_pos_init=batch.ligand_pos,
                                    ligand_bond_index=batch.ligand_bond_index,
                                    ligand_bond_type=None,
                                    ligand_num_node=batch.ligand_num_node,
                                    ligand_batch=batch.ligand_atom_feature_batch,
                                    protein_atom_type=batch.protein_atom_feature.float(),
                                    protein_atom_feature_full=batch.protein_atom_feature_full.float(),
                                    protein_pos=batch.protein_pos,
                                    protein_bond_index=batch.protein_bond_index,
                                    protein_backbone_mask=None,
                                    protein_batch=batch.protein_atom_feature_full_batch,
                                    num_graphs=batch.num_graphs,
                                    extend_order=False,
                                    n_steps=args.n_steps,
                                    step_lr=1e-6,
                                    w_global_pos=args.w_global_pos,
                                    w_global_node=args.w_global_node,
                                    w_local_pos=args.w_local_pos,
                                    w_local_node=args.w_local_node,
                                    global_start_sigma=args.global_start_sigma,
                                    clip=args.clip,
                                    clip_local=clip_local,
                                    sampling_type=args.sampling_type,
                                    eta=args.eta,
                                    context=context,
                                    w_clash=args.w_clash,
                                    clash_start_sigma=10,
                                    # SPM-related parameters
                                    spm_model=spm_model,
                                    spm_eval_interval=args.spm_eval_interval,
                                    # Enable SPO guidance
                                    use_spo=True,
                                    spo_resample_interval=args.spo_resample_interval,
                                    spo_num_candidates=args.spo_num_candidates,
                                    spo_start_step=args.spo_start_step,
                                    # QED and SA calculation parameters
                                    dataset_info=dataset_info,
                                    atomic_numbers=atomic_numbers,
                                    # Scoring weight parameters
                                    score_weight_spm=args.score_weight_spm,
                                    score_weight_qed=args.score_weight_qed,
                                    score_weight_sa=args.score_weight_sa,
                                    score_weight_clash=args.score_weight_clash,
                                    clash_threshold=args.clash_threshold,
                                    spo_use_spm=args.spo_use_spm,
                                    use_lipinski=args.use_lipinski,
                                )
                            )
                            logger.info(f"[COMPARISON] SPO sampling completed")
                        else:
                            # Original single-pass sampling logic
                            pos_gen, pos_gen_traj, atom_type, atom_traj, spm_scores, spo_decisions = (
                                model.langevin_dynamics_sample(
                                    ligand_atom_type=batch.ligand_atom_feature,
                                    ligand_pos_init=batch.ligand_pos,
                                    ligand_bond_index=batch.ligand_bond_index,
                                    ligand_bond_type=None,
                                    ligand_num_node=batch.ligand_num_node,
                                    ligand_batch=batch.ligand_atom_feature_batch,
                                    protein_atom_type=batch.protein_atom_feature.float(),
                                    protein_atom_feature_full=batch.protein_atom_feature_full.float(),
                                    protein_pos=batch.protein_pos,
                                    protein_bond_index=batch.protein_bond_index,
                                    protein_backbone_mask=None,
                                    protein_batch=batch.protein_atom_feature_full_batch,
                                    num_graphs=batch.num_graphs,
                                    extend_order=False,  # Done in transforms.
                                    n_steps=args.n_steps,
                                    step_lr=1e-6,  # 1e-6
                                    w_global_pos=args.w_global_pos,
                                    w_global_node=args.w_global_node,
                                    w_local_pos=args.w_local_pos,
                                    w_local_node=args.w_local_node,
                                    global_start_sigma=args.global_start_sigma,
                                    clip=args.clip,
                                    clip_local=clip_local,
                                    sampling_type=args.sampling_type,
                                    eta=args.eta,
                                    context=context,
                                    w_clash=args.w_clash,
                                    clash_start_sigma=10,
                                    # SPM-related parameters
                                    spm_model=spm_model,
                                    spm_eval_interval=args.spm_eval_interval,
                                    # SPO resampling parameters
                                    use_spo=args.use_spo,
                                    spo_resample_interval=args.spo_resample_interval,
                                    spo_num_candidates=args.spo_num_candidates,
                                    spo_start_step=args.spo_start_step,
                                    # QED and SA calculation parameters
                                    dataset_info=dataset_info,
                                    atomic_numbers=atomic_numbers,
                                    # Scoring weight parameters
                                    score_weight_spm=args.score_weight_spm,
                                    score_weight_qed=args.score_weight_qed,
                                    score_weight_sa=args.score_weight_sa,
                                    score_weight_clash=args.score_weight_clash,
                                    clash_threshold=args.clash_threshold,
                                    spo_use_spm=args.spo_use_spm,
                                    use_lipinski=args.use_lipinski,
                                )
                            )

                    # Process SPO sampling results (or standard sampling results)
                    pos_list = unbatch(pos_gen, batch.ligand_atom_feature_batch)
                    atom_list = unbatch(atom_type, batch.ligand_atom_feature_batch)

                    # If in comparison mode, also process baseline results
                    if args.run_mode == "comparison":
                        pos_list_baseline = unbatch(pos_gen_baseline, batch.ligand_atom_feature_batch)
                        atom_list_baseline = unbatch(atom_type_baseline, batch.ligand_atom_feature_batch)
                        logger.info(f"[COMPARISON] Processing both baseline and SPO results with paired validation...")
                        
                        # ========== Paired processing mode ==========
                        # Ensure that corresponding baseline and SPO samples either both succeed or both fail
                        # Generate a global unique ID for each sample pair (accounting for pocket index)
                        pocket_idx = len(baseline_results) // args.num_samples if baseline_results else 0
                        
                        for m in range(num_samples):
                            pos_baseline = pos_list_baseline[m].detach().cpu()
                            atom_type_baseline_m = atom_list_baseline[m].detach().cpu()
                            pos_spo = pos_list[m].detach().cpu()
                            atom_type_spo_m = atom_list[m].detach().cpu()
                            
                            # Try to build the baseline molecule
                            gmol_baseline, water_pos_baseline, baseline_success = try_build_molecule(
                                pos_baseline, atom_type_baseline_m,
                                args.build_method, atomic_numbers, atomic_numbers_crossdock,
                                dataset_info, largest_mol_flag=(try_num < 10)
                            )
                            
                            # Try to build the SPO molecule
                            gmol_spo, water_pos_spo, spo_success = try_build_molecule(
                                pos_spo, atom_type_spo_m,
                                args.build_method, atomic_numbers, atomic_numbers_crossdock,
                                dataset_info, largest_mol_flag=(try_num < 10)
                            )
                            
                            # Only save if both succeed
                            if baseline_success and spo_success:
                                g_smile_baseline = Chem.MolToSmiles(gmol_baseline)
                                g_smile_spo = Chem.MolToSmiles(gmol_spo)
                                
                                logger.info(f"[COMPARISON] ✅ Sample {m} paired success")
                                logger.info(f"  Baseline: {g_smile_baseline}")
                                logger.info(f"  SPO:      {g_smile_spo}")
                                
                                # Filter SPM scores
                                sample_spm_scores_baseline = None
                                if args.use_spm and spm_scores_baseline:
                                    sample_spm_scores_baseline = [
                                        score_dict for score_dict in spm_scores_baseline 
                                        if score_dict.get('sample_idx') == m
                                    ]
                                
                                sample_spm_scores_spo = None
                                if args.use_spm and spm_scores:
                                    sample_spm_scores_spo = [
                                        score_dict for score_dict in spm_scores 
                                        if score_dict.get('sample_idx') == m
                                    ]
                                
                                sample_spo_decisions = None
                                if spo_decisions:
                                    sample_spo_decisions = [
                                        decision for decision in spo_decisions
                                        if decision.get('sample_idx') == m
                                    ]
                                
                                # Compute global unique ID
                                global_sample_id = len(baseline_results)
                                
                                # Save baseline result
                                baseline_result = {
                                    "atom_type": atom_type_baseline_m.detach().cpu(),
                                    "pos": pos_baseline.detach().cpu(),
                                    "smile": g_smile_baseline,
                                    "protein_file": data.protein_filename,
                                    "ligand_file": data.ligand_filename,
                                    "mol": gmol_baseline,
                                    "water_pos": water_pos_baseline,
                                    "spm_scores": sample_spm_scores_baseline,
                                    "spo_decisions": None,
                                    "sample_idx": m,  # Local sample index (within pocket)
                                    "global_id": global_sample_id,  # Global unique ID
                                    "pocket_idx": pocket_idx,  # Pocket index
                                }
                                baseline_results.append(baseline_result)
                                
                                # Save SPO result
                                spo_result = {
                                    "atom_type": atom_type_spo_m.detach().cpu(),
                                    "pos": pos_spo.detach().cpu(),
                                    "smile": g_smile_spo,
                                    "protein_file": data.protein_filename,
                                    "ligand_file": data.ligand_filename,
                                    "mol": gmol_spo,
                                    "water_pos": water_pos_spo,
                                    "spm_scores": sample_spm_scores_spo,
                                    "spo_decisions": sample_spo_decisions,
                                    "sample_idx": m,  # Local sample index (within pocket)
                                    "global_id": global_sample_id,  # Global unique ID
                                    "pocket_idx": pocket_idx,  # Pocket index
                                }
                                spo_results.append(spo_result)
                                
                                valid += 1
                                num_samples -= 1
                                
                                if num_samples == 0:
                                    break
                            else:
                                # At least one failed, skip this sample pair
                                logger.warning(f"[COMPARISON] ❌ Sample {m} skipped (baseline={'OK' if baseline_success else 'FAIL'}, SPO={'OK' if spo_success else 'FAIL'})")
                        
                        # Paired processing complete, skip the subsequent independent processing loop
                        continue
                    
                    # ========== Non-comparison mode: independent processing ==========
                    # atom_charge_list = atom_charge.reshape(num_samples, -1, 1)
                    for m in range(num_samples):
                        try:
                            pos = pos_list[m].detach().cpu()
                            # pos = pos+torch.mean(data.protein_pos,0)
                            atom_type = atom_list[m].detach().cpu()

                            new_ligand = torch.zeros(
                                [pos.size(0), len(ATOM_FAMILIES)], dtype=np.long
                            )

                            a = 0
                            num_atom_type = len(atomic_numbers)
                            if args.build_method == "reconstruct":
                                new_element = torch.tensor(
                                    [
                                        atomic_numbers_crossdock[m]
                                        for m in torch.argmax(atom_type[:, :8], dim=1)
                                    ]
                                )
                                indicators_elements = torch.argmax(
                                    atom_type[:, 8:], dim=1
                                )
                                indicators = torch.zeros(
                                    [pos.size(0), len(ATOM_FAMILIES)], dtype=np.long
                                )
                                for i, n in enumerate(indicators_elements):
                                    indicators[i, n] = 1

                                gmol = reconstruct_from_generated(
                                    pos, new_element, indicators
                                )
                                water_pos = None  # the reconstruct method does not return water_pos

                            elif args.build_method == "build":
                                new_element = torch.argmax(
                                    atom_type[:, :num_atom_type], dim=1
                                )
                                gmol, water_pos = make_mol_openbabel(
                                    pos, new_element, dataset_info
                                )

                            # gen_mol = set_rdmol_positions(rdmol, data.ligand_pos)
                            g_smile = Chem.MolToSmiles(gmol)
                            print("generated smile:", g_smile)

                            if g_smile is not None:
                                FINISHED = True
                                if "." not in g_smile:
                                    stable += 1
                                if g_smile.count(".") > 1:
                                    raise MolReconsError()

                                if try_num < 10:
                                    largest_mol_flag = True
                                # if try_num<10:
                                #     args.sampling_type = 'ddpm_noisy'
                                if largest_mol_flag:
                                    mol_frags = Chem.rdmolops.GetMolFrags(
                                        gmol, asMols=True, sanitizeFrags=False
                                    )
                                    gmol = max(
                                        mol_frags,
                                        default=gmol,
                                        key=lambda m: m.GetNumAtoms(),
                                    )
                                    g_smile = Chem.MolToSmiles(gmol)
                                    print("largest generated smile part:", g_smile)
                                    if g_smile is None:
                                        raise MolReconsError()
                                if g_smile.count(".") > 0:
                                    raise MolReconsError()
                                if len(g_smile) < 4:
                                    raise MolReconsError()
                                if save_sdf_flag:
                                    save_sdf(gmol, sdf_dir, gen_file_name)
                                valid += 1
                                num_samples -= 1
                                smile_list.append(g_smile)
                                print(
                                    "Successfully generate molecule for {}, remining {} samples generated".format(
                                        pdb_name, num_samples
                                    )
                                )

                            else:
                                raise MolReconsError()

                            if save_results:
                                # Filter out the SPM scores belonging to the current sample
                                sample_spm_scores = None
                                if args.use_spm and spm_scores:
                                    sample_spm_scores = [
                                        score_dict for score_dict in spm_scores 
                                        if score_dict.get('sample_idx') == m
                                    ]
                                
                                # Filter out the SPO decision info belonging to the current sample
                                sample_spo_decisions = None
                                if args.use_spo and spo_decisions:
                                    sample_spo_decisions = [
                                        decision for decision in spo_decisions
                                        if decision.get('sample_idx') == m
                                    ]
                                
                                # metrics = {'SA':g_sa,'QED':g_qed,'logP':g_logP,'Lipinski':g_Lipinski,
                                #            'vina':g_vina_score,'high_affinity':g_high_affinity}
                                result = {
                                    "atom_type": atom_type.detach().cpu(),
                                    "pos": pos.detach().cpu(),
                                    "smile": g_smile,
                                    # 'l_smile':lg_smile,
                                    "protein_file": data.protein_filename,
                                    "ligand_file": data.ligand_filename,
                                    # 'generated_ligand_sdf': gen_file_name,
                                    "mol": gmol,
                                    # 'l_mol':largest_mol,
                                    # 'metric_result':metrics}
                                    "water_pos": water_pos,
                                    "spm_scores": sample_spm_scores,
                                    "spo_decisions": sample_spo_decisions,
                                }
                                # In comparison mode, SPO results go into spo_results; otherwise into results
                                if args.run_mode == "comparison":
                                    spo_results.append(result)
                                else:
                                    results.append(result)
                            if num_samples == 0:
                                break

                        except (
                            RuntimeError,
                            MolReconsError,
                            TypeError,
                            IndexError,
                            OverflowError,
                        ):  # MolReconsError,TypeError,IndexError,OverflowError
                            print("Invalid,continue")

                except (
                    FloatingPointError
                ):  # ,MolReconsError,TypeError,IndexError,OverflowError
                    clip_local = 20
                    logger.warning(
                        "Ignoring, because reconstruction error encountered or retrying with local clipping or vina error."
                    )
                    print("Resample the number of the atoms and regenerate!")

            # ========== Baseline results are already completed in paired processing ==========
            # The old independent processing logic has been removed; paired processing is now used to ensure a fair comparison

            time_list.append(time.time() - t_pocket_start)
            logger.info(
                str(data.protein_filename)
                + "takes {} seconds".format(time.time() - t_pocket_start)
            )
    times_arr = torch.tensor(time_list)
    try:
        logger.info(
            f"Time per pocket: {times_arr.mean():.3f} \pm "
            f"{times_arr.std(unbiased=False):.2f}"
        )
    except:
        logger.info(torch.mean(times_arr))

    logger.info(f"Total trials: {trial}")

    if save_results:
        if args.run_mode == "comparison":
            # ========== Comparison mode: save two separate result files ==========
            baseline_path = os.path.join(output_dir, "baseline_results.pkl")
            spo_path = os.path.join(output_dir, "spo_results.pkl")

            logger.info("=" * 60)
            logger.info("COMPARISON MODE - Saving results:")
            logger.info(f"  Baseline: {baseline_path} ({len(baseline_results)} samples)")
            logger.info(f"  SPO:      {spo_path} ({len(spo_results)} samples)")

            with open(baseline_path, "wb") as f:
                pickle.dump(baseline_results, f)

            with open(spo_path, "wb") as f:
                pickle.dump(spo_results, f)

            logger.info("Results saved successfully!")

            # ========== Generate comparison visualizations ==========
            logger.info("=" * 60)
            logger.info("Generating comparison visualizations...")
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            try:
                visualize_spm_comparison(baseline_results, spo_results, vis_dir)
                logger.info(f"Visualizations saved to: {vis_dir}")
                logger.info("  - spm_score_comparison.png")
                logger.info("  - final_score_distribution.png")
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {e}")
                import traceback
                traceback.print_exc()

            logger.info("=" * 60)
            logger.info("COMPARISON COMPLETE!")
            logger.info(f"Results directory: {output_dir}")
            logger.info("You can now evaluate both results with evaluate.py:")
            logger.info(f"  python evaluate.py --path {baseline_path}")
            logger.info(f"  python evaluate.py --path {spo_path}")
            logger.info("=" * 60)
        else:
            # Single mode (spo_only or baseline_only): save a single result file
            if args.run_mode == "spo_only":
                save_path = os.path.join(output_dir, "spo_results.pkl")
            elif args.run_mode == "baseline_only":
                save_path = os.path.join(output_dir, "baseline_results.pkl")
            else:
                save_path = os.path.join(output_dir, "samples_all.pkl")
            logger.info("Saving samples to: %s" % save_path)

            with open(save_path, "wb") as f:
                pickle.dump(results, f)
                f.close()

        save_time_path = os.path.join(output_dir, "time.pkl")
        logger.info("Saving time to: %s" % save_time_path)
        with open(save_time_path, "wb") as f:
            pickle.dump(time_list, f)
            f.close()
