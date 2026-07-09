import argparse
import pickle


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

# Geometry optimization imports
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit warnings during optimization
RDLogger.DisableLog('rdApp.*')

# OpenMM imports for protein-aware optimization
OPENMM_AVAILABLE = False
try:
    import openmm
    from openmm import app, unit
    from openmm.app import PDBFile, Modeller, ForceField as OpenMMForceField
    from openmm import LangevinMiddleIntegrator, LocalEnergyMinimizer
    OPENMM_AVAILABLE = True
except ImportError:
    pass

# OpenFF imports for small molecule parameterization
OPENFF_AVAILABLE = False
try:
    from openff.toolkit import Molecule as OFFMolecule
    from openff.toolkit import ForceField as OpenFFForceField
    from openff.interchange import Interchange
    OPENFF_AVAILABLE = True
except ImportError:
    pass

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


def optimize_with_openmm(ligand_mol, protein_pdb_path, max_iters=500,
                         tolerance=10.0, add_hs=True):
    """
    Optimize ligand geometry using OpenMM with protein environment constraints.

    This function performs energy minimization of the ligand while considering
    the protein environment. The protein atoms are fixed during optimization.

    Args:
        ligand_mol: RDKit molecule object (the ligand to optimize)
        protein_pdb_path: Path to the protein PDB file
        max_iters: Maximum iterations for energy minimization
        tolerance: Energy tolerance for convergence (kJ/mol/nm)
        add_hs: Whether to add hydrogens to ligand

    Returns:
        tuple: (optimized_mol, success, energy_before, energy_after)
    """
    if not OPENMM_AVAILABLE:
        print("OpenMM not available. Please install: conda install -c conda-forge openmm")
        return ligand_mol, False, None, None

    if not OPENFF_AVAILABLE:
        print("OpenFF not available. Please install: conda install -c conda-forge openff-toolkit openff-interchange")
        return ligand_mol, False, None, None

    if ligand_mol is None:
        return None, False, None, None

    try:
        import tempfile
        import numpy as np

        # 1. Prepare ligand
        mol_copy = Chem.RWMol(ligand_mol)
        try:
            Chem.SanitizeMol(mol_copy)
        except:
            Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

        if add_hs:
            mol_copy = Chem.AddHs(mol_copy, addCoords=True)

        # Ensure conformer exists
        if mol_copy.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_copy, randomSeed=42)

        # Get original ligand coordinates
        conf = mol_copy.GetConformer()
        ligand_positions_original = []
        for i in range(mol_copy.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            ligand_positions_original.append([pos.x, pos.y, pos.z])
        ligand_positions_original = np.array(ligand_positions_original)

        # 2. Load protein
        protein_pdb = PDBFile(protein_pdb_path)

        # 3. Create OpenFF molecule from RDKit mol
        try:
            off_mol = OFFMolecule.from_rdkit(mol_copy, allow_undefined_stereo=True)
        except Exception as e:
            print(f"OpenFF molecule creation failed: {e}")
            return ligand_mol, False, None, None

        # 4. Parameterize ligand with OpenFF
        try:
            openff_forcefield = OpenFFForceField('openff-2.1.0.offxml')
            ligand_interchange = Interchange.from_smirnoff(
                force_field=openff_forcefield,
                topology=[off_mol]
            )
        except Exception as e:
            print(f"OpenFF parameterization failed: {e}")
            return ligand_mol, False, None, None

        # 5. Create protein system with OpenMM force field
        protein_forcefield = OpenMMForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        # Create modeller for protein
        modeller = Modeller(protein_pdb.topology, protein_pdb.positions)

        # Add missing hydrogens to protein
        modeller.addHydrogens(protein_forcefield)

        # 6. Create combined system
        # Get protein system
        protein_system = protein_forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None
        )

        # Get ligand system from interchange
        ligand_system = ligand_interchange.to_openmm(combine_nonbonded_forces=True)

        # 7. Create combined topology and positions
        # For simplicity, we'll optimize ligand-only with protein as external potential
        # This is a simplified approach - full integration would require merging topologies

        # Get number of protein atoms
        num_protein_atoms = modeller.topology.getNumAtoms()
        num_ligand_atoms = mol_copy.GetNumAtoms()

        # Create a simple Lennard-Jones potential between ligand and protein
        # Using CustomExternalForce to add protein influence

        # Get protein positions as numpy array
        protein_positions = np.array([[p.x, p.y, p.z] for p in modeller.positions.value_in_unit(unit.nanometer)])

        # Add custom non-bonded interaction between ligand and protein
        # Using a soft repulsive potential
        custom_force = openmm.CustomExternalForce(
            "k * step(r0 - r) * (r0 - r)^2; r = sqrt((x-px)^2 + (y-py)^2 + (z-pz)^2)"
        )
        custom_force.addGlobalParameter("k", 1000.0)  # force constant kJ/mol/nm^2
        custom_force.addGlobalParameter("r0", 0.25)   # clash distance in nm (2.5 Angstrom)
        custom_force.addPerParticleParameter("px")
        custom_force.addPerParticleParameter("py")
        custom_force.addPerParticleParameter("pz")

        # For each ligand atom, add repulsion from nearby protein atoms
        for i in range(num_ligand_atoms):
            lig_pos = ligand_positions_original[i] / 10.0  # convert to nm
            # Find nearby protein atoms
            for j in range(num_protein_atoms):
                prot_pos = protein_positions[j]
                dist = np.sqrt(np.sum((lig_pos - prot_pos)**2))
                if dist < 0.8:  # within 8 Angstrom
                    custom_force.addParticle(i, [prot_pos[0], prot_pos[1], prot_pos[2]])

        # Add custom force to ligand system
        ligand_system.addForce(custom_force)

        # 8. Create simulation for ligand only
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds
        )

        # Get ligand topology from interchange
        ligand_topology = ligand_interchange.to_openmm_topology()

        simulation = app.Simulation(ligand_topology, ligand_system, integrator)

        # Set ligand positions (convert from Angstrom to nm)
        ligand_positions_nm = [(pos[0]/10.0, pos[1]/10.0, pos[2]/10.0) for pos in ligand_positions_original]
        simulation.context.setPositions(ligand_positions_nm * unit.nanometer)

        # 9. Calculate energy before minimization
        state_before = simulation.context.getState(getEnergy=True)
        energy_before = state_before.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # 10. Minimize energy
        LocalEnergyMinimizer.minimize(
            simulation.context,
            tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_iters
        )

        # 11. Get final state
        state_after = simulation.context.getState(getPositions=True, getEnergy=True)
        energy_after = state_after.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        final_positions = state_after.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        # 12. Update RDKit molecule with optimized coordinates
        result_mol = Chem.RWMol(mol_copy)
        conf = result_mol.GetConformer()

        for i in range(result_mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(final_positions[i][0]),
                                     float(final_positions[i][1]),
                                     float(final_positions[i][2])))

        # Remove hydrogens if they were added
        if add_hs:
            result_mol = Chem.RemoveHs(result_mol)

        success = True
        print(f"OpenMM optimization: energy {energy_before:.2f} -> {energy_after:.2f} kJ/mol")

        return result_mol.GetMol(), success, energy_before, energy_after

    except Exception as e:
        print(f"OpenMM optimization failed: {e}")
        import traceback
        traceback.print_exc()
        # Return original molecule without optimization
        return ligand_mol, False, None, None


class LearnableUnconditionalProtein(nn.Module):
    """可学习的无条件蛋白质表示 - 与train_modify1.py中的定义保持一致"""
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
    
    # Fixed number of atoms
    parser.add_argument("--fixed_num_atoms", type=int, default=None, help="fix the number of atoms in generated ligands (if None, use reference ligand's atom count)")

    # Geometry optimization parameters (OpenMM with protein environment)
    parser.add_argument("--optimize_geometry", action="store_true",
                        help="whether to apply OpenMM geometry optimization with protein environment after molecule reconstruction")
    parser.add_argument("--optimize_max_iters", type=int, default=500,
                        help="maximum iterations for OpenMM energy minimization")
    parser.add_argument("--optimize_add_hs", action="store_true",
                        help="add hydrogens before optimization (recommended for better results)")
    parser.add_argument("--optimize_tolerance", type=float, default=10.0,
                        help="energy tolerance for OpenMM optimization (kJ/mol/nm)")

    args = parser.parse_args()
    
    # 验证SPO参数
    if args.use_spo and not args.use_spm:
        print("Error: --use_spo requires --use_spm to be enabled")
        print("SPO resampling needs SPM model to evaluate candidates")
        sys.exit(1)

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
    tag = "result"
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

    # Log geometry optimization settings
    if args.optimize_geometry:
        logger.info(f"Geometry optimization ENABLED: OpenMM (protein-aware), "
                    f"max_iters={args.optimize_max_iters}, tolerance={args.optimize_tolerance}, add_hs={args.optimize_add_hs}")
        if not OPENMM_AVAILABLE:
            logger.warning("OpenMM not installed! Install with: conda install -c conda-forge openmm")
        if not OPENFF_AVAILABLE:
            logger.warning("OpenFF not installed! Install with: conda install -c conda-forge openff-toolkit openff-interchange")
    else:
        logger.info("Geometry optimization DISABLED")

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
    for i, data in enumerate(testset):
        if not (args.start_idx <= i < args.end_idx):
            continue
        test_set_selected.append(data)
        # break

    print("test_set_selected: ", len(test_set_selected))

    with open(os.path.join(log_dir, "pocket_info.txt"), "a") as f:
        f.write(data.protein_filename + "\n")

    logger.info("Building model...")
    logger.info(config.model["network"])
    print(config.model)
    model = get_model(config.model).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # 加载SPM模型（如果启用）
    spm_model = None
    if args.use_spm:
        logger.info(f"Loading SPM model from: {args.spm_checkpoint}")
        
        # 直接从完整路径导入
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
    
    # 加载可学习的无条件蛋白质生成器（用于CFG采样）
    uncond_protein_gen = None
    if 'uncond_protein_gen' in ckpt and args.guidance_scale != 1.0:
        protein_feature_dim = 31  # 默认值
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
            # 使用fixed_num_atoms参数或参考配体的原子数
            num_points = data.ligand_element.size(0)
            
            # num_points_fix用于固定所有样本的原子数
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
                    # 只在未指定fixed_num_atoms时才允许变化
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

                    # CFG 采样：如果 guidance_scale != 1.0 且有 uncond_protein_gen
                    if args.guidance_scale != 1.0 and uncond_protein_gen is not None:
                        logger.info(f"Using CFG sampling with guidance_scale={args.guidance_scale}")

                        # 生成无条件蛋白质
                        uncond_features, uncond_pos, uncond_batch_idx, uncond_backbone = \
                            uncond_protein_gen(batch.num_graphs, device)

                        # 使用 CFG 采样方法（在每个去噪步骤中组合预测）
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
                        # 标准采样（不使用CFG）
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
                                # SPM相关参数
                                spm_model=spm_model,
                                spm_eval_interval=args.spm_eval_interval,
                                # SPO重采样参数
                                use_spo=args.use_spo,
                                spo_resample_interval=args.spo_resample_interval,
                                spo_num_candidates=args.spo_num_candidates,
                            )
                        )

                    pos_list = unbatch(pos_gen, batch.ligand_atom_feature_batch)
                    atom_list = unbatch(atom_type, batch.ligand_atom_feature_batch)
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
                                water_pos = None  # reconstruct method doesn't return water_pos

                            elif args.build_method == "build":
                                new_element = torch.argmax(
                                    atom_type[:, :num_atom_type], dim=1
                                )
                                gmol, water_pos = make_mol_openbabel(
                                    pos, new_element, dataset_info
                                )

                            # Apply OpenMM geometry optimization if enabled (protein-aware)
                            opt_success = False
                            energy_before, energy_after = None, None
                            if args.optimize_geometry and gmol is not None:
                                # OpenMM optimization with protein environment
                                protein_pdb_path = data.protein_filename
                                gmol, opt_success, energy_before, energy_after = optimize_with_openmm(
                                    gmol,
                                    protein_pdb_path=protein_pdb_path,
                                    max_iters=args.optimize_max_iters,
                                    tolerance=args.optimize_tolerance,
                                    add_hs=args.optimize_add_hs,
                                )

                                if opt_success and energy_before is not None and energy_after is not None:
                                    print(f"Geometry optimized (OpenMM): energy {energy_before:.2f} -> {energy_after:.2f} kJ/mol")
                                elif not opt_success:
                                    print("Geometry optimization failed or did not converge")

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
                                # 筛选出属于当前样本的SPM分数
                                sample_spm_scores = None
                                if args.use_spm and spm_scores:
                                    sample_spm_scores = [
                                        score_dict for score_dict in spm_scores 
                                        if score_dict.get('sample_idx') == m
                                    ]
                                
                                # 筛选出属于当前样本的SPO决策信息
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
                                    # Geometry optimization info (OpenMM with protein environment)
                                    "geometry_optimized": args.optimize_geometry,
                                    "optimization_method": "openmm" if args.optimize_geometry else None,
                                    "optimization_converged": opt_success if args.optimize_geometry else None,
                                    "energy_before": energy_before,
                                    "energy_after": energy_after,
                                }
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
        save_path = os.path.join(output_dir, "samples_all.pkl")
        logger.info("Saving samples to: %s" % save_path)

        # save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, "wb") as f:
            pickle.dump(results, f)
            f.close()

        save_time_path = os.path.join(output_dir, "time.pkl")
        logger.info("Saving time to: %s" % save_path)
        with open(save_time_path, "wb") as f:
            pickle.dump(time_list, f)
            f.close()
