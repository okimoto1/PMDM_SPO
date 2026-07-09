"""
Dataset for ligand pair preference learning.

Loads pairs of ligand molecules from SDF files with preference labels.
"""

import os
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# Atom encoder for CrossDocked dataset (same as ligand_encoder.py)
ATOMIC_NUMBER_TO_INDEX = {
    1: 0,   # H
    6: 1,   # C
    7: 2,   # N
    8: 3,   # O
    9: 4,   # F
    15: 5,  # P
    16: 6,  # S
    17: 7,  # Cl
    34: 8,  # Se
}

try:
    from rdkit import Chem
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install rdkit to use this dataset.")


class LigandPairDataset(Dataset):
    """
    Dataset for ligand pair preference learning.

    Each sample contains:
        - Two ligand molecules (from SDF files)
        - Preference label (which one is better)

    Args:
        data_dir: Directory containing ligand SDF files
        db_path: Path to SQLite database with pair annotations (optional)
        split: 'train', 'val', or 'test'
        cache_dir: Directory to cache processed data
    """

    def __init__(self, data_dir, db_path=None, split='train', cache_dir=None,
                 train_ratio=0.8, val_ratio=0.1, random_seed=42):
        super().__init__()

        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for this dataset. Please install rdkit.")

        self.data_dir = data_dir
        self.db_path = db_path
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        # Load pair annotations
        self.pairs = self._load_pairs()

        print(f"Loaded {len(self.pairs)} ligand pairs for {split} split")

    def _load_pairs(self):
        """
        Load ligand pair annotations from database.

        Returns:
            pairs: List of (ligand_0_path, ligand_1_path, label_0, label_1) tuples
        """
        pairs = []

        if self.db_path and os.path.exists(self.db_path):
            # Load from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query: Get all labeled pairs with ligand paths
            query = """
            SELECT
                la.name AS ligand_a_name,
                lb.name AS ligand_b_name,
                la.sdf_path AS ligand_a_sdf,
                lb.sdf_path AS ligand_b_sdf,
                lab.winner
            FROM label lab
            JOIN pair p ON lab.pair_id = p.id
            JOIN ligand la ON p.ligand_a_id = la.id
            JOIN ligand lb ON p.ligand_b_id = lb.id
            """

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            print(f"Loaded {len(rows)} labeled pairs from database")

            # Process each row
            for row in rows:
                ligand_a_name, ligand_b_name, sdf_a_rel, sdf_b_rel, winner = row

                # Convert relative paths to absolute paths
                # sdf_path in database is like "backend/data/namiki/xxx.sdf"
                # We need to map this to actual file location
                sdf_a_path = self._resolve_sdf_path(sdf_a_rel, ligand_a_name)
                sdf_b_path = self._resolve_sdf_path(sdf_b_rel, ligand_b_name)

                # Skip if files don't exist
                if not os.path.exists(sdf_a_path) or not os.path.exists(sdf_b_path):
                    continue

                # Convert winner to labels
                # winner = 'A': ligand_a is better -> label_0 = 1, label_1 = 0
                # winner = 'B': ligand_b is better -> label_0 = 0, label_1 = 1
                # winner = 'skip': skip this pair
                if winner == 'skip':
                    # Skip pairs marked as skip
                    continue
                elif winner == 'A':
                    label_0, label_1 = 1.0, 0.0
                elif winner == 'B':
                    label_0, label_1 = 0.0, 1.0
                else:
                    # Unknown winner, treat as tie
                    label_0, label_1 = 0.5, 0.5

                pairs.append((sdf_a_path, sdf_b_path, label_0, label_1))

            # Split data by protein ID for proper generalization
            pairs = self._split_data(pairs)

        else:
            # Fallback: discover pairs from directory structure
            print(f"Database not found at {self.db_path}, using directory-based discovery")

            if not os.path.exists(self.data_dir):
                raise ValueError(f"Data directory not found: {self.data_dir}")

            # Scan directories
            for pocket_dir in os.listdir(self.data_dir):
                pocket_path = os.path.join(self.data_dir, pocket_dir)
                if not os.path.isdir(pocket_path):
                    continue

                # Find all SDF files in this pocket
                sdf_files = [f for f in os.listdir(pocket_path) if f.endswith('.sdf')]

                # Create pairs from same pocket (pairwise comparison)
                for i in range(len(sdf_files)):
                    for j in range(i + 1, len(sdf_files)):
                        sdf_0 = os.path.join(pocket_path, sdf_files[i])
                        sdf_1 = os.path.join(pocket_path, sdf_files[j])

                        # Use dummy labels (all ties)
                        pairs.append((sdf_0, sdf_1, 0.5, 0.5))

        return pairs

    def _resolve_sdf_path(self, sdf_rel_path, ligand_name):
        """
        Resolve SDF path from database to actual file location.

        Database paths are like "backend/data/namiki/xxx.sdf" or
        "backend/data/4tqp_A_rec_1dvz_ofl_lig_tt_min_0_pocket10/xxx.sdf".

        We need to find the actual file in self.data_dir.

        Args:
            sdf_rel_path: Relative path from database
            ligand_name: Ligand name

        Returns:
            Absolute path to SDF file
        """
        # Strategy 1: Try to extract pocket directory from path
        if sdf_rel_path:
            # Extract the last two components: pocket_dir/filename.sdf
            parts = sdf_rel_path.split('/')
            if len(parts) >= 2:
                pocket_dir = parts[-2]
                filename = parts[-1]

                # Check if this directory exists in data_dir
                candidate = os.path.join(self.data_dir, pocket_dir, filename)
                if os.path.exists(candidate):
                    return candidate

        # Strategy 2: Search by ligand name
        # Ligand name might be the filename without extension
        candidate = os.path.join(self.data_dir, f"{ligand_name}.sdf")
        if os.path.exists(candidate):
            return candidate

        # Strategy 3: Search subdirectories for matching filename
        for pocket_dir in os.listdir(self.data_dir):
            pocket_path = os.path.join(self.data_dir, pocket_dir)
            if os.path.isdir(pocket_path):
                candidate = os.path.join(pocket_path, f"{ligand_name}.sdf")
                if os.path.exists(candidate):
                    return candidate

        # Not found - return a path that will fail existence check
        return os.path.join(self.data_dir, ligand_name + ".sdf")

    def _split_data(self, pairs):
        """
        Split data into train/val/test by shuffling pairs.

        Since we're doing preference learning, we simply shuffle all pairs
        and split them deterministically.

        Args:
            pairs: List of all pairs

        Returns:
            List of pairs for current split
        """
        # Set random seed for reproducibility
        rng = np.random.RandomState(self.random_seed)

        # Shuffle pairs
        indices = np.arange(len(pairs))
        rng.shuffle(indices)

        # Calculate split points
        n_total = len(pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        # Split
        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            selected_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Return selected pairs
        return [pairs[i] for i in selected_indices]

    def _parse_sdf(self, sdf_path):
        """
        Parse an SDF file and extract molecular features.

        Args:
            sdf_path: Path to SDF file

        Returns:
            element: [N] numpy array of atomic numbers
            pos: [N, 3] numpy array of coordinates
            has_9plus_ring: float, 1.0 if molecule has ring with size >= 9, else 0.0
        """
        # Read molecule
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        rdmol = next(iter(suppl))

        if rdmol is None:
            raise ValueError(f"Failed to parse SDF file: {sdf_path}")

        rdmol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(rdmol)

        # Get periodic table
        ptable = Chem.GetPeriodicTable()

        # Extract atoms and coordinates
        conf = rdmol.GetConformer()
        element = []
        pos = []

        for atom in rdmol.GetAtoms():
            atomic_num = atom.GetAtomicNum()

            # Replace Br with Cl (as per PMDM)
            if atomic_num == 35:
                atomic_num = 17

            element.append(atomic_num)

            # Get coordinates
            atom_pos = conf.GetAtomPosition(atom.GetIdx())
            pos.append([atom_pos.x, atom_pos.y, atom_pos.z])

        element = np.array(element, dtype=np.int64)
        pos = np.array(pos, dtype=np.float32)

        # Detect 9+ ring (ring with size >= 9)
        ring_info = rdmol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        has_9plus_ring = 1.0 if any(size >= 9 for size in ring_sizes) else 0.0

        return element, pos, has_9plus_ring

    def _featurize(self, element, pos):
        """
        Convert atomic numbers to one-hot encoded features.

        Args:
            element: [N] array of atomic numbers
            pos: [N, 3] array of coordinates

        Returns:
            node_attr: [N, 10] tensor of one-hot features
            pos: [N, 3] tensor of coordinates
        """
        num_atoms = len(element)
        node_attr = np.zeros([num_atoms, 10], dtype=np.float32)

        for i, atomic_num in enumerate(element):
            if atomic_num in ATOMIC_NUMBER_TO_INDEX:
                node_attr[i, ATOMIC_NUMBER_TO_INDEX[atomic_num]] = 1.0
            else:
                # Unknown -> 'others' (index 9)
                node_attr[i, 9] = 1.0

        node_attr = torch.from_numpy(node_attr)
        pos = torch.from_numpy(pos)

        return node_attr, pos

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a ligand pair sample.

        Returns:
            dict with keys:
                - node_attr_0: [N0, 10] atom features for ligand 0
                - pos_0: [N0, 3] coordinates for ligand 0
                - node_attr_1: [N1, 10] atom features for ligand 1
                - pos_1: [N1, 3] coordinates for ligand 1
                - label_0: scalar, preference label for ligand 0
                - label_1: scalar, preference label for ligand 1
                - ring_label_0: scalar, 1.0 if ligand 0 has 9+ ring, else 0.0
                - ring_label_1: scalar, 1.0 if ligand 1 has 9+ ring, else 0.0
                - sdf_0: path to SDF file for ligand 0
                - sdf_1: path to SDF file for ligand 1
        """
        sdf_0, sdf_1, label_0, label_1 = self.pairs[idx]

        # Parse SDFs (now returns ring label as well)
        element_0, pos_0, ring_label_0 = self._parse_sdf(sdf_0)
        element_1, pos_1, ring_label_1 = self._parse_sdf(sdf_1)

        # Featurize
        node_attr_0, pos_0 = self._featurize(element_0, pos_0)
        node_attr_1, pos_1 = self._featurize(element_1, pos_1)

        return {
            'node_attr_0': node_attr_0,
            'pos_0': pos_0,
            'node_attr_1': node_attr_1,
            'pos_1': pos_1,
            'label_0': torch.tensor(label_0, dtype=torch.float32),
            'label_1': torch.tensor(label_1, dtype=torch.float32),
            'ring_label_0': torch.tensor(ring_label_0, dtype=torch.float32),
            'ring_label_1': torch.tensor(ring_label_1, dtype=torch.float32),
            'sdf_0': sdf_0,
            'sdf_1': sdf_1,
        }


def collate_ligand_pairs(batch):
    """
    Collate function for ligand pair dataset.

    Batches multiple ligand pairs together using PyG batching.

    Args:
        batch: List of samples from LigandPairDataset

    Returns:
        Batched dict with:
            - node_attr_0, pos_0, batch_0: Batched ligand 0 data
            - node_attr_1, pos_1, batch_1: Batched ligand 1 data
            - label_0, label_1: [B] stacked labels
            - ring_label_0, ring_label_1: [B] stacked ring labels (1.0 if has 9+ ring)
            - sdf_0, sdf_1: Lists of SDF paths
    """
    # Separate ligand 0 and ligand 1
    node_attr_0_list = []
    pos_0_list = []
    node_attr_1_list = []
    pos_1_list = []
    label_0_list = []
    label_1_list = []
    ring_label_0_list = []
    ring_label_1_list = []
    sdf_0_list = []
    sdf_1_list = []

    for sample in batch:
        node_attr_0_list.append(sample['node_attr_0'])
        pos_0_list.append(sample['pos_0'])
        node_attr_1_list.append(sample['node_attr_1'])
        pos_1_list.append(sample['pos_1'])
        label_0_list.append(sample['label_0'])
        label_1_list.append(sample['label_1'])
        ring_label_0_list.append(sample['ring_label_0'])
        ring_label_1_list.append(sample['ring_label_1'])
        sdf_0_list.append(sample['sdf_0'])
        sdf_1_list.append(sample['sdf_1'])

    # Create batch indices
    batch_0 = []
    for i, node_attr in enumerate(node_attr_0_list):
        batch_0.append(torch.full((len(node_attr),), i, dtype=torch.long))
    batch_0 = torch.cat(batch_0)

    batch_1 = []
    for i, node_attr in enumerate(node_attr_1_list):
        batch_1.append(torch.full((len(node_attr),), i, dtype=torch.long))
    batch_1 = torch.cat(batch_1)

    # Concatenate
    node_attr_0 = torch.cat(node_attr_0_list, dim=0)
    pos_0 = torch.cat(pos_0_list, dim=0)
    node_attr_1 = torch.cat(node_attr_1_list, dim=0)
    pos_1 = torch.cat(pos_1_list, dim=0)

    # Stack labels
    label_0 = torch.stack(label_0_list)
    label_1 = torch.stack(label_1_list)
    ring_label_0 = torch.stack(ring_label_0_list)
    ring_label_1 = torch.stack(ring_label_1_list)

    return {
        'node_attr_0': node_attr_0,
        'pos_0': pos_0,
        'batch_0': batch_0,
        'node_attr_1': node_attr_1,
        'pos_1': pos_1,
        'batch_1': batch_1,
        'label_0': label_0,
        'label_1': label_1,
        'ring_label_0': ring_label_0,
        'ring_label_1': ring_label_1,
        'sdf_0': sdf_0_list,
        'sdf_1': sdf_1_list,
    }


class BalancedLigandPairDataset(Dataset):
    """
    Dataset for balanced ligand pair preference learning.

    Loads pairs from the balanced dataset (balanced_dataset.sqlite) which contains:
    - 6 pair types with equal proportions
    - Distribution-matched ligands from NS, Non-NS, and ENA-50k

    Args:
        db_path: Path to balanced_dataset.sqlite
        data_dir: Base directory for resolving SDF paths (optional)
        split: 'train', 'val', or 'test'
        pair_types: List of pair types to include (default: all)
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        random_seed: Random seed for reproducibility
    """

    # All available pair types
    ALL_PAIR_TYPES = [
        'ns_non_ns', 'ns_ena_50k', 'non_ns_ena_50k',
        'ns_ns', 'non_ns_non_ns', 'ena_50k_ena_50k'
    ]

    def __init__(self, db_path, data_dir=None, split='train', pair_types=None,
                 train_ratio=0.8, val_ratio=0.1, random_seed=42):
        super().__init__()

        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for this dataset. Please install rdkit.")

        self.db_path = db_path
        self.data_dir = data_dir
        self.split = split
        self.pair_types = pair_types or self.ALL_PAIR_TYPES
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        # Load pair annotations
        self.pairs = self._load_pairs()

        print(f"Loaded {len(self.pairs)} ligand pairs for {split} split")

    def _load_pairs(self):
        """
        Load ligand pair annotations from balanced database.

        Returns:
            pairs: List of (ligand_0_path, ligand_1_path, label_0, label_1) tuples
        """
        pairs = []

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build pair type filter
        pair_type_filter = ','.join([f"'{pt}'" for pt in self.pair_types])

        # Query: Get all labeled pairs with ligand paths
        query = f"""
        SELECT
            la.name AS ligand_a_name,
            lb.name AS ligand_b_name,
            la.sdf_path AS ligand_a_sdf,
            lb.sdf_path AS ligand_b_sdf,
            lab.winner,
            p.pair_type
        FROM label lab
        JOIN pair p ON lab.pair_id = p.id
        JOIN ligand la ON p.ligand_a_id = la.id
        JOIN ligand lb ON p.ligand_b_id = lb.id
        WHERE p.pair_type IN ({pair_type_filter})
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        print(f"Loaded {len(rows)} labeled pairs from database")

        # Count by pair type
        pair_type_counts = {}

        # Process each row
        for row in rows:
            ligand_a_name, ligand_b_name, sdf_a_path, sdf_b_path, winner, pair_type = row

            # Resolve SDF paths
            sdf_a_path = self._resolve_sdf_path(sdf_a_path, ligand_a_name)
            sdf_b_path = self._resolve_sdf_path(sdf_b_path, ligand_b_name)

            # Skip if files don't exist
            if not os.path.exists(sdf_a_path) or not os.path.exists(sdf_b_path):
                continue

            # Convert winner to labels
            if winner == 'A':
                label_0, label_1 = 1.0, 0.0
            elif winner == 'B':
                label_0, label_1 = 0.0, 1.0
            else:
                label_0, label_1 = 0.5, 0.5

            pairs.append((sdf_a_path, sdf_b_path, label_0, label_1))

            # Count
            pair_type_counts[pair_type] = pair_type_counts.get(pair_type, 0) + 1

        # Print pair type distribution
        print("Pair type distribution:")
        for pt, count in sorted(pair_type_counts.items()):
            print(f"  {pt}: {count}")

        # Split data
        pairs = self._split_data(pairs)

        return pairs

    def _resolve_sdf_path(self, sdf_path, ligand_name):
        """
        Resolve SDF path to actual file location.

        The balanced database stores paths in Docker format (/workspace/...).
        This method handles path resolution for different environments.
        """
        if sdf_path and os.path.exists(sdf_path):
            return sdf_path

        # Try with data_dir prefix
        if self.data_dir:
            # Try direct join
            candidate = os.path.join(self.data_dir, os.path.basename(sdf_path or ''))
            if os.path.exists(candidate):
                return candidate

            # Try with ligand name
            candidate = os.path.join(self.data_dir, f"{ligand_name}.sdf")
            if os.path.exists(candidate):
                return candidate

            # Search in subdirectories
            for subdir in os.listdir(self.data_dir):
                subdir_path = os.path.join(self.data_dir, subdir)
                if os.path.isdir(subdir_path):
                    candidate = os.path.join(subdir_path, f"{ligand_name}.sdf")
                    if os.path.exists(candidate):
                        return candidate

        # Return original path (will fail existence check if invalid)
        return sdf_path or f"{ligand_name}.sdf"

    def _split_data(self, pairs):
        """Split data into train/val/test by shuffling pairs."""
        rng = np.random.RandomState(self.random_seed)

        indices = np.arange(len(pairs))
        rng.shuffle(indices)

        n_total = len(pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            selected_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        return [pairs[i] for i in selected_indices]

    def _parse_sdf(self, sdf_path):
        """Parse an SDF file and extract molecular features."""
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        rdmol = next(iter(suppl))

        if rdmol is None:
            raise ValueError(f"Failed to parse SDF file: {sdf_path}")

        rdmol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(rdmol)

        conf = rdmol.GetConformer()
        element = []
        pos = []

        for atom in rdmol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 35:  # Replace Br with Cl
                atomic_num = 17
            element.append(atomic_num)

            atom_pos = conf.GetAtomPosition(atom.GetIdx())
            pos.append([atom_pos.x, atom_pos.y, atom_pos.z])

        element = np.array(element, dtype=np.int64)
        pos = np.array(pos, dtype=np.float32)

        # Detect 9+ ring
        ring_info = rdmol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        has_9plus_ring = 1.0 if any(size >= 9 for size in ring_sizes) else 0.0

        return element, pos, has_9plus_ring

    def _featurize(self, element, pos):
        """Convert atomic numbers to one-hot encoded features."""
        num_atoms = len(element)
        node_attr = np.zeros([num_atoms, 10], dtype=np.float32)

        for i, atomic_num in enumerate(element):
            if atomic_num in ATOMIC_NUMBER_TO_INDEX:
                node_attr[i, ATOMIC_NUMBER_TO_INDEX[atomic_num]] = 1.0
            else:
                node_attr[i, 9] = 1.0

        return torch.from_numpy(node_attr), torch.from_numpy(pos)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get a ligand pair sample."""
        sdf_0, sdf_1, label_0, label_1 = self.pairs[idx]

        element_0, pos_0, ring_label_0 = self._parse_sdf(sdf_0)
        element_1, pos_1, ring_label_1 = self._parse_sdf(sdf_1)

        node_attr_0, pos_0 = self._featurize(element_0, pos_0)
        node_attr_1, pos_1 = self._featurize(element_1, pos_1)

        return {
            'node_attr_0': node_attr_0,
            'pos_0': pos_0,
            'node_attr_1': node_attr_1,
            'pos_1': pos_1,
            'label_0': torch.tensor(label_0, dtype=torch.float32),
            'label_1': torch.tensor(label_1, dtype=torch.float32),
            'ring_label_0': torch.tensor(ring_label_0, dtype=torch.float32),
            'ring_label_1': torch.tensor(ring_label_1, dtype=torch.float32),
            'sdf_0': sdf_0,
            'sdf_1': sdf_1,
        }
