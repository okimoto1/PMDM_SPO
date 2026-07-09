import os
import pickle

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..data import ProteinLigandData, torchify_dict
from ..protein_ligand import PDBProtein, parse_sdf_file, parse_water_pdb


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, dataset="crossdock", transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip("/")
        if dataset == "pdbind":
            self.file_path = "./data/pdbind/"
            self.file_path = self.raw_path
            self.index_path = os.path.join(self.file_path, "index.pkl")
            self.processed_path = os.path.join(self.file_path, "pdbind_processed.lmdb")
            # print(self.processed_path)
            self.name2id_path = os.path.join(self.file_path, "pdbind_name2id.pt")
        elif dataset == "crossdock_pdbind":
            self.file_path = "./data/crossdock_pdbind/"
            if os.environ.get("WATER") == "ligand":
                self.index_path = os.path.join(self.file_path, "index_ligand_water.pkl")
                self.processed_path = os.path.join(
                    self.file_path, "crossdock_pdbind_processed_ligand_water.lmdb"
                )
                self.name2id_path = os.path.join(
                    self.file_path, "crossdock_pdbind_name2id_ligand_water.pt"
                )
            elif os.environ.get("WATER") == "protein":
                self.index_path = os.path.join(
                    self.file_path, "index_protein_water.pkl"
                )
                self.processed_path = os.path.join(
                    self.file_path, "crossdock_pdbind_processed_protein_water.lmdb"
                )
                self.name2id_path = os.path.join(
                    self.file_path, "crossdock_pdbind_name2id_protein_water.pt"
                )
            else:
                self.index_path = os.path.join(self.file_path, "index.pkl")
                self.processed_path = os.path.join(
                    self.file_path, "crossdock_pdbind_processed_ligand_water.lmdb"
                )
                self.name2id_path = os.path.join(
                    self.file_path, "crossdock_pdbind_name2id_ligand_water.pt"
                )

        else:
            self.file_path = self.raw_path
            self.index_path = os.path.join(
                self.raw_path, "index.pkl"
            )  # crossdock 'crossdock_cutoff/'+
            self.processed_path = os.path.join(
                (self.raw_path+"/")
                + os.path.basename(self.raw_path)
                + "_processed.lmdb",
            )
            self.name2id_path = os.path.join(
                (self.raw_path+"/")
                + os.path.basename(self.raw_path)
                + "_name2id.pt",
            )
            print("file_path:", self.file_path)
            print("index_path:", self.index_path)
            print("processed_path:", self.processed_path)
            print("name2id_path:", self.name2id_path)

        # self.name2id_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

        # print(self.processed_path)

        # print(os.path.exists(self.processed_path))
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()
        self.name2id = torch.load(self.name2id_path)

        self.id2name = {v: k for k, v in self.name2id.items()}

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, "A connection has already been opened."
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, "rb") as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
                if pocket_fn is None:
                    continue
                try:
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_pos = ligand_dict["pos"]
                    #   pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    pocket_dict = PDBProtein(
                        os.path.join(self.raw_path, pocket_fn)
                    ).to_dict_atom_cutoff(ligand_pos, 8.0)

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn

                    # Add water element
                    if os.environ.get("WATER") == "ligand":
                        assert (
                            "pocket10" in pocket_fn
                        )  # 入力タンパク質に結晶水は含まれない
                        pocket_fn = pocket_fn.replace(
                            "pocket10", "pocketW3"
                        )  # フィルタリングされた結晶水
                        w3_path = os.path.join(self.raw_path, pocket_fn)

                        if os.path.exists(w3_path):
                            water_pos = parse_water_pdb(w3_path)
                            if len(water_pos) > 0:
                                data.ligand_element = torch.cat(
                                    [
                                        data.ligand_element,
                                        torch.tensor([1000] * len(water_pos)),
                                    ],
                                    dim=0,
                                ).to(data.ligand_element.device)
                                data.ligand_pos = torch.cat(
                                    [data.ligand_pos, torch.tensor(water_pos)], dim=0
                                ).to(data.ligand_pos.device)
                                water_feature = torch.zeros(
                                    (len(water_pos), data.ligand_atom_feature.shape[1])
                                ).to(data.ligand_atom_feature.device)
                                water_feature[:, -1] = 1
                                data.ligand_atom_feature = torch.cat(
                                    [
                                        data.ligand_atom_feature,
                                        water_feature,
                                    ],
                                    dim=0,
                                ).to(data.ligand_atom_feature.device)

                    txn.put(key=str(i).encode(), value=pickle.dumps(data))
                except Exception as e:
                    num_skipped += 1
                    import traceback

                    print("".join(traceback.format_tb(e.__traceback__)))
                    print(f"Error: {e}")
                    print(
                        "Skipping (%d) %s"
                        % (
                            num_skipped,
                            ligand_fn,
                        )
                    )
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), "Indexing"):
            # if i<63340:
            #     continue
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        # print(idx)
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        # if not data.protein_pos.size(0)>0:
        #     print(idx)
        #     print(key)
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)

        data.name = self.id2name[idx]

        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    args.path = "./data/pdbind/"
    dataset = PocketLigandPairDataset(args.path)
