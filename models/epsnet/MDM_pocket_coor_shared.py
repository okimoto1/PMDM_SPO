import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

# QED and SA calculation imports
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from rdkit.Chem import Descriptors, Crippen, Lipinski
from evaluation.sascorer import compute_sa_score
from utils.reconstruct_mdm import make_mol_openbabel


def compute_lipinski_score(mol):
    """
    计算Lipinski五规则分数，归一化到[0,1]
    参考 evaluation/score_func.py 中的 obey_lipinski 函数
    返回: 满足的规则数 / 5
    """
    try:
        from copy import deepcopy
        mol = deepcopy(mol)
        Chem.SanitizeMol(mol)
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        logp = Crippen.MolLogP(mol)
        rule_4 = (logp >= -2) and (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
        satisfied = sum([int(r) for r in [rule_1, rule_2, rule_3, rule_4, rule_5]])
        return satisfied / 5.0  # 归一化到 [0, 1]
    except Exception:
        return 0.0  # 计算失败返回0

from utils.chem import BOND_TYPES
from .diffusion import get_num_embedding
from ..common import MultiLayerPerceptron, extend_graph_order_radius, get_edges
from ..encoders import (
    EGNN_Sparse_Network,
    SchNetEncoder,
    SchNetEncoder_protein,
    get_edge_encoder,
)
from ..encoders.attention import BasicTransformerBlock
from ..geometry import eq_transform, get_distance


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class MDM_full_pocket_coor_shared(nn.Module):

    def __init__(self, config):
        super(MDM_full_pocket_coor_shared, self).__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        # self.hidden_dim = config.hidden_dim
        self.atom_type_input_dim = (
            config.num_atom if "num_atom" in config else 8
        )  # contains simple tmb or charge or not qm9:5+1(charge) geom: 16+1(charge)
        self.atom_out_dim = (
            config.num_atom if "num_atom" in config else 8
        )  # contains charge or not
        self.time_emb = config.time_emb if "time_emb" in config else True
        self.atom_num_emb = config.atom_num_emb if "atom_num_emb" in config else False
        self.vae_context = config.vae_context if "vae_context" in config else False
        self.context = config.context if "context" in config else []
        self.protein_input_dim = (
            config.protein_feature_dim if "protein_feature_dim" in config else 27
        )
        self.hidden_dim = config.hidden_dim

        self.ligand_emblin = nn.Linear(self.atom_type_input_dim, self.hidden_dim)
        self.protein_emblin = nn.Linear(self.atom_type_input_dim, self.hidden_dim)
        self.atten_layer = BasicTransformerBlock(
            self.hidden_dim, 4, self.hidden_dim // 4, 0.1, self.hidden_dim
        )
        """
        timestep embedding
        """
        if self.time_emb:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList(
                [
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                    torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
                ]
            )
            # self.temb_proj = torch.nn.Linear(self.hidden_dim*4,
            #                                 self.hidden_dim//4)
            self.temb_proj = torch.nn.Linear(
                self.hidden_dim * 4, self.hidden_dim
            )  # -config.protein_hidden_dim

        """
        atom numbers embedding
        """
        if self.atom_num_emb:
            self.nemb = nn.Module()
            self.nemb.dense = nn.ModuleList(
                [
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                    torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
                ]
            )
            # self.temb_proj = torch.nn.Linear(self.hidden_dim*4,
            #                                 self.hidden_dim//4)
            self.nemb_proj = torch.nn.Linear(
                self.hidden_dim * 4, self.hidden_dim
            )  # -config.protein_hidden_dim

        """
        The graph neural network that extracts node-wise features.
        """
        if self.vae_context:
            self.context_encoder = SchNetEncoder(
                hidden_channels=self.hidden_dim,
                num_filters=self.hidden_dim,
                num_interactions=config.num_convs,
                edge_channels=self.edge_encoder_global.out_channels,
                cutoff=10,  # config.cutoff
                smooth=config.smooth_conv,
                input_dim=self.atom_type_input_dim,
                time_emb=False,
                context=True,
            )
            self.atom_type_input_dim = self.atom_type_input_dim * 2
        if self.context is not None and type(self.context) is not str:
            ctx_nf = len(self.context)
            self.atom_type_input_dim = self.atom_type_input_dim + ctx_nf

        # Protein encoder
        self.protein_encoder = SchNetEncoder_protein(
            hidden_channels=config.protein_hidden_dim,  # 128
            num_filters=config.protein_hidden_dim,  # 128
            num_interactions=config.protein_num_convs,  # 2
            edge_channels=self.edge_encoder_global.out_channels,  # 128
            cutoff=config.encoder_cutoff,  # 6
            input_dim=self.protein_input_dim,  # 31
        )

        # Ligand encoder
        self.ligand_encoder = SchNetEncoder_protein(
            hidden_channels=config.protein_hidden_dim,  # 128
            num_filters=config.protein_hidden_dim,  # 128
            num_interactions=config.protein_num_convs,  # 2
            edge_channels=self.edge_encoder_global.out_channels,  # 128
            cutoff=config.encoder_cutoff,  # 6
            input_dim=self.atom_type_input_dim,  # 10
        )


        # EGNN
        self.encoder_global = EGNN_Sparse_Network(
            n_layers=config.num_convs,
            feats_input_dim=self.atom_type_input_dim,
            feats_dim=config.hidden_dim,
            edge_attr_dim=config.hidden_dim,
            m_dim=config.hidden_dim,
            soft_edge=config.soft_edge,
            norm_coors=config.norm_coors,
        )

        # EGNN
        self.encoder_local = EGNN_Sparse_Network(
            n_layers=config.num_convs_local,
            feats_input_dim=self.atom_type_input_dim,
            feats_dim=config.hidden_dim,
            edge_attr_dim=config.hidden_dim,
            m_dim=config.hidden_dim,
            soft_edge=config.soft_edge,
            norm_coors=config.norm_coors,
        )

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1) and node type.
        """

        # if edge attr, then 2*, else 1*
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            1 * 3,
            [self.hidden_dim // 2, self.hidden_dim // 4, 1],
            activation=config.mlp_act,
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            1 * 3,
            [self.hidden_dim // 2, self.hidden_dim // 4, 1],
            activation=config.mlp_act,
        )

        self.grad_global_node_mlp = MultiLayerPerceptron(
            1 * self.hidden_dim,
            [self.hidden_dim, self.hidden_dim // 2, self.atom_out_dim],
            activation=config.mlp_act,
        )

        self.grad_local_node_mlp = MultiLayerPerceptron(
            1 * self.hidden_dim,
            [self.hidden_dim, self.hidden_dim // 2, self.atom_out_dim],
            activation=config.mlp_act,
        )
        """
        Incorporate parameters together
        """
        self.model_global = nn.ModuleList(
            [
                self.edge_encoder_global,
                self.encoder_global,
                self.grad_global_node_mlp,
                self.grad_global_dist_mlp,
            ]
        )
        self.model_local = nn.ModuleList(
            [
                self.edge_encoder_local,
                self.encoder_local,
                self.grad_local_node_mlp,
                self.grad_local_dist_mlp,
            ]
        )
        # self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp])
        # self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp])

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        ## variances
        alphas = (1.0 - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def net(
        self,
        ligand_atom_type,
        ligand_pos,
        ligand_bond_index,
        ligand_bond_type,
        ligand_batch,
        protein_embeddings,
        protein_atom_feature,
        protein_pos,
        protein_backbone_mask,
        protein_batch,
        time_step,
        num_node_ctx=None,
        edge_index=None,
        edge_type=None,
        edge_length=None,
        return_edges=False,
        extend_order=True,
        extend_radius=True,
        is_sidechain=None,
        property_context=None,
        vae_noise=None,
        linker_mask=None,
    ):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            pos: atom coordinates
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        # print(num_node_ctx)
        N = ligand_atom_type.size(0)
        if not self.time_emb:
            time_step = time_step / self.num_timesteps
            time_emb = time_step.index_select(0, ligand_batch).unsqueeze(1)
            ligand_atom_type = torch.cat([ligand_atom_type, time_emb], dim=1)

        """
        VAE noise
        """
        if self.vae_context:
            if self.training:
                ligand_edge_length = get_distance(
                    ligand_pos, ligand_bond_index
                ).unsqueeze(-1)
                m, log_var = self.context_encoder(
                    z=ligand_atom_type,
                    edge_index=ligand_bond_index,
                    edge_length=ligand_edge_length,
                    edge_attr=None,
                    embed_node=False,  # default is True
                )
                std = torch.exp(log_var * 0.5)
                z = torch.randn_like(log_var)
                ctx = m + std * z
                ligand_atom_type = torch.cat([ligand_atom_type, ctx], dim=1)
                kl_loss = 0.5 * torch.sum(torch.exp(log_var) + m**2 - 1.0 - log_var)

            else:
                ctx = torch.randn_like(ligand_atom_type)  # N(0,1)
                # ctx = torch.clamp(torch.randn_like(atom_type), min=-3, max=3) # clip N(0,1)
                # ctx = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device) # N(0,3)
                # ctx = torch.zeros_like(atom_type).uniform_(-1,+1) # U(-1,+1)
                # ctx = vae_noise
                ligand_atom_type = torch.cat([ligand_atom_type, ctx], dim=1)
                kl_loss = 0

        if (
            len(self.context) > 0
            and self.context is not None
            and type(self.context) is not str
        ):
            print("Context:", self.context)
            print(type(self.context))
            ligand_atom_type = torch.cat([ligand_atom_type, context], dim=1)

        # ligand_atom_type = torch.cat([ligand_atom_type,protein_ctx],dim=1)
        protein_ctx = scatter_mean(protein_embeddings, protein_batch, dim=0)
        protein_ctx = protein_ctx.index_select(0, ligand_batch)
        context = protein_ctx

        """
        Time embedding
        """
        if self.time_emb:
            nonlinearity = nn.ReLU()
            temb = get_num_embedding(time_step, self.config.hidden_dim)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
            temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
            # time_ctx = temb.index_select(0, pocket_batch)
            time_ctx = temb.index_select(0, ligand_batch)
            # context = time_ctx
            context = time_ctx + context

        """
        Atom numbers embedding
        """
        if self.atom_num_emb:
            context = context + num_node_ctx

        ligand_atom_feature = (
            self.ligand_encoder(  # cutoff = 6.0
                node_attr=ligand_atom_type,
                pos=ligand_pos,
                batch=ligand_batch,
            )
            + context
        )
        # ligand_atom_feature = self.ligand_emblin(ligand_atom_type)+context
        # protein_atom_feature = self.protein_emblin(protein_atom_feature)
        ligand_atom_feature, protein_embeddings = self.atten_layer(
            ligand_atom_feature, protein_embeddings
        )
        # ligand_atom_feature = self.atten_layer(ligand_atom_feature, protein_embeddings)
        # ligand_atom_feature  = self.atten_layer(ligand_atom_feature, protein_atom_feature)

        pocket_atom = torch.cat([ligand_atom_feature, protein_embeddings], dim=0)
        pocket_pos = torch.cat([ligand_pos, protein_pos], dim=0)
        pocket_batch = torch.cat([ligand_batch, protein_batch])
        pocket_mask = (
            torch.cat(
                [
                    linker_mask,
                    torch.zeros(protein_pos.size(0), dtype=torch.bool).to(
                        linker_mask.device
                    ),
                ]
            )
            if linker_mask is not None
            else None
        )

        if edge_index is None or edge_type is None or edge_length is None:
            full_bond_type = torch.ones(ligand_bond_index.size(1), dtype=torch.long).to(
                ligand_bond_index.device
            )

            # Construct local and global edges
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=ligand_pos,
                edge_index=ligand_bond_index,
                edge_type=full_bond_type,
                batch=ligand_batch,
                order=self.config.edge_order,  # 3
                cutoff=self.config.cutoff,  # 3.0
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(ligand_pos, edge_index).unsqueeze(-1)  # (E, 1)
            ligand_bond_index = None  # comment if fix the edge

        local_pocket_edge = get_edges(
            pocket_pos,
            pocket_batch,
            ligand_batch,
            self.config.cutoff,
            self.config.cutoff,
            ligand_bond_index,
        )  # ligand_bond_index
        global_pocket_edge = get_edges(
            pocket_pos,
            pocket_batch,
            ligand_batch,
            self.config.g_cutoff,
            self.config.g_cutoff,
        )  # self.config.g_cutoff
        local_pocket_edge_length = get_distance(
            pocket_pos, local_pocket_edge
        ).unsqueeze(-1)
        global_pocket_edge_length = get_distance(
            pocket_pos, global_pocket_edge
        ).unsqueeze(-1)

        if ligand_bond_type is not None:
            local_edge_mask = is_local_edge(ligand_bond_type)
        else:
            local_edge_mask = is_radius_edge(edge_type)

        edge_attr_global = self.edge_encoder_global(
            edge_length=global_pocket_edge_length, edge_type=None
        )  # Embed edges

        # EGNN
        node_attr_global, pos_attr_global = self.encoder_global(
            z=pocket_atom,
            pos=pocket_pos,
            edge_index=global_pocket_edge,
            edge_attr=edge_attr_global,
            batch=pocket_batch,
            ligand_batch=ligand_batch,
            context=context,
            linker_mask=pocket_mask,
        )

        # Encoding local
        # edge_attr_local = self.edge_encoder_local(
        #     edge_length=edge_length,
        #     edge_type=edge_type
        # )   # Embed edges
        edge_attr_local = self.edge_encoder_local(
            edge_length=local_pocket_edge_length, edge_type=None
        )  # Embed edges
        # if self.time_emb:
        #     # edge_attr_local += temb_edge
        #     edge_attr_local += l_ptemb_edge

        # # GIN
        # node_attr_local = self.encoder_local(
        #     z=ligand_atom_type,
        #     edge_index=edge_index[:, local_edge_mask],
        #     edge_attr=edge_attr_local[local_edge_mask],
        # )

        # EGNN
        node_attr_local, pos_attr_local = self.encoder_local(
            z=pocket_atom,
            pos=pocket_pos,
            edge_index=local_pocket_edge,
            edge_attr=edge_attr_local,
            batch=pocket_batch,
            ligand_batch=ligand_batch,
            context=context,
            linker_mask=pocket_mask,
        )

        node_score_global = self.grad_global_node_mlp(node_attr_global)
        node_score_local = self.grad_local_node_mlp(node_attr_local)

        if self.vae_context:
            return (
                pos_attr_global,
                pos_attr_local,
                node_score_global,
                node_score_local,
                edge_index,
                edge_type,
                edge_length,
                local_edge_mask,
                kl_loss,
            )
        else:
            return (
                pos_attr_global,
                pos_attr_local,
                node_score_global,
                node_score_local,
                edge_index,
                edge_type,
                edge_length,
                local_edge_mask,
            )

    def forward(
        self,
        batch,
        context=None,
        return_unreduced_loss=False,
        return_unreduced_edge_loss=False,
        extend_order=True,
        extend_radius=True,
        is_sidechain=None,
    ):

        ligand_atom_type = (
            batch.ligand_atom_feature.float()
        )  # full feature or not # e.g 405 * 10
        # print(ligand_atom_type)
        ligand_pos = batch.ligand_pos  # e.g 405 * 3
        ligand_bond_index = batch.ligand_bond_index  # e.g 2* 10908
        ligand_bond_type = batch.ligand_bond_type  # e.g shape = 10908
        ligand_batch = batch.ligand_element_batch  # e.g shape = 405
        ligand_num_atom = batch.num_nodes_per_graph  # e.g shape = 405
        protein_atom_feature = (
            batch.protein_atom_feature.float()
        )  # full feature or not # e.g shape = 3535 * 10
        protein_atom_feature_full = (
            batch.protein_atom_feature_full.float()
        )  # full feature or not # e.g shape = 3535 * 10
        protein_pos = batch.protein_pos  # e.g shape = 3535 * 3
        protein_batch = batch.protein_element_batch  # e.g shape = 3535
        protein_backbone_mask = batch.protein_is_backbone  # e.g shape = 3535

        N = ligand_atom_type.size(0)  # e.g 405
        node2graph = ligand_batch  # e.g shape = 405
        num_graphs = node2graph[-1] + 1  # e.g num_graphs = 16

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=ligand_pos.device
        )

        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[
            :num_graphs
        ]  # e.g shape = 16

        a = self.alphas.index_select(0, time_step)  # (G, ) e.g shape = 16
        a_pos = a.index_select(0, node2graph).unsqueeze(
            -1
        )  # (N, 1) e.g shape = 405 * 1

        # Independently noise
        pos_noise = torch.randn(
            size=ligand_pos.size(), device=ligand_pos.device
        )  # e.g shape = 405 * 3
        atom_noise = torch.randn(
            size=ligand_atom_type.size(), device=ligand_atom_type.device
        )  # e.g shape = 405 * 10


        # Move the ligand to COM, and move the protein to the ligand-COM
        ligand_pos, protein_pos = center_pos_pl(
            ligand_pos, protein_pos, ligand_batch, protein_batch
        )
        ## Perterb pos
        ligand_pos_perturbed = (
            ligand_pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        )
        # Move to the COM again
        ligand_pos_perturbed, protein_pos = center_pos_pl(
            ligand_pos_perturbed, protein_pos, ligand_batch, protein_batch
        )
        ## Perterb atom
        ligand_atom_perturbed = (
            a_pos.sqrt() * ligand_atom_type + (1.0 - a_pos).sqrt() * atom_noise
        )

        """
        Protein embedding
        """
        # protein_pos = center_pos(protein_pos,protein_batch)
        protein_ctx = self.protein_encoder(
            node_attr=protein_atom_feature_full,  # e.g shape = 3535 * 10
            pos=protein_pos,  # e.g shape = 3535 * 3
            batch=protein_batch,  # e.g shape = 3535
        )  # e.g shape = 3535 * 10
        # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)

        """
        Atom numbers embedding
        """
        num_node_ctx = None
        if self.atom_num_emb:
            nonlinearity = nn.ReLU()
            nemb = get_num_embedding(ligand_num_atom, self.config.hidden_dim)
            nemb = self.nemb.dense[0](nemb)
            nemb = nonlinearity(nemb)
            nemb = self.nemb.dense[1](nemb)
            nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
            num_node_ctx = nemb.index_select(0, ligand_batch)

        net_out = self.net(
            ligand_atom_type=ligand_atom_perturbed,
            ligand_pos=ligand_pos_perturbed,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            ligand_batch=ligand_batch,
            protein_embeddings=protein_ctx,
            protein_atom_feature=protein_atom_feature,
            protein_pos=protein_pos,
            protein_backbone_mask=protein_backbone_mask,
            protein_batch=protein_batch,
            time_step=time_step,
            num_node_ctx=num_node_ctx,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
            is_sidechain=is_sidechain,
            property_context=context,
            vae_noise=None,
        )  # (E_global, 1), (E_local, 1)

        if self.vae_context:
            (
                pos_eq_global,
                pos_eq_local,
                node_score_global,
                node_score_local,
                edge_index,
                edge_type,
                edge_length,
                local_edge_mask,
            ) = net_out[:-1]
        else:
            (
                pos_eq_global,
                pos_eq_local,
                node_score_global,
                node_score_local,
                edge_index,
                edge_type,
                edge_length,
                local_edge_mask,
            ) = net_out
        edge2graph = node2graph.index_select(0, edge_index[0])
        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(ligand_pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length

        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        if self.config.edge_encoder == "gaussian":
            # Distances must be greater than 0
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float("inf"))

        d_target = (
            (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        )  # (E_global, 1), denoising direction
        # d_target = (d_perturbed - d_gt* a_edge.sqrt()) / (1.0 - a_edge).sqrt()   # (E_global, 1), denoising direction
        # d_target = -1*(d_perturbed - d_gt* a_edge.sqrt()) / (1.0 - a_edge)

        global_mask = torch.logical_and(
            torch.logical_or(
                torch.logical_and(
                    d_perturbed > self.config.cutoff,
                    d_perturbed <= self.config.g_cutoff,
                ),
                local_edge_mask.unsqueeze(-1),
            ),
            ~local_edge_mask.unsqueeze(-1),
        )

        # score matching
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        target_pos_global = eq_transform(
            target_d_global, ligand_pos_perturbed, edge_index, edge_length
        )

        # score matching
        target_pos_local = eq_transform(
            d_target[local_edge_mask],
            ligand_pos_perturbed,
            edge_index[:, local_edge_mask],
            edge_length[local_edge_mask],
        )
        loss_pos = F.mse_loss(
            pos_eq_global + pos_eq_local,
            target_pos_global + target_pos_local,
            reduction="none",
        )
        loss_pos = 1 * torch.sum(loss_pos, dim=-1, keepdim=True)

        loss_node = F.mse_loss(
            node_score_global + node_score_local, atom_noise, reduction="none"
        )
        loss_node = 1 * torch.sum(loss_node, dim=-1, keepdim=True)
        # loss for atomic eps regression
        # loss = loss_global + loss_local + loss_node
        if self.vae_context:
            vae_KL_loss = net_out[-1]
            loss = loss_pos + loss_node + vae_KL_loss
        else:
            loss = loss_pos + loss_node
        # loss_pos = scatter_add(loss_pos.squeeze(), node2graph)  # (G, 1)

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            if self.vae_context:
                return loss, loss_pos, loss_pos, loss_node, loss_node, vae_KL_loss
            return loss, loss_pos, loss_pos, loss_node, loss_node
        else:
            return loss

    def _denoise_one_step(
        self,
        ligand_pos, ligand_atom_type, protein_pos,
        ligand_bond_index, ligand_bond_type, ligand_batch,
        protein_ctx, protein_atom_type, protein_backbone_mask, protein_batch,
        t, num_node_ctx, timestep_i, timestep_j,
        sigmas, local_start_sigma, global_start_sigma,
        clip_local, clip, w_local_pos, w_global_pos, w_local_node, w_global_node,
        w_clash, clash_start_sigma,
        extend_order, extend_radius, is_sidechain, context,
        step_lr, sampling_type, kwargs_dict
    ):
        """
        执行一步去噪操作
        返回: (new_ligand_pos, new_ligand_atom_type, new_protein_pos)
        """
        def compute_alpha(beta, t_val):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t_val + 1)
            return a
        
        # 调用net获取预测
        net_out = self.net(
            ligand_atom_type=ligand_atom_type,
            ligand_pos=ligand_pos,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            ligand_batch=ligand_batch,
            protein_embeddings=protein_ctx,
            time_step=t,
            num_node_ctx=num_node_ctx,
            protein_atom_feature=protein_atom_type,
            protein_pos=protein_pos,
            protein_backbone_mask=protein_backbone_mask,
            protein_batch=protein_batch,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
            is_sidechain=is_sidechain,
            property_context=context,
            vae_noise=None,
        )
        
        if self.vae_context:
            (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
             edge_index, edge_type, edge_length, local_edge_mask) = net_out[:-1]
        else:
            (pos_eq_global, pos_eq_local, node_score_global, node_score_local,
             edge_index, edge_type, edge_length, local_edge_mask) = net_out
        
        # Local
        if sigmas[timestep_i] < local_start_sigma:
            node_eq_local = pos_eq_local
            if clip_local is not None:
                node_eq_local = clip_norm(node_eq_local, limit=clip_local)
        else:
            node_eq_local = 0
            node_score_local = 0
        
        # Global
        if sigmas[timestep_i] < global_start_sigma:
            node_eq_global = pos_eq_global
            node_eq_global = clip_norm(node_eq_global, limit=clip)
        else:
            node_eq_global = 0
            node_score_global = 0
        
        # Sum
        eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global
        eps_node = w_local_node * node_score_local + w_global_node * node_score_global
        
        # Clash guidance
        if sigmas[timestep_i] < clash_start_sigma and w_clash > 0:
            clash_grad = compute_clash_guidance_gradient(
                ligand_pos=ligand_pos,
                protein_pos=protein_pos,
                sigma=1.0,
            )
            eps_pos = eps_pos + w_clash * clash_grad
        
        # 更新
        noise = torch.randn_like(ligand_pos)
        noise_node = torch.randn_like(ligand_atom_type)
        b = self.betas
        t_val = t[0]
        next_t = (torch.ones(1) * timestep_j).to(ligand_pos.device)
        at = compute_alpha(b, t_val.long())
        at_next = compute_alpha(b, next_t.long())
        
        if sampling_type == "generalized":
            eta = kwargs_dict.get("eta", 1.0)
            et = -eps_pos
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            step_size_pos_ld = step_lr * (sigmas[timestep_i] / 0.01) ** 2 / sigmas[timestep_i]
            step_size_pos_generalized = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
            step_size_pos = step_size_pos_ld if step_size_pos_ld < step_size_pos_generalized else step_size_pos_generalized
            step_size_noise_ld = torch.sqrt((step_lr * (sigmas[timestep_i] / 0.01) ** 2) * 2)
            step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
            step_size_noise = step_size_noise_ld if step_size_noise_ld < step_size_noise_generalized else step_size_noise_generalized
            w = 1
            eps_node = eps_node / (1 - at).sqrt()
            pos_next = ligand_pos - et * step_size_pos + w * noise * step_size_noise
            atom_next = ligand_atom_type - eps_node * step_size_pos + w * noise_node * step_size_noise
        elif sampling_type == "ddpm_noisy":
            atm1 = at_next
            beta_t = 1 - at / atm1
            e = -eps_pos
            mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
            mask = 1 - (t_val == 0).float()
            logvar = beta_t.log()
            pos_next = mean + mask * torch.exp(0.5 * logvar) * noise
            e = eps_node
            node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (1.0 / at - 1).sqrt() * e
            mean_eps = ((atm1.sqrt() * beta_t) * node0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type) / (1.0 - at)
            mean = mean_eps
            mask = 1 - (t_val == 0).float()
            logvar = beta_t.log()
            atom_next = mean + mask * torch.exp(0.5 * logvar) * noise_node
        elif sampling_type == "ld":
            step_size = step_lr * (sigmas[timestep_i] / 0.01) ** 2
            pos_next = ligand_pos + step_size * eps_pos / sigmas[timestep_i] + noise * torch.sqrt(step_size * 2)
            eps_node = eps_node / (1 - at).sqrt()
            atom_next = ligand_atom_type - step_size * eps_node / sigmas[timestep_i] + noise_node * torch.sqrt(step_size * 2)
        else:
            raise ValueError("Unknown sampling type")
        
        # Center
        pos_next, protein_pos = center_pos_pl(pos_next, protein_pos, ligand_batch, protein_batch)
        
        return pos_next, atom_next, protein_pos
    
    def langevin_dynamics_sample(
        self,
        ligand_atom_type,
        ligand_pos_init,
        ligand_bond_index,
        ligand_bond_type,
        ligand_num_node,
        ligand_batch,
        protein_atom_type,
        protein_atom_feature_full,
        protein_pos,
        protein_backbone_mask,
        protein_batch,
        num_graphs,
        context,
        extend_order,
        extend_radius=True,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        min_sigma=0,
        is_sidechain=None,
        global_start_sigma=float("inf"),
        local_start_sigma=float("inf"),
        w_global_pos=0.2,
        w_global_node=0.2,
        w_local_pos=0.2,
        w_local_node=0.2,
        w_reg=1.0,
        # ========================================================
        # === [ 新增部分 2: 添加碰撞指导的超参数 ]           ===
        # ========================================================
        # w_clash: 指导强度 (lambda in paper)
        # clash_start_sigma: 从哪个噪声水平开始应用指导
        w_clash=0.0,
        clash_start_sigma=float("inf"),
        # ========================================================
        # SPM评分参数
        spm_model=None,
        spm_eval_interval=100,
        # SPO重采样参数
        use_spo=False,
        spo_resample_interval=5,
        spo_num_candidates=5,
        spo_start_step=0,  # 从第几步开始应用SPO
        # QED and SA calculation parameters (for SPO scoring)
        dataset_info=None,
        atomic_numbers=None,
        # SPO评分权重参数
        score_weight_spm=1.0,
        score_weight_qed=2.0,
        score_weight_sa=1.0,
        score_weight_clash=1.0,  # clash惩罚权重
        clash_threshold=3.0,  # 小于此距离开始惩罚
        spo_use_spm=True,  # 是否在SPO评分中使用SPM分数
        use_lipinski=False,  # 是否使用Lipinski作为乘性因子
        # ========================================================
        **kwargs
    ):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        # SPM评分记录
        spm_scores_list = []  # 存储 (timestep, score, sample_idx) 
        
        # SPO决策记录
        spo_decisions_list = []  # 存储每次重采样的决策信息

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            print(skip)
            seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            # seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            protein_ori = protein_pos
            protein_com = scatter_mean(protein_pos, protein_batch, dim=0)
            ligand_pos, protein_pos = center_pos_pl(
                ligand_pos_init + protein_com[ligand_batch],
                protein_pos,
                ligand_batch,
                protein_batch,
            )
            # ligand_pos = center_pos(ligand_pos_init, ligand_batch)
            # pos = center_pos(pos_init* sigmas[-1], batch)
            # pos = center_pos(pos_init, batch)* sigmas[-1]

            # VAE noise
            vae_noise = torch.zeros_like(ligand_atom_type).uniform_(-1, +1)
            # vae_noise = torch.randn_like(atom_type)
            # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3)
            # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device)

            """
            Protein embedding
            """
            protein_ctx = self.protein_encoder(
                node_attr=protein_atom_feature_full,
                pos=protein_pos,
                batch=protein_batch,
            )
            # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)
            # protein_ctx = protein_ctx.index_select(0, ligand_batch)

            """
            Atom numbers embedding
            """
            num_node_ctx = None
            if "atom_num_emb" not in self.__dict__.keys():
                self.atom_num_emb = False
            if self.atom_num_emb:
                nonlinearity = nn.ReLU()
                nemb = get_num_embedding(ligand_num_node, self.config.hidden_dim)
                nemb = self.nemb.dense[0](nemb)
                nemb = nonlinearity(nemb)
                nemb = self.nemb.dense[1](nemb)
                nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
                num_node_ctx = nemb.index_select(0, ligand_batch)

            step_counter = 0
            seq_list = list(zip(reversed(seq), reversed(seq_next)))
            total_steps = len(seq_list)
            idx_in_seq = 0
            
            # ========== 日志：记录采样配置 ==========
            print(f"\n{'='*60}")
            print(f"[SAMPLING] Configuration:")
            print(f"  use_spo: {use_spo}")
            print(f"  spo_use_spm: {spo_use_spm}")
            print(f"  spo_start_step: {spo_start_step}")
            print(f"  spo_resample_interval: {spo_resample_interval}")
            print(f"  n_steps: {n_steps}")
            print(f"  self.num_timesteps: {self.num_timesteps}")
            print(f"  skip: {skip}")
            print(f"  seq length: {len(seq)}")
            print(f"  total_steps: {total_steps}")
            if use_spo:
                if spo_use_spm:
                    base_formula = f"{score_weight_spm}*SPM + {score_weight_qed}*QED + {score_weight_sa}*SA + {score_weight_clash}*clash"
                else:
                    base_formula = f"{score_weight_qed}*QED + {score_weight_sa}*SA + {score_weight_clash}*clash"
                if use_lipinski:
                    print(f"  SPO scoring: Lipinski * ({base_formula})")
                else:
                    print(f"  SPO scoring: {base_formula}")
            print(f"{'='*60}\n")
            
            while idx_in_seq < total_steps:
                i, j = seq_list[idx_in_seq]
                
                # 检查是否需要SPO重采样
                # 注意：当spo_use_spm=False时，不需要spm_model也可以触发SPO
                should_resample = (
                    use_spo and
                    (spm_model is not None or not spo_use_spm) and  # 使用SPM时需要模型，不使用SPM时不需要
                    step_counter >= spo_start_step and  # 检查是否到达启动步数
                    step_counter > 0 and
                    step_counter % spo_resample_interval == 0 and
                    idx_in_seq + spo_resample_interval <= total_steps  # 确保还有足够步数
                )
                
                if should_resample:
                    # ========== SPO重采样阶段 ==========
                    print(f"\n[SPO] Triggering resampling:")
                    print(f"  step_counter: {step_counter}")
                    print(f"  idx_in_seq: {idx_in_seq}/{total_steps}")
                    print(f"  timestep: {i}")
                    print(f"  will skip next {spo_resample_interval} steps")
                    
                    # 保存当前状态
                    state_backup = {
                        'ligand_pos': ligand_pos.clone(),
                        'ligand_atom_type': ligand_atom_type.clone(),
                        'protein_pos': protein_pos.clone()
                    }
                    
                    candidates = []
                    
                    # 生成多个候选轨迹
                    for cand_idx in range(spo_num_candidates):
                        # 恢复到起点状态
                        cand_ligand_pos = state_backup['ligand_pos'].clone()
                        cand_ligand_atom_type = state_backup['ligand_atom_type'].clone()
                        cand_protein_pos = state_backup['protein_pos'].clone()
                        
                        # 采样接下来的spo_resample_interval步
                        for sub_step in range(spo_resample_interval):
                            sub_i, sub_j = seq_list[idx_in_seq + sub_step]
                            
                            t_sub = torch.full(
                                size=(num_graphs,),
                                fill_value=sub_i,
                                dtype=torch.long,
                                device=cand_ligand_pos.device,
                            )
                            
                            # 执行一步去噪（这里需要调用net和更新逻辑）
                            cand_ligand_pos, cand_ligand_atom_type, cand_protein_pos = self._denoise_one_step(
                                cand_ligand_pos, cand_ligand_atom_type, cand_protein_pos,
                                ligand_bond_index, ligand_bond_type, ligand_batch,
                                protein_ctx, protein_atom_type, protein_backbone_mask, protein_batch,
                                t_sub, num_node_ctx, sub_i, sub_j,
                                sigmas, local_start_sigma, global_start_sigma,
                                clip_local, clip, w_local_pos, w_global_pos, w_local_node, w_global_node,
                                w_clash, clash_start_sigma,
                                extend_order, extend_radius, is_sidechain, context,
                                step_lr, sampling_type, kwargs
                            )
                        
                        # 用SPM评估这条轨迹的终点（仅当spo_use_spm=True且spm_model存在时）
                        if spo_use_spm and spm_model is not None:
                            with torch.no_grad():
                                final_t = torch.full(
                                    (num_graphs,),
                                    seq_list[idx_in_seq + spo_resample_interval - 1][0],
                                    dtype=torch.long,
                                    device=cand_ligand_pos.device
                                )

                                _, _, quality_1, quality_2 = spm_model(
                                    cand_ligand_atom_type, cand_ligand_pos, ligand_batch,
                                    cand_ligand_atom_type, cand_ligand_pos, ligand_batch,
                                    final_t,
                                    return_scores=True
                                )

                                spm_scores = (quality_1 + quality_2) / 2.0  # [num_graphs, 1]
                        else:
                            # 不使用SPM时，设置为None或零张量
                            spm_scores = torch.zeros(num_graphs, 1, device=cand_ligand_pos.device)

                        # ========== 尝试计算QED、SA和Lipinski ==========
                        qed_sa_scores = []  # 存储每个样本的(qed, sa)，失败则为None
                        qed_sa_success = []  # 存储每个样本是否成功计算
                        lipinski_scores = []  # 存储每个样本的Lipinski分数 (归一化到[0,1])

                        # ========== 计算clash惩罚分数 ==========
                        clash_scores = []  # 存储每个样本的clash惩罚分数

                        for graph_idx in range(num_graphs):
                            # 提取该样本的配体数据
                            mask = (ligand_batch == graph_idx)
                            sample_pos = cand_ligand_pos[mask].detach().cpu()
                            sample_atom_type = cand_ligand_atom_type[mask].detach().cpu()

                            # 提取该样本的蛋白质数据
                            protein_mask = (protein_batch == graph_idx)
                            sample_protein_pos = cand_protein_pos[protein_mask].detach().cpu()

                            # 计算clash惩罚分数
                            # Step 1: 计算每个ligand原子到所有protein原子的距离
                            # sample_pos: [n_ligand, 3], sample_protein_pos: [n_protein, 3]
                            # 计算距离矩阵: [n_ligand, n_protein]
                            dist_matrix = torch.cdist(sample_pos, sample_protein_pos)

                            # Step 2: 每个ligand原子到最近protein原子的距离
                            min_dists_per_ligand, _ = dist_matrix.min(dim=1)  # [n_ligand]

                            # Step 3: 取所有ligand原子中的最小距离
                            min_dist = min_dists_per_ligand.min().item()

                            # Step 4: 计算惩罚 score = -max(threshold - min_dist, 0)
                            clash_penalty = -max(clash_threshold - min_dist, 0.0)
                            clash_scores.append(clash_penalty)

                            try:
                                # 使用build模式构建分子
                                if dataset_info is not None and atomic_numbers is not None:
                                    # 从atom_type提取element索引
                                    num_atom_type = len(atomic_numbers)
                                    element_indices = torch.argmax(sample_atom_type[:, :num_atom_type], dim=1)

                                    # 构建分子 (build模式)
                                    mol, _ = make_mol_openbabel(sample_pos, element_indices, dataset_info)

                                    # 计算QED
                                    qed_score = qed(mol)

                                    # 计算SA (归一化)
                                    _, sa_norm = compute_sa_score(mol)

                                    qed_sa_scores.append((qed_score, sa_norm))
                                    qed_sa_success.append(True)

                                    # 计算Lipinski分数 (归一化到[0,1])
                                    if use_lipinski:
                                        lipinski_score = compute_lipinski_score(mol)
                                        lipinski_scores.append(lipinski_score)
                                    else:
                                        lipinski_scores.append(1.0)  # 不使用时默认为1.0（不影响乘法）
                                else:
                                    # 缺少必要参数，无法计算
                                    qed_sa_scores.append(None)
                                    qed_sa_success.append(False)
                                    lipinski_scores.append(0.0)  # 失败时为0

                            except Exception as e:
                                # 构建失败或计算失败
                                qed_sa_scores.append(None)
                                qed_sa_success.append(False)
                                lipinski_scores.append(0.0)  # 失败时为0

                        candidates.append({
                            'ligand_pos': cand_ligand_pos,
                            'ligand_atom_type': cand_ligand_atom_type,
                            'protein_pos': cand_protein_pos,
                            'spm_scores': spm_scores,  # [num_graphs, 1] - SPM原始分数
                            'qed_sa_scores': qed_sa_scores,  # list of (qed, sa) or None
                            'qed_sa_success': qed_sa_success,  # list of bool
                            'clash_scores': clash_scores,  # list of float (clash惩罚分数)
                            'lipinski_scores': lipinski_scores,  # list of float (归一化到[0,1])
                            'candidate_idx': cand_idx
                        })
                    
                    # ========== 为每个样本独立选择最佳候选 ==========
                    best_cand_indices = []  # 存储每个样本选择的候选索引

                    for graph_idx in range(num_graphs):
                        # ========== Fallback逻辑：检查是否所有候选都成功计算QED/SA ==========
                        all_candidates_success = all(
                            cand['qed_sa_success'][graph_idx] for cand in candidates
                        )

                        if spo_use_spm:
                            # ========== 使用SPM的评分模式 ==========
                            if all_candidates_success:
                                # ✅ 所有候选都成功 → 使用组合分数: [Lipinski *] (w_spm*SPM + w_qed*QED + w_sa*SA + w_clash*clash)
                                cand_scores = []
                                for cand in candidates:
                                    spm_score = cand['spm_scores'][graph_idx].item()
                                    qed_score, sa_score = cand['qed_sa_scores'][graph_idx]
                                    clash_score = cand['clash_scores'][graph_idx]
                                    base_score = (score_weight_spm * spm_score +
                                                  score_weight_qed * qed_score +
                                                  score_weight_sa * sa_score +
                                                  score_weight_clash * clash_score)
                                    # 如果启用Lipinski，作为乘性因子
                                    if use_lipinski:
                                        lipinski_factor = cand['lipinski_scores'][graph_idx]
                                        combined_score = lipinski_factor * base_score
                                    else:
                                        combined_score = base_score
                                    cand_scores.append(combined_score)

                                base_formula = f"{score_weight_spm}*SPM + {score_weight_qed}*QED + {score_weight_sa}*SA + {score_weight_clash}*clash"
                                if use_lipinski:
                                    scoring_method = f"combined (Lipinski * ({base_formula}))"
                                else:
                                    scoring_method = f"combined ({base_formula})"
                            else:
                                # ❌ 至少有一个候选失败 → 回退到SPM分数 + clash惩罚 (不使用Lipinski)
                                cand_scores = []
                                for cand in candidates:
                                    spm_score = score_weight_spm * cand['spm_scores'][graph_idx].item()
                                    clash_score = score_weight_clash * cand['clash_scores'][graph_idx]
                                    cand_scores.append(spm_score + clash_score)
                                scoring_method = f"SPM+clash (fallback, weights: SPM={score_weight_spm}, clash={score_weight_clash})"
                        else:
                            # ========== 不使用SPM的评分模式 ==========
                            if all_candidates_success:
                                # ✅ 所有候选都成功 → 使用组合分数: [Lipinski *] (w_qed*QED + w_sa*SA + w_clash*clash)
                                cand_scores = []
                                for cand in candidates:
                                    qed_score, sa_score = cand['qed_sa_scores'][graph_idx]
                                    clash_score = cand['clash_scores'][graph_idx]
                                    base_score = (score_weight_qed * qed_score +
                                                  score_weight_sa * sa_score +
                                                  score_weight_clash * clash_score)
                                    # 如果启用Lipinski，作为乘性因子
                                    if use_lipinski:
                                        lipinski_factor = cand['lipinski_scores'][graph_idx]
                                        combined_score = lipinski_factor * base_score
                                    else:
                                        combined_score = base_score
                                    cand_scores.append(combined_score)

                                base_formula = f"{score_weight_qed}*QED + {score_weight_sa}*SA + {score_weight_clash}*clash"
                                if use_lipinski:
                                    scoring_method = f"combined_no_spm (Lipinski * ({base_formula}))"
                                else:
                                    scoring_method = f"combined_no_spm ({base_formula})"
                            else:
                                # ❌ 至少有一个候选失败 → 只使用clash惩罚 (不使用Lipinski)
                                cand_scores = []
                                for cand in candidates:
                                    clash_score = score_weight_clash * cand['clash_scores'][graph_idx]
                                    cand_scores.append(clash_score)
                                scoring_method = f"clash_only (fallback, weight: clash={score_weight_clash})"

                        best_cand_idx = np.argmax(cand_scores)
                        best_cand_indices.append(best_cand_idx)

                        # 记录决策信息（包括使用的评分方法）
                        spo_decisions_list.append({
                            'timestep': seq_list[idx_in_seq + spo_resample_interval - 1][0],
                            'step': step_counter + spo_resample_interval,
                            'sample_idx': graph_idx,
                            'num_candidates': spo_num_candidates,
                            'candidate_scores': cand_scores,
                            'selected_candidate': best_cand_idx,
                            'best_score': cand_scores[best_cand_idx],
                            'worst_score': min(cand_scores),
                            'score_range': max(cand_scores) - min(cand_scores),
                            'scoring_method': scoring_method,  # 记录使用的评分方法
                            'qed_sa_available': all_candidates_success,  # 是否有QED/SA数据
                            'clash_scores': [cand['clash_scores'][graph_idx] for cand in candidates],  # 记录各候选的clash分数
                            'lipinski_scores': [cand['lipinski_scores'][graph_idx] for cand in candidates] if use_lipinski else None,  # 记录各候选的Lipinski分数
                            'use_lipinski': use_lipinski,  # 是否使用Lipinski
                        })

                    # ========== 重组batch：每个样本使用自己选择的候选 ==========
                    # 1. 找出每个原子/蛋白质原子属于哪个样本
                    # ligand_batch: [total_ligand_atoms] 每个原子的样本索引
                    # protein_batch: [total_protein_atoms] 每个蛋白质原子的样本索引

                    # 2. 为每个原子确定应该从哪个候选中提取
                    ligand_atom_to_candidate = torch.tensor(
                        [best_cand_indices[graph_idx.item()] for graph_idx in ligand_batch],
                        device=ligand_batch.device
                    )
                    protein_atom_to_candidate = torch.tensor(
                        [best_cand_indices[graph_idx.item()] for graph_idx in protein_batch],
                        device=protein_batch.device
                    )

                    # 3. 组装新的配体状态
                    new_ligand_pos = torch.zeros_like(candidates[0]['ligand_pos'])
                    new_ligand_atom_type = torch.zeros_like(candidates[0]['ligand_atom_type'])
                    new_protein_pos = torch.zeros_like(candidates[0]['protein_pos'])

                    for cand_idx in range(spo_num_candidates):
                        # 找出应该使用该候选的配体原子
                        ligand_mask = (ligand_atom_to_candidate == cand_idx)
                        new_ligand_pos[ligand_mask] = candidates[cand_idx]['ligand_pos'][ligand_mask]
                        new_ligand_atom_type[ligand_mask] = candidates[cand_idx]['ligand_atom_type'][ligand_mask]

                        # 找出应该使用该候选的蛋白质原子
                        protein_mask = (protein_atom_to_candidate == cand_idx)
                        new_protein_pos[protein_mask] = candidates[cand_idx]['protein_pos'][protein_mask]

                    ligand_pos = new_ligand_pos
                    ligand_atom_type = new_ligand_atom_type
                    protein_pos = new_protein_pos

                    # 4. 打印统计信息
                    candidate_counts = torch.bincount(
                        torch.tensor(best_cand_indices, device='cpu'),
                        minlength=spo_num_candidates
                    ).tolist()

                    # 统计使用完整评分的样本数量（4维帕累托或组合分数）
                    num_full_scoring = sum(
                        1 for decision in spo_decisions_list[-num_graphs:]
                        if decision['qed_sa_available']
                    )

                    # 计算最终选择的平均分数
                    avg_final_score = sum(
                        spo_decisions_list[-num_graphs + i]['best_score']
                        for i in range(num_graphs)
                    ) / num_graphs

                    print(f"[SPO] Independent selection at step {step_counter}:")
                    print(f"  - Candidate usage: {candidate_counts} (out of {num_graphs} samples)")
                    if spo_use_spm:
                        print(f"  - Scoring: {num_full_scoring}/{num_graphs} used combined (SPM+QED+SA+clash)")
                    else:
                        print(f"  - Scoring: {num_full_scoring}/{num_graphs} used combined (QED+SA+clash, no SPM)")
                    print(f"  - Average final score: {avg_final_score:.4f}")
                    if spo_use_spm and spm_model is not None:
                        print(f"  - SPM score range: [{min(cand['spm_scores'].min().item() for cand in candidates):.4f}, "
                              f"{max(cand['spm_scores'].max().item() for cand in candidates):.4f}]")
                    
                    # ========== 记录SPM分数（重采样后的状态）==========
                    # 注意：我们刚刚完成了spo_resample_interval步的采样
                    # 应该记录最终状态的SPM分数
                    if spm_model is not None:
                        # 重采样后状态对应的时间步
                        final_timestep = seq_list[idx_in_seq + spo_resample_interval - 1][0]
                        # 记录每个样本的SPM分数
                        for graph_idx in range(num_graphs):
                            best_cand = candidates[best_cand_indices[graph_idx]]
                            score = best_cand['spm_scores'][graph_idx].item()
                            spm_scores_list.append({
                                'timestep': final_timestep,
                                'score': score,
                                'step': step_counter + spo_resample_interval - 1,
                                'sample_idx': graph_idx,
                                'is_spo_point': True  # 标记这是SPO重采样点
                            })
                    
                    # 跳过已经采样过的步骤
                    idx_in_seq += spo_resample_interval
                    step_counter += spo_resample_interval
                    print(f"[SPO] After resampling: idx_in_seq={idx_in_seq}, step_counter={step_counter}")
                    continue
                
                # ========== 正常采样阶段 ==========
                t = torch.full(
                    size=(num_graphs,),
                    fill_value=i,
                    dtype=torch.long,
                    device=ligand_pos.device,
                )
                
                # SPM评分：每隔spm_eval_interval步评估一次
                if spm_model is not None and step_counter % spm_eval_interval == 0:
                    try:
                        # 评估当前采样状态的配体质量
                        # 当前的ligand_pos和ligand_atom_type已经是带有时间步i对应噪声的状态
                        with torch.no_grad():
                            timestep_spm = t  # [num_graphs] 当前时间步
                            
                            # SPM模型需要成对输入，我们复制一份作为第二个配体
                            # 这样可以获得单个配体的质量分数
                            _, _, quality_1, quality_2 = spm_model(
                                ligand_atom_type, ligand_pos, ligand_batch,
                                ligand_atom_type, ligand_pos, ligand_batch,
                                timestep_spm,
                                return_scores=True
                            )
                            
                            # 两个质量分数应该相同（因为是同一个配体）
                            # 取平均以防数值误差
                            quality_scores = (quality_1 + quality_2) / 2.0  # [num_graphs, 1]
                            
                            # 记录每个样本的分数（quality_scores有num_graphs个元素）
                            for graph_idx in range(num_graphs):
                                score = quality_scores[graph_idx].item()
                                spm_scores_list.append({
                                    'timestep': i,
                                    'score': score,
                                    'step': step_counter,
                                    'sample_idx': graph_idx  # 标记是第几个样本
                                })
                    except Exception as e:
                        print(f"Warning: SPM scoring failed at step {step_counter}: {e}")
                
                # Note: step_counter is incremented at the end of the loop (line ~1465)
                # Do NOT increment here to avoid double counting
                # step_counter += 1

                net_out = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type,
                    protein_pos=protein_pos,
                    protein_backbone_mask=protein_backbone_mask,
                    protein_batch=protein_batch,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None,
                )  # (E_global, 1), (E_local, 1)
                if self.vae_context:
                    (
                        pos_eq_global,
                        pos_eq_local,
                        node_score_global,
                        node_score_local,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out[:-1]
                else:
                    (
                        pos_eq_global,
                        pos_eq_local,
                        node_score_global,
                        node_score_local,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out
                # Local float('inf')

                # local_start_sigma = random.uniform(0.1,1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = pos_eq_local
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    node_eq_global = pos_eq_global
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = (
                    w_local_pos * node_eq_local + w_global_pos * node_eq_global
                )  # + eps_pos_reg * w_reg
                eps_node = (
                    w_local_node * node_score_local + w_global_node * node_score_global
                )
                # eps_pos = 3 * node_eq_local + 1 * node_eq_global  # + eps_pos_reg * w_reg
                # eps_node = 1 * node_score_local + 1 * node_score_global
                # Update

                # =========================================================================
                # ===              [ 新增部分 3: 应用碰撞指导 ]                       ===
                # =========================================================================
                # 讲解：
                # 1. 检查当前噪声水平 sigma[i] 是否小于我们设定的阈值 clash_start_sigma。
                #    这避免了在噪声非常大（分子结构混乱）的早期阶段进行不必要的指导。
                # 2. 如果满足条件，调用我们之前定义的函数计算碰撞梯度。
                # 3. 将计算出的梯度乘以指导强度 w_clash，然后加到模型预测的 eps_pos 上。
                #    注意论文中是加号，因为梯度方向本身就是推离蛋白质的方向。

                if sigmas[i] < clash_start_sigma and w_clash > 0:
                    # 计算碰撞梯度
                    clash_grad = compute_clash_guidance_gradient(
                        ligand_pos=ligand_pos,
                        protein_pos=protein_pos,
                        sigma=1.0,  # 可以调整的超参数
                    )

                    # 将指导梯度应用到预测的噪声上
                    eps_pos = eps_pos + w_clash * clash_grad
                # =========================================================================

                sampling_type = kwargs.get(
                    "sampling_type", "ddpm_noisy"
                )  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(ligand_pos)
                noise_node = torch.randn_like(
                    ligand_atom_type
                )  # center_pos(torch.randn_like(pos), batch)
                b = self.betas
                t = t[0]
                next_t = (torch.ones(1) * j).to(ligand_pos.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if sampling_type == "generalized" or sampling_type == "ddpm_noisy":
                    if sampling_type == "generalized":
                        eta = kwargs.get("eta", 1.0)
                        et = -eps_pos

                        c1 = (
                            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        )
                        c2 = ((1 - at_next) - c1**2).sqrt()

                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 3 * (
                            (1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt()
                        )
                        step_size_pos = (
                            step_size_pos_ld
                            if step_size_pos_ld < step_size_pos_generalized
                            else step_size_pos_generalized
                        )

                        step_size_noise_ld = torch.sqrt(
                            (step_lr * (sigmas[i] / 0.01) ** 2) * 2
                        )
                        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
                        step_size_noise = (
                            step_size_noise_ld
                            if step_size_noise_ld < step_size_noise_generalized
                            else step_size_noise_generalized
                        )

                        # w = 1+2 * i/self.num_timesteps
                        w = 1

                        eps_node = eps_node / (1 - at).sqrt()
                        pos_next = (
                            ligand_pos
                            - et * step_size_pos
                            + w * noise * step_size_noise
                        )
                        atom_next = (
                            ligand_atom_type
                            - eps_node * step_size_pos
                            + w * noise_node * step_size_noise
                        )
                    elif sampling_type == "ddpm_noisy":
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        # pos0_from_e = (1.0 / at).sqrt() * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # pos0_from_e = 1 * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # mean_eps = (
                        #     (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_pos
                        # ) / (1.0 - at)
                        # mean = mean_eps
                        # mean = pos-beta_t/(1-at).sqrt()*e
                        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = (
                            mean + mask * torch.exp(0.5 * logvar) * noise
                        )  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                        e = eps_node
                        node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (
                            1.0 / at - 1
                        ).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * node0_from_e
                            + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        atom_next = (
                            mean + mask * torch.exp(0.5 * logvar) * noise_node
                        )  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == "ld":
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = (
                        ligand_pos
                        + step_size * eps_pos / sigmas[i]
                        + noise * torch.sqrt(step_size * 2)
                    )
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = (
                        ligand_atom_type
                        - step_size * eps_node / sigmas[i]
                        + noise_node * torch.sqrt(step_size * 2)
                    )
                else:
                    raise ValueError(
                        "Unknown sampling type, it should be one of [generalized, ddpm_noisy, ld]"
                    )

                ligand_pos = pos_next
                ligand_atom_type = atom_next

                if torch.isnan(ligand_pos).any():
                    print("NaN detected. Please restart.")
                    print(node_eq_local)
                    print(node_eq_global)
                    raise FloatingPointError()
                # ligand_pos = center_pos(ligand_pos, ligand_batch)
                ligand_pos, protein_pos = center_pos_pl(
                    ligand_pos, protein_pos, ligand_batch, protein_batch
                )
                if clip_pos is not None:
                    ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(ligand_pos.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())
                
                # 更新循环索引
                idx_in_seq += 1
                step_counter += 1
                
                # 周期性日志（每50步）
                if step_counter % 50 == 0:
                    print(f"[PROGRESS] step_counter={step_counter}, idx_in_seq={idx_in_seq}/{total_steps}, timestep={i}")
                
        # ========== 日志：采样完成总结 ==========
        print(f"\n{'='*60}")
        print(f"[SAMPLING] Completed:")
        print(f"  Total steps executed: {step_counter}")
        print(f"  Expected steps (n_steps): {n_steps}")
        print(f"  idx_in_seq final: {idx_in_seq}/{total_steps}")
        print(f"  SPM evaluations: {len(spm_scores_list)}")
        if use_spo:
            print(f"  SPO decisions made: {len(spo_decisions_list)}")
        print(f"{'='*60}\n")
        
        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        protein_pos = protein_pos + (protein_com - protein_final)[protein_batch]
        ligand_pos = ligand_pos + (protein_com - protein_final)[ligand_batch]
        # atom_type = torch.cat([atom_type[:,:-1]*4,atom_type[:,-1:]*10], dim=1)
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj, spm_scores_list, spo_decisions_list

    def langevin_dynamics_sample_cfg(
        self,
        ligand_atom_type,
        ligand_pos_init,
        ligand_bond_index,
        ligand_bond_type,
        ligand_num_node,
        ligand_batch,
        # Conditional protein
        protein_atom_type_cond,
        protein_atom_feature_full_cond,
        protein_pos_cond,
        protein_backbone_mask_cond,
        protein_batch_cond,
        # Unconditional protein
        protein_atom_type_uncond,
        protein_atom_feature_full_uncond,
        protein_pos_uncond,
        protein_backbone_mask_uncond,
        protein_batch_uncond,
        num_graphs,
        context,
        guidance_scale=1.0,
        extend_order=False,
        extend_radius=True,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        min_sigma=0,
        is_sidechain=None,
        global_start_sigma=float("inf"),
        local_start_sigma=float("inf"),
        w_global_pos=0.2,
        w_global_node=0.2,
        w_local_pos=0.2,
        w_local_node=0.2,
        w_reg=1.0,
        w_clash=0.0,
        clash_start_sigma=float("inf"),
        **kwargs
    ):
        """
        Classifier-Free Guidance sampling for diffusion models.

        This method implements CFG by:
        1. Running the model with conditional protein at each step
        2. Running the model with unconditional protein at each step
        3. Combining predictions: eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        Args:
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
        """
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            print(f"CFG sampling with guidance_scale={guidance_scale}, skip={skip}")
            seq = range(0, self.num_timesteps, skip)
            seq_next = [-1] + list(seq[:-1])

            # Center positions
            protein_com_cond = scatter_mean(protein_pos_cond, protein_batch_cond, dim=0)
            ligand_pos, protein_pos_cond_centered = center_pos_pl(
                ligand_pos_init + protein_com_cond[ligand_batch],
                protein_pos_cond,
                ligand_batch,
                protein_batch_cond,
            )

            # Also center unconditional protein
            protein_com_uncond = scatter_mean(protein_pos_uncond, protein_batch_uncond, dim=0)
            _, protein_pos_uncond_centered = center_pos_pl(
                ligand_pos_init + protein_com_uncond[ligand_batch],
                protein_pos_uncond,
                ligand_batch,
                protein_batch_uncond,
            )

            # Encode conditional protein
            protein_ctx_cond = self.protein_encoder(
                node_attr=protein_atom_feature_full_cond,
                pos=protein_pos_cond_centered,
                batch=protein_batch_cond,
            )

            # Encode unconditional protein
            protein_ctx_uncond = self.protein_encoder(
                node_attr=protein_atom_feature_full_uncond,
                pos=protein_pos_uncond_centered,
                batch=protein_batch_uncond,
            )

            # Atom numbers embedding (shared for both conditional and unconditional)
            num_node_ctx = None
            if "atom_num_emb" not in self.__dict__.keys():
                self.atom_num_emb = False
            if self.atom_num_emb:
                nonlinearity = nn.ReLU()
                nemb = get_num_embedding(ligand_num_node, self.config.hidden_dim)
                nemb = self.nemb.dense[0](nemb)
                nemb = nonlinearity(nemb)
                nemb = self.nemb.dense[1](nemb)
                nemb = self.nemb_proj(nonlinearity(nemb))
                num_node_ctx = nemb.index_select(0, ligand_batch)

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="CFG sample"):
                t = torch.full(
                    size=(num_graphs,),
                    fill_value=i,
                    dtype=torch.long,
                    device=ligand_pos.device,
                )

                # ===== Conditional forward pass =====
                net_out_cond = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx_cond,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type_cond,
                    protein_pos=protein_pos_cond_centered,
                    protein_backbone_mask=protein_backbone_mask_cond,
                    protein_batch=protein_batch_cond,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None,
                )

                if self.vae_context:
                    (
                        pos_eq_global_cond,
                        pos_eq_local_cond,
                        node_score_global_cond,
                        node_score_local_cond,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out_cond[:-1]
                else:
                    (
                        pos_eq_global_cond,
                        pos_eq_local_cond,
                        node_score_global_cond,
                        node_score_local_cond,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out_cond

                # ===== Unconditional forward pass =====
                net_out_uncond = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx_uncond,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type_uncond,
                    protein_pos=protein_pos_uncond_centered,
                    protein_backbone_mask=protein_backbone_mask_uncond,
                    protein_batch=protein_batch_uncond,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None,
                )

                if self.vae_context:
                    (
                        pos_eq_global_uncond,
                        pos_eq_local_uncond,
                        node_score_global_uncond,
                        node_score_local_uncond,
                        _,
                        _,
                        _,
                        _,
                    ) = net_out_uncond[:-1]
                else:
                    (
                        pos_eq_global_uncond,
                        pos_eq_local_uncond,
                        node_score_global_uncond,
                        node_score_local_uncond,
                        _,
                        _,
                        _,
                        _,
                    ) = net_out_uncond

                # ===== Apply CFG combination =====
                # Local
                if sigmas[i] < local_start_sigma:
                    # CFG for local position
                    pos_eq_local_cond_clipped = clip_norm(pos_eq_local_cond, limit=clip_local) if clip_local is not None else pos_eq_local_cond
                    pos_eq_local_uncond_clipped = clip_norm(pos_eq_local_uncond, limit=clip_local) if clip_local is not None else pos_eq_local_uncond
                    node_eq_local = pos_eq_local_uncond_clipped + guidance_scale * (pos_eq_local_cond_clipped - pos_eq_local_uncond_clipped)

                    # CFG for local node score
                    node_score_local = node_score_local_uncond + guidance_scale * (node_score_local_cond - node_score_local_uncond)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    # CFG for global position
                    pos_eq_global_cond_clipped = clip_norm(pos_eq_global_cond, limit=clip)
                    pos_eq_global_uncond_clipped = clip_norm(pos_eq_global_uncond, limit=clip)
                    node_eq_global = pos_eq_global_uncond_clipped + guidance_scale * (pos_eq_global_cond_clipped - pos_eq_global_uncond_clipped)
                    # CFG for global node score
                    node_score_global = node_score_global_uncond + guidance_scale * (node_score_global_cond - node_score_global_uncond)
                else:
                    node_eq_global = 0
                    node_score_global = 0

                # Sum weighted predictions
                eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global
                eps_node = w_local_node * node_score_local + w_global_node * node_score_global

                # Apply clash guidance if needed (on the combined prediction)
                if sigmas[i] < clash_start_sigma and w_clash > 0:
                    clash_grad = compute_clash_guidance_gradient(
                        ligand_pos=ligand_pos,
                        protein_pos=protein_pos_cond_centered,
                        sigma=1.0,
                    )
                    eps_pos = eps_pos + w_clash * clash_grad

                # Denoising update
                sampling_type = kwargs.get("sampling_type", "ddpm_noisy")
                noise = torch.randn_like(ligand_pos)
                noise_node = torch.randn_like(ligand_atom_type)
                b = self.betas
                t = t[0]
                next_t = (torch.ones(1) * j).to(ligand_pos.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())

                if sampling_type == "generalized" or sampling_type == "ddpm_noisy":
                    if sampling_type == "generalized":
                        eta = kwargs.get("eta", 1.0)
                        et = -eps_pos

                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1**2).sqrt()

                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld < step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld < step_size_noise_generalized else step_size_noise_generalized

                        w = 1
                        eps_node = eps_node / (1 - at).sqrt()
                        pos_next = ligand_pos - et * step_size_pos + w * noise * step_size_noise
                        atom_next = ligand_atom_type - eps_node * step_size_pos + w * noise_node * step_size_noise

                    elif sampling_type == "ddpm_noisy":
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                        e = eps_node
                        node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (1.0 / at - 1).sqrt() * e
                        mean_eps = ((atm1.sqrt() * beta_t) * node0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        atom_next = mean + mask * torch.exp(0.5 * logvar) * noise_node

                elif sampling_type == "ld":
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = ligand_pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size * 2)
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = ligand_atom_type - step_size * eps_node / sigmas[i] + noise_node * torch.sqrt(step_size * 2)
                else:
                    raise ValueError("Unknown sampling type, it should be one of [generalized, ddpm_noisy, ld]")

                ligand_pos = pos_next
                ligand_atom_type = atom_next

                if torch.isnan(ligand_pos).any():
                    print("NaN detected. Please restart.")
                    raise FloatingPointError()

                ligand_pos, protein_pos_cond_centered = center_pos_pl(ligand_pos, protein_pos_cond_centered, ligand_batch, protein_batch_cond)
                _, protein_pos_uncond_centered = center_pos_pl(ligand_pos, protein_pos_uncond_centered, ligand_batch, protein_batch_uncond)

                if clip_pos is not None:
                    ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(ligand_pos.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())

        protein_final = scatter_mean(protein_pos_cond_centered, protein_batch_cond, dim=0)
        protein_pos_cond_centered = protein_pos_cond_centered + (protein_com_cond - protein_final)[protein_batch_cond]
        ligand_pos = ligand_pos + (protein_com_cond - protein_final)[ligand_batch]

        # CFG采样目前不支持SPM评分和SPO重采样，返回空列表
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj, [], []

    # for lead optimization
    def inpainting_sample(
        self,
        ligand_atom_type,
        ligand_pos_init,
        ligand_bond_index,
        ligand_bond_type,
        ligand_num_node,
        ligand_batch,
        frag_mask,
        protein_atom_type,
        protein_pos,
        protein_backbone_mask,
        protein_batch,
        num_graphs,
        context,
        extend_order,
        extend_radius=True,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        is_sidechain=None,
        global_start_sigma=float("inf"),
        local_start_sigma=float("inf"),
        w_global_pos=0.2,
        w_global_node=0.2,
        w_local_pos=0.2,
        w_local_node=0.2,
        w_reg=1.0,
        **kwargs
    ):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            # seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            linker_mask = ~frag_mask
            frag_pos = ligand_pos_init[frag_mask, :]
            ligand_pos = ligand_pos_init
            linker_pos = ligand_pos_init[linker_mask, :]

            frag_batch = ligand_batch[frag_mask]

            protein_pos_ori = protein_pos
            protein_com = scatter_mean(protein_pos, protein_batch, dim=0)
            # ligand_pos = ligand_pos_init + protein_com[ligand_batch] #important

            # subtract the center of mass of the pocket (COM)
            ligand_pos, protein_pos = center_pos_lp(
                ligand_pos, protein_pos, ligand_batch, protein_batch
            )  # important
            # frag_pos = ligand_pos[frag_mask, :]
            # Recover the linker postion
            ligand_pos[linker_mask] = linker_pos  # important
            original_atom_type = ligand_atom_type.clone()

            # VAE noise
            # vae_noise = torch.zeros_like(ligand_atom_type).uniform_(-1, +1)
            # vae_noise = torch.randn_like(atom_type)
            # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3)
            # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device)

            """
            Protein embedding
            """
            protein_ctx = self.protein_encoder(
                node_attr=protein_atom_type,
                pos=protein_pos,
                batch=protein_batch,
            )
            # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)
            # protein_ctx = protein_ctx.index_select(0, ligand_batch)

            """
            Atom numbers embedding
            """
            num_node_ctx = None
            if "atom_num_emb" not in self.__dict__.keys():
                self.atom_num_emb = False
            if self.atom_num_emb:
                nonlinearity = nn.ReLU()
                nemb = get_num_embedding(ligand_num_node, self.config.hidden_dim)
                nemb = self.nemb.dense[0](nemb)
                nemb = nonlinearity(nemb)
                nemb = self.nemb.dense[1](nemb)
                nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
                num_node_ctx = nemb.index_select(0, ligand_batch)

            # linker_pos = ligand_pos_init[linker_mask,:]
            # linker_atom = ligand_atom_type[linker_mask,:]
            # original_atom_type = ligand_atom_type
            ligand_pos_ori = ligand_pos.clone()
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="sample"):
                t = torch.full(
                    size=(num_graphs,),
                    fill_value=i,
                    dtype=torch.long,
                    device=ligand_pos.device,
                )
                b = self.betas
                at = compute_alpha(b, t[0].long())

                pos_noise = torch.randn(
                    size=ligand_pos[frag_mask, :].size(), device=ligand_pos.device
                )
                atom_noise = torch.randn(
                    size=ligand_atom_type[frag_mask, :].size(),
                    device=ligand_atom_type.device,
                )
                mask = 1 - (t[0] == 0).float()

                # linker_pos = ligand_pos
                frag_pos = ligand_pos[frag_mask, :]  # important
                frag_atom_type = ligand_atom_type[frag_mask, :]  # important

                # # fix the atom type
                # frag_pos_perturbed = frag_pos + pos_noise * (1.0 - at).sqrt() / at.sqrt() * mask #important
                # # ligand_pos = torch.cat([linker_pos_perturbed,ligand_pos[~linker_mask,:]])
                # ligand_pos[frag_mask] = frag_pos_perturbed #important

                # ligand_atom_type = torch.cat([linker_atom_perturbed,ligand_atom_type[~linker_mask,:]])
                frag_atom_perturbed = (
                    at.sqrt() * frag_atom_type + (1.0 - at).sqrt() * atom_noise * mask
                )
                ligand_atom_type[frag_mask] = frag_atom_perturbed
                # ligand_pos,protein_pos = center_pos_pl(ligand_pos, protein_pos, ligand_batch, protein_batch)
                net_out = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type[:, :10],
                    protein_pos=protein_pos,
                    protein_backbone_mask=protein_backbone_mask,
                    protein_batch=protein_batch,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None,
                )  # (E_global, 1), (E_local, 1)
                if self.vae_context:
                    (
                        pos_eq_global,
                        pos_eq_local,
                        node_score_global,
                        node_score_local,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out[:-1]
                else:
                    (
                        pos_eq_global,
                        pos_eq_local,
                        node_score_global,
                        node_score_local,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out
                # Local float('inf')

                # local_start_sigma = random.uniform(0.1,1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = pos_eq_local
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    node_eq_global = pos_eq_global
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = (
                    w_local_pos * node_eq_local + w_global_pos * node_eq_global
                )  # + eps_pos_reg * w_reg
                eps_node = (
                    w_local_node * node_score_local + w_global_node * node_score_global
                )
                # eps_pos = 3 * node_eq_local + 1 * node_eq_global  # + eps_pos_reg * w_reg
                # eps_node = 1 * node_score_local + 1 * node_score_global
                # Update

                sampling_type = kwargs.get(
                    "sampling_type", "ddpm_noisy"
                )  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(ligand_pos)
                noise_node = torch.randn_like(
                    ligand_atom_type
                )  # center_pos(torch.randn_like(pos), batch)

                t = t[0]
                next_t = (torch.ones(1) * j).to(ligand_pos.device)
                # at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if sampling_type == "generalized" or sampling_type == "ddpm_noisy":
                    if sampling_type == "generalized":
                        eta = kwargs.get("eta", 1.0)
                        et = -eps_pos

                        c1 = (
                            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        )
                        c2 = ((1 - at_next) - c1**2).sqrt()

                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 3 * (
                            (1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt()
                        )
                        step_size_pos = (
                            step_size_pos_ld
                            if step_size_pos_ld < step_size_pos_generalized
                            else step_size_pos_generalized
                        )

                        step_size_noise_ld = torch.sqrt(
                            (step_lr * (sigmas[i] / 0.01) ** 2) * 2
                        )
                        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
                        step_size_noise = (
                            step_size_noise_ld
                            if step_size_noise_ld < step_size_noise_generalized
                            else step_size_noise_generalized
                        )

                        # w = 1+2 * i/self.num_timesteps
                        w = 1

                        eps_node = eps_node / (1 - at).sqrt()
                        pos_next = (
                            ligand_pos
                            - et * step_size_pos
                            + w * noise * step_size_noise
                        )
                        atom_next = (
                            ligand_atom_type
                            - eps_node * step_size_pos
                            + w * noise_node * step_size_noise
                        )
                    elif sampling_type == "ddpm_noisy":
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        # pos0_from_e = (1.0 / at).sqrt() * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # pos0_from_e = 1 * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # mean_eps = (
                        #     (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_pos
                        # ) / (1.0 - at)
                        # mean = mean_eps
                        # mean = pos-beta_t/(1-at).sqrt()*e
                        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = (
                            mean + mask * torch.exp(0.5 * logvar) * noise
                        )  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                        e = eps_node
                        node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (
                            1.0 / at - 1
                        ).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * node0_from_e
                            + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        atom_next = (
                            mean + mask * torch.exp(0.5 * logvar) * noise_node
                        )  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == "ld":
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = (
                        ligand_pos
                        + step_size * eps_pos / sigmas[i]
                        + noise * torch.sqrt(step_size * 2)
                    )
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = (
                        ligand_atom_type
                        - step_size * eps_node / sigmas[i]
                        + noise_node * torch.sqrt(step_size * 2)
                    )
                else:
                    raise ValueError(
                        "Unknown sampling type, it should be one of [generalized, ddpm_noisy, ld]"
                    )

                ligand_pos = pos_next  # important
                ligand_atom_type = atom_next

                if torch.isnan(ligand_pos).any():
                    print("NaN detected. Please restart.")
                    print(node_eq_local)
                    print(node_eq_global)
                    raise FloatingPointError()
                # ligand_pos = center_pos(ligand_pos, ligand_batch)
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                ligand_pos[frag_mask] = frag_pos  # important
                # ligand_pos = ligand_pos_ori
                # ligand_atom_type = torch.cat([linker_atom_type,ligand_atom_type[~linker_mask,:]])
                ligand_atom_type[frag_mask] = frag_atom_type  # important
                # ligand_atom_type = original_atom_type.clone() #fix the atom type
                ligand_pos, protein_pos = center_pos_pl(
                    ligand_pos, protein_pos, ligand_batch, protein_batch
                )  # important
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                # ligand_atom_type = torch.cat([linker_atom,ligand_atom_type[~linker_mask,:]])
                if clip_pos is not None:
                    ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)

                protein_t = scatter_mean(protein_pos, protein_batch, dim=0)
                move_dist = protein_com - protein_t
                ligand_pos_fix = ligand_pos + move_dist[ligand_batch]
                pos_traj.append(ligand_pos_fix.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())
        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        # protein_final = protein_pos
        protein_pos = protein_pos + (protein_com - protein_final)[protein_batch]
        ligand_pos = (
            ligand_pos + (protein_com - protein_final)[ligand_batch]
        )  # important
        print(torch.equal(protein_pos_ori, protein_pos))
        # ligand_pos = ligand_pos_init
        # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])

        # atom_type = torch.cat([atom_type[:,:-1]*4,atom_type[:,-1:]*10], dim=1)
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj

    # for linker sample
    def linker_sample(
        self,
        ligand_atom_type,
        ligand_pos_init,
        ligand_bond_index,
        ligand_bond_type,
        ligand_num_node,
        ligand_batch,
        frag_mask,
        protein_atom_type,
        protein_pos,
        protein_backbone_mask,
        protein_batch,
        num_graphs,
        context,
        extend_order,
        extend_radius=True,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_local=None,
        clip_pos=None,
        is_sidechain=None,
        global_start_sigma=float("inf"),
        local_start_sigma=float("inf"),
        w_global_pos=0.2,
        w_global_node=0.2,
        w_local_pos=0.2,
        w_local_node=0.2,
        w_reg=1.0,
        **kwargs
    ):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            # seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            linker_mask = ~frag_mask
            frag_pos = ligand_pos_init[frag_mask, :]
            ligand_pos = ligand_pos_init
            linker_pos = ligand_pos_init[linker_mask, :]

            frag_batch = ligand_batch[frag_mask]

            protein_pos_ori = protein_pos
            protein_com = scatter_mean(protein_pos, protein_batch, dim=0)

            # subtract the center of mass of the pocket (COM)
            ligand_pos, protein_pos = center_pos_lp(
                ligand_pos, protein_pos, ligand_batch, protein_batch
            )  # important
            frag_pos = ligand_pos[frag_mask, :]
            # Recover the linker postion
            ligand_pos[linker_mask] = linker_pos  # important

            # VAE noise
            vae_noise = torch.zeros_like(ligand_atom_type).uniform_(-1, +1)
            # vae_noise = torch.randn_like(atom_type)
            # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3)
            # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device)

            """
            Protein embedding
            """
            protein_ctx = self.protein_encoder(
                node_attr=protein_atom_type,
                pos=protein_pos,
                batch=protein_batch,
            )
            # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)
            # protein_ctx = protein_ctx.index_select(0, ligand_batch)

            """
            Atom numbers embedding
            """
            num_node_ctx = None
            if "atom_num_emb" not in self.__dict__.keys():
                self.atom_num_emb = False
            if self.atom_num_emb:
                nonlinearity = nn.ReLU()
                nemb = get_num_embedding(ligand_num_node, self.config.hidden_dim)
                nemb = self.nemb.dense[0](nemb)
                nemb = nonlinearity(nemb)
                nemb = self.nemb.dense[1](nemb)
                nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
                num_node_ctx = nemb.index_select(0, ligand_batch)

            # linker_pos = ligand_pos_init[linker_mask,:]
            # linker_atom = ligand_atom_type[linker_mask,:]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="sample"):
                t = torch.full(
                    size=(num_graphs,),
                    fill_value=i,
                    dtype=torch.long,
                    device=ligand_pos.device,
                )
                b = self.betas
                at = compute_alpha(b, t[0].long())

                pos_noise = torch.randn(
                    size=ligand_pos[frag_mask, :].size(), device=ligand_pos.device
                )
                atom_noise = torch.randn(
                    size=ligand_atom_type[frag_mask, :].size(),
                    device=ligand_atom_type.device,
                )
                mask = 1 - (t[0] == 0).float()

                frag_atom_type = ligand_atom_type[frag_mask, :]
                frag_pos = ligand_pos[frag_mask, :]
                net_out = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type[:, :10],
                    protein_pos=protein_pos,
                    protein_backbone_mask=protein_backbone_mask,
                    protein_batch=protein_batch,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None,
                    linker_mask=linker_mask,
                )  # (E_global, 1), (E_local, 1)
                if self.vae_context:
                    (
                        pos_eq_global,
                        pos_eq_local,
                        node_score_global,
                        node_score_local,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out[:-1]
                else:
                    (
                        pos_eq_global,
                        pos_eq_local,
                        node_score_global,
                        node_score_local,
                        edge_index,
                        edge_type,
                        edge_length,
                        local_edge_mask,
                    ) = net_out
                # Local float('inf')

                # local_start_sigma = random.uniform(0.1,1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = pos_eq_local
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    node_eq_global = pos_eq_global
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = (
                    w_local_pos * node_eq_local + w_global_pos * node_eq_global
                )  # + eps_pos_reg * w_reg
                eps_node = (
                    w_local_node * node_score_local + w_global_node * node_score_global
                )
                # eps_pos = 3 * node_eq_local + 1 * node_eq_global  # + eps_pos_reg * w_reg
                # eps_node = 1 * node_score_local + 1 * node_score_global
                # Update

                sampling_type = kwargs.get(
                    "sampling_type", "ddpm_noisy"
                )  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(ligand_pos)
                noise_node = torch.randn_like(
                    ligand_atom_type
                )  # center_pos(torch.randn_like(pos), batch)

                t = t[0]
                next_t = (torch.ones(1) * j).to(ligand_pos.device)
                # at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if sampling_type == "generalized" or sampling_type == "ddpm_noisy":
                    if sampling_type == "generalized":
                        eta = kwargs.get("eta", 1.0)
                        et = -eps_pos

                        c1 = (
                            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        )
                        c2 = ((1 - at_next) - c1**2).sqrt()

                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 3 * (
                            (1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt()
                        )
                        step_size_pos = (
                            step_size_pos_ld
                            if step_size_pos_ld < step_size_pos_generalized
                            else step_size_pos_generalized
                        )

                        step_size_noise_ld = torch.sqrt(
                            (step_lr * (sigmas[i] / 0.01) ** 2) * 2
                        )
                        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
                        step_size_noise = (
                            step_size_noise_ld
                            if step_size_noise_ld < step_size_noise_generalized
                            else step_size_noise_generalized
                        )

                        # w = 1+2 * i/self.num_timesteps
                        w = 1

                        eps_node = eps_node / (1 - at).sqrt()
                        pos_next = (
                            ligand_pos
                            - et * step_size_pos
                            + w * noise * step_size_noise
                        )
                        atom_next = (
                            ligand_atom_type
                            - eps_node * step_size_pos
                            + w * noise_node * step_size_noise
                        )
                    elif sampling_type == "ddpm_noisy":
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        # pos0_from_e = (1.0 / at).sqrt() * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # pos0_from_e = 1 * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # mean_eps = (
                        #     (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_pos
                        # ) / (1.0 - at)
                        # mean = mean_eps
                        # mean = pos-beta_t/(1-at).sqrt()*e
                        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = (
                            mean + mask * torch.exp(0.5 * logvar) * noise
                        )  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                        e = eps_node
                        node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (
                            1.0 / at - 1
                        ).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * node0_from_e
                            + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        atom_next = (
                            mean + mask * torch.exp(0.5 * logvar) * noise_node
                        )  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == "ld":
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = (
                        ligand_pos
                        + step_size * eps_pos / sigmas[i]
                        + noise * torch.sqrt(step_size * 2)
                    )
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = (
                        ligand_atom_type
                        - step_size * eps_node / sigmas[i]
                        + noise_node * torch.sqrt(step_size * 2)
                    )
                else:
                    raise ValueError(
                        "Unknown sampling type, it should be one of [generalized, ddpm_noisy, ld]"
                    )

                ligand_pos = pos_next  # important
                ligand_atom_type = atom_next

                if torch.isnan(ligand_pos).any():
                    print("NaN detected. Please restart.")
                    print(node_eq_local)
                    print(node_eq_global)
                    raise FloatingPointError()
                # ligand_pos = center_pos(ligand_pos, ligand_batch)
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                ligand_pos[frag_mask] = frag_pos  # important
                # ligand_atom_type = torch.cat([linker_atom_type,ligand_atom_type[~linker_mask,:]])
                ligand_atom_type[frag_mask] = frag_atom_type

                ligand_pos, protein_pos = center_pos_pl(
                    ligand_pos, protein_pos, ligand_batch, protein_batch
                )  # important
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                # ligand_atom_type = torch.cat([linker_atom,ligand_atom_type[~linker_mask,:]])
                if clip_pos is not None:
                    ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)

                protein_t = scatter_mean(protein_pos, protein_batch, dim=0)
                move_dist = protein_com - protein_t
                ligand_pos_fix = ligand_pos + move_dist[ligand_batch]
                pos_traj.append(ligand_pos_fix.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())
        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        # protein_final = protein_pos
        protein_pos = protein_pos + (protein_com - protein_final)[protein_batch]

        ligand_pos = (
            ligand_pos + (protein_com - protein_final)[ligand_batch]
        )  # important
        ligand_pos = ligand_pos
        # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])

        # atom_type = torch.cat([atom_type[:,:-1]*4,atom_type[:,-1:]*10], dim=1)
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj


def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


# def is_radius_edge(edge_type):
#     return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0
    # return edge_type == 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(
            is_sidechain[edge_index[0]], is_sidechain[edge_index[1]]
        )


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float("inf")) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def center_pos_pl(ligand_pos, pocket_pos, ligand_batch, pocket_batch):
    ligand_pos_center = (
        ligand_pos - scatter_mean(ligand_pos, ligand_batch, dim=0)[ligand_batch]
    )
    pocket_pos_center = (
        pocket_pos - scatter_mean(ligand_pos, ligand_batch, dim=0)[pocket_batch]
    )
    return ligand_pos_center, pocket_pos_center


def center_pos_lp(ligand_pos, pocket_pos, ligand_batch, pocket_batch):
    ligand_pos_center = (
        ligand_pos - scatter_mean(pocket_pos, pocket_batch, dim=0)[ligand_batch]
    )
    pocket_pos_center = (
        pocket_pos - scatter_mean(pocket_pos, pocket_batch, dim=0)[pocket_batch]
    )
    return ligand_pos_center, pocket_pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom


# =========================================================================
# ===                  [ 新增部分 1: 碰撞损失函数 ]                    ===
# =========================================================================
def compute_clash_guidance_gradient(
    ligand_pos, protein_pos, sigma=1.0, use_softmin=True, clash_threshold=2.5
):
    """
    计算配体和蛋白质之间的碰撞损失梯度。
    Args:
        ligand_pos (Tensor): 配体原子坐标, shape [N_l, 3].
        protein_pos (Tensor): 蛋白质原子坐标, shape [N_p, 3].
        sigma (float): Softmin函数的平滑因子。
        use_softmin (bool): 是否使用softmin来近似最近距离。
        clash_threshold (float): 碰撞阈值，只惩罚距离小于此值的情况。
    Returns:
        Tensor: 碰撞梯度的向量, shape [N_l, 3].
    """
    # 即使在 torch.no_grad() 上下文中，我们也需要临时开启梯度计算
    with torch.enable_grad():
        # 克隆 ligand_pos 并设置 requires_grad=True，使其成为计算梯度的目标
        ligand_pos_with_grad = ligand_pos.clone().requires_grad_(True)

        # 计算所有配体-蛋白质原子对之间的距离的平方
        dist_sq = torch.sum(
            (ligand_pos_with_grad.unsqueeze(1) - protein_pos.unsqueeze(0)) ** 2, dim=-1
        )

        # 计算实际距离
        dist = torch.sqrt(dist_sq + 1e-8)  # 添加小值避免数值不稳定

        # 只对距离小于阈值的情况进行惩罚
        clash_mask = dist < clash_threshold

        if use_softmin:
            # 只对碰撞的原子对计算损失
            # 将非碰撞的距离设为很大的值，使其在softmin中贡献很小
            masked_dist_sq = torch.where(
                clash_mask, dist_sq, torch.full_like(dist_sq, 1e6)
            )

            # MolSnapper论文中的公式 (3) & (4)
            log_sum_exp_val = torch.logsumexp(-masked_dist_sq / sigma, dim=1)
            clash_loss = -sigma * torch.sum(log_sum_exp_val)
        else:
            raise NotImplementedError("use_softmin must be True")

        # 使用 torch.autograd.grad 直接计算梯度
        # 这是计算输出相对于输入的梯度的推荐方法，而不是使用 .backward()
        # outputs: clash_loss
        # inputs: ligand_pos_with_grad
        grad_outputs = torch.ones_like(clash_loss)  # 对于标量损失，梯度输出为1
        (grad,) = torch.autograd.grad(
            outputs=clash_loss,
            inputs=ligand_pos_with_grad,
            grad_outputs=grad_outputs,
            create_graph=False,  # 我们不需要二阶梯度，所以设为False以提高效率
            retain_graph=False,
        )

    # 防止梯度过大导致不稳定
    if grad is not None:
        grad_norm = torch.norm(grad, dim=1, keepdim=True)
        # 避免除以零
        safe_norm = torch.where(grad_norm > 1e-8, grad_norm, torch.ones_like(grad_norm))
        grad = grad / safe_norm  # 归一化梯度
    else:
        grad = torch.zeros_like(ligand_pos)

    return grad.detach()  # 返回梯度，并切断其计算图
