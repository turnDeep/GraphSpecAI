import argparse
import copy
import datetime
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, DataStructs, Descriptors, Draw, MolToSmiles, MurckoScaffold, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit.Chem.rdchem import BondType
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           precision_recall_fscore_support, silhouette_score)
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              random_split)
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from tqdm import tqdm
# mpl_toolkits is part of matplotlib but imported this way
from mpl_toolkits.mplot3d import Axes3D
# torch.nn, torch.nn.functional, torch.optim are usually imported like this
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Content from gsai0501-1.py ---

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Atom type mapping and constants
ATOM_TYPE_TO_INDEX = {
    6: 0,  # C
    1: 1,  # H
    7: 2,  # N
    8: 3,  # O
    9: 4,  # F
    16: 5, # S
    15: 6, # P
    17: 7, # Cl
    35: 8, # Br
    53: 9  # I
}
UNKNOWN_ATOM_INDEX_TARGET = -100  # Using -100 for CrossEntropyLoss ignore_index
NUM_ATOM_CLASSES = 10 # C, H, N, O, F, S, P, Cl, Br, I (matches decoder output)


# モデル定数
HIDDEN_DIM = 256
LATENT_DIM = 128
SPECTRUM_DIM = 2000  # m/zの最大値
MAX_ATOMS = 100  # 1分子あたりの最大原子数
MAX_MOTIFS = 20  # 1分子あたりの最大モチーフ数
ATOM_FEATURE_DIM = 150  # 原子特徴量の次元
BOND_FEATURE_DIM = 10  # 結合特徴量の次元
MOTIF_FEATURE_DIM = 20  # モチーフ特徴量の次元

# 拡散モデル定数
DIFFUSION_STEPS = 1000
DIFFUSION_BETA_START = 1e-4
DIFFUSION_BETA_END = 0.02

# モチーフの種類
MOTIF_TYPES = [
    "ester", "amide", "amine", "urea", "ether", "olefin", 
    "aromatic", "heterocycle", "lactam", "lactone", "carbonyl"
]

# 破壊モード
BREAK_MODES = ["single_cleavage", "multiple_cleavage", "ring_opening"]

#------------------------------------------------------
# データ構造の定義
#------------------------------------------------------

class Fragment:
    """質量分析のフラグメントを表すクラス"""
    
    def __init__(self, atoms: List[int], mol: Chem.Mol, parent_mol: Chem.Mol = None, 
                lost_hydrogens: int = 0, charge: int = 1):
        """
        フラグメントを初期化する
        
        Args:
            atoms: フラグメントに含まれる原子のインデックスリスト
            mol: フラグメントのRDKit分子オブジェクト
            parent_mol: 親分子のRDKit分子オブジェクト（なければNone）
            lost_hydrogens: 失われた水素原子の数
            charge: フラグメントの電荷（デフォルトは+1）
        """
        self.atoms = atoms
        self.mol = mol
        self.parent_mol = parent_mol
        self.lost_hydrogens = lost_hydrogens
        self.charge = charge
        
        # フラグメントの質量と化学式を計算
        self.mass = self._calculate_mass()
        self.formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        
        # 電子数や安定性などの特性を計算
        self.electron_count = sum(atom.GetAtomicNum() - atom.GetFormalCharge() for atom in mol.GetAtoms())
        self.stability = self._calculate_stability()
        self.ionization_efficiency = self._calculate_ionization_efficiency()
        
    def _calculate_mass(self) -> float:
        """フラグメントの質量を計算（水素損失を考慮）"""
        # 正確な分子量を計算
        exact_mass = Chem.rdMolDescriptors.CalcExactMolWt(self.mol)
        
        # 失われた水素原子の質量を差し引く
        hydrogen_mass = 1.00782503  # 水素原子の正確な質量
        adjusted_mass = exact_mass - (hydrogen_mass * self.lost_hydrogens)
        
        # m/z値を計算（電荷で割る）
        mz = adjusted_mass / self.charge
        
        return mz
    
    def _calculate_stability(self) -> float:
        """フラグメントの安定性を評価（0〜1の値）"""
        stability_score = 0.5  # デフォルト値
        
        # 1. 芳香環の数（安定性に寄与）
        aromatic_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(self.mol)
        stability_score += 0.1 * aromatic_rings
        
        # 2. 不対電子の存在（不安定化要因）
        radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in self.mol.GetAtoms())
        stability_score -= 0.15 * radical_electrons
        
        # 3. 共役システムの存在（安定性に寄与）
        conjugated_bonds = sum(bond.GetIsConjugated() for bond in self.mol.GetBonds())
        stability_score += 0.05 * conjugated_bonds
        
        # 4. フラグメントサイズ（小さすぎると不安定）
        n_atoms = self.mol.GetNumAtoms()
        if n_atoms < 3:
            stability_score -= 0.2
        elif n_atoms < 5:
            stability_score -= 0.1
        
        # 5. 閉殻電子配置（安定性に寄与）
        closed_shell = all(atom.GetNumRadicalElectrons() == 0 for atom in self.mol.GetAtoms())
        if closed_shell:
            stability_score += 0.2
        
        # 最終スコアを0〜1の範囲に正規化
        stability_score = max(0.0, min(1.0, stability_score))
        
        return stability_score
    
    def _calculate_ionization_efficiency(self) -> float:
        """フラグメントのイオン化効率を計算（0〜1の値）"""
        # デフォルト効率
        efficiency = 0.5
        
        # 1. 電子供与基/吸引基の存在をチェック
        donors = ['NH2', 'OH', 'OCH3', 'CH3']
        acceptors = ['NO2', 'CN', 'CF3', 'COOR', 'COR']
        
        donor_count = sum(self.mol.HasSubstructMatch(Chem.MolFromSmarts(d)) for d in donors)
        acceptor_count = sum(self.mol.HasSubstructMatch(Chem.MolFromSmarts(a)) for a in acceptors)
        
        # 電子供与基は正イオン化を促進
        efficiency += 0.05 * donor_count
        
        # 2. 特定原子の存在をチェック（N, O, Sなど）
        heteroatom_count = sum(1 for atom in self.mol.GetAtoms() 
                              if atom.GetAtomicNum() in [7, 8, 16])
        efficiency += 0.02 * heteroatom_count
        
        # 3. 不飽和度（二重結合、三重結合、環の数）
        unsaturation = Chem.rdMolDescriptors.CalcNumUnsaturations(self.mol)
        efficiency += 0.03 * unsaturation
        
        # 4. π電子系の存在
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in self.mol.GetAtoms())
        efficiency += 0.05 * (aromatic_atoms > 0)
        
        # 最終効率を0〜1の範囲に正規化
        efficiency = max(0.0, min(1.0, efficiency))
        
        return efficiency
    
    def __repr__(self) -> str:
        """フラグメントの文字列表現"""
        return f"Fragment(mass={self.mass:.4f}, formula={self.formula}, stability={self.stability:.2f})"

class FragmentNode:
    """フラグメントツリーのノードクラス"""
    
    def __init__(self, fragment: Fragment, parent=None, break_mode: str = "single_cleavage", 
                 broken_bonds: List[int] = None):
        """
        フラグメントノードを初期化する
        
        Args:
            fragment: ノードに対応するフラグメント
            parent: 親ノード（なければNone）
            break_mode: このノードを生成した破壊モード
            broken_bonds: 切断された結合のリスト
        """
        self.fragment = fragment
        self.parent = parent
        self.children = []
        self.break_mode = break_mode
        self.broken_bonds = broken_bonds or []
        self.intensity = None  # 後で計算される相対強度
        
    def add_child(self, child_node):
        """子ノードを追加"""
        self.children.append(child_node)
        child_node.parent = self
    
    def get_path_from_root(self) -> List["FragmentNode"]:
        """ルートからこのノードまでのパスを取得"""
        path = []
        current = self
        while current:
            path.insert(0, current)
            current = current.parent
        return path
    
    def get_all_fragments(self) -> List[Fragment]:
        """このノード以下のすべてのフラグメントを取得（深さ優先探索）"""
        fragments = [self.fragment]
        for child in self.children:
            fragments.extend(child.get_all_fragments())
        return fragments
    
    def __repr__(self) -> str:
        """ノードの文字列表現"""
        return f"FragmentNode({self.fragment}, children={len(self.children)}, mode={self.break_mode})"

#------------------------------------------------------
# 拡散モデル基本コンポーネント
#------------------------------------------------------

class DiffusionModel:
    """拡散モデルの基本クラス"""
    
    def __init__(self, num_steps=DIFFUSION_STEPS, beta_start=DIFFUSION_BETA_START, beta_end=DIFFUSION_BETA_END):
        """拡散モデルの初期化"""
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # スケジューリングパラメータの計算
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def _get_beta_schedule(self):
        """ベータスケジュールを計算"""
        return torch.linspace(self.beta_start, self.beta_end, self.num_steps)
    
    def q_sample(self, x_0, t, noise=None):
        """前方拡散過程: x_0 から x_t を生成"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_sample(self, model, x_t, t, t_index):
        """逆拡散過程の単一ステップ: x_t から x_{t-1} を生成"""
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        # モデルによるノイズ予測
        pred_noise = model(x_t, t)
        
        # ノイズからの復元
        mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_variance_t) * noise
        
    def p_sample_loop(self, model, shape, device):
        """完全な逆拡散過程: ノイズからサンプルを生成"""
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.num_steps)), total=self.num_steps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)
            
        return x
    
    def _extract(self, a, t, shape):
        """t時点でのパラメータを抽出"""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(shape) - 1))).to(t.device)

#------------------------------------------------------
# グラフニューラルネットワークコンポーネント
#------------------------------------------------------

class StructureEncoder(nn.Module):
    """化学構造をエンコードするモジュール（モチーフベースGNN）"""
    
    def __init__(self, atom_fdim, bond_fdim, motif_fdim, hidden_dim, latent_dim):
        super(StructureEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 原子特徴量エンコーダ
        self.atom_encoder = nn.Linear(atom_fdim, hidden_dim)
        
        # 結合特徴量エンコーダ
        self.bond_encoder = nn.Linear(bond_fdim, hidden_dim)
        
        # モチーフ特徴量エンコーダ
        self.motif_encoder = nn.Linear(motif_fdim, hidden_dim)
        
        # グラフ畳み込み層
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # モチーフGNN層
        self.gin_layers = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )) for _ in range(3)
        ])
        
        # グローバルアテンション
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 最終潜在表現への射影
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, molecule_data_list: List['MoleculeData']):
        """順伝播: 化学構造から潜在表現を生成"""
        device = self.atom_encoder.weight.device

        if not molecule_data_list:
            return torch.empty(0, self.latent_dim).to(device)

        pyg_data_list = []
        for md in molecule_data_list:
            graph_data = md.graph_data
            
            def get_tensor(key, default_shape_fn, dtype=torch.float32):
                tensor = graph_data.get(key)
                if tensor is None or tensor.numel() == 0:
                    if dtype == torch.long:
                         return default_shape_fn(device).long()
                    return default_shape_fn(device) # Default to float
                return tensor.to(device)

            bond_fdim = self.bond_encoder.in_features
            motif_fdim = self.motif_encoder.in_features
            motif_edge_attr_fdim = 6 

            data_obj = Data(
                x=get_tensor('x', lambda dev: torch.empty(0, self.atom_encoder.in_features).to(dev)),
                edge_index=get_tensor('edge_index', lambda dev: torch.empty(2, 0).to(dev), dtype=torch.long),
                edge_attr=get_tensor('edge_attr', lambda dev: torch.empty(0, bond_fdim).to(dev)),
                motif_x=get_tensor('motif_x', lambda dev: torch.empty(0, motif_fdim).to(dev)),
                motif_edge_index=get_tensor('motif_edge_index', lambda dev: torch.empty(2, 0).to(dev), dtype=torch.long),
                motif_edge_attr=get_tensor('motif_edge_attr', lambda dev: torch.empty(0, motif_edge_attr_fdim).to(dev))
            )
            if data_obj.motif_edge_attr is not None and data_obj.motif_edge_attr.shape[0] == 0:
                 data_obj.motif_edge_attr = None
            
            pyg_data_list.append(data_obj)

        batch = Batch.from_data_list(pyg_data_list)

        atom_features_encoded = self.atom_encoder(batch.x)
        
        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            bond_features_encoded = self.bond_encoder(batch.edge_attr)
        else:
            bond_features_encoded = torch.empty(0, self.hidden_dim).to(device)

        if batch.motif_x is not None and batch.motif_x.numel() > 0:
            motif_features_encoded = self.motif_encoder(batch.motif_x)
        else:
            motif_features_encoded = torch.empty(0, self.hidden_dim).to(device)

        atom_embeddings = atom_features_encoded
        for gcn in self.gcn_layers:
            # As per subtask point 6, edge_attr is not passed to gcn()
            atom_embeddings = F.relu(gcn(atom_embeddings, batch.edge_index))
        
        motif_embeddings = motif_features_encoded
        if batch.motif_x is not None and batch.motif_x.numel() > 0 :
            current_motif_edge_index = batch.motif_edge_index if batch.motif_edge_index is not None and batch.motif_edge_index.numel() > 0 else torch.empty(2,0).long().to(device)
            for gin in self.gin_layers:
                 # As per subtask point 6, edge_attr is not passed to gin()
                motif_embeddings = F.relu(gin(motif_embeddings, current_motif_edge_index))
        else:
             motif_embeddings = torch.empty(0, self.hidden_dim).to(device)

        atom_pooled = global_mean_pool(atom_embeddings, batch.batch)
        atom_attn_input = atom_pooled.unsqueeze(1) 
        atom_attn, _ = self.attention(atom_attn_input, atom_attn_input, atom_attn_input)
        atom_global = torch.mean(atom_attn, dim=1) # Point 7

        if motif_embeddings.numel() > 0:
            motif_batch_vector = batch.motif_batch if hasattr(batch, 'motif_batch') and batch.motif_batch is not None else None
            if motif_batch_vector is not None and motif_batch_vector.numel() > 0 :
                 motif_pooled = global_mean_pool(motif_embeddings, motif_batch_vector)
                 motif_attn_input = motif_pooled.unsqueeze(1)
                 motif_attn, _ = self.attention(motif_attn_input, motif_attn_input, motif_attn_input)
                 motif_global = torch.mean(motif_attn, dim=1) # Point 7
            else: 
                 motif_global = torch.zeros(batch.num_graphs, self.hidden_dim, device=device)
        else:
            motif_global = torch.zeros(batch.num_graphs, self.hidden_dim, device=device)
        
        combined = torch.cat([atom_global, motif_global], dim=1) # Point 8
        
        latent = self.projector(combined)
        
        return latent

class StructureDecoder(nn.Module):
    """潜在表現から化学構造を生成するモジュール"""
    
    def __init__(self, latent_dim, hidden_dim):
        super(StructureDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 潜在表現からの拡張
        self.expander = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (MAX_ATOMS + MAX_MOTIFS) * self.hidden_dim)
        )
        
        # グラフ生成モジュール
        self.graph_generator = nn.ModuleDict({
            # ノード（原子）の存在確率を予測
            'node_existence': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            
            # ノード（原子）の種類を予測
            'node_type': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 10)  # C, H, N, O, F, S, P, Cl, Br, I
            ),
            
            # エッジ（結合）の存在確率を予測
            'edge_existence': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            
            # エッジ（結合）の種類を予測
            'edge_type': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)  # 単結合、二重結合、三重結合、芳香族結合
            )
        })
        
        # モチーフ生成モジュール
        self.motif_generator = nn.ModuleDict({
            # モチーフの存在確率を予測
            'motif_existence': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            
            # モチーフの種類を予測
            'motif_type': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(MOTIF_TYPES))
            )
        })
    
    def forward(self, latent, max_atoms=MAX_ATOMS):
        """順伝播: 潜在表現から化学構造を生成"""
        batch_size = latent.size(0)
        device = latent.device

        # 潜在表現を拡張
        expanded_flat = self.expander(latent) # Shape: (batch_size, (MAX_ATOMS + MAX_MOTIFS) * self.hidden_dim)
        expanded = expanded_flat.view(batch_size, MAX_ATOMS + MAX_MOTIFS, self.hidden_dim)
        
        # ノード（原子）の特徴量を生成
        # Note: max_atoms from args might be different from MAX_ATOMS used in expander's output size.
        # We will slice up to the max_atoms used in generation, ensuring it does not exceed MAX_ATOMS.
        current_max_atoms = min(max_atoms, MAX_ATOMS)
        node_hiddens = expanded[:, :current_max_atoms, :]  # Shape: (batch_size, current_max_atoms, self.hidden_dim)
        
        # ノード（原子）の存在確率
        node_exists = self.graph_generator['node_existence'](node_hiddens) # Output: (batch_size, current_max_atoms, 1)
        
        # ノード（原子）の種類
        node_types = self.graph_generator['node_type'](node_hiddens) # Output: (batch_size, current_max_atoms, num_node_types)
        
        # エッジ（結合）の特徴量を生成 (Batch-aware)
        if current_max_atoms < 2:
            # Ensure correct shape for empty edge features, matching expected input dim for subsequent layers
            edge_hiddens = torch.empty(batch_size, 0, self.hidden_dim * 2, device=device)
            # Correspondingly, edge_exists and edge_types should also be empty but correctly shaped.
            # For edge_existence, output is (batch_size, num_edges, 1)
            edge_exists = torch.empty(batch_size, 0, 1, device=device)
            # For edge_type, output is (batch_size, num_edges, num_edge_types)
            # Assuming num_edge_types is the output dim of self.graph_generator['edge_type']'s last linear layer
            num_edge_types = self.graph_generator['edge_type'][-1].out_features
            edge_types = torch.empty(batch_size, 0, num_edge_types, device=device)
        else:
            idx_i, idx_j = torch.triu_indices(current_max_atoms, current_max_atoms, offset=1, device=device)
            h_i = node_hiddens[:, idx_i, :]  # Shape: (batch_size, num_edges, hidden_dim)
            h_j = node_hiddens[:, idx_j, :]  # Shape: (batch_size, num_edges, hidden_dim)
            edge_hiddens = torch.cat([h_i, h_j], dim=-1) # Shape: (batch_size, num_edges, hidden_dim * 2)
            
            # エッジ（結合）の存在確率
            edge_exists = self.graph_generator['edge_existence'](edge_hiddens) # Output: (batch_size, num_edges, 1)
            
            # エッジ（結合）の種類
            edge_types = self.graph_generator['edge_type'](edge_hiddens) # Output: (batch_size, num_edges, num_edge_types)

        # モチーフの特徴量を生成
        # Ensure MAX_MOTIFS is used for slicing consistently with expander
        if MAX_MOTIFS > 0:
            motif_hiddens = expanded[:, MAX_ATOMS : MAX_ATOMS + MAX_MOTIFS, :] # Shape: (batch_size, MAX_MOTIFS, self.hidden_dim)
            # モチーフの存在確率
            motif_exists = self.motif_generator['motif_existence'](motif_hiddens) # Output: (batch_size, MAX_MOTIFS, 1)
            # モチーフの種類
            motif_types = self.motif_generator['motif_type'](motif_hiddens) # Output: (batch_size, MAX_MOTIFS, num_motif_types)
        else:
            # Handle case with no motifs
            motif_hiddens = torch.empty(batch_size, 0, self.hidden_dim, device=device)
            motif_exists = torch.empty(batch_size, 0, 1, device=device)
            num_motif_types = self.motif_generator['motif_type'][-1].out_features
            motif_types = torch.empty(batch_size, 0, num_motif_types, device=device)
        
        return {
            'node_exists': node_exists,
            'node_types': node_types,
            'edge_exists': edge_exists,
            'edge_types': edge_types,
            'motif_exists': motif_exists,
            'motif_types': motif_types
        }

class SpectrumEncoder(nn.Module):
    """マススペクトルをエンコードするモジュール"""
    
    def __init__(self, spectrum_dim, hidden_dim, latent_dim):
        super(SpectrumEncoder, self).__init__()
        self.spectrum_dim = spectrum_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # スペクトル入力の次元削減
        self.dim_reducer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 特徴抽出用の変換器レイヤー
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 最終潜在表現への射影
        self.projector = nn.Sequential(
            nn.Linear(128 * (spectrum_dim // 8), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, spectrum):
        """順伝播: マススペクトルから潜在表現を生成"""
        # 入力形状を調整
        x = spectrum.unsqueeze(1)  # [batch_size, 1, spectrum_dim]
        
        # 次元削減
        x = self.dim_reducer(x)  # [batch_size, 128, spectrum_dim/8]
        
        # 特徴抽出
        x = x.transpose(1, 2)  # [batch_size, spectrum_dim/8, 128]
        x = self.transformer_encoder(x)  # [batch_size, spectrum_dim/8, 128]
        
        # 平坦化
        x = x.reshape(x.size(0), -1)  # [batch_size, 128 * (spectrum_dim/8)]
        
        # 潜在表現に射影
        latent = self.projector(x)  # [batch_size, latent_dim]
        
        return latent

class SpectrumDecoder(nn.Module):
    """潜在表現からマススペクトルを生成するモジュール"""
    
    def __init__(self, latent_dim, hidden_dim, spectrum_dim):
        super(SpectrumDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.spectrum_dim = spectrum_dim
        
        # 潜在表現からの拡張
        self.expander = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        # アップサンプリング
        self.upsampler = nn.Sequential(
            nn.Linear(hidden_dim * 2, spectrum_dim // 8 * 128),
            nn.ReLU(),
            nn.Unflatten(1, (spectrum_dim // 8, 128)),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()  # スペクトル強度は0-1に正規化
        )
    
    def forward(self, latent):
        """順伝播: 潜在表現からマススペクトルを生成"""
        # 潜在表現を拡張
        expanded = self.expander(latent)
        
        # アップサンプリングでスペクトルを生成
        spectrum = self.upsampler(expanded).squeeze(1)  # [batch_size, spectrum_dim]
        
        return spectrum

class StructureNoisePredictor(nn.Module):
    """化学構造の拡散モデル用ノイズ予測器"""
    
    def __init__(self, latent_dim, hidden_dim, time_dim=128):
        super(StructureNoisePredictor, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        # 時間埋め込み
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # ノイズ予測ネットワーク
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x, t):
        """順伝播: ノイズ予測"""
        # 時間埋め込み
        time_emb = self.time_mlp(t)
        
        # 入力と時間埋め込みを結合
        x_with_time = torch.cat([x, time_emb], dim=1)
        
        # ノイズ予測
        predicted_noise = self.noise_predictor(x_with_time)
        
        return predicted_noise

class SpectrumNoisePredictor(nn.Module):
    """マススペクトルの拡散モデル用ノイズ予測器"""
    
    def __init__(self, spectrum_dim, hidden_dim, time_dim=128):
        super(SpectrumNoisePredictor, self).__init__()
        self.spectrum_dim = spectrum_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        # 時間埋め込み
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 1D CNN特徴抽出
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        
        # 時間条件付き特徴処理
        self.time_processor = nn.Sequential(
            nn.Linear(128 + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # アップサンプリングでスペクトルノイズを予測
        self.noise_predictor = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=1, padding=3)
        )
    
    def forward(self, x, t):
        """順伝播: ノイズ予測"""
        # 入力形状を調整
        x = x.unsqueeze(1)  # [batch_size, 1, spectrum_dim]
        
        # 特徴抽出
        features = self.feature_extractor(x)  # [batch_size, 128, spectrum_dim]
        
        # 時間埋め込み
        time_emb = self.time_mlp(t).unsqueeze(2).expand(-1, -1, self.spectrum_dim)
        
        # 特徴と時間埋め込みを結合
        features_with_time = torch.cat([features, time_emb], dim=1)
        
        # 時間条件付き特徴処理
        processed = self.time_processor(features_with_time.permute(0, 2, 1)).permute(0, 2, 1)
        
        # ノイズ予測
        predicted_noise = self.noise_predictor(processed).squeeze(1)  # [batch_size, spectrum_dim]
        
        return predicted_noise

class SinusoidalPositionEmbeddings(nn.Module):
    """サイン波ベースの位置埋め込み"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#------------------------------------------------------
# 双方向自己成長型モデル
#------------------------------------------------------

class BidirectionalSelfGrowingModel(nn.Module):
    """構造-スペクトル間の双方向自己成長型モデル"""
    
    def __init__(self, atom_fdim, bond_fdim, motif_fdim, spectrum_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super(BidirectionalSelfGrowingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.spectrum_dim = spectrum_dim
        
        # 構造→スペクトル方向
        self.structure_encoder = StructureEncoder(atom_fdim, bond_fdim, motif_fdim, hidden_dim, latent_dim)
        self.spectrum_decoder = SpectrumDecoder(latent_dim, hidden_dim, spectrum_dim)
        
        # スペクトル→構造方向
        self.spectrum_encoder = SpectrumEncoder(spectrum_dim, hidden_dim, latent_dim)
        self.structure_decoder = StructureDecoder(latent_dim, hidden_dim)
        
        # 拡散モデル
        self.diffusion = DiffusionModel()
        
        # 構造用ノイズ予測器
        self.structure_noise_predictor = StructureNoisePredictor(latent_dim, hidden_dim)
        
        # スペクトル用ノイズ予測器
        self.spectrum_noise_predictor = SpectrumNoisePredictor(spectrum_dim, hidden_dim)
        
        # 潜在空間アライメント
        self.structure_to_spectrum_aligner = nn.Linear(latent_dim, latent_dim)
        self.spectrum_to_structure_aligner = nn.Linear(latent_dim, latent_dim)
    
    def structure_to_spectrum(self, structure_data):
        """構造からスペクトルを予測"""
        # 構造をエンコード
        latent = self.structure_encoder(structure_data)
        
        # 潜在表現を調整
        aligned_latent = self.structure_to_spectrum_aligner(latent)
        
        # スペクトルをデコード
        spectrum = self.spectrum_decoder(aligned_latent)
        
        return spectrum, aligned_latent
    
    def spectrum_to_structure(self, spectrum):
        """スペクトルから構造を予測"""
        # スペクトルをエンコード
        latent = self.spectrum_encoder(spectrum)
        
        # 潜在表現を調整
        aligned_latent = self.spectrum_to_structure_aligner(latent)
        
        # 構造をデコード
        structure = self.structure_decoder(aligned_latent)
        
        return structure, aligned_latent
    
    def forward(self, data, direction="bidirectional"):
        """順伝播"""
        if not isinstance(data, dict):
            raise TypeError(f"Input 'data' must be a dictionary, got {type(data)}")

        results = {}
        
        if direction in ["structure_to_spectrum", "bidirectional"]:
            if "structure" not in data:
                raise ValueError("Missing 'structure' key in data for structure_to_spectrum")
            if not isinstance(data["structure"], list):
                raise TypeError(f"data['structure'] must be a list of MoleculeData objects, got {type(data['structure'])}")
            # Optional: Check type of first element if list is not empty
            if data["structure"] and not hasattr(data["structure"][0], 'mol'): # Check for a known attribute like 'mol'
                raise TypeError(f"Elements of data['structure'] must be objects with a 'mol' attribute, got {type(data['structure'][0])}")

            # 構造→スペクトル方向
            predicted_spectrum, structure_latent = self.structure_to_spectrum(data["structure"])
            results["predicted_spectrum"] = predicted_spectrum
            results["structure_latent"] = structure_latent
        
        if direction in ["spectrum_to_structure", "bidirectional"]:
            if "spectrum" not in data:
                raise ValueError("Missing 'spectrum' key in data for spectrum_to_structure")
            if not isinstance(data["spectrum"], torch.Tensor):
                raise TypeError(f"data['spectrum'] must be a PyTorch tensor, got {type(data['spectrum'])}")

            # スペクトル→構造方向
            predicted_structure, spectrum_latent = self.spectrum_to_structure(data["spectrum"])
            results["predicted_structure"] = predicted_structure
            results["spectrum_latent"] = spectrum_latent
        
        return results
    
    def diffusion_training_step(self, x, domain="structure"):
        """拡散モデルのトレーニングステップ"""
        # バッチサイズ
        batch_size = x.shape[0]
        
        # ランダムなタイムステップ
        t = torch.randint(0, self.diffusion.num_steps, (batch_size,), device=x.device, dtype=torch.long)
        
        # ノイズを追加
        x_noisy, noise = self.diffusion.q_sample(x, t)
        
        # ノイズを予測
        if domain == "structure":
            predicted_noise = self.structure_noise_predictor(x_noisy, t)
        else:  # domain == "spectrum"
            predicted_noise = self.spectrum_noise_predictor(x_noisy, t)
        
        # 損失を計算
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def sample_structure(self, batch_size=1, device=device):
        """構造潜在表現のサンプリング"""
        shape = (batch_size, self.latent_dim)
        return self.diffusion.p_sample_loop(self.structure_noise_predictor, shape, device)
    
    def sample_spectrum(self, batch_size=1, device=device):
        """スペクトル潜在表現のサンプリング"""
        shape = (batch_size, self.spectrum_dim)
        return self.diffusion.p_sample_loop(self.spectrum_noise_predictor, shape, device)
    
    def cycle_consistency_loss(self, structure_data, spectrum):
        """サイクル一貫性損失の計算"""
        # 構造→スペクトル→構造 サイクル
        predicted_spectrum, structure_latent = self.structure_to_spectrum(structure_data)
        predicted_structure, _ = self.spectrum_to_structure(predicted_spectrum)
        
        # スペクトル→構造→スペクトル サイクル
        predicted_structure2, spectrum_latent = self.spectrum_to_structure(spectrum)
        predicted_spectrum2, _ = self.structure_to_spectrum(structure_data)
        
        # 構造サイクル損失
        structure_cycle_loss = F.mse_loss(
            predicted_structure["node_exists"], 
            structure_data["node_exists"]
        )
        
        # スペクトルサイクル損失
        spectrum_cycle_loss = F.mse_loss(predicted_spectrum2, spectrum)
        
        # 潜在表現の一貫性損失
        latent_consistency_loss = F.mse_loss(structure_latent, spectrum_latent)
        
        return structure_cycle_loss + spectrum_cycle_loss + 0.1 * latent_consistency_loss

# --- Content from gsai0501-2.py ---

# ロギング設定
log_filename = f"self_growing_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfGrowingModel")

#------------------------------------------------------
# データ構造とデータセット
#------------------------------------------------------

class MoleculeData:
    """分子データを処理するクラス"""
    
    def __init__(self, mol, spectrum=None):
        """
        分子データを初期化
        
        Args:
            mol: RDKit分子オブジェクト
            spectrum: 分子のマススペクトル（あれば）
        """
        self.mol = mol
        self.spectrum = spectrum
        
        # 分子の基本情報
        self.smiles = Chem.MolToSmiles(mol)
        self.formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        
        # 原子と結合の特徴量
        self.atom_features = self._get_atom_features()
        self.bond_features, self.adjacency_list = self._get_bond_features_and_adjacency()
        
        # モチーフの抽出と特徴量
        self.motifs, self.motif_types = self._extract_motifs()
        self.motif_features = self._get_motif_features()
        self.motif_graph, self.motif_edge_features = self._build_motif_graph()
        
        # グラフデータ構造
        self.graph_data = self._build_graph_data()
    
    def _get_atom_features(self):
        """原子の特徴量を抽出"""
        features = []
        for atom in self.mol.GetAtoms():
            # 原子番号（one-hot）
            atom_type = atom.GetAtomicNum()
            atom_type_oh = [0] * 119
            atom_type_oh[atom_type] = 1
            
            # 形式電荷
            charge = atom.GetFormalCharge()
            charge_oh = [0] * 11  # -5 ~ +5
            charge_oh[charge + 5] = 1
            
            # 混成軌道状態
            hybridization = atom.GetHybridization()
            hybridization_types = [Chem.rdchem.HybridizationType.SP, 
                                  Chem.rdchem.HybridizationType.SP2,
                                  Chem.rdchem.HybridizationType.SP3, 
                                  Chem.rdchem.HybridizationType.SP3D, 
                                  Chem.rdchem.HybridizationType.SP3D2]
            hybridization_oh = [int(hybridization == i) for i in hybridization_types]
            
            # 水素の数
            h_count = atom.GetTotalNumHs()
            h_count_oh = [0] * 9
            h_count_oh[min(h_count, 8)] = 1
            
            # 特性フラグ
            is_aromatic = atom.GetIsAromatic()
            is_in_ring = atom.IsInRing()
            
            # 特徴量を結合
            atom_features = atom_type_oh + charge_oh + hybridization_oh + h_count_oh + [is_aromatic, is_in_ring]
            features.append(atom_features)
        
        return np.array(features, dtype=np.float32)
    
    def _get_bond_features_and_adjacency(self):
        """結合の特徴量と隣接リストを取得"""
        bond_features = []
        adjacency_list = [[] for _ in range(self.mol.GetNumAtoms())]
        
        for bond in self.mol.GetBonds():
            # 結合のインデックス
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # 結合タイプ（one-hot）
            bond_type = bond.GetBondType()
            bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
            bond_type_oh = [int(bond_type == i) for i in bond_types]
            
            # 特性フラグ
            is_in_ring = bond.IsInRing()
            is_conjugated = bond.GetIsConjugated()
            is_stereo = bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE
            
            # 特徴量を結合
            bond_feature = bond_type_oh + [is_in_ring, is_conjugated, is_stereo]
            bond_features.append(bond_feature)
            
            # 隣接リストに追加
            bond_idx = len(bond_features) - 1
            adjacency_list[begin_idx].append((end_idx, bond_idx))
            adjacency_list[end_idx].append((begin_idx, bond_idx))
        
        return np.array(bond_features, dtype=np.float32), adjacency_list
    
    def _extract_motifs(self, motif_size_threshold=3, max_motifs=MAX_MOTIFS):
        """分子からモチーフを抽出"""
        motifs = []
        motif_types = []
        
        # 1. BRICS分解によるモチーフ抽出
        try:
            brics_frags = list(BRICS.BRICSDecompose(self.mol, keepNonLeafNodes=True))
            for frag_smiles in brics_frags:
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol and frag_mol.GetNumAtoms() >= motif_size_threshold:
                    # モチーフに含まれる原子のインデックスを特定
                    substructure = self.mol.GetSubstructMatch(frag_mol)
                    if substructure and len(substructure) > 0:
                        motifs.append(list(substructure))
                        
                        # モチーフタイプを判定
                        motif_type = self._determine_motif_type(frag_mol)
                        motif_types.append(motif_type)
        except:
            pass  # BRICSが失敗する場合はスキップ
        
        # 2. 環系モチーフの抽出
        try:
            rings = self.mol.GetSSSR()
            for ring in rings:
                if len(ring) >= motif_size_threshold:
                    ring_atoms = list(ring)
                    if ring_atoms not in motifs:
                        motifs.append(ring_atoms)
                        
                        # 環タイプを判定
                        ring_mol = Chem.PathToSubmol(self.mol, ring, atomMap={})
                        ring_type = "aromatic" if any(atom.GetIsAromatic() for atom in ring_mol.GetAtoms()) else "aliphatic_ring"
                        motif_types.append(ring_type)
        except:
            pass  # 環抽出が失敗する場合はスキップ
        
        # 3. 機能性グループの抽出
        functional_groups = {
            "carboxyl": "[CX3](=O)[OX2H1]",
            "hydroxyl": "[OX2H]",
            "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
            "amide": "[NX3][CX3](=[OX1])",
            "ether": "[OD2]([#6])[#6]",
            "ester": "[#6][CX3](=O)[OX2][#6]",
            "carbonyl": "[CX3]=[OX1]"
        }
        
        for group_name, smarts in functional_groups.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = self.mol.GetSubstructMatches(pattern)
                    for match in matches:
                        if len(match) >= motif_size_threshold and list(match) not in motifs:
                            motifs.append(list(match))
                            motif_types.append(group_name)
            except:
                continue  # パターンマッチングが失敗する場合はスキップ
        
        # 最大モチーフ数を制限
        if len(motifs) > max_motifs:
            # サイズで並べ替えて大きいものを優先
            sorted_pairs = sorted(zip(motifs, motif_types), key=lambda x: len(x[0]), reverse=True)
            motifs, motif_types = zip(*sorted_pairs[:max_motifs])
            motifs, motif_types = list(motifs), list(motif_types)
        
        return motifs, motif_types
    
    def _determine_motif_type(self, motif_mol):
        """モチーフの化学的タイプを判定"""
        # デフォルトタイプ
        motif_type = "other"
        
        # 芳香族環の検出
        if any(atom.GetIsAromatic() for atom in motif_mol.GetAtoms()):
            motif_type = "aromatic"
            
            # ヘテロ環の検出
            if any(atom.GetAtomicNum() != 6 for atom in motif_mol.GetAtoms() if atom.IsInRing()):
                motif_type = "heterocycle"
        
        # 官能基の検出
        functional_groups = {
            "ester": "[#6][CX3](=O)[OX2][#6]",
            "amide": "[NX3][CX3](=[OX1])",
            "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
            "urea": "[NX3][CX3](=[OX1])[NX3]",
            "ether": "[OD2]([#6])[#6]",
            "olefin": "[CX3]=[CX3]",
            "carbonyl": "[CX3]=[OX1]",
            "lactam": "[NX3R][CX3R](=[OX1])",
            "lactone": "[#6R][CX3R](=[OX1])[OX2R][#6R]"
        }
        
        for group_name, smarts in functional_groups.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and motif_mol.HasSubstructMatch(pattern):
                    motif_type = group_name
                    break
            except:
                continue
                
        return motif_type
    
    def _get_motif_features(self):
        """モチーフの特徴量を計算"""
        features = []
        
        for i, (motif, motif_type) in enumerate(zip(self.motifs, self.motif_types)):
            # 基本特徴量
            size = len(motif) / self.mol.GetNumAtoms()  # 正規化サイズ
            
            # モチーフタイプ（one-hot）
            type_oh = [0] * len(MOTIF_TYPES)
            if motif_type in MOTIF_TYPES:
                type_oh[MOTIF_TYPES.index(motif_type)] = 1
            
            # 環構造フラグ
            is_ring = all(self.mol.GetAtomWithIdx(atom_idx).IsInRing() for atom_idx in motif)
            
            # 芳香族フラグ
            is_aromatic = any(self.mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in motif)
            
            # ヘテロ原子を含むかのフラグ
            has_heteroatom = any(self.mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 6 for atom_idx in motif)
            
            # 特徴量を結合
            motif_features = [size] + type_oh + [is_ring, is_aromatic, has_heteroatom]
            features.append(motif_features)
        
        # モチーフがない場合は空の配列を返す
        if not features:
            return np.zeros((0, 1 + len(MOTIF_TYPES) + 3), dtype=np.float32)
        
        return np.array(features, dtype=np.float32)
    
    def _build_motif_graph(self):
        """モチーフグラフと特徴量を構築"""
        n_motifs = len(self.motifs)
        motif_edges = []
        motif_edge_features = []
        
        # 各モチーフペアについて処理
        for i in range(n_motifs):
            for j in range(i+1, n_motifs):
                # モチーフ間に共有原子があるか確認
                shared_atoms = set(self.motifs[i]) & set(self.motifs[j])
                has_shared_atoms = len(shared_atoms) > 0
                
                # モチーフ間に結合があるか確認
                boundary_bonds = []
                for atom_i in self.motifs[i]:
                    for atom_j in self.motifs[j]:
                        bond = self.mol.GetBondBetweenAtoms(atom_i, atom_j)
                        if bond is not None:
                            boundary_bonds.append(bond)
                
                has_bond = len(boundary_bonds) > 0
                
                # モチーフ間に接続があれば（共有原子または結合）、エッジを追加
                if has_shared_atoms or has_bond:
                    motif_edges.append((i, j))
                    
                    # エッジ特徴量を計算
                    n_shared_atoms = len(shared_atoms) / 10.0  # 正規化
                    n_bonds = len(boundary_bonds) / 5.0  # 正規化
                    
                    # 結合タイプのカウント
                    bond_type_counts = [0] * 4  # SINGLE, DOUBLE, TRIPLE, AROMATIC
                    for bond in boundary_bonds:
                        bond_type = bond.GetBondType()
                        if bond_type == BondType.SINGLE:
                            bond_type_counts[0] += 1
                        elif bond_type == BondType.DOUBLE:
                            bond_type_counts[1] += 1
                        elif bond_type == BondType.TRIPLE:
                            bond_type_counts[2] += 1
                        elif bond_type == BondType.AROMATIC:
                            bond_type_counts[3] += 1
                    
                    # 結合タイプの割合を計算
                    if boundary_bonds:
                        bond_type_ratios = [count / len(boundary_bonds) for count in bond_type_counts]
                    else:
                        bond_type_ratios = [0, 0, 0, 0]
                    
                    # エッジ特徴量を結合
                    edge_features = [n_shared_atoms, n_bonds] + bond_type_ratios
                    motif_edge_features.append(edge_features)
        
        return motif_edges, np.array(motif_edge_features, dtype=np.float32) if motif_edge_features else np.zeros((0, 6), dtype=np.float32)
    
    def _build_graph_data(self):
        """PyTorch Geometricのグラフデータ構造を構築"""
        # 原子特徴量
        x = torch.FloatTensor(self.atom_features)
        
        # 結合インデックス（エッジインデックス）
        edge_index = []
        for i, neighbors in enumerate(self.adjacency_list):
            for j, _ in neighbors:
                edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        
        # 結合特徴量
        if len(self.bond_features) > 0:
    edge_attr = torch.FloatTensor(self.bond_features)
else:
    # Assuming bond_feature_dim is 7 (bond_type_oh[4] + 3 flags)
    edge_attr = torch.zeros((0, 7), dtype=torch.float32)
        
        # モチーフインデックス
        motif_index = []
        for i, motif in enumerate(self.motifs):
            for atom in motif:
                motif_index.append([atom, i])
        
        motif_index = torch.tensor(motif_index, dtype=torch.long).t().contiguous() if motif_index else torch.zeros((2, 0), dtype=torch.long)
        
        # モチーフ特徴量
        motif_x = torch.FloatTensor(self.motif_features) if len(self.motif_features) > 0 else torch.zeros((0, self.motif_features.shape[1] if self.motif_features.shape[0] > 0 else 1 + len(MOTIF_TYPES) + 3))
        
        # モチーフエッジインデックス
        motif_edge_index = []
        for i, j in self.motif_graph:
            motif_edge_index.append([i, j])
            motif_edge_index.append([j, i])  # 両方向
        
        motif_edge_index = torch.tensor(motif_edge_index, dtype=torch.long).t().contiguous() if motif_edge_index else torch.zeros((2, 0), dtype=torch.long)
        
        # モチーフエッジ特徴量
        if self.motif_edge_features.size > 0: # Check if numpy array is not empty (size > 0 for (0,6) is false)
            # self.motif_edge_features would be (num_edges, 6)
            motif_edge_attr = torch.FloatTensor(self.motif_edge_features)
        else:
            # self.motif_edge_features would be an array like np.zeros((0,6))
            # The number of features is 6.
            motif_edge_attr = torch.zeros((0, 6), dtype=torch.float32)
        
        # スペクトル
        spectrum = torch.FloatTensor(self.spectrum) if self.spectrum is not None else None
        
        # グラフデータを構築
        data = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'motif_index': motif_index,
            'motif_x': motif_x,
            'motif_edge_index': motif_edge_index,
            'motif_edge_attr': motif_edge_attr,
            'spectrum': spectrum,
            'smiles': self.smiles,
            'formula': self.formula
        }
        
        return data

def normalize_spectrum(peaks: List[Tuple[int, int]], max_mz: int = SPECTRUM_DIM, threshold: float = 0.01, top_n: int = 20) -> np.ndarray:
    """マススペクトルを正規化してベクトル形式に変換"""
    spectrum = np.zeros(max_mz)
    
    # ピークがない場合は空のスペクトルを返す
    if not peaks:
        return spectrum
    
    # 最大強度を見つける
    max_intensity = max([intensity for mz, intensity in peaks if mz < max_mz])
    if max_intensity <= 0:
        return spectrum
    
    # 相対強度の閾値を計算
    intensity_threshold = max_intensity * threshold
    
    # 閾値以上のピークを抽出
    filtered_peaks = [(mz, intensity) for mz, intensity in peaks 
                     if mz < max_mz and intensity >= intensity_threshold]
    
    # 上位N個のピークのみを保持
    if top_n > 0 and len(filtered_peaks) > top_n:
        # 強度の降順でソート
        filtered_peaks.sort(key=lambda x: x[1], reverse=True)
        # 上位N個のみを保持
        filtered_peaks = filtered_peaks[:top_n]
    
    # 選択されたピークをスペクトルに設定
    for mz, intensity in filtered_peaks:
        spectrum[mz] = intensity / max_intensity
    
    return spectrum

class ChemicalStructureSpectumDataset(Dataset):
    """化学構造とマススペクトルのデータセット"""
    
    def __init__(self, structures=None, spectra=None, structure_spectrum_pairs=None):
        """
        データセットを初期化
        
        Args:
            structures: 構造のリスト（教師なしデータ）
            spectra: スペクトルのリスト（教師なしデータ）
            structure_spectrum_pairs: 構造とスペクトルのペアのリスト（教師ありデータ）
        """
        self.structures = structures or []
        self.spectra = spectra or []
        self.structure_spectrum_pairs = structure_spectrum_pairs or []
        
        # データのインデックス管理
        self.n_pairs = len(self.structure_spectrum_pairs)
        self.n_structures = len(self.structures)
        self.n_spectra = len(self.spectra)
        self.total = self.n_pairs + self.n_structures + self.n_spectra
    
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        # 教師ありデータ（構造-スペクトルペア）
        if idx < self.n_pairs:
            structure, spectrum = self.structure_spectrum_pairs[idx]
            return {
                'type': 'supervised',
                'structure': structure,
                'spectrum': spectrum
            }
        
        # 教師なし構造データ
        elif idx < self.n_pairs + self.n_structures:
            structure = self.structures[idx - self.n_pairs]
            return {
                'type': 'unsupervised_structure',
                'structure': structure,
                'spectrum': None
            }
        
        # 教師なしスペクトルデータ
        else:
            spectrum = self.spectra[idx - self.n_pairs - self.n_structures]
            return {
                'type': 'unsupervised_spectrum',
                'structure': None,
                'spectrum': spectrum
            }
    
    def add_structure_spectrum_pair(self, structure, spectrum):
        """構造-スペクトルペアを追加"""
        self.structure_spectrum_pairs.append((structure, spectrum))
        self.n_pairs += 1
        self.total += 1
    
    def add_structure(self, structure):
        """構造を追加"""
        self.structures.append(structure)
        self.n_structures += 1
        self.total += 1
    
    def add_spectrum(self, spectrum):
        """スペクトルを追加"""
        self.spectra.append(spectrum)
        self.n_spectra += 1
        self.total += 1

def collate_fn(batch):
    """バッチ処理用の関数"""
    batch_dict = {
        'type': [],
        'structure': [],
        'spectrum': []
    }
    
    for data in batch:
        for key in batch_dict:
            batch_dict[key].append(data[key])
    
    # スペクトルをスタック（あれば）
    spectra_processed = []
    # batch_dict['spectrum'] contains a list of items, where each item can be a spectrum or None
    for item in batch_dict['spectrum']: 
        if item is not None:
            if isinstance(item, np.ndarray):
                spectra_processed.append(torch.FloatTensor(item))
            elif isinstance(item, torch.Tensor):
                # Ensure it's a FloatTensor if it's not already
                spectra_processed.append(item.float() if not item.is_floating_point() else item)
            # Add other potential type checks if necessary, e.g. for lists of numbers
            # else:
            #     # Potentially handle other types or raise an error
            #     pass # For now, only ndarray and Tensor are explicitly handled as per plan

    if spectra_processed:
        try:
            batch_dict['spectrum_tensor'] = torch.stack(spectra_processed)
        except Exception as e:
            # Log error or handle cases where stacking might fail (e.g. inconsistent tensor shapes)
            # For now, re-raise or set to None as per original logic for failure
            print(f"Error stacking spectra in collate_fn: {e}") # Or use logger
            batch_dict['spectrum_tensor'] = None 
    else:
        batch_dict['spectrum_tensor'] = None
    
    return batch_dict

#------------------------------------------------------
# 自己成長トレーニングループとアルゴリズム
#------------------------------------------------------

class SelfGrowingTrainer:
    """自己成長型モデルのトレーナー"""
    
    def __init__(self, model, device, config):
        """
        トレーナーを初期化
        
        Args:
            model: 双方向自己成長型モデル
            device: トレーニングに使用するデバイス
            config: トレーニング設定
        """
        self.model = model
        self.device = device
        self.config = config
        
        # オプティマイザ
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
        # 学習率スケジューラ
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # メトリクス追跡
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'structure_to_spectrum_loss': [],
            'spectrum_to_structure_loss': [],
            'cycle_consistency_loss': [],
            'diffusion_loss': [],
            'pseudo_labeling_accuracy': []
        }
        
        # 疑似ラベル付けで使用する信頼度閾値
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
        # サイクル一貫性のウェイト
        self.cycle_weight = config.get('cycle_weight', 1.0)
        
        # 拡散モデルのウェイト
        self.diffusion_weight = config.get('diffusion_weight', 0.1)

    def _create_structural_targets(self, molecule_data_list: List[MoleculeData], max_atoms_target: int, device: torch.device) -> Dict[str, torch.Tensor]:
        batch_node_exists = []
        batch_node_types = []

        for md in molecule_data_list:
            num_actual_atoms = md.mol.GetNumAtoms()
            
            # Node exists target
            node_exists_single = torch.zeros(max_atoms_target, 1, device=device)
            if num_actual_atoms > 0: # Ensure slicing is valid
                node_exists_single[:min(num_actual_atoms, max_atoms_target)] = 1.0
            batch_node_exists.append(node_exists_single)

            # Node types target
            node_types_single = torch.full((max_atoms_target,), UNKNOWN_ATOM_INDEX_TARGET, dtype=torch.long, device=device) 
            for i in range(min(num_actual_atoms, max_atoms_target)): # Iterate up to padding limit
                atom = md.mol.GetAtomWithIdx(i)
                atomic_num = atom.GetAtomicNum()
                node_types_single[i] = ATOM_TYPE_TO_INDEX.get(atomic_num, UNKNOWN_ATOM_INDEX_TARGET)
            batch_node_types.append(node_types_single)

        return {
            'node_exists': torch.stack(batch_node_exists), # (B, max_atoms_target, 1)
            'node_types': torch.stack(batch_node_types)    # (B, max_atoms_target)
        }
    
    def train_supervised(self, dataloader, epochs=1):
        """教師あり学習"""
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Supervised Training Epoch {epoch+1}/{epochs}"):
                # 教師ありデータのみを処理
                supervised_indices = [i for i, t in enumerate(batch['type']) if t == 'supervised']
                if not supervised_indices:
                    continue
                
                # バッチから教師ありデータを抽出
                supervised_structures = [batch['structure'][i] for i in supervised_indices]
                supervised_spectra = torch.stack([batch['spectrum'][i] for i in supervised_indices]).to(self.device)
                
                # データを辞書にまとめる
                data = {
                    'structure': supervised_structures,
                    'spectrum': supervised_spectra
                }
                
                # 順伝播
                self.optimizer.zero_grad()
                outputs = self.model(data, direction="bidirectional")
                
                # 損失計算
                loss_s2p = F.mse_loss(outputs['predicted_spectrum'], supervised_spectra)

                # Create structural targets
                structural_targets = self._create_structural_targets(supervised_structures, MAX_ATOMS, self.device)
                target_node_exists = structural_targets['node_exists']
                target_node_types = structural_targets['node_types']

                predicted_structure = outputs['predicted_structure']
                
                node_exists_loss = F.binary_cross_entropy(
                    predicted_structure['node_exists'],
                    target_node_exists
                )
                
                predicted_node_types_for_loss = predicted_structure['node_types'].permute(0, 2, 1)
                node_types_loss = F.cross_entropy(
                    predicted_node_types_for_loss,
                    target_node_types,
                    ignore_index=UNKNOWN_ATOM_INDEX_TARGET
                )
                
                loss_p2s = node_exists_loss + node_types_loss
                
                # 合計損失
                loss = loss_s2p + loss_p2s
                
                # 勾配計算と最適化
                loss.backward()
                self.optimizer.step()
                
                # 損失を追跡
                epoch_loss += loss.item()
                
                # メトリクスに追加
                self.metrics['structure_to_spectrum_loss'].append(loss_s2p.item())
                # Ensure loss_p2s is a scalar item for logging
                self.metrics['spectrum_to_structure_loss'].append(loss_p2s.item() if torch.is_tensor(loss_p2s) else loss_p2s)

            avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            total_loss += avg_epoch_loss
            
            logger.info(f"Supervised Epoch {epoch+1}/{epochs} Loss: {avg_epoch_loss:.4f}")
        
        # 平均損失を返す
        return total_loss / epochs if epochs > 0 else 0.0
    
    def train_cycle_consistency(self, dataloader, epochs=1):
        """サイクル一貫性を使った自己教師あり学習"""
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Cycle Consistency Training Epoch {epoch+1}/{epochs}"):
                # すべてのデータを処理
                structures = [s for s in batch['structure'] if s is not None]
                spectra = [s for s in batch['spectrum'] if s is not None]
                
                if not structures or not spectra:
                    continue
                
                # 順伝播
                self.optimizer.zero_grad()
                
                # サイクル一貫性損失の計算
                cycle_losses = []
                
                # 各構造-スペクトルペアについてサイクル一貫性を計算
                for structure in structures:
                    for spectrum in spectra:
                        spectrum_tensor = torch.FloatTensor(spectrum).to(self.device)
                        cycle_loss = self.model.cycle_consistency_loss(structure, spectrum_tensor)
                        cycle_losses.append(cycle_loss)
                
                # 平均サイクル損失
                if cycle_losses:
                    avg_cycle_loss = sum(cycle_losses) / len(cycle_losses)
                    weighted_loss = self.cycle_weight * avg_cycle_loss
                    
                    # 勾配計算と最適化
                    weighted_loss.backward()
                    self.optimizer.step()
                    
                    # 損失を追跡
                    epoch_loss += weighted_loss.item()
                    
                    # メトリクスに追加
                    self.metrics['cycle_consistency_loss'].append(avg_cycle_loss.item())
            
            avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            total_loss += avg_epoch_loss
            
            logger.info(f"Cycle Consistency Epoch {epoch+1}/{epochs} Loss: {avg_epoch_loss:.4f}")
        
        # 平均損失を返す
        return total_loss / epochs
    
    def train_diffusion(self, dataloader, epochs=1):
        """拡散モデルの訓練"""
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Diffusion Training Epoch {epoch+1}/{epochs}"):
                # 構造とスペクトルのデータを取得
                structures = [s for s in batch['structure'] if s is not None]
                spectra = [s for s in batch['spectrum'] if s is not None]
                
                # 構造潜在表現の拡散訓練
                if structures:
                    structure_latents = []
                    for structure in structures:
                        with torch.no_grad():
                            latent = self.model.structure_encoder(structure)
                        structure_latents.append(latent)
                    
                    structure_latents = torch.stack(structure_latents).to(self.device)
                    
                    # 拡散損失の計算
                    self.optimizer.zero_grad()
                    structure_diffusion_loss = self.model.diffusion_training_step(structure_latents, domain="structure")
                    
                    # 勾配計算と最適化
                    weighted_loss = self.diffusion_weight * structure_diffusion_loss
                    weighted_loss.backward()
                    self.optimizer.step()
                    
                    # 損失を追跡
                    epoch_loss += weighted_loss.item()
                
                # スペクトル潜在表現の拡散訓練
                if spectra:
                    spectrum_tensors = torch.stack([torch.FloatTensor(s) for s in spectra]).to(self.device)
                    
                    # 拡散損失の計算
                    self.optimizer.zero_grad()
                    spectrum_diffusion_loss = self.model.diffusion_training_step(spectrum_tensors, domain="spectrum")
                    
                    # 勾配計算と最適化
                    weighted_loss = self.diffusion_weight * spectrum_diffusion_loss
                    weighted_loss.backward()
                    self.optimizer.step()
                    
                    # 損失を追跡
                    epoch_loss += weighted_loss.item()
                    
                    # メトリクスに追加
                    self.metrics['diffusion_loss'].append((structure_diffusion_loss.item() if structures else 0) +
                                                         (spectrum_diffusion_loss.item() if spectra else 0))
            
            avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            total_loss += avg_epoch_loss
            
            logger.info(f"Diffusion Epoch {epoch+1}/{epochs} Loss: {avg_epoch_loss:.4f}")
        
        # 平均損失を返す
        return total_loss / epochs
    
    def generate_pseudo_labels(self, dataloader):
        """疑似ラベルを生成"""
        self.model.eval()
        pseudo_labeled_data = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating Pseudo Labels"):
                # 構造のみのデータを処理
                structure_indices = [i for i, t in enumerate(batch['type']) 
                                     if t == 'unsupervised_structure']
                
                for idx in structure_indices:
                    structure = batch['structure'][idx] # structure is a single MoleculeData obj
                    
                    # 構造からスペクトルを予測
                    # Pass as a list (batch of 1)
                    structure_data_for_model = {'structure': [structure]} 
                    outputs = self.model(structure_data_for_model, direction="structure_to_spectrum")
                    # outputs['predicted_spectrum'] will have shape (1, spectrum_dim)
                    single_predicted_spectrum = outputs['predicted_spectrum'][0] # Get the first (only) item
                    
                    # For confidence calculation
                    conf_input_dict = {'predicted_spectrum': single_predicted_spectrum}
                    if 'structure_latent' in outputs: # If latent is also batched (1, latent_dim)
                        conf_input_dict['structure_latent'] = outputs['structure_latent'][0]
                    
                    conf = self._calculate_confidence(conf_input_dict, "spectrum")

                    # 疑似ラベルとして追加
                    pseudo_labeled_data.append({
                        'structure': structure, # This is the original single MoleculeData
                        'spectrum': single_predicted_spectrum.cpu().numpy(), # Use the unbatched tensor
                        'confidence': conf
                    })
                
                # スペクトルのみのデータを処理
                spectrum_indices = [i for i, t in enumerate(batch['type']) 
                                   if t == 'unsupervised_spectrum']
                
                for idx in spectrum_indices:
                    spectrum = torch.FloatTensor(batch['spectrum'][idx]).to(self.device)
                    
                    # スペクトルから構造を予測
                    # The input spectrum is already a single tensor here from torch.FloatTensor(batch['spectrum'][idx]).to(self.device)
                    # Model's spectrum_to_structure expects a batched tensor.
                    spectrum_data_for_model = {'spectrum': spectrum.unsqueeze(0)} # Add batch dimension
                    outputs = self.model(spectrum_data_for_model, direction="spectrum_to_structure")
                    
                    # outputs['predicted_structure'] is a dict of tensors, each (1, MAX_ATOMS, Features) or (1, num_edges, Features)
                    # We need to extract the prediction for the single input.
                    single_predicted_structure_dict = {}
                    for key, value_batch_tensor in outputs['predicted_structure'].items():
                        single_predicted_structure_dict[key] = value_batch_tensor[0] # Take the first item

                    converted_mol = self._convert_to_molecule(single_predicted_structure_dict)
                    
                    conf_input_dict = {'predicted_structure': single_predicted_structure_dict}
                    if 'spectrum_latent' in outputs: # If latent is also batched (1, latent_dim)
                         conf_input_dict['spectrum_latent'] = outputs['spectrum_latent'][0]

                    conf = self._calculate_confidence(conf_input_dict, "structure")
                    
                    # 疑似ラベルとして追加
                    pseudo_labeled_data.append({
                        'structure': converted_mol, # This is an RDKit Mol object
                        'spectrum': batch['spectrum'][idx], # Original spectrum (numpy array or list)
                        'confidence': conf
                    })
        
        return pseudo_labeled_data
    
    def filter_high_confidence_pseudo_labels(self, pseudo_labeled_data):
        """高信頼度の疑似ラベルをフィルタリング"""
        filtered_data = [
            (data['structure'], data['spectrum'])
            for data in pseudo_labeled_data
            if data['confidence'] >= self.confidence_threshold
        ]
        
        logger.info(f"Filtered {len(filtered_data)} high confidence pseudo labels out of {len(pseudo_labeled_data)}")
        
        return filtered_data
    
    def _calculate_confidence(self, single_prediction_output: Dict, prediction_type: str) -> float:
        """予測の信頼度を計算 (単一の予測結果に対して)"""
        if prediction_type == "spectrum":
            # スペクトル予測の場合
            # single_prediction_output is {'predicted_spectrum': single_spectrum_tensor}
            single_spectrum_tensor = single_prediction_output['predicted_spectrum'] # Shape: (spectrum_dim,)
            if single_spectrum_tensor.numel() == 0: # Handle empty spectrum
                return 0.0
            entropy = -(single_spectrum_tensor * torch.log(single_spectrum_tensor + 1e-9)).sum() # Use 1e-9 for stability
            max_entropy = -torch.log(torch.tensor(1.0 / single_spectrum_tensor.size(0), device=single_spectrum_tensor.device))
            if max_entropy == 0: # Avoid division by zero if spectrum_dim is 1 (though unlikely for spectra)
                return 0.0 if entropy > 0 else 1.0
            confidence = 1.0 - (entropy / max_entropy)
            return confidence.item()

        elif prediction_type == "structure":
            # 構造予測の場合
            # single_prediction_output is {'predicted_structure': single_predicted_structure_dict}
            single_predicted_structure_dict = single_prediction_output['predicted_structure']
            node_exists = single_predicted_structure_dict['node_exists'] # Shape: (MAX_ATOMS, 1) or (MAX_ATOMS,)
            
            if node_exists.numel() == 0: # Handle empty node_exists (e.g. MAX_ATOMS = 0, though unlikely)
                return 0.0

            # Ensure node_exists is 1D for entropy calculation if it's (N,1)
            if node_exists.ndim > 1 and node_exists.shape[-1] == 1:
                node_exists = node_exists.squeeze(-1)

            # Clamp node_exists probabilities to avoid log(0) for perfect predictions
            node_exists_clamped = torch.clamp(node_exists, 1e-9, 1.0 - 1e-9)
            
            node_exists_entropy = -(node_exists_clamped * torch.log(node_exists_clamped) + \
                                   (1 - node_exists_clamped) * torch.log(1 - node_exists_clamped)).mean()
            confidence = torch.exp(-node_exists_entropy)
            return confidence.item()
        else:
            raise ValueError(f"Unknown prediction_type for confidence calculation: {prediction_type}")

    
    def _convert_to_molecule(self, predicted_structure):
        """予測された構造を分子に変換"""
        # 予測から分子を構築する処理
        # 実際の実装では、予測された原子タイプと結合を使用してRDKit分子を構築
        
        if isinstance(predicted_structure, dict):
            node_exists_t = predicted_structure['node_exists']    # Shape e.g. (MAX_ATOMS, 1)
            node_types_t = predicted_structure['node_types']      # Shape e.g. (MAX_ATOMS, NUM_CLASSES)
            edge_exists_t = predicted_structure['edge_exists']    # Shape e.g. (num_edges, 1)
            edge_types_t = predicted_structure['edge_types']      # Shape e.g. (num_edges, NUM_EDGE_CLASSES)
        else:
            # This path is for robustness, assuming object-like access
            node_exists_t = predicted_structure.node_exists
            node_types_t = predicted_structure.node_types
            edge_exists_t = predicted_structure.edge_exists
            edge_types_t = predicted_structure.edge_types

        # Process tensors, assuming they are for a single molecule now
        node_exists = node_exists_t.squeeze(-1).cpu().numpy() > 0.5  # From (N, 1) to (N,)
        node_types = node_types_t.argmax(dim=-1).cpu().numpy()       # From (N, C) to (N,)
        
        # Handle edge_exists: from (num_edges, 1) to (num_edges,)
        if edge_exists_t.ndim > 1 and edge_exists_t.shape[-1] == 1:
             edge_exists_t = edge_exists_t.squeeze(-1)
        edge_exists = edge_exists_t.cpu().numpy() > 0.5
        
        # Handle edge_types: from (num_edges, C) to (num_edges,)
        # If edge_types_t could already be (num_edges,), argmax only if it has more than 1 dim.
        if edge_types_t.ndim > 1: # Has class dimension
            edge_types = edge_types_t.argmax(dim=-1).cpu().numpy()
        else: # Assuming it's already (num_edges,)
            edge_types = edge_types_t.cpu().numpy()
        
        # RWMolオブジェクトを作成
        mol = Chem.RWMol()
        
        # 原子を追加
        atom_map = {}
        atom_counter = 0
        for i, (exists, atom_type) in enumerate(zip(node_exists, node_types)):
            if exists:
                atom_idx = atom_counter
                atom_counter += 1
                atom_map[i] = atom_idx
                
                # 原子タイプから元素を決定
                element_map = {0: 6, 1: 1, 2: 7, 3: 8, 4: 9, 5: 16, 6: 15, 7: 17, 8: 35, 9: 53}
                atomic_num = element_map.get(atom_type, 6)  # デフォルトは炭素
                
                # 原子を追加
                atom = Chem.Atom(atomic_num)
                mol.AddAtom(atom)
        
        # 結合を追加
        edge_counter = 0
        for i, j in itertools.combinations(range(len(node_exists)), 2):
            if i in atom_map and j in atom_map:
                if edge_exists[edge_counter]:
                    # 結合タイプを決定
                    bond_type_map = {
                        0: Chem.BondType.SINGLE,
                        1: Chem.BondType.DOUBLE,
                        2: Chem.BondType.TRIPLE,
                        3: Chem.BondType.AROMATIC
                    }
                    bond_type = bond_type_map.get(edge_types[edge_counter], Chem.BondType.SINGLE)
                    
                    # 結合を追加
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)
                
                edge_counter += 1
        
        # 分子を整える
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except:
            # 構築に失敗した場合はデフォルト分子を返す
            return Chem.MolFromSmiles("C")
    
    def train_semi_supervised(self, labeled_dataloader, pseudo_labeled_data, epochs=1):
        """半教師あり学習"""
        # 高信頼度の疑似ラベルをフィルタリング
        high_confidence_data = self.filter_high_confidence_pseudo_labels(pseudo_labeled_data)
        
        # 教師ありデータと疑似ラベルデータを組み合わせる
        combined_dataset = ConcatDataset([
            labeled_dataloader.dataset,
            ChemicalStructureSpectumDataset(structure_spectrum_pairs=high_confidence_data)
        ])
        
        combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # 組み合わせたデータセットで教師あり学習を実行
        return self.train_supervised(combined_dataloader, epochs)
    
    def self_growing_train_loop(self, labeled_dataloader, unlabeled_dataloader, val_dataloader=None, 
                               num_iterations=10, supervised_epochs=5, cycle_epochs=3, diffusion_epochs=2):
        """自己成長トレーニングループ"""
        best_val_loss = float('inf')
        best_model_state = None
        
        for iteration in range(num_iterations):
            logger.info(f"=== Self-Growing Iteration {iteration+1}/{num_iterations} ===")
            
            # 1. 教師あり学習
            logger.info("Step 1: Supervised Training")
            supervised_loss = self.train_supervised(labeled_dataloader, epochs=supervised_epochs)
            
            # 2. 拡散モデルの訓練
            logger.info("Step 2: Diffusion Model Training")
            diffusion_loss = self.train_diffusion(labeled_dataloader, epochs=diffusion_epochs)
            
            # 3. サイクル一貫性訓練
            logger.info("Step 3: Cycle Consistency Training")
            cycle_loss = self.train_cycle_consistency(labeled_dataloader, epochs=cycle_epochs)
            
            # 4. 疑似ラベルの生成
            logger.info("Step 4: Generating Pseudo Labels")
            pseudo_labeled_data = self.generate_pseudo_labels(unlabeled_dataloader)
            
            # 5. 半教師あり学習
            logger.info("Step 5: Semi-Supervised Training")
            semi_supervised_loss = self.train_semi_supervised(
                labeled_dataloader, 
                pseudo_labeled_data, 
                epochs=supervised_epochs
            )
            
            # バリデーション
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                self.metrics['val_loss'].append(val_loss)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                
                # 学習率の調整
                self.scheduler.step(val_loss)
                
                # モデルの保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            # 平均損失をメトリクスに追加
            self.metrics['train_loss'].append(supervised_loss + semi_supervised_loss)
            
            # メトリクスの表示
            self._display_metrics()
        
        # 最良のモデルを読み込む
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
        
        return self.metrics
    
    def evaluate(self, dataloader):
        """モデルの評価"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 教師ありデータのみを処理
                supervised_indices = [i for i, t in enumerate(batch['type']) if t == 'supervised']
                if not supervised_indices:
                    continue
                
                # バッチから教師ありデータを抽出
                supervised_structures = [batch['structure'][i] for i in supervised_indices]
                supervised_spectra = torch.stack([batch['spectrum'][i] for i in supervised_indices]).to(self.device)
                
                # データを辞書にまとめる
                data = {
                    'structure': supervised_structures,
                    'spectrum': supervised_spectra
                }
                
                # 順伝播
                outputs = self.model(data, direction="bidirectional")
                
                # 損失計算
                loss_s2p = F.mse_loss(outputs['predicted_spectrum'], supervised_spectra)
                
                # Create structural targets
                structural_targets = self._create_structural_targets(supervised_structures, MAX_ATOMS, self.device)
                target_node_exists = structural_targets['node_exists']
                target_node_types = structural_targets['node_types']

                # Predicted structure outputs from decoder (already batched)
                predicted_structure = outputs['predicted_structure']
                
                # Calculate node_exists_loss
                node_exists_loss = F.binary_cross_entropy(
                    predicted_structure['node_exists'],
                    target_node_exists
                )
                
                # Calculate node_types_loss
                predicted_node_types_for_loss = predicted_structure['node_types'].permute(0, 2, 1)
                node_types_loss = F.cross_entropy(
                    predicted_node_types_for_loss,
                    target_node_types,
                    ignore_index=UNKNOWN_ATOM_INDEX_TARGET
                )
                
                loss_p2s = node_exists_loss + node_types_loss
                
                # 合計損失
                loss = loss_s2p + loss_p2s
                total_loss += loss.item()
        
        # 平均損失を返す
        return total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    
    def _display_metrics(self):
        """メトリクスの表示"""
        # 最新のメトリクスを表示
        logger.info("=== Training Metrics ===")
        
        for key, values in self.metrics.items():
            if values:
                logger.info(f"{key}: {values[-1]:.4f}")
        
        # 損失のグラフを描画
        if self.metrics['train_loss']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['train_loss'], label='Train Loss')
            
            if self.metrics['val_loss']:
                plt.plot(self.metrics['val_loss'], label='Validation Loss')
            
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.savefig('training_progress.png')
            plt.close()

# --- Content from gsai0501-3.py (utilities) ---

# 前のパートからのモデルとデータ構造をインポート
# これらは実際には同じファイルまたはモジュールからインポートされるべきだが、
# ここでは説明のためにコメントアウトしている
# from part1 import BidirectionalSelfGrowingModel, Fragment, FragmentNode, DiffusionModel
# from part2 import SelfGrowingTrainer, ChemicalStructureSpectumDataset, MoleculeData, normalize_spectrum, collate_fn

#------------------------------------------------------
# データローディングとプリプロセシング関数
#------------------------------------------------------

def load_msp_file(file_path: str) -> Dict[str, Dict]:
    """MSPファイルを読み込み、パースする"""
    compound_data = {}
    current_compound = None
    current_id = None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Loading MSP file"):
            line = line.strip()
            
            # 空行は無視
            if not line:
                continue
            
            # 新しい化合物の開始
            if line.startswith("Name:"):
                if current_id is not None:
                    compound_data[current_id] = current_compound
                
                current_compound = {
                    'name': line.replace("Name:", "").strip(),
                    'peaks': []
                }
                
            # 化合物IDの取得
            elif line.startswith("ID:"):
                current_id = line.replace("ID:", "").strip()
                
            # マススペクトルピークの取得
            elif re.match(r"^\d+\s+\d+$", line):
                mz, intensity = line.split()
                current_compound['peaks'].append((int(mz), int(intensity)))
                
            # その他のメタデータ
            elif ":" in line:
                key, value = line.split(":", 1)
                current_compound[key.strip()] = value.strip()
    
    # 最後の化合物を追加
    if current_id is not None:
        compound_data[current_id] = current_compound
    
    return compound_data

def load_mol_files(directory: str) -> Dict[str, Chem.Mol]:
    """ディレクトリからMOLファイルを読み込む"""
    mol_data = {}
    
    for filename in tqdm(os.listdir(directory), desc="Loading MOL files"):
        if filename.endswith(".MOL") or filename.endswith(".mol"):
            # ファイル名からIDを抽出
            mol_id = filename.replace(".MOL", "").replace(".mol", "")
            if mol_id.startswith("ID"):
                mol_id = mol_id[2:]  # "ID" プレフィックスを削除
            
            # MOLファイルを読み込む
            mol_path = os.path.join(directory, filename)
            try:
                mol = Chem.MolFromMolFile(mol_path)
                if mol is not None:
                    mol_data[mol_id] = mol
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return mol_data

def prepare_dataset(msp_data: Dict[str, Dict], mol_data: Dict[str, Chem.Mol], 
                   spectrum_dim: int = 2000, test_ratio: float = 0.1, val_ratio: float = 0.1,
                   unlabeled_ratio: float = 0.3, seed: int = 42):
    """データセットを準備する"""
    # 共通のIDを持つ化合物だけを使用
    common_ids = set(msp_data.keys()) & set(mol_data.keys())
    print(f"Found {len(common_ids)} compounds with both MSP and MOL data")
    
    # データを処理
    dataset = []
    for compound_id in tqdm(common_ids, desc="Preparing dataset"):
        try:
            # スペクトルを抽出して正規化
            peaks = msp_data[compound_id]['peaks']
            spectrum = normalize_spectrum(peaks, max_mz=spectrum_dim)
            
            # 分子を処理
            mol = mol_data[compound_id]
            mol_data_obj = MoleculeData(mol, spectrum)
            
            # データセットに追加
            dataset.append((compound_id, mol_data_obj))
        except Exception as e:
            print(f"Error processing compound {compound_id}: {e}")
    
    # 乱数シードを設定
    random.seed(seed)
    
    # データセットをシャッフル
    random.shuffle(dataset)
    
    # 教師なしデータとして使用するデータの割合
    n_unlabeled = int(len(dataset) * unlabeled_ratio)
    
    # 教師なしデータを分離
    unlabeled_data = dataset[:n_unlabeled]
    labeled_data = dataset[n_unlabeled:]
    
    # 教師ありデータを訓練/検証/テストに分割
    n_val = int(len(labeled_data) * val_ratio / (1 - unlabeled_ratio))
    n_test = int(len(labeled_data) * test_ratio / (1 - unlabeled_ratio))
    n_train = len(labeled_data) - n_val - n_test
    
    train_data = labeled_data[:n_train]
    val_data = labeled_data[n_train:n_train+n_val]
    test_data = labeled_data[n_train+n_val:]
    
    # 教師なしデータを構造のみとスペクトルのみに分割
    unlabeled_structures = []
    unlabeled_spectra = []
    
    for _, mol_data_obj in unlabeled_data:
        if random.random() < 0.5:
            # 構造のみのデータ（スペクトルを破棄）
            mol_data_obj.spectrum = None
            unlabeled_structures.append(mol_data_obj)
        else:
            # スペクトルのみのデータ（構造情報は保持）
            unlabeled_spectra.append(mol_data_obj.spectrum)
    
    # 構造-スペクトルのペアを作成
    structure_spectrum_pairs = []
    for _, mol_data_obj in train_data:
        structure_spectrum_pairs.append((mol_data_obj, mol_data_obj.spectrum))
    
    # データセットを作成
    train_dataset = ChemicalStructureSpectumDataset(
        structures=unlabeled_structures,
        spectra=unlabeled_spectra,
        structure_spectrum_pairs=structure_spectrum_pairs
    )
    
    val_pairs = []
    for _, mol_data_obj in val_data:
        val_pairs.append((mol_data_obj, mol_data_obj.spectrum))
    
    val_dataset = ChemicalStructureSpectumDataset(
        structure_spectrum_pairs=val_pairs
    )
    
    test_pairs = []
    for _, mol_data_obj in test_data:
        test_pairs.append((mol_data_obj, mol_data_obj.spectrum))
    
    test_dataset = ChemicalStructureSpectumDataset(
        structure_spectrum_pairs=test_pairs
    )
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    print(f"Train dataset: {len(structure_spectrum_pairs)} supervised pairs, {len(unlabeled_structures)} unsupervised structures, {len(unlabeled_spectra)} unsupervised spectra")
    
    return train_dataset, val_dataset, test_dataset

#------------------------------------------------------
# 可視化関数
#------------------------------------------------------

def visualize_molecule(mol, highlight_atoms=None, highlight_bonds=None, 
                      highlight_atom_colors=None, highlight_bond_colors=None,
                      title=None, size=(400, 300), save_path=None):
    """分子を可視化する"""
    # RDKitドローイングオブジェクトを作成
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    
    # 描画オプション
    d.drawOptions().addAtomIndices = True
    
    # ハイライト設定
    if highlight_atoms is None:
        highlight_atoms = []
    if highlight_bonds is None:
        highlight_bonds = []
    
    # ハイライト色設定
    if highlight_atom_colors is None and highlight_atoms:
        highlight_atom_colors = [(0.7, 0.0, 0.0) for _ in highlight_atoms]
    if highlight_bond_colors is None and highlight_bonds:
        highlight_bond_colors = [(0.0, 0.7, 0.0) for _ in highlight_bonds]
    
    # 分子を描画
    d.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=dict(zip(highlight_atoms, highlight_atom_colors)) if highlight_atom_colors else {},
        highlightBondColors=dict(zip(highlight_bonds, highlight_bond_colors)) if highlight_bond_colors else {}
    )
    d.FinishDrawing()
    
    # 画像をPILオブジェクトに変換
    img_data = d.GetDrawingText()
    img = Image.open(BytesIO(img_data))
    
    # 保存するか表示する
    if save_path:
        img.save(save_path)
    
    # RDKit image saving (PIL Image object)
    if save_path:
        img.save(save_path)

    # Matplotlib display part
    fig = None
    try:
        # Create a new figure for Matplotlib operations
        fig = plt.figure(figsize=(size[0]/100, size[1]/100))
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.tight_layout()

        # Show the plot only if save_path was not used for the PIL image
        # (as per original logic, plt part is mainly for display if PIL not saved)
        if not save_path:
            plt.show()
    finally:
        # Ensure the Matplotlib figure is closed if it was created
        if fig is not None:
            plt.close(fig)
        # Fallback for safety, though fig should cover it if plt.figure was successful
        elif plt.gcf().get_axes(): 
            plt.close(plt.gcf())
            
    return img

def visualize_spectrum(spectrum, max_mz=2000, threshold=0.01, top_n=20, 
                      title=None, size=(10, 5), save_path=None):
    """マススペクトルを可視化する"""
    # ピークを抽出
    peaks = []
    for mz, intensity in enumerate(spectrum):
        if intensity > threshold:
            peaks.append((mz, intensity))
    
    # 強度順にソート
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # 上位N個のピークを選択
    if top_n > 0 and len(peaks) > top_n:
        peaks = peaks[:top_n]
    
    # m/z順にソート
    peaks.sort()
    
    # プロット
    plt.figure(figsize=size)
    mz_values = [p[0] for p in peaks]
    intensities = [p[1] for p in peaks]
    
    plt.stem(mz_values, intensities, markerfmt=" ", basefmt=" ")
    plt.xlabel("m/z")
    plt.ylabel("Relative Intensity")
    
    if title:
        plt.title(title)
    
    plt.xlim(0, max_mz)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close() # Ensure one plt.close() here

def visualize_fragment_tree(fragment_tree, max_depth=3, size=(12, 8), save_path=None):
    """フラグメントツリーを可視化する"""
    # ツリー構造を取得
    def get_tree_structure(node, depth=0):
        if depth > max_depth:
            return []
        
        result = [(depth, node)]
        for child in node.children:
            result.extend(get_tree_structure(child, depth + 1))
        return result
    
    tree_structure = get_tree_structure(fragment_tree)
    
    # プロット設定
    plt.figure(figsize=size)
    
    # 各深さのノード数をカウント
    depth_counts = {}
    for depth, _ in tree_structure:
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    
    # Y座標の調整
    node_positions = {}
    for depth in range(max_depth + 1):
        count = depth_counts.get(depth, 0)
        for i in range(count):
            y_pos = (i + 1) / (count + 1)
            node_positions[(depth, i)] = (depth, y_pos)
    
    # ノードとエッジを描画
    node_index = {depth: 0 for depth in range(max_depth + 1)}
    
    for depth, node in tree_structure:
        # ノードの位置
        x, y = node_positions[(depth, node_index[depth])]
        node_index[depth] += 1
        
        # ノード情報
        mass = f"{node.fragment.mass:.2f}"
        formula = node.fragment.formula
        break_mode = node.break_mode
        
        # ノード描画
        plt.scatter(x, y, s=100, alpha=0.7)
        plt.annotate(
            f"m/z: {mass}\n{formula}\n{break_mode}",
            (x, y),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.1)
        )
        
        # 親へのエッジを描画
        if node.parent:
            parent_depth = depth - 1
            parent_idx = 0
            for i, (d, n) in enumerate(tree_structure):
                if d == parent_depth and n == node.parent:
                    parent_idx = node_index[parent_depth] - 1
                    break
            
            parent_x, parent_y = node_positions[(parent_depth, parent_idx)]
            plt.plot([x, parent_x], [y, parent_y], 'k-', alpha=0.5)
    
    # プロット設定
    plt.xlim(-0.5, max_depth + 0.5)
    plt.ylim(0, 1.1)
    plt.xticks(range(max_depth + 1), [f"Depth {i}" for i in range(max_depth + 1)])
    plt.yticks([])
    plt.title("Fragment Tree")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close() # Ensure one plt.close() here

def visualize_latent_space(model, dataset, n_samples=100, perplexity=30, 
                          title=None, size=(10, 8), save_path=None):
    """潜在空間を可視化する（t-SNE）"""
    model.eval()
    
    # データをサンプリング
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    sampled_data = [dataset[i] for i in indices]
    
    # 構造とスペクトルから潜在表現を取得
    structure_latents = []
    spectrum_latents = []
    categories = []
    
    with torch.no_grad():
        for data in tqdm(sampled_data, desc="Encoding latent representations"):
            # データの種類を取得
            if data['type'] == 'supervised':
                categories.append('Supervised')
                
                # 構造からの潜在表現
                structure_embedding = model.structure_encoder(data['structure'])
                structure_latents.append(structure_embedding.cpu().numpy())
                
                # スペクトルからの潜在表現
                spectrum = torch.FloatTensor(data['spectrum']).to(model.device)
                spectrum_embedding = model.spectrum_encoder(spectrum)
                spectrum_latents.append(spectrum_embedding.cpu().numpy())
                
            elif data['type'] == 'unsupervised_structure':
                categories.append('Structure Only')
                
                # 構造からの潜在表現
                structure_embedding = model.structure_encoder(data['structure'])
                structure_latents.append(structure_embedding.cpu().numpy())
                
            elif data['type'] == 'unsupervised_spectrum':
                categories.append('Spectrum Only')
                
                # スペクトルからの潜在表現
                spectrum = torch.FloatTensor(data['spectrum']).to(model.device)
                spectrum_embedding = model.spectrum_encoder(spectrum)
                spectrum_latents.append(spectrum_embedding.cpu().numpy())
    
    # 潜在表現をスタック
    all_latents = []
    all_latents.extend(structure_latents)
    all_latents.extend(spectrum_latents)
    
    # カテゴリラベルを拡張
    latent_categories = []
    latent_categories.extend(['Structure Latent'] * len(structure_latents))
    latent_categories.extend(['Spectrum Latent'] * len(spectrum_latents))
    
    # t-SNEで次元削減
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latents_2d = tsne.fit_transform(all_latents)
    
    # プロット
    plt.figure(figsize=size)
    
    # カテゴリ別に色分け
    category_colors = {
        'Structure Latent (Supervised)': 'blue',
        'Spectrum Latent (Supervised)': 'red',
        'Structure Latent (Structure Only)': 'cyan',
        'Spectrum Latent (Spectrum Only)': 'orange'
    }
    
    combined_categories = []
    for i in range(len(latent_categories)):
        if i < len(structure_latents):
            idx = i
            if categories[idx] == 'Supervised':
                combined_categories.append('Structure Latent (Supervised)')
            else:
                combined_categories.append('Structure Latent (Structure Only)')
        else:
            idx = i - len(structure_latents)
            if idx < len(categories) and categories[idx] == 'Supervised':
                combined_categories.append('Spectrum Latent (Supervised)')
            else:
                combined_categories.append('Spectrum Latent (Spectrum Only)')
    
    # カテゴリごとにプロット
    for category, color in category_colors.items():
        indices = [i for i, c in enumerate(combined_categories) if c == category]
        if indices:
            plt.scatter(
                latents_2d[indices, 0],
                latents_2d[indices, 1],
                c=color,
                label=category,
                alpha=0.7
            )
    
    plt.legend()
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    if title:
        plt.title(title)
    else:
        plt.title("Latent Space Visualization (t-SNE)")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close() # Ensure one plt.close() here

def visualize_metrics(trainer, size=(12, 8), save_path=None):
    """トレーニングメトリクスを可視化する"""
    metrics = trainer.metrics
    
    # プロット設定
    plt.figure(figsize=size)
    
    # サブプロット配置
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols
    
    # 各メトリクスをプロット
    for i, (metric_name, values) in enumerate(metrics.items()):
        if not values:
            continue
            
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(values)
        plt.title(metric_name)
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close() # Added

def create_spectrum_prediction_report(model, test_dataset, n_samples=5, save_dir=None):
    """スペクトル予測レポートを作成"""
    model.eval()
    
    # ディレクトリの作成
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # テストデータをサンプリング
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))
    sampled_data = [test_dataset[i] for i in indices]
    
    results = []
    
    with torch.no_grad():
        for i, data in enumerate(sampled_data):
            # 教師ありデータのみを使用
            if data['type'] != 'supervised':
                continue
            
            # 構造からスペクトルを予測
            structure_data = {'structure': data['structure']}
            outputs = model(structure_data, direction="structure_to_spectrum")
            predicted_spectrum = outputs['predicted_spectrum'].cpu().numpy()
            
            # 実際のスペクトル
            true_spectrum = data['spectrum']
            
            # 比較メトリクス
            cosine_similarity = F.cosine_similarity(
                torch.FloatTensor(predicted_spectrum).unsqueeze(0),
                torch.FloatTensor(true_spectrum).unsqueeze(0)
            ).item()
            
            # 結果を保存
            result = {
                'index': i,
                'predicted_spectrum': predicted_spectrum,
                'true_spectrum': true_spectrum,
                'cosine_similarity': cosine_similarity,
                'structure': data['structure']
            }
            results.append(result)
            
            # 可視化と保存
            if save_dir:
                # 分子の可視化
                mol = data['structure'].mol
                mol_path = os.path.join(save_dir, f"mol_{i}.png")
                visualize_molecule(mol, title=f"Compound {i}", save_path=mol_path)
                
                # 実際のスペクトル
                true_path = os.path.join(save_dir, f"true_spectrum_{i}.png")
                visualize_spectrum(true_spectrum, title=f"True Spectrum {i}", save_path=true_path)
                
                # 予測スペクトル
                pred_path = os.path.join(save_dir, f"pred_spectrum_{i}.png")
                visualize_spectrum(predicted_spectrum, title=f"Predicted Spectrum {i} (CS: {cosine_similarity:.3f})", save_path=pred_path)
                
                # 比較プロット
                plt.figure(figsize=(12, 6))
                
                plt.subplot(2, 1, 1)
                mz_values_true = [mz for mz, intensity in enumerate(true_spectrum) if intensity > 0.01]
                intensities_true = [intensity for intensity in true_spectrum if intensity > 0.01]
                plt.stem(mz_values_true, intensities_true, markerfmt=" ", basefmt=" ", linefmt="b-")
                plt.title(f"True Spectrum {i}")
                plt.ylabel("Intensity")
                plt.ylim(0, 1.05)
                
                plt.subplot(2, 1, 2)
                mz_values_pred = [mz for mz, intensity in enumerate(predicted_spectrum) if intensity > 0.01]
                intensities_pred = [intensity for intensity in predicted_spectrum if intensity > 0.01]
                plt.stem(mz_values_pred, intensities_pred, markerfmt=" ", basefmt=" ", linefmt="r-")
                plt.title(f"Predicted Spectrum {i} (Cosine Similarity: {cosine_similarity:.3f})")
                plt.xlabel("m/z")
                plt.ylabel("Intensity")
                plt.ylim(0, 1.05)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"comparison_{i}.png"))
                plt.close()
    
    # 結果の要約
    if results:
        similarities = [r['cosine_similarity'] for r in results]
        avg_similarity = np.mean(similarities)
        
        summary = {
            'n_samples': len(results),
            'average_cosine_similarity': avg_similarity,
            'min_cosine_similarity': min(similarities),
            'max_cosine_similarity': max(similarities)
        }
        
        print(f"Spectrum Prediction Summary:")
        print(f"Number of samples: {summary['n_samples']}")
        print(f"Average cosine similarity: {summary['average_cosine_similarity']:.4f}")
        print(f"Min cosine similarity: {summary['min_cosine_similarity']:.4f}")
        print(f"Max cosine similarity: {summary['max_cosine_similarity']:.4f}")
        
        if save_dir:
            with open(os.path.join(save_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=4)
        
        return results, summary
    
    return [], {}

def create_structure_prediction_report(model, test_dataset, n_samples=5, save_dir=None):
    """構造予測レポートを作成"""
    model.eval()
    
    # ディレクトリの作成
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # テストデータをサンプリング
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))
    sampled_data = [test_dataset[i] for i in indices]
    
    results = []
    
    with torch.no_grad():
        for i, data in enumerate(sampled_data):
            # 教師ありデータのみを使用
            if data['type'] != 'supervised':
                continue
            
            # スペクトルから構造を予測
            spectrum = torch.FloatTensor(data['spectrum']).unsqueeze(0).to(model.device)
            spectrum_data = {'spectrum': spectrum}
            outputs = model(spectrum_data, direction="spectrum_to_structure")
            predicted_structure = outputs['predicted_structure']
            
            # 予測から分子を構築
            pred_mol = convert_prediction_to_molecule(predicted_structure)
            
            # 実際の分子
            true_mol = data['structure'].mol
            
            # 構造の類似度を計算
            similarity = calculate_structure_similarity(true_mol, pred_mol)
            
            # 結果を保存
            result = {
                'index': i,
                'predicted_mol': pred_mol,
                'true_mol': true_mol,
                'similarity': similarity,
                'spectrum': data['spectrum']
            }
            results.append(result)
            
            # 可視化と保存
            if save_dir:
                # 真の構造
                true_path = os.path.join(save_dir, f"true_mol_{i}.png")
                visualize_molecule(true_mol, title=f"True Structure {i}", save_path=true_path)
                
                # 予測構造
                pred_path = os.path.join(save_dir, f"pred_mol_{i}.png")
                visualize_molecule(pred_mol, title=f"Predicted Structure {i} (Sim: {similarity:.3f})", save_path=pred_path)
                
                # スペクトル
                spectrum_path = os.path.join(save_dir, f"spectrum_{i}.png")
                visualize_spectrum(data['spectrum'], title=f"Input Spectrum {i}", save_path=spectrum_path)
                
                # 比較レポート
                with open(os.path.join(save_dir, f"structure_report_{i}.txt"), "w") as f:
                    f.write(f"Structure Comparison Report for Sample {i}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Similarity: {similarity:.4f}\n\n")
                    f.write(f"True SMILES: {Chem.MolToSmiles(true_mol)}\n")
                    f.write(f"Predicted SMILES: {Chem.MolToSmiles(pred_mol)}\n")
    
    # 結果の要約
    if results:
        similarities = [r['similarity'] for r in results]
        avg_similarity = np.mean(similarities)
        
        summary = {
            'n_samples': len(results),
            'average_similarity': avg_similarity,
            'min_similarity': min(similarities),
            'max_similarity': max(similarities)
        }
        
        print(f"Structure Prediction Summary:")
        print(f"Number of samples: {summary['n_samples']}")
        print(f"Average similarity: {summary['average_similarity']:.4f}")
        print(f"Min similarity: {summary['min_similarity']:.4f}")
        print(f"Max similarity: {summary['max_similarity']:.4f}")
        
        if save_dir:
            with open(os.path.join(save_dir, "structure_summary.json"), "w") as f:
                json.dump(summary, f, indent=4)
        
        return results, summary
    
    return [], {}

def convert_prediction_to_molecule(predicted_structure):
    """予測構造を分子に変換"""
    # この関数の実装は実際のモデル出力形式に依存するため、
    # ここではシンプルな実装を示す
    
    # 予測から原子と結合の情報を抽出
    node_exists = predicted_structure['node_exists'].cpu().numpy() > 0.5
    node_types = predicted_structure['node_types'].argmax(dim=1).cpu().numpy()
    edge_exists = predicted_structure['edge_exists'].cpu().numpy() > 0.5
    edge_types = predicted_structure['edge_types'].argmax(dim=1).cpu().numpy()
    
    # RWMolオブジェクトを作成
    mol = Chem.RWMol()
    
    # 原子マップ（予測インデックス → 実際のRDKit原子インデックス）
    atom_map = {}
    
    # 原子を追加
    for i, (exists, atom_type) in enumerate(zip(node_exists, node_types)):
        if exists:
            # 原子タイプから元素を決定
            # 例: 0=C, 1=H, 2=N, 3=O, 4=F, ...
            element_map = {0: 6, 1: 1, 2: 7, 3: 8, 4: 9, 5: 16, 6: 15, 7: 17, 8: 35, 9: 53}
            atomic_num = element_map.get(atom_type, 6)  # デフォルトは炭素
            
            # 原子を追加
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            atom_map[i] = atom_idx
    
    # 結合を追加
    edge_idx = 0
    for i in range(len(node_exists)):
        for j in range(i+1, len(node_exists)):
            if i in atom_map and j in atom_map:
                if edge_idx < len(edge_exists) and edge_exists[edge_idx]:
                    # 結合タイプを決定
                    bond_type = Chem.BondType.SINGLE
                    if edge_idx < len(edge_types):
                        bond_type_map = {
                            0: Chem.BondType.SINGLE,
                            1: Chem.BondType.DOUBLE,
                            2: Chem.BondType.TRIPLE,
                            3: Chem.BondType.AROMATIC
                        }
                        bond_type = bond_type_map.get(edge_types[edge_idx], Chem.BondType.SINGLE)
                    
                    # 結合を追加
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)
                
                edge_idx += 1
    
    # 分子を整える
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
    except:
        # 構造が妥当でない場合はデフォルト分子を返す
        mol = Chem.MolFromSmiles("C")
    
    return mol

def calculate_structure_similarity(mol1, mol2, method="morgan"):
    """2つの分子構造の類似度を計算"""
    if method == "morgan":
        # MorganフィンガープリントとTanimoto類似度を使用
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    elif method == "maccs":
        # MACCSキーとTanimoto類似度を使用
        fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
        fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    elif method == "smiles":
        # SMILES文字列の編集距離に基づく類似度
        smiles1 = Chem.MolToSmiles(mol1)
        smiles2 = Chem.MolToSmiles(mol2)
        # Levenshtein is not a standard library, so using a placeholder for now
        # In a real scenario, you'd install python-Levenshtein
        # For this exercise, let's assume a simple equality check if Levenshtein is not available.
        try:
            import Levenshtein
            distance = Levenshtein.distance(smiles1, smiles2)
            max_len = max(len(smiles1), len(smiles2))
            similarity = 1 - (distance / max_len) if max_len > 0 else 0
        except ImportError:
            similarity = 1.0 if smiles1 == smiles2 else 0.0 # Fallback
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity

#------------------------------------------------------
# メイン実行関数
#------------------------------------------------------

def main(args):
    """メイン実行関数"""
    # 設定の読み込み
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        # デフォルト設定
        config = {
            "data": {
                "msp_file": "data/NIST17.MSP",
                "mol_dir": "data/mol_files",
                "spectrum_dim": 2000,
                "test_ratio": 0.1,
                "val_ratio": 0.1,
                "unlabeled_ratio": 0.3,
                "seed": 42
            },
            "model": {
                "hidden_dim": 256,
                "latent_dim": 128,
                "atom_fdim": 150,
                "bond_fdim": 10,
                "motif_fdim": 20
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_iterations": 10,
                "supervised_epochs": 5,
                "cycle_epochs": 3,
                "diffusion_epochs": 2,
                "confidence_threshold": 0.8,
                "cycle_weight": 1.0,
                "diffusion_weight": 0.1
            },
            "evaluation": {
                "n_samples": 10
            },
            "output": {
                "model_dir": "models",
                "results_dir": "results"
            }
        }
    
    # ディレクトリの作成
    os.makedirs(config["output"]["model_dir"], exist_ok=True)
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    
    # ロギング設定
    log_path = os.path.join(config["output"]["results_dir"], f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("SelfGrowingModel")
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # データの読み込み
    logger.info("Loading data...")
    msp_data = load_msp_file(config["data"]["msp_file"])
    mol_data = load_mol_files(config["data"]["mol_dir"])
    
    # データセットの準備
    logger.info("Preparing dataset...")
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        msp_data,
        mol_data,
        spectrum_dim=config["data"]["spectrum_dim"],
        test_ratio=config["data"]["test_ratio"],
        val_ratio=config["data"]["val_ratio"],
        unlabeled_ratio=config["data"]["unlabeled_ratio"],
        seed=config["data"]["seed"]
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # モデルの初期化
    logger.info("Initializing model...")
    model = BidirectionalSelfGrowingModel(
        atom_fdim=config["model"]["atom_fdim"],
        bond_fdim=config["model"]["bond_fdim"],
        motif_fdim=config["model"]["motif_fdim"],
        spectrum_dim=config["data"]["spectrum_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    
    # トレーナーの初期化
    trainer = SelfGrowingTrainer(
        model=model,
        device=device,
        config=config["training"]
    )
    
    # 訓練実行
    if not args.eval_only:
        logger.info("Starting training...")
        trainer.self_growing_train_loop(
            labeled_dataloader=train_loader,
            unlabeled_dataloader=train_loader,
            val_dataloader=val_loader,
            num_iterations=config["training"]["num_iterations"],
            supervised_epochs=config["training"]["supervised_epochs"],
            cycle_epochs=config["training"]["cycle_epochs"],
            diffusion_epochs=config["training"]["diffusion_epochs"]
        )
        
        # モデルの保存
        model_path = os.path.join(config["output"]["model_dir"], f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # トレーニングメトリクスの可視化
        metrics_path = os.path.join(config["output"]["results_dir"], "training_metrics.png")
        visualize_metrics(trainer, save_path=metrics_path)
    elif args.model_path:
        # 既存のモデルを読み込む
        logger.info(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 評価
    logger.info("Evaluating model...")
    
    # スペクトル予測評価
    spectrum_report_dir = os.path.join(config["output"]["results_dir"], "spectrum_prediction")
    os.makedirs(spectrum_report_dir, exist_ok=True)
    
    spectrum_results, spectrum_summary = create_spectrum_prediction_report(
        model=model,
        test_dataset=test_dataset,
        n_samples=config["evaluation"]["n_samples"],
        save_dir=spectrum_report_dir
    )
    
    # 構造予測評価
    structure_report_dir = os.path.join(config["output"]["results_dir"], "structure_prediction")
    os.makedirs(structure_report_dir, exist_ok=True)
    
    structure_results, structure_summary = create_structure_prediction_report(
        model=model,
        test_dataset=test_dataset,
        n_samples=config["evaluation"]["n_samples"],
        save_dir=structure_report_dir
    )
    
    # 潜在空間の可視化
    latent_space_path = os.path.join(config["output"]["results_dir"], "latent_space.png")
    visualize_latent_space(
        model=model,
        dataset=test_dataset,
        n_samples=min(100, len(test_dataset)),
        save_path=latent_space_path
    )
    
    logger.info("Evaluation complete")

# --- Main execution block ---

if __name__ == "__main__":
    # コマンドライン引数
    parser = argparse.ArgumentParser(description="Chemical Structure-Mass Spectrum Self-Growing Model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--model-path", type=str, help="Path to pre-trained model")
    parser.add_argument("--cpu", action="store_true", help="Use CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    main(args)
