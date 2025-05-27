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
from rdkit.Chem import AllChem, BRICS, Descriptors, Draw, MolToSmiles, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit.Chem import DataStructs
from rdkit.Chem.rdchem import BondType
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           precision_recall_fscore_support, silhouette_score)
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              random_split)
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv
from tqdm import tqdm
# mpl_toolkits is part of matplotlib but imported this way
from mpl_toolkits.mplot3d import Axes3D
# torch.nn, torch.nn.functional, torch.optim are usually imported like this
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# インポート部分の後に追加
import logging
from rdkit import RDLogger

# RDKitの警告を抑制
RDLogger.DisableLog('rdApp.*')

# ログレベル設定を調整
logging.getLogger().setLevel(logging.INFO)  # DEBUGレベルの大量ログを抑制
# --- Content from gsai0501-1.py ---

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Configuration ---
class ModelConfig:
    def __init__(self):
        self.HIDDEN_DIM = 256
        self.LATENT_DIM = 128
        self.SPECTRUM_DIM = 2000  # m/zの最大値
        self.MAX_ATOMS = 100  # 1分子あたりの最大原子数
        self.MAX_MOTIFS = 20  # 1分子あたりの最大モチーフ数
        
        # 特徴量次元を正確に定義
        self.ATOM_FEATURE_DIM = 149  # 119+11+8+9+2
        self.BOND_FEATURE_DIM = 9   # 6+1+1+1
        self.MOTIF_FEATURE_DIM = 1 + len(MOTIF_TYPES) + 3  # size + types + flags

MODEL_CONFIG = ModelConfig()

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

# Atom type mapping for target creation in cycle consistency
ATOM_TYPES_TARGET_LIST = [6, 1, 7, 8, 9, 16, 15, 17, 35, 53]  # C, H, N, O, F, S, P, Cl, Br, I
ATOM_TYPE_TO_INDEX = {atomic_num: i for i, atomic_num in enumerate(ATOM_TYPES_TARGET_LIST)}
UNKNOWN_ATOM_INDEX_TARGET = len(ATOM_TYPES_TARGET_LIST) # Index for unknown atom types

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
        # CRITICAL FIX: Validate charge before division
        if self.charge == 0:
            # logger.error(f"Fragment charge cannot be zero. Fragment atoms: {self.atoms}, SMILES: {Chem.MolToSmiles(self.mol) if self.mol else 'N/A'}")
            raise ValueError(f"Fragment charge cannot be zero. Fragment atoms: {self.atoms}")

        # 正確な分子量を計算
        exact_mass = Chem.rdMolDescriptors.CalcExactMolWt(self.mol)

        # 失われた水素原子の質量を差し引く
        hydrogen_mass = 1.00782503  # 水素原子の正確な質量
        adjusted_mass = exact_mass - (hydrogen_mass * self.lost_hydrogens)

        # Ensure adjusted mass is not negative
        if adjusted_mass < 0:
            logger.warning(f"Negative adjusted mass for fragment atoms: {self.atoms} (SMILES: {Chem.MolToSmiles(self.mol) if self.mol else 'N/A'}). Setting to 0.0. Original exact_mass: {exact_mass}, lost_hydrogens: {self.lost_hydrogens}")
            adjusted_mass = 0.0

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
        acceptors = ['[N+](=O)[O-]', 'C#N', 'C(F)(F)F', 'C(=O)O[#6]', 'C(=O)[#6]'] # Assuming 'COOR' and 'COR' are valid SMARTS or will be handled if not.
                                                       # For RDKit, these might need to be more specific like '[CX3](=O)O[#6]' for COOR.
                                                       # However, the task is to fix the None check, not validate SMARTS strings themselves.

        # Pre-compile SMARTS patterns and filter out None results
        donor_mols = [Chem.MolFromSmarts(d) for d in donors]
        acceptor_mols = [Chem.MolFromSmarts(a) for a in acceptors]

        # Sum matches only for valid Mol objects
        donor_count = sum(self.mol.HasSubstructMatch(mol_pattern) for mol_pattern in donor_mols if mol_pattern is not None)
        acceptor_count = sum(self.mol.HasSubstructMatch(mol_pattern) for mol_pattern in acceptor_mols if mol_pattern is not None)
        
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
    
    def forward(self, data):
        # dataがMoleculeDataオブジェクトの場合の処理
        if hasattr(data, 'graph_data'):
            graph_data = data.graph_data
        else:
            graph_data = data
        
        # atom_featuresをテンソルに変換
        # Ensure np and torch are imported.
        # self.atom_encoder.weight.device should be valid.
        if isinstance(graph_data['atom_features'], np.ndarray):
            atom_features = torch.FloatTensor(graph_data['atom_features']).to(self.atom_encoder.weight.device)
        else:
            atom_features = graph_data['atom_features']
        
        encoded_atom_features = self.atom_encoder(atom_features)
        
        # The rest of the original forward method should follow, using the new encoded_atom_features.
        # The issue description only provides the modification for atom_features.
        # I will append the rest of the original method.
        
        # Original code after encoded_atom_features:
        # モチーフ特徴量をエンコード
        # This part needs to use graph_data which has been defined.
        encoded_motif_features = self.motif_encoder(graph_data['motif_features'])

        # Atom processing
        if encoded_atom_features.shape[0] == 0:
            # Ensure self.hidden_dim is available.
            atom_global = torch.zeros(self.hidden_dim, device=encoded_atom_features.device)
        else:
            atom_embeddings = encoded_atom_features
            if atom_embeddings.shape[0] > 0 and graph_data['edge_index'].shape[1] > 0:
                 for gcn in self.gcn_layers:
                    # Ensure F (torch.nn.functional) is imported
                    atom_embeddings = F.relu(gcn(atom_embeddings, graph_data['edge_index']))
            elif atom_embeddings.shape[0] > 0: 
                pass 

            atom_embeddings_for_attn = atom_embeddings.unsqueeze(0) 
            atom_attn_output, _ = self.attention(atom_embeddings_for_attn, atom_embeddings_for_attn, atom_embeddings_for_attn)
            atom_attn_output = atom_attn_output.squeeze(0) 
            atom_global = torch.mean(atom_attn_output, dim=0)

        # Motif processing
        if encoded_motif_features.shape[0] == 0:
            motif_global = torch.zeros(self.hidden_dim, device=encoded_motif_features.device)
        else:
            motif_embeddings = encoded_motif_features
            if motif_embeddings.shape[0] > 0 and graph_data['motif_edge_index'].shape[1] > 0:
                for gin in self.gin_layers:
                    motif_embeddings = F.relu(gin(motif_embeddings, graph_data['motif_edge_index']))
            elif motif_embeddings.shape[0] > 0: 
                pass 
            
            motif_embeddings_for_attn = motif_embeddings.unsqueeze(0)
            motif_attn_output, _ = self.attention(motif_embeddings_for_attn, motif_embeddings_for_attn, motif_embeddings_for_attn)
            motif_attn_output = motif_attn_output.squeeze(0) 
            motif_global = torch.mean(motif_attn_output, dim=0)
        
        combined = torch.cat([atom_global, motif_global], dim=0)
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
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        # New layer for projecting node features
        self.node_feature_projector = nn.Linear(hidden_dim * 2 + 1, hidden_dim)

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

    def forward(self, latent, max_atoms=MODEL_CONFIG.MAX_ATOMS): # MAX_ATOMS is global
        expanded = self.expander(latent)
    
        if len(latent.shape) == 1:  # Single sample
            latent = latent.unsqueeze(0)  # Add batch dimension
            expanded = self.expander(latent) # Re-expand with batch dimension
    
        batch_size = latent.shape[0]
        
        # ノード特徴量の生成（各原子に対して独立した特徴量を生成）
        node_features = []
        for i in range(max_atoms):
            # 位置エンコーディングを追加
            position_encoding = torch.tensor([i / max_atoms], device=latent.device).expand(batch_size, 1)
            atom_input = torch.cat([expanded, position_encoding], dim=-1) 
            node_features.append(atom_input)
        
        node_hiddens = torch.stack(node_features, dim=1) # [B, max_atoms, H*2+1]
        node_hiddens = self.node_feature_projector(node_hiddens) # [B, max_atoms, H]
        
        node_exists = self.graph_generator['node_existence'](node_hiddens)
        node_types = self.graph_generator['node_type'](node_hiddens)

        edge_hiddens_list = []
        for b_idx in range(batch_size):
            current_sample_edges = []
            for i in range(max_atoms): 
                for j in range(i + 1, max_atoms): 
                    combined = torch.cat([node_hiddens[b_idx, i], node_hiddens[b_idx, j]], dim=0) 
                    current_sample_edges.append(combined)
            if current_sample_edges: # Only stack if there are edges (max_atoms >= 2)
                edge_hiddens_list.append(torch.stack(current_sample_edges))
            else: # For max_atoms < 2, this item in batch has no edges
                edge_hiddens_list.append(torch.empty((0, self.hidden_dim * 2), device=node_hiddens.device, dtype=node_hiddens.dtype))

        # Stack the list of edge features for each batch item
        # All items will have max_atoms * (max_atoms - 1) / 2 edges if max_atoms >=2
        if batch_size > 0 and max_atoms >=2 : 
            edge_hiddens = torch.stack(edge_hiddens_list) # [B, num_edges, H*2]
        elif batch_size > 0 : # Batch exists, but no edges possible (max_atoms < 2)
            edge_hiddens = torch.empty((batch_size, 0, self.hidden_dim *2), device=node_hiddens.device, dtype=node_hiddens.dtype)
        else: # No batch, no edges
            edge_hiddens = torch.empty((0, 0, self.hidden_dim*2), device=node_hiddens.device, dtype=node_hiddens.dtype)

        edge_exists = self.graph_generator['edge_existence'](edge_hiddens)
        edge_types = self.graph_generator['edge_type'](edge_hiddens)

        # Motif generation: using a slice of `expanded` and repeating for MAX_MOTIFS
        # `expanded` has shape [batch_size, hidden_dim * 2]
        # `self.hidden_dim` is defined in __init__
        # `MAX_MOTIFS` is a global constant
        motif_hiddens_base = expanded[:, self.hidden_dim:] # Shape: [batch_size, hidden_dim]
        if batch_size > 0:
            motif_hiddens = motif_hiddens_base.unsqueeze(1).repeat(1, MAX_MOTIFS, 1) # Shape: [B, MAX_MOTIFS, H]
        else: # Handle empty batch case
            motif_hiddens = torch.empty((0, MAX_MOTIFS, self.hidden_dim), device=expanded.device, dtype=expanded.dtype)

        motif_exists = self.motif_generator['motif_existence'](motif_hiddens)
        motif_types = self.motif_generator['motif_type'](motif_hiddens)
        
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
        
        if self.dim == 0:
            return torch.zeros((time.shape[0], 0), device=device)

        half_dim = self.dim // 2

        # Handle cases where the formula is problematic (dim < 4)
        if self.dim < 4: # Covers half_dim = 0 and half_dim = 1
            # For very small dimensions, the formula for div_term breaks down or is ill-defined.
            # Returning zeros as a safe fallback.
            # A more specific embedding could be designed for these small dims if needed.
            # print(f"Warning: SinusoidalPositionEmbeddings dim {self.dim} is small, returning zeros.")
            return torch.zeros((time.shape[0], self.dim), device=device)

        # Original formula for div_term, safe now because half_dim > 1
        # Note: In the original code, 'embeddings' variable was reused. Renaming for clarity.
        div_term_val = math.log(10000) / (half_dim - 1) 
        
        # Exponents for div_term
        exponents = torch.arange(half_dim, device=device, dtype=torch.float32) * -div_term_val
        div_term_tensor = torch.exp(exponents)

        # Apply to time
        # time shape: (batch_size,) -> make it (batch_size, 1) for broadcasting
        # div_term_tensor shape: (half_dim,) -> make it (1, half_dim) for broadcasting
        scaled_time = time.unsqueeze(1) * div_term_tensor.unsqueeze(0) # broadcasts to (batch_size, half_dim)

        # Concatenate sin and cos
        # Result shape: (batch_size, 2 * half_dim)
        embeddings = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=-1)

        # If self.dim is odd, the above produces a tensor of shape (batch_size, self.dim - 1)
        # Add padding if necessary to match self.dim
        if self.dim % 2 == 1:
            # This condition implies embeddings.shape[1] would be self.dim - 1
            padding = torch.zeros((time.shape[0], 1), device=device)
            embeddings = torch.cat((embeddings, padding), dim=-1)
            
        return embeddings

#------------------------------------------------------
# 双方向自己成長型モデル
#------------------------------------------------------

class BidirectionalSelfGrowingModel(nn.Module):
    """構造-スペクトル間の双方向自己成長型モデル"""
    
    def __init__(self, atom_fdim, bond_fdim, motif_fdim, spectrum_dim, hidden_dim=MODEL_CONFIG.HIDDEN_DIM, latent_dim=MODEL_CONFIG.LATENT_DIM):
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
        results = {}
        
        if direction in ["structure_to_spectrum", "bidirectional"]:
            # 構造→スペクトル方向
            predicted_spectrum, structure_latent = self.structure_to_spectrum(data["structure"])
            results["predicted_spectrum"] = predicted_spectrum
            results["structure_latent"] = structure_latent
        
        if direction in ["spectrum_to_structure", "bidirectional"]:
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

        # CRITICAL FIX: Convert predicted structure to MoleculeData format for structure_to_spectrum
        # This is a simplified conversion - in practice, you'd need proper structure-to-MoleculeData conversion
        try:
            # For now, we'll compute cycle loss in latent space to avoid complex conversion
            # Get latent from predicted_structure2
            # predicted_structure2_latent = spectrum_latent  # Already computed above
            # The user feedback implies spectrum_latent is from the s->p direction.
            # For p->s->p, the latent from predicted_structure2 is what we need.
            # self.spectrum_to_structure returns (structure_dict, aligned_latent)
            # So spectrum_latent from its output is correct here for predicted_structure2.

            # Get what the spectrum should reconstruct to using the alignment and decoder
            raw_spectrum_latent_for_cycle2 = self.spectrum_encoder(spectrum) # spectrum_latent comes from the p->s path, so it's already aligned for structure. We need raw spec_latent.
            align_for_spectrum_decoder = self.structure_to_spectrum_aligner(raw_spectrum_latent_for_cycle2) # Align this raw latent for the spectrum decoder
            predicted_spectrum2_reconstructed = self.spectrum_decoder(align_for_spectrum_decoder)

        except Exception as e:
            # Fallback: skip spectrum cycle if conversion fails
            logger.warning(f"Spectrum cycle computation failed: {e}")
            predicted_spectrum2_reconstructed = spectrum  # Use original spectrum as fallback

        # Structure cycle loss:
        # This part calculates loss for the structure predicted from the first cycle (structure -> spectrum -> structure)
        # predicted_structure is available from:
        # predicted_spectrum, structure_latent = self.structure_to_spectrum(structure_data)
        # predicted_structure, _ = self.spectrum_to_structure(predicted_spectrum)

        if isinstance(structure_data, dict) and 'x' in structure_data and            predicted_structure and 'node_exists' in predicted_structure:
            num_actual_atoms = structure_data['x'].shape[0]
            # node_exists is typically [max_atoms, 1] after sigmoid. Summing gives a soft count.
            num_predicted_atoms = predicted_structure['node_exists'].sum()
            
            structure_cycle_loss = F.mse_loss(
                num_predicted_atoms.float(),
                torch.tensor(float(num_actual_atoms), device=spectrum.device) # Use spectrum's device
            )
        else:
            # Fallback if data is not in expected format or prediction is missing
            structure_cycle_loss = torch.tensor(0.0, device=spectrum.device) 


        # スペクトルサイクル損失
        spectrum_cycle_loss = F.mse_loss(predicted_spectrum2_reconstructed, spectrum)

        # 潜在表現の一貫性損失
        raw_structure_latent = self.structure_encoder(structure_data)
        raw_spectrum_latent = self.spectrum_encoder(spectrum)
        latent_consistency_loss = F.mse_loss(raw_structure_latent, raw_spectrum_latent)

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

# import logging # Already imported at top
# import re # Already imported at top
# import numpy as np # Already imported at top
# import torch # Already imported at top
# from rdkit import Chem # Already imported at top
# from rdkit.Chem import BRICS # Already imported at top
# from rdkit.Chem.rdchem import BondType # Already imported at top

# Assuming MODEL_CONFIG and MOTIF_TYPES are defined globally in the file gsai0527.py
# For example:
# class ModelConfig: ...
# MODEL_CONFIG = ModelConfig()
# MOTIF_TYPES = ["type1", "type2", ...]

class MoleculeData:
    """分子データを処理するクラス - 完全版"""

    # クラス定数として特徴量次元を定義
    ATOM_TYPE_DIM = 119
    CHARGE_DIM = 11
    HYBRIDIZATION_DIM = 8
    H_COUNT_DIM = 9
    PROPERTY_DIM = 2
    ATOM_FEATURE_DIM = ATOM_TYPE_DIM + CHARGE_DIM + HYBRIDIZATION_DIM + H_COUNT_DIM + PROPERTY_DIM

    BOND_TYPE_DIM = 6
    BOND_PROPERTY_DIM = 3
    BOND_FEATURE_DIM = BOND_TYPE_DIM + BOND_PROPERTY_DIM

    def __init__(self, mol, spectrum=None, max_atoms=None, max_motifs=None):
        """分子データを初期化"""
        self.mol = mol
        self.spectrum = spectrum

        # デフォルト値の安全な設定
        self.max_atoms = max_atoms if max_atoms is not None else getattr(MODEL_CONFIG, 'MAX_ATOMS', 100)
        self.max_motifs = max_motifs if max_motifs is not None else getattr(MODEL_CONFIG, 'MAX_MOTIFS', 20)


        # 基本情報の安全な取得
        try:
            if self.mol is not None:
                self.smiles = Chem.MolToSmiles(self.mol)
                self.formula = Chem.rdMolDescriptors.CalcMolFormula(self.mol)
            else:
                raise ValueError("Input RDKit molecule is None")
        except Exception as e:
            logging.warning(f"Failed to generate SMILES/formula for input molecule: {e}")
            self.smiles = "INVALID"
            self.formula = "INVALID"

        # 特徴量の段階的生成
        self._initialize_features()

    def _initialize_features(self):
        """特徴量を段階的に初期化"""
        try:
            if self.mol is None: # Ensure mol is not None before feature extraction
                raise ValueError("Cannot initialize features, RDKit molecule is None.")
            self.atom_features = self._get_atom_features()
            self.bond_features, self.adjacency_list = self._get_bond_features_and_adjacency()
            self.motifs, self.motif_types = self._extract_motifs()
            self.motif_features = self._get_motif_features()
            self.motif_graph, self.motif_edge_features = self._build_motif_graph()
            self.graph_data = self._build_graph_data()
        except Exception as e:
            logging.error(f"Failed to initialize features for {self.smiles}: {e}")
            self._create_fallback_features()

    def _create_fallback_features(self):
        """フォールバック特徴量を作成"""
        self.atom_features = np.zeros((self.max_atoms, self.ATOM_FEATURE_DIM), dtype=np.float32)
        self.bond_features = np.zeros((0, self.BOND_FEATURE_DIM), dtype=np.float32)
        self.adjacency_list = [[] for _ in range(self.max_atoms)]
        self.motifs = []
        self.motif_types = []
        # Ensure MOTIF_TYPES is accessible here, it's a global variable.
        global MOTIF_TYPES # Make sure MOTIF_TYPES is defined globally or passed
        self.motif_features = np.zeros((0, 1 + len(MOTIF_TYPES) + 3), dtype=np.float32)
        self.motif_graph = []
        self.motif_edge_features = np.zeros((0, 6), dtype=np.float32)
        # graph_data build should ideally handle empty/fallback features correctly.
        # Call _build_graph_data ensuring it can handle this state.
        self.graph_data = self._build_graph_data(fallback_mode=True)


    def _get_atom_features(self):
        """改善された原子特徴量抽出"""
        features = []
        # Ensure self.mol is valid before proceeding
        if self.mol is None:
            for _ in range(self.max_atoms):
                 features.append([0] * self.ATOM_FEATURE_DIM)
            return np.array(features, dtype=np.float32)

        num_atoms = self.mol.GetNumAtoms()

        hybridization_types = [
            Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER
        ]
        hybridization_map = {h: i for i, h in enumerate(hybridization_types)}

        for i in range(self.max_atoms):
            if i < num_atoms:
                atom = self.mol.GetAtomWithIdx(i)
                atom_type = min(atom.GetAtomicNum(), self.ATOM_TYPE_DIM - 1)
                atom_type_oh = [0] * self.ATOM_TYPE_DIM; atom_type_oh[atom_type] = 1
                charge = max(-5, min(5, atom.GetFormalCharge()))
                charge_oh = [0] * self.CHARGE_DIM; charge_oh[charge + 5] = 1
                hybridization = atom.GetHybridization()
                hybridization_idx = hybridization_map.get(hybridization, hybridization_map[Chem.rdchem.HybridizationType.OTHER])
                hybridization_oh = [0] * self.HYBRIDIZATION_DIM; hybridization_oh[hybridization_idx] = 1
                h_count = min(atom.GetTotalNumHs(), self.H_COUNT_DIM - 1)
                h_count_oh = [0] * self.H_COUNT_DIM; h_count_oh[h_count] = 1
                is_aromatic = int(atom.GetIsAromatic())
                is_in_ring = int(atom.IsInRing())
                atom_f = atom_type_oh + charge_oh + hybridization_oh + h_count_oh + [is_aromatic, is_in_ring]
                features.append(atom_f)
            else:
                features.append([0] * self.ATOM_FEATURE_DIM)
        return np.array(features, dtype=np.float32)

    def _get_bond_features_and_adjacency(self):
        """改善された結合特徴量と隣接リスト取得"""
        bond_features = []
        if self.mol is None: # Guard against None mol
            return np.array(bond_features, dtype=np.float32), [[] for _ in range(self.max_atoms)]

        num_actual_atoms = self.mol.GetNumAtoms()
        adjacency_list = [[] for _ in range(num_actual_atoms)]


        bond_types_enum = [
            BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC,
            BondType.UNSPECIFIED, BondType.OTHER
        ]
        bond_type_map = {bt: i for i, bt in enumerate(bond_types_enum)}

        for bond in self.mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type_val = bond.GetBondType()
            bond_type_idx = bond_type_map.get(bond_type_val, bond_type_map[BondType.OTHER])
            bond_type_oh = [0] * self.BOND_TYPE_DIM; bond_type_oh[bond_type_idx] = 1
            is_in_ring = int(bond.IsInRing())
            is_conjugated = int(bond.GetIsConjugated())
            is_stereo = int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)
            bond_feature = bond_type_oh + [is_in_ring, is_conjugated, is_stereo]
            bond_features.append(bond_feature)
            if begin_idx < num_actual_atoms and end_idx < num_actual_atoms: # Check indices
                bond_idx_in_list = len(bond_features) - 1
                adjacency_list[begin_idx].append((end_idx, bond_idx_in_list))
                adjacency_list[end_idx].append((begin_idx, bond_idx_in_list))
        return np.array(bond_features, dtype=np.float32), adjacency_list

    def _extract_motifs(self, motif_size_threshold=3):
        """改善されたモチーフ抽出"""
        motifs = []
        motif_types = []
        if self.mol is None: return motifs, motif_types

        try:
            brics_frags_smiles = list(BRICS.BRICSDecompose(self.mol, keepNonLeafNodes=True))
            for frag_smi in brics_frags_smiles:
                try:
                    cleaned_smi = re.sub(r'\[\*\d*\]', '', frag_smi)
                    if not cleaned_smi or len(cleaned_smi) < 2: continue
                    frag_mol = Chem.MolFromSmiles(cleaned_smi)
                    if frag_mol and frag_mol.GetNumAtoms() >= motif_size_threshold:
                        matches = self.mol.GetSubstructMatches(frag_mol)
                        for match_indices in matches[:3]:
                            match_list = sorted(list(match_indices))
                            if match_list not in motifs:
                                motifs.append(match_list)
                                motif_types.append(self._determine_motif_type(frag_mol))
                except Exception as e_frag: logging.debug(f"BRICS frag '{frag_smi}' proc fail: {e_frag}")
        except Exception as e_brics: logging.warning(f"BRICS decomp fail for {self.smiles}: {e_brics}")

        try:
            ring_info = self.mol.GetRingInfo()
            for ring_atom_indices_tuple in ring_info.AtomRings():
                ring_atom_indices = list(ring_atom_indices_tuple)
                if len(ring_atom_indices) >= motif_size_threshold:
                    ring_list_sorted = sorted(ring_atom_indices)
                    if ring_list_sorted not in motifs:
                        motifs.append(ring_list_sorted)
                        is_aromatic_ring = all(self.mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atom_indices)
                        motif_types.append("aromatic" if is_aromatic_ring else "aliphatic_ring")
        except Exception as e_ring: logging.warning(f"Ring extraction fail for {self.smiles}: {e_ring}")

        functional_groups = {
            "carboxyl": "[CX3](=O)[OX2H1]", "ester": "[CX3](=O)[OX2][#6]",
            "amide": "[NX3][CX3](=[OX1])", "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
            "hydroxyl": "[OX2H]", "carbonyl": "[CX3]=[OX1]"
        }
        for group_name, smarts in functional_groups.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches_tuples = self.mol.GetSubstructMatches(pattern)
                    for match_tuple in matches_tuples:
                        match_indices = list(match_tuple)
                        if len(match_indices) >= motif_size_threshold:
                            match_list_sorted = sorted(match_indices)
                            if match_list_sorted not in motifs:
                                motifs.append(match_list_sorted)
                                motif_types.append(group_name)
            except Exception as e_fg: logging.debug(f"FG '{group_name}' match fail: {e_fg}")

        if len(motifs) > self.max_motifs:
            sorted_pairs = sorted(zip(motifs, motif_types), key=lambda x_pair: len(x_pair[0]), reverse=True)
            motifs = [p[0] for p in sorted_pairs[:self.max_motifs]]
            motif_types = [p[1] for p in sorted_pairs[:self.max_motifs]]
        return motifs, motif_types

    def _determine_motif_type(self, motif_mol):
        """改善されたモチーフタイプ判定"""
        m_type = "other"
        if motif_mol is None: return m_type
        try:
            if any(atom.GetIsAromatic() for atom in motif_mol.GetAtoms()):
                m_type = "aromatic"
                if any(atom.GetAtomicNum() != 6 for atom in motif_mol.GetAtoms() if atom.IsInRing()):
                    m_type = "heterocycle"
            fg_defs = {
                "ester": "[#6][CX3](=O)[OX2][#6]", "amide": "[NX3][CX3](=[OX1])",
                "urea": "[NX3][CX3](=[OX1])[NX3]", "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
                "ether": "[OD2]([#6])[#6]", "olefin": "[CX3]=[CX3]",
                "carbonyl": "[CX3]=[OX1]", "lactam": "[NX3R][CX3R](=[OX1])",
                "lactone": "[#6R][CX3R](=[OX1])[OX2R][#6R]"
            }
            for name, smarts_str in fg_defs.items():
                try:
                    pattern_mol = Chem.MolFromSmarts(smarts_str)
                    if pattern_mol and motif_mol.HasSubstructMatch(pattern_mol):
                        m_type = name; break
                except: continue # Smarts parsing error
        except Exception as e_det: logging.debug(f"Motif type determination error: {e_det}")
        return m_type

    def _get_motif_features(self):
        """改善されたモチーフ特徴量計算"""
        global MOTIF_TYPES # Ensure MOTIF_TYPES is global or passed
        if not self.motifs or self.mol is None:
            return np.zeros((0, 1 + len(MOTIF_TYPES) + 3), dtype=np.float32)
        
        features = []
        num_mol_atoms = self.mol.GetNumAtoms()
        if num_mol_atoms == 0: # Avoid division by zero if mol has no atoms
             for _ in self.motifs: # Add fallback for each motif
                features.append([0.0] + [0]*len(MOTIF_TYPES) + [0,0,0])
             return np.array(features, dtype=np.float32)


        for motif_atom_indices, current_motif_type_str in zip(self.motifs, self.motif_types):
            try:
                size_feat = len(motif_atom_indices) / num_mol_atoms
                type_oh_feat = [0] * len(MOTIF_TYPES)
                if current_motif_type_str in MOTIF_TYPES:
                    type_oh_feat[MOTIF_TYPES.index(current_motif_type_str)] = 1
                
                # Check if all atoms in motif are part of ANY ring in the parent molecule
                is_ring_feat = all(self.mol.GetAtomWithIdx(idx).IsInRing() for idx in motif_atom_indices)
                is_aromatic_feat = any(self.mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in motif_atom_indices)
                has_hetero_feat = any(self.mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in motif_atom_indices)
                
                motif_f_val = [size_feat] + type_oh_feat + [int(is_ring_feat), int(is_aromatic_feat), int(has_hetero_feat)]
                features.append(motif_f_val)
            except Exception as e_mf:
                logging.debug(f"Motif feature calc error for {current_motif_type_str}: {e_mf}")
                features.append([0.0] + [0]*len(MOTIF_TYPES) + [0,0,0])
        return np.array(features, dtype=np.float32)

    def _build_motif_graph(self):
        """改善されたモチーフグラフ構築"""
        n_motifs_val = len(self.motifs)
        motif_edges_list = []
        motif_edge_features_list = []
        if n_motifs_val < 2 or self.mol is None:
            return motif_edges_list, np.zeros((0, 6), dtype=np.float32)

        for i in range(n_motifs_val):
            for j in range(i + 1, n_motifs_val):
                try:
                    shared_atoms_set = set(self.motifs[i]) & set(self.motifs[j])
                    has_shared_atoms_bool = len(shared_atoms_set) > 0
                    boundary_bonds_list = []
                    for atom_i_idx in self.motifs[i]:
                        for atom_j_idx in self.motifs[j]:
                            bond_obj = self.mol.GetBondBetweenAtoms(atom_i_idx, atom_j_idx)
                            if bond_obj: boundary_bonds_list.append(bond_obj)
                    has_bond_bool = len(boundary_bonds_list) > 0

                    if has_shared_atoms_bool or has_bond_bool:
                        motif_edges_list.append((i, j))
                        n_shared_feat = min(len(shared_atoms_set) / 10.0, 1.0)
                        n_bonds_feat = min(len(boundary_bonds_list) / 5.0, 1.0)
                        counts = [0,0,0,0] # S,D,T,A
                        for b in boundary_bonds_list:
                            bt = b.GetBondType()
                            if bt == BondType.SINGLE: counts[0]+=1
                            elif bt == BondType.DOUBLE: counts[1]+=1
                            elif bt == BondType.TRIPLE: counts[2]+=1
                            elif bt == BondType.AROMATIC: counts[3]+=1
                        ratios = [c/len(boundary_bonds_list) for c in counts] if boundary_bonds_list else [0,0,0,0]
                        motif_edge_features_list.append([n_shared_feat, n_bonds_feat] + ratios)
                except Exception as e_bmg: logging.debug(f"Motif graph edge {i}-{j} error: {e_bmg}")
        return motif_edges_list, np.array(motif_edge_features_list, dtype=np.float32) if motif_edge_features_list else np.zeros((0,6), dtype=np.float32)


    def _build_graph_data(self, fallback_mode=False):
        """改善されたグラフデータ構築"""
        global MOTIF_TYPES # Ensure MOTIF_TYPES is accessible
        try:
            if fallback_mode or self.mol is None: # Handle fallback or None mol
                num_actual_atoms = 0
                x_tensor = torch.FloatTensor(self.atom_features) # Use full padded features
            else:
                num_actual_atoms = self.mol.GetNumAtoms()
                x_tensor = torch.FloatTensor(self.atom_features[:num_actual_atoms]) if num_actual_atoms > 0 else torch.zeros((0, self.ATOM_FEATURE_DIM))

            edge_indices_list = []
            if not fallback_mode and self.mol is not None:
                for src_idx, neighbors_list in enumerate(self.adjacency_list): # adj_list from actual atoms
                    for dest_idx, _ in neighbors_list:
                        if src_idx < num_actual_atoms and dest_idx < num_actual_atoms: # Should always be true if adj_list is correct
                            edge_indices_list.append([src_idx, dest_idx])
            edge_index_tensor = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous() if edge_indices_list else torch.zeros((2,0), dtype=torch.long)
            
            edge_attr_tensor = torch.FloatTensor(self.bond_features) if len(self.bond_features) > 0 else torch.zeros((0, self.BOND_FEATURE_DIM))

            motif_indices_list = []
            if not fallback_mode and self.mol is not None:
                for motif_i, atom_indices_in_motif in enumerate(self.motifs):
                    for atom_idx_in_motif in atom_indices_in_motif:
                        if atom_idx_in_motif < num_actual_atoms:
                             motif_indices_list.append([atom_idx_in_motif, motif_i])
            motif_index_tensor = torch.tensor(motif_indices_list, dtype=torch.long).t().contiguous() if motif_indices_list else torch.zeros((2,0), dtype=torch.long)

            motif_x_tensor = torch.FloatTensor(self.motif_features) if hasattr(self, 'motif_features') and len(self.motif_features) > 0 else torch.zeros((0, 1 + len(MOTIF_TYPES) + 3))

            motif_edge_indices_list = []
            if not fallback_mode and hasattr(self, 'motif_graph'):
                for src_m_idx, dest_m_idx in self.motif_graph:
                    motif_edge_indices_list.extend([[src_m_idx, dest_m_idx], [dest_m_idx, src_m_idx]])
            motif_edge_index_tensor = torch.tensor(motif_edge_indices_list, dtype=torch.long).t().contiguous() if motif_edge_indices_list else torch.zeros((2,0), dtype=torch.long)
            
            motif_edge_attr_tensor = torch.FloatTensor(np.tile(self.motif_edge_features, (2,1))) if hasattr(self, 'motif_edge_features') and len(self.motif_edge_features) > 0 else torch.zeros((0,6))
            
            spectrum_tensor = torch.FloatTensor(self.spectrum) if self.spectrum is not None else None

            return {
                'x': x_tensor, 'edge_index': edge_index_tensor, 'edge_attr': edge_attr_tensor,
                'motif_index': motif_index_tensor, 'motif_x': motif_x_tensor,
                'motif_edge_index': motif_edge_index_tensor, 'motif_edge_attr': motif_edge_attr_tensor,
                'spectrum': spectrum_tensor, 'smiles': self.smiles, 'formula': self.formula,
                'atom_features': torch.FloatTensor(self.atom_features), 
                'motif_features': motif_x_tensor # Use the tensor version
            }
        except Exception as e_graph:
            logging.error(f"Graph data construction error for {self.smiles}: {e_graph}")
            atom_feats_fallback = self.atom_features if hasattr(self, 'atom_features') else np.zeros((self.max_atoms, self.ATOM_FEATURE_DIM))
            motif_feats_fallback = self.motif_features if hasattr(self, 'motif_features') else np.zeros((0, 1 + len(MOTIF_TYPES) + 3))
            return {
                'x': torch.zeros((0, self.ATOM_FEATURE_DIM)),
                'edge_index': torch.zeros((2,0), dtype=torch.long), 'edge_attr': torch.zeros((0, self.BOND_FEATURE_DIM)),
                'motif_index': torch.zeros((2,0), dtype=torch.long), 'motif_x': torch.zeros((0, 1 + len(MOTIF_TYPES) + 3)),
                'motif_edge_index': torch.zeros((2,0), dtype=torch.long), 'motif_edge_attr': torch.zeros((0,6)),
                'spectrum': None, 'smiles': self.smiles, 'formula': self.formula,
                'atom_features': torch.FloatTensor(atom_feats_fallback),
                'motif_features': torch.FloatTensor(motif_feats_fallback)
            }

    def validate(self):
        """データの妥当性を検証"""
        issues = []
        if self.mol is None:
            issues.append("Molecule object (self.mol) is None"); return issues
        if self.mol.GetNumAtoms() == 0 and self.smiles != "INVALID":
            issues.append("Molecule has no atoms")
        
        if not hasattr(self, 'atom_features') or not isinstance(self.atom_features, np.ndarray):
            issues.append("atom_features is not a numpy array or not initialized")
        elif self.atom_features.shape[0] != self.max_atoms:
             issues.append(f"Atom features row count: expected {self.max_atoms}, got {self.atom_features.shape[0]}")
        elif self.atom_features.shape[1] != self.ATOM_FEATURE_DIM:
            issues.append(f"Atom features dim: expected {self.ATOM_FEATURE_DIM}, got {self.atom_features.shape[1]}")

        if hasattr(self, 'bond_features') and isinstance(self.bond_features, np.ndarray) and self.bond_features.size > 0:
            if self.bond_features.shape[1] != self.BOND_FEATURE_DIM:
                issues.append(f"Bond features dim: expected {self.BOND_FEATURE_DIM}, got {self.bond_features.shape[1]}")
        
        if self.spectrum is not None:
            if not isinstance(self.spectrum, np.ndarray): issues.append("Spectrum is not numpy array")
            elif np.all(self.spectrum == 0) and self.smiles != "INVALID" : issues.append("Spectrum is all zeros") # Check if all elements are zero
        return issues

    def __repr__(self):
        num_actual_atoms = self.mol.GetNumAtoms() if self.mol else 0
        return f"MoleculeData(smiles='{self.smiles}', atoms={num_actual_atoms}, motifs={len(self.motifs) if hasattr(self, 'motifs') else 0})"

def normalize_spectrum(peaks: List[Tuple[int, int]], max_mz: int = MODEL_CONFIG.SPECTRUM_DIM, threshold: float = 0.01, top_n: int = 20) -> np.ndarray:
    """マススペクトルを正規化してベクトル形式に変換"""
    spectrum = np.zeros(max_mz)

    # ピークがない場合は空のスペクトルを返す
    if not peaks:
        return spectrum

    # Adhering to user's provided function structure for Issue #16:
    
    # Step 1 from user's code logic (for max_intensity calc):
    # Get intensities of peaks that are within the mz range
    intensities_in_range = [intensity for mz_val, intensity in peaks if mz_val < max_mz]
    if not intensities_in_range:
        logger.warning(f"No peaks found with mz < {max_mz}. Input peaks (first 5): {peaks[:5]}")
        return spectrum

    # 最大強度を見つける (Step 2 & 3 from user's code)
    max_intensity = max(intensities_in_range) # max() on non-empty list of intensities_in_range
    if max_intensity <= 0:
        logger.warning(f"All peak intensities within mz range are <= 0. Max intensity: {max_intensity}")
        return spectrum

    # 相対強度の閾値を計算 (Step 4)
    intensity_threshold = max_intensity * threshold

    # 閾値以上のピークを抽出 (Step 5) - also re-applying mz < max_mz
    filtered_peaks = [(mz, intensity) for mz, intensity in peaks 
                     if mz < max_mz and intensity >= intensity_threshold]

    # CRITICAL FIX: Handle case where no peaks pass the threshold (Step 6 from user's code)
    if not filtered_peaks:
        logger.warning(f"No peaks passed intensity threshold {intensity_threshold:.4f} (max_intensity in range: {max_intensity:.4f}).")
        return spectrum

    # 上位N個のピークのみを保持 (Step 7)
    if top_n > 0 and len(filtered_peaks) > top_n:
        # 強度の降順でソート
        filtered_peaks.sort(key=lambda x: x[1], reverse=True)
        # 上位N個のみを保持
        filtered_peaks = filtered_peaks[:top_n]

    # 選択されたピークをスペクトルに設定 (Step 8 & 9)
    for mz, intensity in filtered_peaks:
        # The user code has an additional `0 <= mz` check here.
        # Since mz for spectra is usually non-negative, this is fine.
        # max_mz is exclusive for array indexing (0 to max_mz-1)
        if 0 <= mz < max_mz:  # Ensure mz is within the bounds of the spectrum array
            spectrum[mz] = intensity / max_intensity # Intensities are normalized by the true max_intensity in range

    return spectrum

class ChemicalStructureSpectumDataset(Dataset):
    """化学構造とマススペクトルのデータセット"""
    
    def __init__(self, structures=None, spectra=None, structure_spectrum_pairs=None, lazy_load=True):
        self.lazy_load = lazy_load
        
        if self.lazy_load:
            # In lazy_load mode, 'structures', 'spectra', and 'structure_spectrum_pairs'
            # are expected to be lists of paths or path pairs.
            self.structure_paths = structures if structures is not None else []
            self.spectra_paths = spectra if spectra is not None else []
            self.structure_spectrum_pair_paths = structure_spectrum_pairs if structure_spectrum_pairs is not None else []
            
            self.n_pairs = len(self.structure_spectrum_pair_paths)
            self.n_structures = len(self.structure_paths)
            self.n_spectra = len(self.spectra_paths)
        else:
            # In non-lazy_load mode, 'structures', 'spectra', and 'structure_spectrum_pairs'
            # are expected to be lists of loaded data objects.
            self.structures = structures if structures is not None else [] # List of MoleculeData
            self.spectra = spectra if spectra is not None else []          # List of np.array
            self.structure_spectrum_pairs = structure_spectrum_pairs if structure_spectrum_pairs is not None else [] # List of (MoleculeData, np.array)

            self.n_pairs = len(self.structure_spectrum_pairs)
            self.n_structures = len(self.structures)
            self.n_spectra = len(self.spectra)
            
        self.total = self.n_pairs + self.n_structures + self.n_spectra
    
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        if self.lazy_load:
            # --- LAZY LOADING LOGIC ---
            if idx < self.n_pairs:
                struct_path, spec_path = self.structure_spectrum_pair_paths[idx]
                
                mol = Chem.MolFromMolFile(struct_path)
                if mol is None:
                    logging.error(f"LazyLoad Error: Failed to load MOL file: {struct_path} for supervised pair.")
                    # Return a structure that collate_fn can somewhat handle, or that indicates error.
                    # Depending on collate_fn, this might need adjustment.
                    # For now, returning None for data fields.
                    return {'type': 'supervised', 'structure': None, 'spectrum': None}

                peaks = []
                try:
                    with open(spec_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 2:
                                peaks.append((int(float(parts[0])), int(float(parts[1])))) # mz can be float in some MSP, int for normalize
                except Exception as e:
                    logging.error(f"LazyLoad Error: Failed to load/parse spectrum file {spec_path} for supervised pair: {e}")
                    # Create MoleculeData for structure if mol is valid, but spectrum is None.
                    structure_obj = MoleculeData(mol) if mol else None
                    return {'type': 'supervised', 'structure': structure_obj, 'spectrum': None}

                # Assuming normalize_spectrum uses MODEL_CONFIG.SPECTRUM_DIM internally for default max_mz
                spectrum_array = normalize_spectrum(peaks) 
                structure_obj = MoleculeData(mol, spectrum_array) # Pass spectrum to MoleculeData

                return {
                    'type': 'supervised',
                    'structure': structure_obj,
                    'spectrum': spectrum_array
                }

            elif idx < self.n_pairs + self.n_structures:
                struct_path = self.structure_paths[idx - self.n_pairs]
                mol = Chem.MolFromMolFile(struct_path)
                if mol is None:
                    logging.error(f"LazyLoad Error: Failed to load MOL file for unsupervised structure: {struct_path}")
                    return {'type': 'unsupervised_structure', 'structure': None, 'spectrum': None}
                
                structure_obj = MoleculeData(mol)

                return {
                    'type': 'unsupervised_structure',
                    'structure': structure_obj,
                    'spectrum': None
                }
            else: # Unsupervised spectrum
                spec_path = self.spectra_paths[idx - self.n_pairs - self.n_structures]
                peaks = []
                try:
                    with open(spec_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 2:
                                peaks.append((int(float(parts[0])), int(float(parts[1]))))
                except Exception as e:
                    logging.error(f"LazyLoad Error: Failed to load/parse spectrum file {spec_path} for unsupervised spectrum: {e}")
                    return {'type': 'unsupervised_spectrum', 'structure': None, 'spectrum': None}

                spectrum_array = normalize_spectrum(peaks)

                return {
                    'type': 'unsupervised_spectrum',
                    'structure': None,
                    'spectrum': spectrum_array
                }
        else:
            # --- EXISTING NON-LAZY LOGIC ---
            if idx < self.n_pairs:
                # structure and spectrum are already loaded MoleculeData and np.array
                structure, spectrum = self.structure_spectrum_pairs[idx]
                return {
                    'type': 'supervised',
                    'structure': structure,
                    'spectrum': torch.FloatTensor(spectrum) if isinstance(spectrum, np.ndarray) else spectrum
                }
            elif idx < self.n_pairs + self.n_structures:
                structure = self.structures[idx - self.n_pairs] # MoleculeData
                return {
                    'type': 'unsupervised_structure',
                    'structure': structure,
                    'spectrum': None
                }
            else: # Unsupervised spectrum
                spectrum = self.spectra[idx - self.n_pairs - self.n_structures] # np.array
                return {
                    'type': 'unsupervised_spectrum',
                    'structure': None,
                    'spectrum': torch.FloatTensor(spectrum) if isinstance(spectrum, np.ndarray) else spectrum
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
    spectra = []
    for item in batch_dict['spectrum']:
        if item is not None:
            if isinstance(item, torch.Tensor):
                spectra.append(item)
            elif isinstance(item, np.ndarray):
                spectra.append(torch.FloatTensor(item))
            else:
                # その他の形式の場合もFloatTensorに変換を試みる
                try:
                    spectra.append(torch.FloatTensor(item))
                except:
                    pass
    
    if spectra:
        batch_dict['spectrum_tensor'] = torch.stack(spectra)
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
                supervised_structures_filtered = []
                supervised_spectra_to_stack = []

                for i in supervised_indices:
                    spectrum_item = batch['spectrum'][i]
                    structure_item = batch['structure'][i]
                    
                    # 両方が有効な場合のみ追加
                    if spectrum_item is not None and structure_item is not None:
                        supervised_structures_filtered.append(structure_item)
                        if isinstance(spectrum_item, torch.Tensor):
                            supervised_spectra_to_stack.append(spectrum_item)
                        else:
                            # Ensure torch is imported
                            supervised_spectra_to_stack.append(torch.FloatTensor(spectrum_item))
                
                # If, after filtering, there are no valid supervised samples in this batch, skip
                if not supervised_spectra_to_stack:
                    continue
                
                supervised_spectra = torch.stack(supervised_spectra_to_stack).to(self.device)
                supervised_structures = supervised_structures_filtered # Use the filtered list of structures
                
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
                
                # New calculation for loss_p2s:
                if outputs.get('predicted_structure') and supervised_structures: # Check if prediction and targets are available
                    structural_targets = self._create_structural_targets(supervised_structures, MODEL_CONFIG.MAX_ATOMS, self.device)

                    # Ensure predicted_structure contains the expected keys
                    pred_node_exists = outputs['predicted_structure'].get('node_exists')
                    pred_node_types = outputs['predicted_structure'].get('node_types')

                    if pred_node_exists is not None and pred_node_types is not None:
                        node_exists_loss = F.binary_cross_entropy(
                            pred_node_exists,
                            structural_targets['node_exists']
                        )

                        predicted_node_types_permuted = pred_node_types.permute(0, 2, 1)
                        node_types_loss = F.cross_entropy(
                            predicted_node_types_permuted,
                            structural_targets['node_types']
                        )
                        loss_p2s = node_exists_loss + node_types_loss
                    else:
                        # Handle case where predicted structure or its components might be missing
                        loss_p2s = torch.tensor(0.0, device=self.device) # Or log a warning
                else:
                    loss_p2s = torch.tensor(0.0, device=self.device) # Or log a warning

                # 合計損失
                loss = loss_s2p + loss_p2s
                
                # 勾配計算と最適化
                loss.backward()
                self.optimizer.step()
                
                # 損失を追跡
                epoch_loss += loss.item()
                
                # メトリクスに追加
                self.metrics['structure_to_spectrum_loss'].append(loss_s2p.item())
                self.metrics['spectrum_to_structure_loss'].append(loss_p2s.item())
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            total_loss += avg_epoch_loss
            
            logger.info(f"Supervised Epoch {epoch+1}/{epochs} Loss: {avg_epoch_loss:.4f}")
        
        # 平均損失を返す
        return total_loss / epochs
    
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

    def _create_structural_targets(self, molecule_data_list: List[MoleculeData], max_atoms_target: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Create structural targets for loss computation"""
        batch_node_exists = []
        batch_node_types = []

        # Ensure ATOM_TYPE_TO_INDEX and UNKNOWN_ATOM_INDEX_TARGET are accessible
        # These should be defined globally or passed to this class/method.
        # Assuming they are globally defined as per Part 1.

        for md in molecule_data_list:
            num_actual_atoms = md.mol.GetNumAtoms()

            # Node exists target
            node_exists_single = torch.zeros(max_atoms_target, 1, device=device)
            if num_actual_atoms > 0:
                node_exists_single[:min(num_actual_atoms, max_atoms_target)] = 1.0
            batch_node_exists.append(node_exists_single)

            # Node types target
            # Ensure UNKNOWN_ATOM_INDEX_TARGET is an int.
            node_types_single = torch.full((max_atoms_target,), UNKNOWN_ATOM_INDEX_TARGET, dtype=torch.long, device=device) 
            for i in range(min(num_actual_atoms, max_atoms_target)):
                atom = md.mol.GetAtomWithIdx(i)
                atomic_num = atom.GetAtomicNum()
                # Ensure ATOM_TYPE_TO_INDEX is a dict {atomic_num: index}.
                node_types_single[i] = ATOM_TYPE_TO_INDEX.get(atomic_num, UNKNOWN_ATOM_INDEX_TARGET)
            batch_node_types.append(node_types_single)

        return {
            'node_exists': torch.stack(batch_node_exists),
            'node_types': torch.stack(batch_node_types)
        }
    
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
                    structure = batch['structure'][idx]
                    
                    # 構造からスペクトルを予測
                    structure_data = {'structure': structure}
                    outputs = self.model(structure_data, direction="structure_to_spectrum")
                    predicted_spectrum = outputs['predicted_spectrum']
                    
                    # 疑似ラベルとして追加
                    pseudo_labeled_data.append({
                        'structure': structure,
                        'spectrum': predicted_spectrum.cpu().numpy(),
                        'confidence': self._calculate_confidence(outputs)
                    })
                
                # スペクトルのみのデータを処理
                spectrum_indices = [i for i, t in enumerate(batch['type']) 
                                   if t == 'unsupervised_spectrum']
                
                for idx in spectrum_indices:
                    spectrum_input = torch.FloatTensor(batch['spectrum'][idx]).unsqueeze(0).to(self.device)
                    
                    # スペクトルから構造を予測
                    spectrum_data = {'spectrum': spectrum_input}
                    outputs = self.model(spectrum_data, direction="spectrum_to_structure")
                    predicted_structure_batch = outputs['predicted_structure']
                    
                    # Create a dictionary for a single prediction by indexing/squeezing the batch dimension
                    single_pred_data_dict = {}
                    for key, batched_tensor in predicted_structure_batch.items():
                        if isinstance(batched_tensor, torch.Tensor):
                            single_pred_data_dict[key] = batched_tensor.squeeze(0) # Remove the batch dim of 1
                        else:
                            single_pred_data_dict[key] = batched_tensor # Handle non-tensor data if any

                    # Now call _convert_to_molecule with the single prediction data
                    molecule_object = self._convert_to_molecule(single_pred_data_dict)
                    
                    # MoleculeDataオブジェクトでラップ
                    if molecule_object is not None:
                        molecule_data = MoleculeData(molecule_object)
                    else:
                        continue  # 変換に失敗した場合はスキップ
                    
                    # 疑似ラベルとして追加
                    pseudo_labeled_data.append({
                        'structure': molecule_data,  # MoleculeDataオブジェクト
                        'spectrum': batch['spectrum'][idx], # This is the original spectrum
                        'confidence': self._calculate_confidence(outputs)
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
    
    def _calculate_confidence(self, outputs):
        """予測の信頼度を計算"""
        # ここでは単純な例として、予測値の確率分布のエントロピーを使用
        # 実際のアプリケーションでは、より洗練された信頼度の計算が必要
        if 'predicted_spectrum' in outputs:
            # スペクトル予測の場合
            spectrum = outputs['predicted_spectrum']
            entropy = -(spectrum * torch.log(spectrum + 1e-10)).sum()
            max_entropy = -torch.log(torch.tensor(1.0 / spectrum.size(0)))
            confidence = 1.0 - (entropy / max_entropy)
        else:
            # 構造予測の場合
            structure = outputs['predicted_structure']
            node_exists = structure['node_exists']
            node_exists_entropy = -(node_exists * torch.log(node_exists + 1e-10) + 
                                   (1 - node_exists) * torch.log(1 - node_exists + 1e-10)).mean()
            confidence = torch.exp(-node_exists_entropy).item()
        
        return confidence.item()
    
    def _convert_to_molecule(self, predicted_structure):
        """予測された構造を分子に変換"""
        # 予測から分子を構築する処理
        # 実際の実装では、予測された原子タイプと結合を使用してRDKit分子を構築
        
        # ここではサンプル実装として、予測された原子と結合を使用
        node_exists = predicted_structure['node_exists'].cpu().numpy() > 0.5
        node_types = predicted_structure['node_types'].argmax(dim=1).cpu().numpy()
        edge_exists = predicted_structure['edge_exists'].cpu().numpy() > 0.5
        edge_types = predicted_structure['edge_types'].argmax(dim=1).cpu().numpy()
        
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
            ChemicalStructureSpectumDataset(
                structure_spectrum_pairs=high_confidence_data,
                lazy_load=False  # MoleculeDataオブジェクトを直接渡しているため
            )
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
                
                if outputs.get('predicted_structure') and supervised_structures: # Check if prediction and targets are available
                    # supervised_structures is a list of MoleculeData objects from the batch
                    # MAX_ATOMS is a global constant
                    structural_targets = self._create_structural_targets(supervised_structures, MODEL_CONFIG.MAX_ATOMS, self.device)

                    pred_node_exists = outputs['predicted_structure'].get('node_exists')
                    pred_node_types = outputs['predicted_structure'].get('node_types')

                    if pred_node_exists is not None and pred_node_types is not None:
                        # Calculate node_exists_loss (Binary Cross Entropy)
                        node_exists_loss = F.binary_cross_entropy(
                            pred_node_exists,
                            structural_targets['node_exists']
                        )

                        # Calculate node_types_loss (Cross Entropy)
                        # Input shape for cross_entropy: (N, C, ...) where C is number of classes
                        # pred_node_types is likely [batch_size, max_atoms, num_classes]
                        # structural_targets['node_types'] is [batch_size, max_atoms] (class indices)
                        predicted_node_types_permuted = pred_node_types.permute(0, 2, 1)
                        node_types_loss = F.cross_entropy(
                            predicted_node_types_permuted,
                            structural_targets['node_types']
                        )
                        
                        loss_p2s = node_exists_loss + node_types_loss
                    else:
                        loss_p2s = torch.tensor(0.0, device=self.device) # Fallback if structure components are missing
                else:
                    loss_p2s = torch.tensor(0.0, device=self.device) # Fallback if no predicted structure or no supervised structures

                # The total loss for the batch
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

def prepare_dataset(msp_data: Dict[str, Dict], mol_dict: Dict[str, Chem.Mol], 
                   spectrum_dim: int = MODEL_CONFIG.SPECTRUM_DIM, test_ratio: float = 0.1, val_ratio: float = 0.1, # Assuming SPECTRUM_DIM is global
                   unlabeled_ratio: float = 0.3, seed: int = 42):
    """データセットを準備する"""
    # 共通のIDを持つ化合物だけを使用
    common_ids = set(msp_data.keys()) & set(mol_dict.keys())
    # logger.info(f"Found {len(common_ids)} compounds with both MSP and MOL data") # Changed print to logger.info
    # Using print as per user's snippet for now.
    print(f"Found {len(common_ids)} compounds with both MSP and MOL data")


    # データを処理
    dataset = []
    # Make sure tqdm is imported: from tqdm import tqdm
    for compound_id in tqdm(common_ids, desc="Preparing dataset"):
        try:
            # スペクトルを抽出して正規化
            peaks = msp_data[compound_id]['peaks']
            # normalize_spectrum should be available
            spectrum = normalize_spectrum(peaks, max_mz=spectrum_dim)

            # 分子を処理
            mol = mol_dict[compound_id]
            # MoleculeData should be available
            mol_data_obj = MoleculeData(mol, spectrum)

            # データセットに追加
            dataset.append((compound_id, mol_data_obj))
        except Exception as e:
            # logger.error(f"Error processing compound {compound_id}: {e}") # Changed print to logger.error
            print(f"Error processing compound {compound_id}: {e}")


    # CRITICAL FIX: Validate ratios before processing
    if unlabeled_ratio + test_ratio + val_ratio >= 1.0:
        raise ValueError(f"Combined ratios ({unlabeled_ratio + test_ratio + val_ratio:.2f}) must be < 1.0. unlabeled_ratio={unlabeled_ratio}, test_ratio={test_ratio}, val_ratio={val_ratio}")

    if len(dataset) == 0:
        # logger.critical("No valid compounds found in dataset after processing common_ids.") # Changed print to logger
        raise ValueError("No valid compounds found in dataset")

    # 乱数シードを設定
    # Make sure random is imported
    random.seed(seed)

    # データセットをシャッフル
    random.shuffle(dataset)

    # CRITICAL FIX: Correct data splitting logic
    total_samples = len(dataset)

    # Calculate absolute numbers
    n_unlabeled = int(total_samples * unlabeled_ratio)
    n_test = int(total_samples * test_ratio)
    n_val = int(total_samples * val_ratio)
    n_train = total_samples - n_unlabeled - n_test - n_val

    # Ensure we have enough samples
    if n_train <= 0:
        # logger.error(f"Not enough samples for training. Total: {total_samples}, Train: {n_train}, Unlabeled: {n_unlabeled}, Test: {n_test}, Val: {n_val}")
        raise ValueError(f"Not enough samples for training ({n_train}). Total: {total_samples}, "
                        f"Unlabeled: {n_unlabeled}, Test: {n_test}, Val: {n_val}. Check ratios.")

    # Split data
    unlabeled_data = dataset[:n_unlabeled]
    # Slicing must be correct:
    idx_start_test = n_unlabeled
    idx_end_test = n_unlabeled + n_test
    test_data = dataset[idx_start_test:idx_end_test]

    idx_start_val = idx_end_test
    idx_end_val = idx_end_test + n_val
    val_data = dataset[idx_start_val:idx_end_val]
    
    idx_start_train = idx_end_val
    train_data = dataset[idx_start_train:]


    # Verify splits
    assert len(unlabeled_data) + len(test_data) + len(val_data) + len(train_data) == total_samples,         f"Data split sanity check failed. Sum of splits: {len(unlabeled_data) + len(test_data) + len(val_data) + len(train_data)}, Total: {total_samples}"
    assert len(train_data) > 0, f"Training set is empty after split: {len(train_data)}"
    # logger.info(f"Data split - Total: {total_samples}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, Unlabeled: {len(unlabeled_data)}")
    print(f"Data split - Total: {total_samples}, Train: {len(train_data)}, "
          f"Val: {len(val_data)}, Test: {len(test_data)}, Unlabeled: {len(unlabeled_data)}")


    # 教師なしデータを構造のみとスペクトルのみに分割
    unlabeled_structures = []
    unlabeled_spectra = []

    for _, mol_data_obj_unlabeled in unlabeled_data: # renamed var to avoid conflict
        if random.random() < 0.5:
            # 構造のみのデータ（スペクトルを破棄）
            mol_data_obj_unlabeled.spectrum = None # Make sure this modification is okay (shallow copy vs deep copy of mol_data_obj)
                                         # If mol_data_obj is referenced elsewhere, this might be an issue.
                                         # Assuming it's fine for this context as it's for unsupervised learning.
            unlabeled_structures.append(mol_data_obj_unlabeled)
        else:
            # スペクトルのみのデータ（構造情報は保持）
            unlabeled_spectra.append(mol_data_obj_unlabeled.spectrum)

    # 構造-スペクトルのペアを作成
    structure_spectrum_pairs = []
    for _, mol_data_obj_train in train_data: # renamed var
        structure_spectrum_pairs.append((mol_data_obj_train, mol_data_obj_train.spectrum))

    # データセットを作成
    # ChemicalStructureSpectumDataset should be available
    train_dataset = ChemicalStructureSpectumDataset(
        structures=unlabeled_structures,
        spectra=unlabeled_spectra,
        structure_spectrum_pairs=structure_spectrum_pairs,
        lazy_load=False  # MoleculeDataオブジェクトを直接渡しているため
    )

    val_pairs = []
    for _, mol_data_obj_val in val_data: # renamed var
        val_pairs.append((mol_data_obj_val, mol_data_obj_val.spectrum))

    val_dataset = ChemicalStructureSpectumDataset(
        structure_spectrum_pairs=val_pairs,
        lazy_load=False  # MoleculeDataオブジェクトを直接渡しているため
    )

    test_pairs = []
    for _, mol_data_obj_test in test_data: # renamed var
        test_pairs.append((mol_data_obj_test, mol_data_obj_test.spectrum))

    test_dataset = ChemicalStructureSpectumDataset(
        structure_spectrum_pairs=test_pairs,
        lazy_load=False  # MoleculeDataオブジェクトを直接渡しているため
    )
    # logger.info(f"Final dataset objects - Train: {len(train_dataset)} items, Val: {len(val_dataset)} items, Test: {len(test_dataset)} items")
    # logger.info(f"Train dataset composition: {len(structure_spectrum_pairs)} supervised pairs, {len(unlabeled_structures)} unsupervised structures, {len(unlabeled_spectra)} unsupervised spectra")
    print(f"Dataset split (dataset objects) - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Train dataset composition: {len(structure_spectrum_pairs)} supervised pairs, {len(unlabeled_structures)} unsupervised structures, {len(unlabeled_spectra)} unsupervised spectra")


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
    
    # タイトルを設定
    plt.figure(figsize=(size[0]/100, size[1]/100))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    
    if not save_path:
        plt.show()
    plt.close()
    
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
                "spectrum_dim": MODEL_CONFIG.SPECTRUM_DIM,
                "test_ratio": 0.1,
                "val_ratio": 0.1,
                "unlabeled_ratio": 0.3,
                "seed": 42
            },
            "model": {
                "hidden_dim": MODEL_CONFIG.HIDDEN_DIM,
                "latent_dim": MODEL_CONFIG.LATENT_DIM,
                "atom_fdim": MODEL_CONFIG.ATOM_FEATURE_DIM,
                "bond_fdim": MODEL_CONFIG.BOND_FEATURE_DIM,
                "motif_fdim": MODEL_CONFIG.MOTIF_FEATURE_DIM
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

# Additional helper functions (should be at the end of the file)
# import logging # Ensure logging is imported if not already - ALREADY IMPORTED
# from rdkit import Chem # Ensure Chem is imported - ALREADY IMPORTED

def safe_mol_from_smiles(smiles, sanitize=True):
    """安全にSMILESから分子を作成"""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            logging.warning(f"Failed to create molecule from SMILES: {smiles}")
        return mol # Return mol whether it's None or a valid molecule
    except Exception as e:
        logging.error(f"Error creating molecule from SMILES {smiles}: {e}")
        return None

def safe_mol_from_mol_file(file_path, sanitize=True):
    """安全にMOLファイルから分子を作成"""
    try:
        mol = Chem.MolFromMolFile(file_path, sanitize=sanitize)
        if mol is None:
            logging.warning(f"Failed to load MOL file: {file_path}")
        return mol # Return mol whether it's None or a valid molecule
    except Exception as e:
        logging.error(f"Error loading MOL file {file_path}: {e}")
        return None
