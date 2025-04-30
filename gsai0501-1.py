import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch_geometric.nn import GCNConv, GINConv
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, MolToSmiles, MurckoScaffold, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit.Chem.rdchem import BondType
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import random
import copy
import math
from typing import Dict, List, Tuple, Set, Union, Optional
from collections import defaultdict, deque

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        self.electron_count = sum(atom.GetNumElectrons() for atom in mol.GetAtoms())
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
    
    def forward(self, data):
        """順伝播: 化学構造から潜在表現を生成"""
        # 原子特徴量をエンコード
        atom_features = self.atom_encoder(data['atom_features'])
        
        # 結合特徴量をエンコード
        bond_features = self.bond_encoder(data['bond_features'])
        
        # モチーフ特徴量をエンコード
        motif_features = self.motif_encoder(data['motif_features'])
        
        # GCN層で原子グラフを更新
        atom_embeddings = atom_features
        for gcn in self.gcn_layers:
            atom_embeddings = F.relu(gcn(atom_embeddings, data['edge_index']))
        
        # GIN層でモチーフグラフを更新
        motif_embeddings = motif_features
        for gin in self.gin_layers:
            motif_embeddings = F.relu(gin(motif_embeddings, data['motif_edge_index']))
        
        # グローバルアテンション適用
        atom_attn, _ = self.attention(atom_embeddings, atom_embeddings, atom_embeddings)
        motif_attn, _ = self.attention(motif_embeddings, motif_embeddings, motif_embeddings)
        
        # グローバル表現の作成
        atom_global = torch.mean(atom_attn, dim=0)
        motif_global = torch.mean(motif_attn, dim=0)
        
        # 原子とモチーフの表現を結合
        combined = torch.cat([atom_global, motif_global], dim=0)
        
        # 潜在表現に射影
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
        # 潜在表現を拡張
        expanded = self.expander(latent)
        
        # ノード（原子）の特徴量を生成
        node_hiddens = expanded[:max_atoms, :self.hidden_dim]
        
        # ノード（原子）の存在確率
        node_exists = self.graph_generator['node_existence'](node_hiddens)
        
        # ノード（原子）の種類
        node_types = self.graph_generator['node_type'](node_hiddens)
        
        # エッジ（結合）の特徴量を生成
        edge_hiddens = []
        for i in range(max_atoms):
            for j in range(i+1, max_atoms):
                # 両方のノード特徴量を結合
                combined = torch.cat([node_hiddens[i], node_hiddens[j]], dim=0)
                edge_hiddens.append(combined)
        
        edge_hiddens = torch.stack(edge_hiddens) if edge_hiddens else torch.zeros((0, self.hidden_dim * 2))
        
        # エッジ（結合）の存在確率
        edge_exists = self.graph_generator['edge_existence'](edge_hiddens)
        
        # エッジ（結合）の種類
        edge_types = self.graph_generator['edge_type'](edge_hiddens)
        
        # モチーフの特徴量を生成
        motif_hiddens = expanded[max_atoms:, :self.hidden_dim]
        
        # モチーフの存在確率
        motif_exists = self.motif_generator['motif_existence'](motif_hiddens)
        
        # モチーフの種類
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