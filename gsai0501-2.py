# パート2: トレーニングインフラストラクチャと自己成長メカニズム

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, MolToSmiles, MurckoScaffold, Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import BondType
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import random
import copy
import math
from typing import Dict, List, Tuple, Set, Union, Optional
from collections import defaultdict, deque
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import itertools
import logging
import datetime

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
        edge_attr = torch.FloatTensor(self.bond_features) if len(self.bond_features) > 0 else torch.zeros((0, self.bond_features.shape[1] if self.bond_features.shape[0] > 0 else 7))
        
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
        motif_edge_attr = torch.FloatTensor(self.motif_edge_features) if len(self.motif_edge_features) > 0 else torch.zeros((0, self.motif_edge_features.shape[1] if self.motif_edge_features.shape[0] > 0 else 6))
        
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
    spectra = [item for item in batch_dict['spectrum'] if item is not None]
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
                
                structure_losses = []
                for i, structure_output in enumerate(outputs['predicted_structure']):
                    # 構造に関する損失を計算
                    node_exists_loss = F.binary_cross_entropy(
                        structure_output['node_exists'],
                        torch.tensor([s['node_exists'] for s in supervised_structures], device=self.device)
                    )
                    node_types_loss = F.cross_entropy(
                        structure_output['node_types'],
                        torch.tensor([s['node_types'] for s in supervised_structures], device=self.device)
                    )
                    structure_loss = node_exists_loss + node_types_loss
                    structure_losses.append(structure_loss)
                
                loss_p2s = sum(structure_losses) / len(structure_losses) if structure_losses else 0
                
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
                    spectrum = torch.FloatTensor(batch['spectrum'][idx]).to(self.device)
                    
                    # スペクトルから構造を予測
                    spectrum_data = {'spectrum': spectrum}
                    outputs = self.model(spectrum_data, direction="spectrum_to_structure")
                    predicted_structure = outputs['predicted_structure']
                    
                    # 疑似ラベルとして追加
                    pseudo_labeled_data.append({
                        'structure': self._convert_to_molecule(predicted_structure),
                        'spectrum': batch['spectrum'][idx],
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
                
                structure_losses = []
                for structure_output in outputs['predicted_structure']:
                    # 構造に関する損失を計算
                    node_exists_loss = F.binary_cross_entropy(
                        structure_output['node_exists'],
                        torch.tensor([s['node_exists'] for s in supervised_structures], device=self.device)
                    )
                    node_types_loss = F.cross_entropy(
                        structure_output['node_types'],
                        torch.tensor([s['node_types'] for s in supervised_structures], device=self.device)
                    )
                    structure_loss = node_exists_loss + node_types_loss
                    structure_losses.append(structure_loss)
                
                loss_p2s = sum(structure_losses) / len(structure_losses) if structure_losses else 0
                
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