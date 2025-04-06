import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import re
import random
from typing import Dict, List, Tuple, Set, Union

# GPU使用設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# パス設定
DATA_DIR = "data"
MOL_DIR = os.path.join(DATA_DIR, "mol_files")
MSP_FILE = os.path.join(DATA_DIR, "NIST17.MSP")

# 定数
MAX_MZ = 2000  # m/zの最大値
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
HIDDEN_DIM = 128
DEPTH = 3  # DMPNNの深さ

# 非金属元素の原子番号リスト
NON_METAL_ATOMIC_NUMS = [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53]  # H, C, N, O, F, P, S, Cl, Se, Br, I

#------------------------------------------------------
# 1. データの読み込みと前処理
#------------------------------------------------------

class MoleculeGraph:
    """分子構造のグラフ表現を扱うクラス"""
    
    def __init__(self, mol):
        """RDKitの分子オブジェクトからグラフ表現を構築"""
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()
        self.n_bonds = mol.GetNumBonds()
        
        # 原子の特徴量の初期化
        self.atom_features = self._get_atom_features()
        
        # 結合の特徴量と隣接リストの初期化
        self.bond_features, self.adjacency_list = self._get_bond_features_and_adjacency()
        
    def _get_atom_features(self) -> np.ndarray:
        """原子の特徴量を取得"""
        features = []
        for atom in self.mol.GetAtoms():
            # 原子の特徴量をベクトル化
            atom_features = []
            
            # 原子番号（one-hot）
            atom_type = atom.GetAtomicNum()
            atom_features.extend([int(atom_type == i) for i in range(1, 119)])  # 周期表の全元素
            
            # 形式電荷
            charge = atom.GetFormalCharge()
            atom_features.extend([int(charge == i) for i in range(-5, 6)])  # -5から+5まで
            
            # 混成軌道状態
            hybridization = atom.GetHybridization()
            hybridization_types = [Chem.rdchem.HybridizationType.SP, 
                                  Chem.rdchem.HybridizationType.SP2,
                                  Chem.rdchem.HybridizationType.SP3, 
                                  Chem.rdchem.HybridizationType.SP3D, 
                                  Chem.rdchem.HybridizationType.SP3D2]
            atom_features.extend([int(hybridization == i) for i in hybridization_types])
            
            # 水素の数
            h_count = atom.GetTotalNumHs()
            atom_features.extend([int(h_count == i) for i in range(5)])
            
            # 芳香族性
            atom_features.append(int(atom.GetIsAromatic()))
            
            # 環構造の一部か
            atom_features.append(int(atom.IsInRing()))
            
            features.append(atom_features)
        
        return np.array(features, dtype=np.float32)
    
    def _get_bond_features_and_adjacency(self) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """結合の特徴量と隣接リストを取得"""
        bond_features = []
        adjacency_list = [[] for _ in range(self.n_atoms)]
        
        for bond in self.mol.GetBonds():
            # 結合のインデックス
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # 結合特徴量のベクトル化
            bond_feature = []
            
            # 結合タイプ（one-hot）
            bond_type = bond.GetBondType()
            bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
            bond_feature.extend([int(bond_type == i) for i in bond_types])
            
            # 環構造の一部か
            bond_feature.append(int(bond.IsInRing()))
            
            # 共役か
            bond_feature.append(int(bond.GetIsConjugated()))
            
            # ステレオ化学
            bond_feature.append(int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE))
            
            # 両方向に追加（無向グラフ）
            bond_idx = len(bond_features)
            adjacency_list[begin_idx].append((end_idx, bond_idx))
            adjacency_list[end_idx].append((begin_idx, bond_idx))
            
            bond_features.append(bond_feature)
        
        if len(bond_features) == 0:
            # 結合がない場合（単一原子分子など）
            return np.zeros((0, 7), dtype=np.float32), adjacency_list
        else:
            return np.array(bond_features, dtype=np.float32), adjacency_list

def read_mol_file(file_path: str) -> MoleculeGraph:
    """MOLファイルを読み込み、MoleculeGraphオブジェクトを返す"""
    mol = Chem.MolFromMolFile(file_path)
    if mol is None:
        raise ValueError(f"Failed to parse MOL file: {file_path}")
    return MoleculeGraph(mol)

def is_non_metal_only(mol) -> bool:
    """分子が非金属のみで構成されているかどうかを判定"""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in NON_METAL_ATOMIC_NUMS:
            return False
    return True

def parse_msp_file(file_path: str) -> Dict[str, Dict]:
    """MSPファイルを解析し、各化合物のマススペクトルデータを返す"""
    compound_data = {}
    current_compound = None
    current_id = None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Parsing MSP file"):
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

def normalize_spectrum(peaks: List[Tuple[int, int]], max_mz: int = MAX_MZ, threshold: float = 0.01, top_n: int = 20) -> np.ndarray:
    """
    マススペクトルを正規化してベクトル形式に変換（相対強度でフィルタリングし、上位N個のピークのみ保持）
    
    Args:
        peaks: (m/z, intensity)のリスト
        max_mz: 最大m/z値
        threshold: 相対閾値（ベースピークに対する割合、デフォルト1%）
        top_n: 保持する上位ピーク数（デフォルト20、指定した場合は上位N個のみ保持）
    """
    spectrum = np.zeros(max_mz + 1)
    
    # ピークがない場合は空のスペクトルを返す
    if not peaks:
        return spectrum
    
    # 最大強度を見つける
    max_intensity = max([intensity for mz, intensity in peaks if mz <= max_mz])
    if max_intensity <= 0:
        return spectrum
    
    # 相対強度の閾値を計算
    intensity_threshold = max_intensity * threshold
    
    # 閾値以上のピークを抽出
    filtered_peaks = [(mz, intensity) for mz, intensity in peaks 
                     if mz <= max_mz and intensity >= intensity_threshold]
    
    # 上位N個のピークのみを保持
    if top_n > 0 and len(filtered_peaks) > top_n:
        # 強度の降順でソート
        filtered_peaks.sort(key=lambda x: x[1], reverse=True)
        # 上位N個のみを保持
        filtered_peaks = filtered_peaks[:top_n]
    
    # 選択されたピークをスペクトルに設定
    for mz, intensity in filtered_peaks:
        spectrum[mz] = intensity
    
    # 最大値で正規化
    spectrum = spectrum / max_intensity
    
    return spectrum

class MassSpectrumDataset(Dataset):
    """マススペクトルデータセット"""
    
    def __init__(self, spectrum_data, mol_dir, non_metal_only=True):
        self.spectrum_data = spectrum_data
        self.mol_dir = mol_dir
        self.non_metal_only = non_metal_only
        self.ids = []
        
        print("Filtering compounds...")
        # 非金属のみの分子をフィルタリング
        for compound_id in tqdm(list(spectrum_data.keys())):
            mol_file = os.path.join(self.mol_dir, f"ID{compound_id}.MOL")
            if os.path.exists(mol_file):
                try:
                    mol = Chem.MolFromMolFile(mol_file)
                    if mol is not None:
                        if not non_metal_only or is_non_metal_only(mol):
                            self.ids.append(compound_id)
                except Exception as e:
                    print(f"Error loading MOL file for compound {compound_id}: {e}")
        
        print(f"Total compounds after filtering: {len(self.ids)}")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        compound_id = self.ids[idx]
        compound = self.spectrum_data[compound_id]
        
        # マススペクトルの処理（1%閾値フィルタリングと上位20ピーク保持を適用）
        peaks = compound['peaks']
        spectrum = normalize_spectrum(peaks, threshold=0.01, top_n=20)
        
        # MOLファイルの読み込み
        mol_file = os.path.join(self.mol_dir, f"ID{compound_id}.MOL")
        try:
            molecule_graph = read_mol_file(mol_file)
            
            # PyTorchテンソルに変換
            atom_features = torch.FloatTensor(molecule_graph.atom_features)
            bond_features = torch.FloatTensor(molecule_graph.bond_features)
            adjacency_list = molecule_graph.adjacency_list
            
            return {
                'id': compound_id,
                'spectrum': torch.FloatTensor(spectrum),
                'atom_features': atom_features,
                'bond_features': bond_features,
                'adjacency_list': adjacency_list,
                'n_atoms': molecule_graph.n_atoms,
                'peaks': peaks  # 元のピークデータも保持
            }
        except Exception as e:
            print(f"Error processing compound {compound_id}: {e}")
            # エラーが発生した場合はスキップ（実際の実装ではより適切なエラーハンドリングが必要）
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    """バッチ処理用の関数"""
    # IDのリスト
    ids = [item['id'] for item in batch]
    
    # スペクトルをスタック
    spectra = torch.stack([item['spectrum'] for item in batch])
    
    # グラフデータは各サンプルで構造が異なるため、リストで保持
    atom_features = [item['atom_features'] for item in batch]
    bond_features = [item['bond_features'] for item in batch]
    adjacency_lists = [item['adjacency_list'] for item in batch]
    n_atoms = [item['n_atoms'] for item in batch]
    
    # 元のピークデータも保持
    peaks = [item['peaks'] for item in batch]
    
    return {
        'ids': ids,
        'spectra': spectra,
        'atom_features': atom_features,
        'bond_features': bond_features,
        'adjacency_lists': adjacency_lists,
        'n_atoms': n_atoms,
        'peaks': peaks
    }

#------------------------------------------------------
# 2. DMPNNモデルの定義
#------------------------------------------------------

class DMPNN(nn.Module):
    """有向メッセージパッシングニューラルネットワーク"""
    
    def __init__(self, hidden_dim, depth, atom_fdim, bond_fdim, spectrum_dim):
        """
        初期化メソッド
        
        Args:
            hidden_dim: 隠れ層の次元
            depth: メッセージパッシングの繰り返し回数
            atom_fdim: 原子特徴量の次元
            bond_fdim: 結合特徴量の次元
            spectrum_dim: マススペクトルの次元
        """
        super(DMPNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        
        # 原子特徴量エンコーダ
        self.atom_encoder = nn.Linear(atom_fdim, hidden_dim)
        
        # 結合特徴量エンコーダ
        self.bond_encoder = nn.Linear(bond_fdim, hidden_dim)
        
        # メッセージパッシングネットワーク
        self.W_message = nn.Linear(hidden_dim, hidden_dim)
        
        # アップデートネットワーク
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 読み出しネットワーク
        self.W_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # スペクトル予測ネットワーク
        self.spectrum_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, spectrum_dim)
        )
        
        # スペクトル入力エンコーダ（逆方向のタスク用）
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(spectrum_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        
        # 原子/結合特徴量予測ネットワーク（逆方向のタスク用）
        self.atom_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, atom_fdim)
        )
    
    def forward(self, atom_features, bond_features, adjacency_lists, n_atoms, spectra=None, predict_spectrum=False):
        """
        フォワードパス
        
        Args:
            atom_features: 原子特徴量のリスト [batch_size, ...]
            bond_features: 結合特徴量のリスト [batch_size, ...]
            adjacency_lists: 隣接リストのリスト [batch_size, ...]
            n_atoms: 各分子の原子数のリスト [batch_size]
            spectra: マススペクトル [batch_size, spectrum_dim]
            predict_spectrum: スペクトル予測モードかどうか
            
        Returns:
            予測結果（タスクによって異なる）
        """
        if predict_spectrum:
            # 構造→スペクトル予測モード
            return self._forward_structure_to_spectrum(atom_features, bond_features, adjacency_lists, n_atoms)
        else:
            # スペクトル→構造予測モード
            assert spectra is not None, "Spectra must be provided for spectrum-to-structure prediction"
            return self._forward_spectrum_to_structure(spectra, n_atoms)
    
    def _forward_structure_to_spectrum(self, atom_features, bond_features, adjacency_lists, n_atoms):
        """構造からスペクトルを予測"""
        # 各分子の埋め込みを計算
        mol_embeddings = []
        
        for i in range(len(atom_features)):
            # 原子特徴量と結合特徴量をエンコード
            atom_vecs = self.atom_encoder(atom_features[i])
            
            if len(bond_features[i]) > 0:  # 結合がある場合
                bond_vecs = self.bond_encoder(bond_features[i])
                
                # メッセージパッシング
                message_vecs = bond_vecs.clone()
                
                for _ in range(self.depth):
                    # メッセージの更新
                    new_messages = []
                    
                    for atom_idx in range(n_atoms[i]):
                        for neighbor_idx, bond_idx in adjacency_lists[i][atom_idx]:
                            # 隣接原子からのメッセージを計算
                            neighbor_message = atom_vecs[neighbor_idx] + message_vecs[bond_idx]
                            new_messages.append((bond_idx, self.W_message(neighbor_message)))
                    
                    # メッセージの更新
                    new_message_vecs = message_vecs.clone()
                    for bond_idx, new_message in new_messages:
                        new_message_vecs[bond_idx] = new_message
                    
                    # GRUによる更新
                    message_vecs = self.gru(
                        new_message_vecs,
                        message_vecs
                    )
                
                # 原子表現の更新
                for atom_idx in range(n_atoms[i]):
                    # 隣接結合からのメッセージを集約
                    neighbors = adjacency_lists[i][atom_idx]
                    if len(neighbors) > 0:
                        neighbor_messages = torch.stack([message_vecs[bond_idx] for _, bond_idx in neighbors])
                        atom_vecs[atom_idx] = atom_vecs[atom_idx] + torch.sum(neighbor_messages, dim=0)
            
            # 分子表現の計算（各原子の表現の平均）
            mol_embedding = torch.mean(atom_vecs, dim=0)
            mol_embeddings.append(mol_embedding)
        
        # バッチ内の分子埋め込みをスタック
        mol_embeddings = torch.stack(mol_embeddings)
        
        # 読み出しネットワークを適用
        mol_embeddings = self.W_readout(mol_embeddings)
        
        # スペクトル予測
        predicted_spectra = self.spectrum_predictor(mol_embeddings)
        
        return predicted_spectra
    
    def _forward_spectrum_to_structure(self, spectra, n_atoms):
        """スペクトルから構造を予測"""
        # スペクトルをエンコード
        spectrum_embeddings = self.spectrum_encoder(spectra)
        
        # 各分子の原子特徴量を予測
        predicted_atom_features = []
        
        for i, n_atom in enumerate(n_atoms):
            # 分子埋め込みを各原子に複製
            molecule_embedding = spectrum_embeddings[i].repeat(n_atom, 1)
            
            # 原子特徴量の予測
            atom_features = self.atom_predictor(molecule_embedding)
            predicted_atom_features.append(atom_features)
        
        return predicted_atom_features

#------------------------------------------------------
# 3. トレーニングと評価
#------------------------------------------------------

def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs, device):
    """モデルのトレーニングと評価"""
    # 結果の記録用
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # トレーニングフェーズ
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # データをデバイスに移動
            spectra = batch['spectra'].to(device)
            
            # バッチサイズが1の場合に備えて次元を確認
            if spectra.dim() == 1:
                spectra = spectra.unsqueeze(0)
            
            atom_features = [af.to(device) for af in batch['atom_features']]
            bond_features = [bf.to(device) for bf in batch['bond_features']]
            adjacency_lists = batch['adjacency_lists']
            n_atoms = batch['n_atoms']
            
            # 重みの勾配をゼロに初期化
            optimizer.zero_grad()
            
            # 予測（構造→スペクトル）
            predicted_spectra = model(atom_features, bond_features, adjacency_lists, n_atoms, predict_spectrum=True)
            loss = criterion(predicted_spectra, spectra)
            
            # 誤差逆伝播
            loss.backward()
            
            # パラメータ更新
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # データをデバイスに移動
                spectra = batch['spectra'].to(device)
                
                # バッチサイズが1の場合に備えて次元を確認
                if spectra.dim() == 1:
                    spectra = spectra.unsqueeze(0)
                
                atom_features = [af.to(device) for af in batch['atom_features']]
                bond_features = [bf.to(device) for bf in batch['bond_features']]
                adjacency_lists = batch['adjacency_lists']
                n_atoms = batch['n_atoms']
                
                # 予測（構造→スペクトル）
                predicted_spectra = model(atom_features, bond_features, adjacency_lists, n_atoms, predict_spectrum=True)
                loss = criterion(predicted_spectra, spectra)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

#------------------------------------------------------
# 4. マススペクトル比較と可視化
#------------------------------------------------------

def calculate_cosine_similarity(measured_peaks, predicted_spectrum, max_mz=MAX_MZ):
    """測定スペクトルと予測スペクトルのコサイン類似度を計算"""
    # 測定スペクトルをベクトル化
    measured_vector = np.zeros(max_mz + 1)
    for mz, intensity in measured_peaks:
        if mz <= max_mz:
            measured_vector[mz] = intensity
    
    # 予測スペクトルはすでにベクトル形式
    predicted_vector = predicted_spectrum
    
    # ノルムが0の場合の処理
    measured_norm = np.linalg.norm(measured_vector)
    predicted_norm = np.linalg.norm(predicted_vector)
    
    if measured_norm == 0 or predicted_norm == 0:
        return 0.0
    
    # コサイン類似度を計算
    similarity = np.dot(measured_vector, predicted_vector) / (measured_norm * predicted_norm)
    
    return similarity

def convert_to_mz_intensity_pairs(spectrum, threshold=0.01, top_n=20):
    """
    一次元スペクトル配列をm/z-強度ペアのリストに変換
    
    Args:
        spectrum: スペクトル配列
        threshold: ベースピークに対する相対強度の閾値（デフォルト1%）
        top_n: 保持する上位ピーク数（デフォルト20、指定した場合は上位N個のみ保持）
    """
    # スペクトルが空の場合
    if len(spectrum) == 0 or np.max(spectrum) == 0:
        return []
    
    max_intensity = np.max(spectrum)
    pairs = []
    
    # 閾値以上のピークを抽出
    for mz, intensity in enumerate(spectrum):
        if intensity > 0 and intensity / max_intensity >= threshold:
            pairs.append((mz, intensity))
    
    # 上位N個のピークのみを保持
    if top_n > 0 and len(pairs) > top_n:
        # 強度の降順でソート
        pairs.sort(key=lambda x: x[1], reverse=True)
        # 上位N個のみを保持
        pairs = pairs[:top_n]
        # m/zでソートし直す（視覚的に見やすくするため）
        pairs.sort(key=lambda x: x[0])
    
    return pairs

def filter_peaks_by_relative_intensity(peaks, threshold=0.01, top_n=None):
    """
    ベースピークの相対強度に基づいてピークをフィルタリングし、さらに上位N個のピークのみを保持
    
    Args:
        peaks: (m/z, intensity)のリスト
        threshold: 相対閾値（ベースピークに対する割合、デフォルト1%）
        top_n: 保持する上位ピーク数（デフォルトはNone、指定した場合は上位N個のみ保持）
    
    Returns:
        フィルタリングされたピークのリスト
    """
    if not peaks:
        return []
    
    # 最大強度を見つける
    max_intensity = max([intensity for _, intensity in peaks])
    if max_intensity <= 0:
        return peaks
    
    # 閾値以上のピークのみを保持
    filtered_peaks = [(mz, intensity) for mz, intensity in peaks 
                        if intensity / max_intensity >= threshold]
    
    # 上位N個のピークのみを保持
    if top_n is not None and top_n > 0 and len(filtered_peaks) > top_n:
        # 強度の降順でソート
        filtered_peaks.sort(key=lambda x: x[1], reverse=True)
        # 上位N個のみを保持
        filtered_peaks = filtered_peaks[:top_n]
        # m/zでソートし直す（視覚的に見やすくするため）
        filtered_peaks.sort(key=lambda x: x[0])
    
    return filtered_peaks

def plot_spectrum_comparison(measured_peaks, predicted_spectrum, compound_id, compound_name="", threshold=0.01, top_n=20):
    """
    測定スペクトルと予測スペクトルの比較プロット
    （相対強度フィルタリングと上位N個のピーク保持を適用）
    
    Args:
        measured_peaks: 測定ピークのリスト
        predicted_spectrum: 予測スペクトル配列
        compound_id: 化合物ID
        compound_name: 化合物名
        threshold: 相対強度の閾値（デフォルト1%）
        top_n: 保持する上位ピーク数（デフォルト20）
    """
    # 測定ピークをフィルタリング（相対強度+上位N個）
    filtered_measured_peaks = filter_peaks_by_relative_intensity(measured_peaks, threshold, top_n)
    
    # 測定ピークが空でないことを確認
    if not filtered_measured_peaks:
        print(f"Warning: No peaks above threshold for compound {compound_id}")
        return
    
    # 測定ピークを正規化
    max_measured_intensity = max([intensity for _, intensity in filtered_measured_peaks])
    measured_mz = [mz for mz, _ in filtered_measured_peaks]
    measured_intensity = [intensity / max_measured_intensity for _, intensity in filtered_measured_peaks]
    
    # 予測スペクトルからピークを抽出（フィルタリング+上位N個）
    predicted_peaks = convert_to_mz_intensity_pairs(predicted_spectrum, threshold, top_n)
    predicted_mz = [mz for mz, _ in predicted_peaks]
    predicted_intensity = [intensity for _, intensity in predicted_peaks]
    
    # コサイン類似度を計算
    similarity = calculate_cosine_similarity(filtered_measured_peaks, predicted_spectrum)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 測定スペクトル
    ax1.stem(measured_mz, measured_intensity, markerfmt=" ", basefmt=" ", linefmt="b-")
    ax1.set_ylabel("Relative Intensity")
    title = f"Measured Mass Spectrum - ID: {compound_id}\n{compound_name}"
    title += f"\n(Filtered: Relative Intensity ≥ {threshold*100}%, Top {top_n} peaks)"
    ax1.set_title(title)
    ax1.set_ylim(0, 1.05)
    
    # 予測スペクトル
    ax2.stem(predicted_mz, predicted_intensity, markerfmt=" ", basefmt=" ", linefmt="r-")
    ax2.set_xlabel("m/z")
    ax2.set_ylabel("Relative Intensity")
    ax2.set_title(f"Predicted Mass Spectrum (Cosine Similarity: {similarity:.4f})")
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f"spectrum_comparison_{compound_id}.png")
    plt.close()
    
    return similarity

def compare_spectra(model, test_dataset, n_samples=10, threshold=0.01, top_n=20):
    """
    テストデータセットからランダムにサンプルを選択してスペクトルを比較
    
    Args:
        model: 訓練済みモデル
        test_dataset: テストデータセット
        n_samples: 比較するサンプル数
        threshold: 相対強度の閾値（デフォルト1%）
        top_n: 保持する上位ピーク数（デフォルト20）
    """
    model.eval()
    
    # ランダムにn_samples個のサンプルを選択
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))
    
    # 類似度のリスト
    similarities = []
    compound_ids = []
    
    with torch.no_grad():
        for idx in indices:
            sample = test_dataset[idx]
            
            # 入力データの準備
            spectrum = sample['spectrum'].unsqueeze(0).to(device)
            atom_features = [sample['atom_features'].to(device)]
            bond_features = [sample['bond_features'].to(device)]
            adjacency_list = [sample['adjacency_list']]
            n_atoms = [sample['n_atoms']]
            
            # 構造からスペクトルを予測
            predicted_spectrum = model(atom_features, bond_features, adjacency_list, n_atoms, predict_spectrum=True)
            predicted_spectrum = predicted_spectrum.squeeze(0).cpu().numpy()
            
            # 測定スペクトルと予測スペクトルを比較
            compound_id = sample['id']
            compound_name = test_dataset.spectrum_data[compound_id].get('name', '')
            
            # フィルタリングと上位N個ピーク保持を適用して比較プロットを作成
            similarity = plot_spectrum_comparison(
                sample['peaks'], 
                predicted_spectrum, 
                compound_id, 
                compound_name,
                threshold=threshold,
                top_n=top_n
            )
            
            if similarity is not None:
                similarities.append(similarity)
                compound_ids.append(compound_id)
                print(f"Compound ID: {compound_id}, Cosine similarity: {similarity:.4f}")
                
                # ピーク数を表示
                measured_peaks = filter_peaks_by_relative_intensity(sample['peaks'], threshold, top_n)
                predicted_peaks = convert_to_mz_intensity_pairs(predicted_spectrum, threshold, top_n)
                print(f"  Number of peaks - Measured: {len(measured_peaks)}, Predicted: {len(predicted_peaks)}")
    
    # 類似度の統計を出力
    if similarities:
        print(f"\nAverage cosine similarity: {np.mean(similarities):.4f}")
        print(f"Min cosine similarity: {np.min(similarities):.4f}")
        print(f"Max cosine similarity: {np.max(similarities):.4f}")
        
        # 最も類似度の高い/低い化合物
        best_idx = np.argmax(similarities)
        worst_idx = np.argmin(similarities)
        print(f"Best match: Compound ID {compound_ids[best_idx]}, Similarity: {similarities[best_idx]:.4f}")
        print(f"Worst match: Compound ID {compound_ids[worst_idx]}, Similarity: {similarities[worst_idx]:.4f}")
    else:
        print("No valid comparisons could be made.")

#------------------------------------------------------
# 5. メイン処理
#------------------------------------------------------

def main():
    """メイン処理"""
    print("Loading and processing data...")
    
    # MSPファイルの解析
    print(f"Parsing MSP file: {MSP_FILE}")
    spectrum_data = parse_msp_file(MSP_FILE)
    print(f"Found {len(spectrum_data)} compounds in MSP file")
    
    # データセットの作成（非金属のみの分子に制限）
    dataset = MassSpectrumDataset(spectrum_data, MOL_DIR, non_metal_only=True)
    
    # データセットの分割
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_test_dataset, [val_size, test_size])
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 特徴量の次元を取得
    sample = dataset[0]
    atom_fdim = sample['atom_features'].shape[1]
    bond_fdim = sample['bond_features'].shape[1] if len(sample['bond_features']) > 0 else 0
    spectrum_dim = sample['spectrum'].shape[0]
    
    print(f"Feature dimensions - Atom: {atom_fdim}, Bond: {bond_fdim}, Spectrum: {spectrum_dim}")
    
    # モデルの初期化
    model = DMPNN(
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        atom_fdim=atom_fdim,
        bond_fdim=bond_fdim,
        spectrum_dim=spectrum_dim
    ).to(device)
    
    # オプティマイザと損失関数の設定
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # トレーニングと評価
    print("Starting training...")
    train_losses, val_losses = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    # モデルの保存
    print("Saving model...")
    torch.save(model.state_dict(), "ms_structure_model.pt")
    
    # 結果の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    # テスト
    print("Testing model...")
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # データをデバイスに移動
            spectra = batch['spectra'].to(device)
            
            # バッチサイズが1の場合に備えて次元を確認
            if spectra.dim() == 1:
                spectra = spectra.unsqueeze(0)
            
            atom_features = [af.to(device) for af in batch['atom_features']]
            bond_features = [bf.to(device) for bf in batch['bond_features']]
            adjacency_lists = batch['adjacency_lists']
            n_atoms = batch['n_atoms']
            
            # 予測（構造→スペクトル）
            predicted_spectra = model(atom_features, bond_features, adjacency_lists, n_atoms, predict_spectrum=True)
            loss = criterion(predicted_spectra, spectra)
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # ランダムに10個の分子を選択し、測定スペクトルと予測スペクトルを比較
    # 相対強度1%以上かつ上位20ピークのみ保持
    print("Comparing measured and predicted spectra for 10 random compounds...")
    print("Applying filtering: 1% relative intensity threshold + top 20 peaks only")
    compare_spectra(model, dataset, n_samples=10, threshold=0.01, top_n=20)
    
    print("Done!")

if __name__ == "__main__":
    main()