import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, GlobalAttention, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
# RDKitの警告を抑制
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # RDKitの全ての警告を無効化

from tqdm import tqdm
import logging
import copy
import random
import math
import gc
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from torch.amp import autocast, GradScaler
import time
import datetime

# ===== メモリ管理関連の関数 =====
def aggressive_memory_cleanup(force_sync=True, percent=70, purge_cache=False):
    """強化版メモリクリーンアップ関数"""
    gc.collect()
    
    if not torch.cuda.is_available():
        return False
    
    # 強制同期してGPUリソースを確実に解放
    if force_sync:
        torch.cuda.synchronize()
    
    torch.cuda.empty_cache()
    
    # メモリ使用率の計算
    gpu_memory_allocated = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_percent = gpu_memory_allocated / total_memory * 100
    
    if gpu_memory_percent > percent:
        logger.warning(f"高いGPUメモリ使用率 ({gpu_memory_percent:.1f}%)。キャッシュをクリアします。")
        
        if purge_cache:
            # データセットキャッシュが存在する場合はクリア
            for obj_name in ['train_dataset', 'val_dataset', 'test_dataset']:
                if obj_name in globals():
                    obj = globals()[obj_name]
                    if hasattr(obj, 'graph_cache') and isinstance(obj.graph_cache, dict):
                        obj.graph_cache.clear()
                        logger.info(f"{obj_name}のグラフキャッシュをクリア")
        
        # もう一度クリーンアップ
        gc.collect()
        torch.cuda.empty_cache()
        
        # PyTorchメモリアロケータをリセット
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        return True
    
    return False

# CUDA割り当てを明示的に制限（メモリ断片化を防ぐ）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# パス設定
DATA_PATH = "data/"
MOL_FILES_PATH = os.path.join(DATA_PATH, "mol_files/")
MSP_FILE_PATH = os.path.join(DATA_PATH, "NIST17.MSP")
CACHE_DIR = os.path.join(DATA_PATH, "cache/")
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints/")

# ディレクトリの作成
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 最大m/z値の設定
MAX_MZ = 2000

# エフェメラル値
EPS = np.finfo(np.float32).eps

# 原子の特徴マッピング
ATOM_FEATURES = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8,
    'Si': 9, 'B': 10, 'Na': 11, 'K': 12, 'Li': 13, 'Mg': 14, 'Ca': 15, 'Fe': 16,
    'Co': 17, 'Ni': 18, 'Cu': 19, 'Zn': 20, 'H': 21, 'OTHER': 22
}

# 結合の特徴マッピング
BOND_FEATURES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}

# フラグメントパターンの数（MACCSキー使用）
NUM_FRAGS = 167  # MACCSキーのビット数

###############################
# データ処理関連の関数
###############################

def process_spec(spec, transform, normalization, eps=EPS):
    """スペクトルにトランスフォームと正規化を適用（離散的性質を保持）"""
    # スペクトルを1000までスケーリング（最大強度を基準）
    max_intensity = torch.max(spec, dim=-1, keepdim=True)[0] + eps
    spec = spec / max_intensity * 1000.
    
    # 離散的な性質を保持するため、ピークのみを処理
    non_zero_mask = (spec > 0).float()
    
    # 信号変換
    if transform == "log10":
        spec = torch.log10(spec + 1) * non_zero_mask
    elif transform == "log10over3":
        spec = torch.log10(spec + 1) / 3 * non_zero_mask
    elif transform == "loge":
        spec = torch.log(spec + 1) * non_zero_mask
    elif transform == "sqrt":
        spec = torch.sqrt(spec) * non_zero_mask
    elif transform == "none":
        pass
    else:
        raise ValueError("invalid transform")
    
    # 正規化
    if normalization == "l1":
        norm = torch.sum(spec, dim=-1, keepdim=True) + eps
        spec = spec / norm
    elif normalization == "l2":
        norm = torch.sqrt(torch.sum(spec * spec, dim=-1, keepdim=True)) + eps
        spec = spec / norm
    elif normalization == "none":
        pass
    else:
        raise ValueError("invalid normalization")
    
    # ゼロ以外の値のみを保持し、他はゼロにする
    spec = spec * non_zero_mask
    
    assert not torch.isnan(spec).any()
    return spec, max_intensity

def unprocess_spec(spec, transform, original_intensity=None):
    """スペクトルの変換を元に戻す"""
    # ゼロ以外のピークのみを処理
    non_zero_mask = (spec > 0).float()
    
    # transform signal
    if transform == "log10":
        untransformed = (10**spec - 1.) * non_zero_mask
    elif transform == "log10over3":
        untransformed = (10**(3 * spec) - 1.) * non_zero_mask
    elif transform == "loge":
        untransformed = (torch.exp(spec) - 1.) * non_zero_mask
    elif transform == "sqrt":
        untransformed = (spec**2) * non_zero_mask
    elif transform == "none":
        untransformed = spec
    else:
        raise ValueError("invalid transform")
    
    # 元のスケールに戻す
    if original_intensity is not None:
        untransformed = untransformed / 1000. * original_intensity
    
    untransformed = torch.clamp(untransformed, min=0.)
    assert not torch.isnan(untransformed).any()
    return untransformed

def convert_to_original_format(model_output, transform="log10over3"):
    """モデル出力を元のマススペクトル形式に変換"""
    # 非ゼロピークだけを処理
    non_zero_mask = (model_output > 0.01).float()  # 小さな値はノイズとして除去
    
    # 非ゼロピークの位置を特定
    peak_indices = torch.nonzero(non_zero_mask).squeeze()
    
    # 変換を適用
    if transform == "log10":
        untransformed = (10**model_output - 1.) * non_zero_mask
    elif transform == "log10over3":
        untransformed = (10**(3 * model_output) - 1.) * non_zero_mask
    elif transform == "loge":
        untransformed = (torch.exp(model_output) - 1.) * non_zero_mask
    elif transform == "sqrt":
        untransformed = (model_output**2) * non_zero_mask
    elif transform == "none":
        untransformed = model_output * non_zero_mask
    
    # 最大値で正規化
    max_val = torch.max(untransformed, dim=-1, keepdim=True)[0]
    if max_val.item() > 0:
        normalized = untransformed / max_val * 100.0
    else:
        normalized = untransformed
    
    return normalized

def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    """前駆体質量によるマスキング"""
    device = raw_prediction.device
    max_idx = raw_prediction.shape[1]
    
    # prec_mass_idxのデータ型を確認し調整
    if prec_mass_idx.dtype != torch.long:
        prec_mass_idx = prec_mass_idx.long()
    
    # エラーチェックを追加
    if not torch.all(prec_mass_idx < max_idx):
        # エラーを回避するために範囲外の値をクリップ
        prec_mass_idx = torch.clamp(prec_mass_idx, max=max_idx-1)
    
    idx = torch.arange(max_idx, device=device)
    mask = (
        idx.unsqueeze(0) <= (
            prec_mass_idx.unsqueeze(1) +
            prec_mass_offset)).float()
    return mask * raw_prediction + (1. - mask) * mask_value

def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    """予測を反転する（双方向予測用）"""
    device = raw_prediction.device
    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    
    # prec_mass_idxのデータ型を確認し調整
    if prec_mass_idx.dtype != torch.long:
        prec_mass_idx = prec_mass_idx.long()
    
    # エラーチェックを追加
    if not torch.all(prec_mass_idx < max_idx):
        # エラーを回避するために範囲外の値をクリップ
        prec_mass_idx = torch.clamp(prec_mass_idx, max=max_idx-1)
    
    rev_prediction = torch.flip(raw_prediction, dims=(1,))
    offset_idx = torch.minimum(
        max_idx * torch.ones_like(prec_mass_idx, device=device),
        prec_mass_idx + prec_mass_offset + 1)
    shifts = - (max_idx - offset_idx)
    gather_idx = torch.arange(
        max_idx,
        device=device).unsqueeze(0).expand(
        batch_size,
        max_idx)
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
    offset_rev_prediction = torch.gather(rev_prediction, 1, gather_idx)
    return offset_rev_prediction

def parse_msp_file(msp_file_path, cache_dir=CACHE_DIR):
    """MSPファイルを解析し、ID->マススペクトルのマッピングを返す（キャッシュ対応）"""
    # キャッシュファイルのパス
    cache_file = os.path.join(cache_dir, f"msp_data_cache_raw_{os.path.basename(msp_file_path)}.pkl")
    
    # キャッシュが存在すれば読み込む
    if os.path.exists(cache_file):
        logger.info(f"キャッシュからMSPデータを読み込み中: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logger.info(f"MSPファイルを解析中: {msp_file_path}")
    msp_data = {}
    current_id = None
    current_peaks = []
    
    with open(msp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # IDを検出
            if line.startswith("ID:"):
                current_id = line.split(":")[1].strip()
                current_id = int(current_id)
            
            # ピーク数を検出（これはピークデータの直前にある）
            elif line.startswith("Num peaks:"):
                current_peaks = []
            
            # 空行は化合物の区切り
            elif line == "" and current_id is not None and current_peaks:
                # マススペクトルをベクトルに変換
                ms_vector = np.zeros(MAX_MZ)
                for mz, intensity in current_peaks:
                    if 0 <= mz < MAX_MZ:
                        ms_vector[mz] = intensity
                
                # 強度を正規化（最大値が100になるよう調整）
                if np.sum(ms_vector) > 0:
                    ms_vector = ms_vector / np.max(ms_vector) * 100
                    
                    # スムージングと小さなピークのフィルタリングを削除
                    # 重要なm/z値の強調も削除
                    
                    msp_data[current_id] = ms_vector
                else:
                    msp_data[current_id] = ms_vector
                
                current_id = None
                current_peaks = []
            
            # ピークデータを処理
            elif current_id is not None and " " in line and not any(line.startswith(prefix) for prefix in ["Name:", "Formula:", "MW:", "ExactMass:", "CASNO:", "Comment:"]):
                try:
                    parts = line.split()
                    if len(parts) == 2:
                        mz = int(parts[0])
                        intensity = float(parts[1])
                        current_peaks.append((mz, intensity))
                except ValueError:
                    pass  # 数値に変換できない行はスキップ
    
    # キャッシュに保存
    logger.info(f"MSPデータをキャッシュに保存中: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(msp_data, f)
    
    return msp_data

###############################
# モデル関連のコンポーネント（最適化版）
###############################

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation ブロック - 最適化版"""
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 8), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        # 高速化のためのシンプルな実装
        y = torch.mean(x, dim=0, keepdim=True).expand(b, c)
        y = self.fc(y).view(b, c)
        return x * y

class ResidualBlock(nn.Module):
    """残差ブロック - 最適化版"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.ln1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        
        # 入力と出力のチャネル数が異なる場合の調整用レイヤー
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels)
            )
            
        # ドロップアウト - 少し小さくして学習加速
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.leaky_relu(self.ln1(self.conv1(x)))
        out = self.dropout(out)
        out = self.ln2(self.conv2(out))
        
        out += residual  # 残差接続
        out = F.leaky_relu(out)
        
        return out

class AttentionBlock(nn.Module):
    """軽量化グローバルアテンションブロック"""
    def __init__(self, in_dim, hidden_dim, heads=2):
        super(AttentionBlock, self).__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        
        # シンプル化したアテンション
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)
        
        self.attn_combine = nn.Linear(hidden_dim, hidden_dim)
        
        # ゲート付きグローバルアテンション - 簡素化
        self.gate_nn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 出力投影 - 簡素化
        self.out_proj = nn.Linear(hidden_dim, in_dim)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, batch):
        device = x.device
        batch_size = torch.max(batch).item() + 1
        
        # マルチヘッドアテンション計算
        q = self.query(x).view(-1, self.heads, self.head_dim)
        k = self.key(x).view(-1, self.heads, self.head_dim)
        v = self.value(x).view(-1, self.heads, self.head_dim)
        
        # GlobalAttentionを使用
        global_attention = GlobalAttention(
            gate_nn=self.gate_nn,
            nn=nn.Identity()  # 簡素化
        )
        
        x_global = global_attention(x, batch)
        
        # 結果の組み合わせ
        out = self.out_proj(x_global)
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out

class TransformerEncoderLayer(nn.Module):
    """軽量化Transformerエンコーダーレイヤー"""
    def __init__(self, hidden_dim, nhead, dim_feedforward=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 2
            
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, src):
        # 正規化とセルフアテンション
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        # FFNとスキップ接続
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

###############################
# 軽量化ハイブリッドモデル
###############################

class OptimizedHybridMSModel(nn.Module):
    """軽量化GNN-Transformerハイブリッドモデル"""
    def __init__(self, node_features, edge_features, hidden_channels, out_channels, num_fragments=NUM_FRAGS,
                 prec_mass_offset=10, bidirectional=True, gate_prediction=True):
        super(OptimizedHybridMSModel, self).__init__()
        
        self.prec_mass_offset = prec_mass_offset
        self.bidirectional = bidirectional
        self.gate_prediction = gate_prediction
        self.global_features_dim = 16
        
        # 次元を小さくして軽量化
        self.hidden_channels = hidden_channels
        self.transformer_dim = 128  # Transformer次元を縮小
        
        # GNNレイヤー - 軽量化
        self.gat1 = GATv2Conv(node_features, hidden_channels, edge_dim=edge_features, heads=2)
        self.gat2 = GATv2Conv(hidden_channels*2, hidden_channels, edge_dim=edge_features, heads=2)
        
        # 後段のGATレイヤーを削減
        self.gat3 = GATv2Conv(hidden_channels*2, hidden_channels, edge_dim=edge_features, heads=2)
        
        # スキップ接続
        self.skip_connection1 = nn.Linear(hidden_channels*2, hidden_channels*2)
        
        # グローバル特徴量処理
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_features_dim, hidden_channels),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_channels)
        )
        
        # 特徴量の次元をTransformer入力用に変換
        self.transformer_projection = nn.Linear(hidden_channels*3, self.transformer_dim)
        
        # 軽量Transformerエンコーダー
        self.transformer_encoder = TransformerEncoderLayer(
            hidden_dim=self.transformer_dim,
            nhead=4,  # ヘッド数を減らす
            dim_feedforward=self.transformer_dim*2,
            dropout=0.1
        )
        
        # Transformer出力を元の次元に戻す
        self.transformer_unprojection = nn.Linear(self.transformer_dim, hidden_channels*2)
        
        # スペクトル予測のための全結合層 - 層を削減
        self.fc_layers = nn.ModuleList([
            ResidualBlock(hidden_channels*2, hidden_channels*2),
            ResidualBlock(hidden_channels*2, hidden_channels)
        ])
        
        # マルチタスク学習: フラグメントパターン予測
        self.fragment_pred = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels//2, num_fragments),
        )
        
        # 双方向予測用レイヤー
        if bidirectional:
            self.forw_out_layer = nn.Linear(hidden_channels, out_channels)
            self.rev_out_layer = nn.Linear(hidden_channels, out_channels)
            self.out_gate = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.Sigmoid()
            )
        else:
            # 通常の出力レイヤー
            self.out_layer = nn.Linear(hidden_channels, out_channels)
            if gate_prediction:
                self.out_gate = nn.Sequential(
                    nn.Linear(hidden_channels, out_channels),
                    nn.Sigmoid()
                )
        
        # レイヤー正規化
        self.ln1 = nn.LayerNorm(hidden_channels*2)
        self.ln2 = nn.LayerNorm(hidden_channels)
        
        self.dropout = nn.Dropout(0.2)
        
        # 重み初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化（収束を高速化）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        device = next(self.parameters()).device
        
        if isinstance(data, dict):  # MassFormer形式の入力
            x = data['graph'].x.to(device)
            edge_index = data['graph'].edge_index.to(device)
            edge_attr = data['graph'].edge_attr.to(device)
            batch = data['graph'].batch.to(device)
            
            global_attr = data['graph'].global_attr.to(device) if hasattr(data['graph'], 'global_attr') else None
            prec_mz_bin = data.get('prec_mz_bin', None)
            if prec_mz_bin is not None:
                prec_mz_bin = prec_mz_bin.to(device)
        else:  # Generalized_model形式の入力
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            
            global_attr = data.global_attr.to(device) if hasattr(data, 'global_attr') else None
            # 前駆体質量のダミー値を作成
            if hasattr(data, 'mass'):
                prec_mz_bin = data.mass.to(device)
            else:
                prec_mz_bin = None
        
        # 特徴量の型変換（最適化）
        x = x.float()
        edge_attr = edge_attr.float()
        
        # グローバル属性のリシェイプとバッチサイズ調整（メモリ効率化）
        if global_attr is not None:
            if len(global_attr.shape) == 1:
                num_graphs = batch[-1].item() + 1
                global_attr = global_attr.view(num_graphs, -1)
                # 期待する次元にパディング
                if global_attr.shape[1] != self.global_features_dim:
                    padded = torch.zeros(num_graphs, self.global_features_dim, device=device)
                    copy_size = min(global_attr.shape[1], self.global_features_dim)
                    padded[:, :copy_size] = global_attr[:, :copy_size]
                    global_attr = padded
        
        # GATレイヤーを通す - 段数削減
        x1 = F.leaky_relu(self.gat1(x, edge_index, edge_attr))
        x1 = self.dropout(x1)
        
        x2 = F.leaky_relu(self.gat2(x1, edge_index, edge_attr))
        x2 = self.dropout(x2)
        
        # スキップ接続1
        x1_transformed = self.skip_connection1(x1)
        x2 = x2 + x1_transformed
        x2 = self.ln1(x2)
        
        x3 = F.leaky_relu(self.gat3(x2, edge_index, edge_attr))
        
        # グローバルプーリング - シンプル化
        x_graph = global_mean_pool(x3, batch)
        
        # グローバル特徴量の処理
        if global_attr is not None:
            global_features = self.global_proj(global_attr)
            # 特徴量の結合
            x_combined = torch.cat([x_graph, global_features], dim=1)
        else:
            # グローバル特徴がない場合はゼロパディング
            dummy_global_features = torch.zeros(x_graph.size(0), x_graph.size(1)//2, device=device)
            x_combined = torch.cat([x_graph, dummy_global_features], dim=1)
        
        # 特徴量の次元をTransformer入力次元に変換
        x_for_transformer = self.transformer_projection(x_combined)
        
        # Transformerエンコーダー処理
        x_transformed = self.transformer_encoder(x_for_transformer)
        
        # Transformer出力を元の次元に戻す
        x_combined = self.transformer_unprojection(x_transformed)
        
        # 残差ブロックを通した特徴抽出
        for i, fc_layer in enumerate(self.fc_layers):
            x_combined = fc_layer(x_combined)
            if i < len(self.fc_layers) - 1:
                x_combined = self.dropout(x_combined)
        
        # マルチタスク学習: フラグメントパターン予測
        fragment_pred = self.fragment_pred(x_combined)
        
        # 双方向予測を使用する場合
        if self.bidirectional and prec_mz_bin is not None:
            # 順方向と逆方向の予測
            ff = self.forw_out_layer(x_combined)
            fr = reverse_prediction(
                self.rev_out_layer(x_combined),
                prec_mz_bin,
                self.prec_mass_offset)
            
            # ゲート機構で重み付け
            fg = self.out_gate(x_combined)
            output = ff * fg + fr * (1. - fg)
            
            # 前駆体質量でマスク
            output = mask_prediction_by_mass(output, prec_mz_bin, self.prec_mass_offset)
        else:
            # 通常の予測
            if hasattr(self, 'out_layer'):
                output = self.out_layer(x_combined)
                
                # ゲート予測を使用する場合
                if self.gate_prediction and hasattr(self, 'out_gate'):
                    fg = self.out_gate(x_combined)
                    output = fg * output
            else:
                # 双方向予測のためのレイヤーが存在しても前駆体質量情報がない場合
                output = self.forw_out_layer(x_combined)
        
        # 出力をReLUで活性化
        output = F.relu(output)
        
        return output, fragment_pred

###############################
# 最適化データセット
###############################

def process_mol_id(mol_id, mol_files_path, msp_data):
    """単一分子IDの処理（シングルプロセス用）"""
    # RDKitの警告を抑制
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # RDKitの全ての警告を無効化
    
    mol_file = os.path.join(mol_files_path, f"ID{mol_id}.MOL")
    try:
        # 分子ファイルが読み込めるか確認
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        if mol is None:
            return None, None
        
        # 分子の基本的なサニタイズを試みる
        try:
            # プロパティキャッシュを更新
            for atom in mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            
            # 部分的なサニタイズ
            Chem.SanitizeMol(mol, 
                           sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                      Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                      Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                      Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                      Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                      Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                           catchErrors=True)
        except Exception:
            return None, None
        
        # MACCSフィンガープリントを計算
        try:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            fragments = np.zeros(NUM_FRAGS)
            for i in range(NUM_FRAGS):
                if maccs.GetBit(i):
                    fragments[i] = 1.0
        except Exception:
            fragments = np.zeros(NUM_FRAGS)
            
        # この分子のスペクトルがあるか確認
        if mol_id not in msp_data:
            return None, None
            
        return mol_id, fragments
    except Exception:
        return None, None

class OptimizedMoleculeGraphDataset(Dataset):
    def __init__(self, mol_ids, mol_files_path, msp_data, transform="log10over3", 
                normalization="l1", augment=False, cache_dir=CACHE_DIR):
        self.mol_ids = mol_ids
        self.mol_files_path = mol_files_path
        self.msp_data = msp_data
        self.augment = augment
        self.transform = transform
        self.normalization = normalization
        self.valid_mol_ids = []
        self.fragment_patterns = {}
        self.cache_dir = cache_dir
        self.graph_cache = {}  # メモリ内キャッシュ
        
        # 前処理で有効な分子IDを抽出（キャッシュ使用）
        self._preprocess_mol_ids()
        
    def _preprocess_mol_ids(self):
        """有効な分子IDのみを抽出する（キャッシュ＋シングルスレッド処理）"""
        # キャッシュファイルのパス
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"preprocessed_data_{hash(str(sorted(self.mol_ids)))}.pkl")
        
        # キャッシュファイルが存在するか確認
        if os.path.exists(cache_file):
            logger.info(f"キャッシュから前処理データを読み込み中: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.valid_mol_ids = cached_data['valid_mol_ids']
                self.fragment_patterns = cached_data['fragment_patterns']
                return
        
        logger.info("分子データの前処理を開始します（シングルプロセス処理）...")
        
        # シングルプロセスでの処理
        valid_ids = []
        fragment_patterns = {}
        
        # 進捗表示用
        with tqdm(total=len(self.mol_ids), desc="分子の検証") as pbar:
            # 処理関数
            process_func = partial(process_mol_id, mol_files_path=self.mol_files_path, msp_data=self.msp_data)
            
            for mol_id in self.mol_ids:
                try:
                    mol_id_result, fragments = process_func(mol_id)
                    if mol_id_result is not None:
                        valid_ids.append(mol_id_result)
                        fragment_patterns[mol_id_result] = fragments
                    
                    # 進捗更新
                    pbar.update(1)
                    
                    # 定期的にガベージコレクション
                    if len(valid_ids) % 100 == 0:
                        gc.collect()
                except Exception as e:
                    logger.warning(f"分子ID {mol_id} の処理中にエラー: {str(e)}")
                    pbar.update(1)
        
        self.valid_mol_ids = valid_ids
        self.fragment_patterns = fragment_patterns
        
        # 結果をキャッシュに保存
        logger.info(f"前処理結果をキャッシュに保存中: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'valid_mol_ids': valid_ids,
                'fragment_patterns': fragment_patterns
            }, f)
        
        logger.info(f"有効な分子: {len(valid_ids)}個 / 全体: {len(self.mol_ids)}個")
        
    def _mol_to_graph(self, mol_file):
        """分子をグラフに変換（キャッシュ対応）"""
        # キャッシュをチェック
        if mol_file in self.graph_cache:
            return self.graph_cache[mol_file]
        
        # キャッシュファイルのパス
        cache_file = os.path.join(self.cache_dir, f"graph_cache_{os.path.basename(mol_file)}.pkl")
        
        # ディスクキャッシュをチェック
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    graph_data = pickle.load(f)
                    # メモリキャッシュに追加
                    self.graph_cache[mol_file] = graph_data
                    return graph_data
            except Exception:
                # キャッシュが壊れている場合は再計算
                pass
        
        # RDKitの警告を抑制 - 正しい形式で呼び出す
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        # RDKitでMOLファイルを読み込む
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        if mol is None:
            raise ValueError(f"Could not read molecule from {mol_file}")
        
        try:
            # プロパティキャッシュを更新して暗黙的な原子価を計算
            for atom in mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            
            # 部分的なサニタイズ
            Chem.SanitizeMol(mol, 
                           sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                      Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                      Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                      Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                      Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                      Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                           catchErrors=True)
            
            # 明示的な水素を追加（安全モード）
            try:
                mol = Chem.AddHs(mol)
            except:
                pass
        except Exception:
            # エラーを無視して処理を続行
            pass
        
        # 原子情報を取得
        num_atoms = mol.GetNumAtoms()
        x = []
        
        # 環情報の取得
        ring_info = mol.GetRingInfo()
        rings = []
        try:
            rings = ring_info.AtomRings()
        except:
            # 環情報取得に失敗した場合は空リストを使う
            pass
        
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_feature_idx = ATOM_FEATURES.get(atom_symbol, ATOM_FEATURES['OTHER'])
            
            # 基本的な原子タイプの特徴
            atom_feature = [0] * len(ATOM_FEATURES)
            atom_feature[atom_feature_idx] = 1
            
            # 安全なメソッド呼び出し
            try:
                degree = atom.GetDegree() / 8.0
            except:
                degree = 0.0
                
            try:
                formal_charge = atom.GetFormalCharge() / 8.0
            except:
                formal_charge = 0.0
                
            try:
                radical_electrons = atom.GetNumRadicalElectrons() / 4.0
            except:
                radical_electrons = 0.0
                
            try:
                is_aromatic = atom.GetIsAromatic() * 1.0
            except:
                is_aromatic = 0.0
                
            try:
                atom_mass = atom.GetMass() / 200.0
            except:
                atom_mass = 0.0
                
            try:
                is_in_ring = atom.IsInRing() * 1.0
            except:
                is_in_ring = 0.0
                
            try:
                hybridization = int(atom.GetHybridization()) / 8.0
            except:
                hybridization = 0.0
                
            try:
                explicit_valence = atom.GetExplicitValence() / 8.0
            except:
                explicit_valence = 0.0
                
            try:
                implicit_valence = atom.GetImplicitValence() / 8.0
            except:
                implicit_valence = 0.0
                
            # 追加の環境特徴量
            try:
                is_in_aromatic_ring = (atom.GetIsAromatic() and atom.IsInRing()) * 1.0
            except:
                is_in_aromatic_ring = 0.0
                
            try:
                ring_size = 0
                atom_idx = atom.GetIdx()
                for ring in rings:
                    if atom_idx in ring:
                        ring_size = max(ring_size, len(ring))
                ring_size = ring_size / 8.0
            except:
                ring_size = 0.0
                
            try:
                num_h = atom.GetTotalNumHs() / 8.0
            except:
                num_h = 0.0
            
            # 簡素化した特徴リスト - 計算効率を向上
            additional_features = [
                degree, formal_charge, radical_electrons, is_aromatic,
                atom_mass, is_in_ring, hybridization, explicit_valence, 
                implicit_valence, is_in_aromatic_ring, ring_size, num_h
            ]
            
            # すべての特徴を結合
            atom_feature.extend(additional_features)
            x.append(atom_feature)
        
        # 結合情報を取得
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            try:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # 結合タイプ
                try:
                    bond_type = BOND_FEATURES.get(bond.GetBondType(), BOND_FEATURES[Chem.rdchem.BondType.SINGLE])
                except:
                    bond_type = BOND_FEATURES[Chem.rdchem.BondType.SINGLE]
                
                # 双方向のエッジを追加
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                
                # 簡素化したボンド特徴量
                bond_feature = [0] * len(BOND_FEATURES)
                bond_feature[bond_type] = 1
                
                # 安全な追加ボンド特徴量の取得
                try:
                    is_in_ring = bond.IsInRing() * 1.0
                except:
                    is_in_ring = 0.0
                    
                try:
                    is_conjugated = bond.GetIsConjugated() * 1.0
                except:
                    is_conjugated = 0.0
                    
                try:
                    is_aromatic = bond.GetIsAromatic() * 1.0
                except:
                    is_aromatic = 0.0
                
                additional_bond_features = [is_in_ring, is_conjugated, is_aromatic]
                
                bond_feature.extend(additional_bond_features)
                edge_attrs.append(bond_feature)
                edge_attrs.append(bond_feature)  # 双方向なので同じ属性
            except Exception:
                continue
        
        # 分子全体の特徴量 - 簡素化
        mol_features = [0.0] * 16
        
        try:
            mol_features[0] = Descriptors.MolWt(mol) / 1000.0  # 分子量
        except:
            pass
            
        try:
            mol_features[1] = Descriptors.NumHAcceptors(mol) / 20.0  # 水素結合アクセプター数
        except:
            pass
            
        try:
            mol_features[2] = Descriptors.NumHDonors(mol) / 10.0  # 水素結合ドナー数
        except:
            pass
            
        try:
            mol_features[3] = Descriptors.TPSA(mol) / 200.0  # トポロジカル極性表面積
        except:
            pass
        
        # エッジが存在するか確認
        if not edge_indices:
            # 単一原子分子の場合や結合情報が取得できない場合、セルフループを追加
            for i in range(num_atoms):
                edge_indices.append([i, i])
                
                bond_feature = [0] * len(BOND_FEATURES)
                bond_feature[BOND_FEATURES[Chem.rdchem.BondType.SINGLE]] = 1
                
                # ダミーの追加特徴量
                additional_bond_features = [0.0, 0.0, 0.0]
                bond_feature.extend(additional_bond_features)
                edge_attrs.append(bond_feature)
        
        # PyTorch Geometricのデータ形式に変換
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attrs)
        global_attr = torch.FloatTensor(mol_features)
        
        # グラフデータを作成
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_attr=global_attr)
        
        # キャッシュに保存
        self.graph_cache[mol_file] = graph_data
        
        # ディスクキャッシュにも保存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(graph_data, f)
        except Exception:
            pass
        
        return graph_data
    
    def _preprocess_spectrum(self, spectrum):
        """スペクトルの前処理（離散的性質を保持）"""
        # スペクトルをPyTorchテンソルに変換
        spec_tensor = torch.FloatTensor(spectrum)
        
        # 離散的性質を保持した処理
        processed_spec, max_intensity = process_spec(
            spec_tensor.unsqueeze(0), 
            self.transform, 
            self.normalization
        )
        
        # 元のスケール情報を保存
        self.original_max_intensities = self.original_max_intensities if hasattr(self, 'original_max_intensities') else {}
        
        # NumPy配列に変換して返す
        return processed_spec.squeeze(0).numpy()
        
    def __len__(self):
        return len(self.valid_mol_ids)
    
    def __getitem__(self, idx):
        mol_id = self.valid_mol_ids[idx]
        mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
        
        # RDKitの警告を抑制 - 正しい形式で呼び出す
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')  # 引数を追加
        
        try:
            # MOLファイルからグラフ表現を生成
            graph_data = self._mol_to_graph(mol_file)
            
            # MSPデータからマススペクトルを取得
            mass_spectrum = self.msp_data.get(mol_id, np.zeros(MAX_MZ))
            mass_spectrum = self._preprocess_spectrum(mass_spectrum)
            
            # フラグメントパターンを取得
            fragment_pattern = self.fragment_patterns.get(mol_id, np.zeros(NUM_FRAGS))
            
            # 前駆体m/zの計算
            peaks = np.nonzero(mass_spectrum)[0]
            if len(peaks) > 0:
                prec_mz = np.max(peaks)
            else:
                prec_mz = 0
                
            prec_mz_bin = prec_mz
            
            # データ拡張（トレーニング時のみ）
            if self.augment and np.random.random() < 0.2:
                # ノイズ追加
                noise_amplitude = 0.01
                graph_data.x = graph_data.x + torch.randn_like(graph_data.x) * noise_amplitude
                graph_data.edge_attr = graph_data.edge_attr + torch.randn_like(graph_data.edge_attr) * noise_amplitude
        
        except Exception as e:
            # エラー発生時のフォールバック処理
            logger.warning(f"分子ID {mol_id} の処理中にエラー: {str(e)}")
            # 最小限のグラフを生成
            x = torch.zeros((1, len(ATOM_FEATURES)+12), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, len(BOND_FEATURES)+3), dtype=torch.float)
            global_attr = torch.zeros(16, dtype=torch.float)
            
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_attr=global_attr)
            mass_spectrum = np.zeros(MAX_MZ)
            fragment_pattern = np.zeros(NUM_FRAGS)
            prec_mz = 0
            prec_mz_bin = 0
        
        return {
            'graph_data': graph_data, 
            'mass_spectrum': torch.FloatTensor(mass_spectrum),
            'fragment_pattern': torch.FloatTensor(fragment_pattern),
            'mol_id': mol_id,
            'prec_mz': prec_mz,
            'prec_mz_bin': prec_mz_bin
        }

def optimized_collate_fn(batch):
    """最適化されたバッチ内のデータ結合"""
    graph_data = [item['graph_data'] for item in batch]
    mass_spectrum = torch.stack([item['mass_spectrum'] for item in batch])
    fragment_pattern = torch.stack([item['fragment_pattern'] for item in batch])
    mol_id = [item['mol_id'] for item in batch]
    prec_mz = torch.tensor([item['prec_mz'] for item in batch], dtype=torch.float32)
    prec_mz_bin = torch.tensor([item['prec_mz_bin'] for item in batch], dtype=torch.long)
    
    # バッチ作成を効率化
    batched_graphs = Batch.from_data_list(graph_data)
    
    return {
        'graph': batched_graphs,
        'spec': mass_spectrum,
        'fragment_pattern': fragment_pattern,
        'mol_id': mol_id,
        'prec_mz': prec_mz,
        'prec_mz_bin': prec_mz_bin
    }

###############################
# 損失関数と類似度計算（最適化）
###############################

def cosine_similarity_loss(y_pred, y_true, eps=EPS):
    """コサイン類似度損失関数 - 重要なm/z値のリストを使わない"""
    # 正規化
    y_pred_norm = F.normalize(y_pred, p=2, dim=1)
    y_true_norm = F.normalize(y_true, p=2, dim=1)
    
    # コサイン類似度（-1〜1の範囲）
    cosine = torch.sum(y_pred_norm * y_true_norm, dim=1)
    
    # 損失を1 - cosineにして、0〜2の範囲に
    loss = 1.0 - cosine
    
    return loss.mean()

def combined_loss(y_pred, y_true, fragment_pred=None, fragment_true=None, 
                 alpha=0.2, beta=0.6, epsilon=0.2):
    """高速化された損失関数 - m/z重要度リスト不使用"""
    # バッチサイズのチェックと調整
    if y_pred.shape[0] != y_true.shape[0]:
        min_batch_size = min(y_pred.shape[0], y_true.shape[0])
        y_pred = y_pred[:min_batch_size]
        y_true = y_true[:min_batch_size]
    
    # 特徴数のチェックと調整
    if y_pred.shape[1] != y_true.shape[1]:
        min_size = min(y_pred.shape[1], y_true.shape[1])
        y_pred = y_pred[:, :min_size]
        y_true = y_true[:, :min_size]
    
    # 1. MSE損失（ピーク重視）
    peak_mask = (y_true > 0).float()
    mse_weights = peak_mask * 10.0 + 1.0
    
    # 重要なm/z値の重み付けを削除
    
    mse_loss = torch.mean(mse_weights * (y_pred - y_true) ** 2)
    
    # 2. コサイン類似度損失
    cosine_loss = cosine_similarity_loss(y_pred, y_true)
    
    # 主要な損失関数の組み合わせ
    main_loss = alpha * mse_loss + beta * cosine_loss
    
    # フラグメントパターン予測がある場合
    if fragment_pred is not None and fragment_true is not None:
        if fragment_pred.shape[0] != fragment_true.shape[0]:
            min_batch_size = min(fragment_pred.shape[0], fragment_true.shape[0])
            fragment_pred = fragment_pred[:min_batch_size]
            fragment_true = fragment_true[:min_batch_size]
        
        fragment_loss = F.binary_cross_entropy_with_logits(fragment_pred, fragment_true)
        return main_loss + epsilon * fragment_loss
    
    return main_loss

def cosine_similarity_score(y_true, y_pred):
    """コサイン類似度スコア計算（最適化）"""
    # バッチサイズチェック
    min_batch = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:min_batch]
    y_pred = y_pred[:min_batch]
    
    # NumPy配列に変換
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    y_true_flat = y_true_np.reshape(y_true_np.shape[0], -1)
    y_pred_flat = y_pred_np.reshape(y_pred_np.shape[0], -1)
    
    # 効率的なバッチ計算
    dot_products = np.sum(y_true_flat * y_pred_flat, axis=1)
    true_norms = np.sqrt(np.sum(y_true_flat**2, axis=1))
    pred_norms = np.sqrt(np.sum(y_pred_flat**2, axis=1))
    
    # ゼロ除算を防ぐ
    true_norms = np.maximum(true_norms, 1e-10)
    pred_norms = np.maximum(pred_norms, 1e-10)
    
    similarities = dot_products / (true_norms * pred_norms)
    
    # NaNや無限大の値を修正
    similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return np.mean(similarities)

###############################
# トレーニングとモデル評価（最適化）
###############################

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
               eval_interval=2, patience=10, grad_clip=1.0, checkpoint_dir=CACHE_DIR):
    """最適化されたモデルのトレーニング（チェックポイント機能付き）"""
    train_losses = []
    val_losses = []
    val_cosine_similarities = []
    best_cosine = 0.0
    early_stopping_counter = 0
    start_epoch = 0
    
    # チェックポイントディレクトリの作成
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 既存のチェックポイントがあれば読み込む
    latest_checkpoint = None
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
            try:
                epoch_num = int(file.split("_")[2])
                if latest_checkpoint is None or epoch_num > start_epoch:
                    latest_checkpoint = file
                    start_epoch = epoch_num
            except:
                continue
    
    if latest_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"チェックポイントを読み込み: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 最適化情報を明示的にデバイスに移動
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            val_cosine_similarities = checkpoint.get('val_cosine_similarities', [])
            best_cosine = checkpoint.get('best_cosine', 0.0)
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            start_epoch = checkpoint['epoch'] + 1  # 次のエポックから開始
            
            # スケジューラの復元
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except:
                    logger.warning("スケジューラの状態を復元できませんでした")
                    
            # メモリクリーンアップ
            del checkpoint
            aggressive_memory_cleanup()
        except Exception as e:
            logger.error(f"チェックポイント読み込みエラー: {e}")
            start_epoch = 0
    
    # Automatic Mixed Precision (AMP)のスケーラー
    scaler = GradScaler('cuda')
    
    # モデルをデバイスに明示的に転送
    model = model.to(device)
    
    # 合計バッチ数の計算
    total_steps = len(train_loader) * (num_epochs - start_epoch)
    logger.info(f"トレーニング開始: 総ステップ数 = {total_steps}, 開始エポック = {start_epoch + 1}")
    
    # バッチ処理中の定期的なメモリ管理
    total_batches = len(train_loader)
    memory_check_interval = max(1, total_batches // 10)  # 10回程度チェック
    
    for epoch in range(start_epoch, num_epochs):
        # 4エポックごとに強力なメモリクリーンアップを実行
        if epoch % 4 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - 定期的なメモリクリーンアップを実行")
            aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        # 訓練モード
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # プログレスバーで進捗管理
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True)
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                # 定期的なメモリチェック
                if batch_idx % memory_check_interval == 0:
                    memory_cleared = aggressive_memory_cleanup(percent=80)
                    if memory_cleared and batch_idx > 0:
                        logger.info("メモリ使用量を削減しました")
                
                # データをGPUに転送（効率化）
                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        processed_batch[k] = v.to(device, non_blocking=True)
                    elif k == 'graph':
                        # グラフデータは別途処理
                        v.x = v.x.to(device, non_blocking=True)
                        v.edge_index = v.edge_index.to(device, non_blocking=True)
                        v.edge_attr = v.edge_attr.to(device, non_blocking=True)
                        v.batch = v.batch.to(device, non_blocking=True)
                        if hasattr(v, 'global_attr'):
                            v.global_attr = v.global_attr.to(device, non_blocking=True)
                        processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                
                # 勾配をゼロに初期化
                optimizer.zero_grad(set_to_none=True)  # メモリ効率のためNoneに設定
                
                # Automatic Mixed Precision (AMP)を使用した順伝播
                with autocast('cuda'):
                    output, fragment_pred = model(processed_batch)
                    loss = criterion(output, processed_batch['spec'], fragment_pred, processed_batch['fragment_pattern'])
                
                # AMP逆伝播
                scaler.scale(loss).backward()
                
                # 勾配クリッピング
                scaler.unscale_(optimizer)  # スケーリングを解除して勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                
                # オプティマイザステップ
                scaler.step(optimizer)
                scaler.update()
                
                # OneCycleLRスケジューラーの更新
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                
                # 損失を追跡
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1
                
                # プログレスバーの更新
                train_pbar.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'avg_loss': f"{epoch_loss/batch_count:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # バッチごとにメモリ解放
                del loss, output, fragment_pred, processed_batch
                torch.cuda.empty_cache()
                
                # 非常に大規模なデータセットでは定期的にキャッシュをクリア
                if len(train_loader.dataset) > 100000 and batch_idx % (memory_check_interval * 2) == 0:
                    if hasattr(train_loader.dataset, 'graph_cache'):
                        # キャッシュサイズが大きくなりすぎたらクリア
                        if len(train_loader.dataset.graph_cache) > 5000:
                            train_loader.dataset.graph_cache.clear()
                            gc.collect()
                
                # バッチチェックポイント（1000バッチごと）
                if (batch_idx + 1) % 1000 == 0:
                    batch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_cosine_similarities': val_cosine_similarities,
                        'best_cosine': best_cosine,
                        'early_stopping_counter': early_stopping_counter
                    }, batch_checkpoint_path)
                    logger.info(f"バッチチェックポイントを保存: {batch_checkpoint_path}")
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"CUDAメモリ不足: {str(e)}")
                    # 緊急メモリクリーンアップ
                    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
                    continue
                else:
                    print(f"バッチ処理エラー: {str(e)}")
                    # スタックトレースを出力（デバッグに役立つ）
                    import traceback
                    traceback.print_exc()
                    continue
        
        # エポック終了時の評価
        if batch_count > 0:
            avg_train_loss = epoch_loss / batch_count
            train_losses.append(avg_train_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - 平均訓練損失: {avg_train_loss:.4f}")
            
            # エポックチェックポイントの保存
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_cosine_similarities': val_cosine_similarities,
                'best_cosine': best_cosine,
                'early_stopping_counter': early_stopping_counter
            }, checkpoint_path)
            logger.info(f"エポックチェックポイントを保存: {checkpoint_path}")
            
            # 定期的な検証（すべてのエポックで行わない）
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                # 評価前にメモリクリーンアップ
                aggressive_memory_cleanup()
                
                # 評価モードで検証
                val_metrics = evaluate_model(model, val_loader, criterion, device, use_amp=True)
                val_loss = val_metrics['loss']
                cosine_sim = val_metrics['cosine_similarity']
                
                val_losses.append(val_loss)
                val_cosine_similarities.append(cosine_sim)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - 検証損失: {val_loss:.4f}, "
                            f"コサイン類似度: {cosine_sim:.4f}")
                
                # ReduceLROnPlateauスケジューラーの更新
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                
                # 最良モデルの保存
                if cosine_sim > best_cosine:
                    best_cosine = cosine_sim
                    early_stopping_counter = 0
                    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"新しい最良モデル保存: コサイン類似度 = {cosine_sim:.4f}")
                else:
                    early_stopping_counter += 1
                    logger.info(f"早期停止カウンター: {early_stopping_counter}/{patience}")
                    
                # 早期停止
                if early_stopping_counter >= patience:
                    logger.info(f"早期停止: {epoch+1}エポック後")
                    break
            
            # 学習中の損失推移をプロットして保存（5エポックごと）
            if (epoch + 1) % 5 == 0:
                try:
                    _plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine)
                except Exception as e:
                    logger.error(f"プロット作成エラー: {str(e)}")
        else:
            logger.warning("このエポックで成功したバッチ処理がありません。")
            train_losses.append(float('inf'))
            
    # 最終的な学習曲線の保存
    try:
        _plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine)
    except Exception as e:
        logger.error(f"最終プロット作成エラー: {str(e)}")
    
    return train_losses, val_losses, val_cosine_similarities, best_cosine

def _plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine):
    """Training progress visualization"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if val_losses:  # 検証損失が存在する場合
        # エポック間隔を調整
        val_epochs = np.linspace(0, len(train_losses)-1, len(val_losses))
        plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    if val_cosine_similarities:  # コサイン類似度が存在する場合
        val_epochs = np.linspace(0, len(train_losses)-1, len(val_cosine_similarities))
        plt.plot(val_epochs, val_cosine_similarities, label='Validation Cosine Similarity')
        plt.axhline(y=best_cosine, color='r', linestyle='--', label=f'Best: {best_cosine:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.title('Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig('hybrid_learning_curves.png')
    plt.close()
    
def evaluate_model(model, data_loader, criterion, device, use_amp=False):
    """モデル評価用の関数"""
    model.eval()
    total_loss = 0
    batch_count = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="評価中", leave=False):
            try:
                # データをGPUに転送
                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        processed_batch[k] = v.to(device, non_blocking=True)
                    elif k == 'graph':
                        # グラフデータは別途処理
                        v.x = v.x.to(device, non_blocking=True)
                        v.edge_index = v.edge_index.to(device, non_blocking=True)
                        v.edge_attr = v.edge_attr.to(device, non_blocking=True)
                        v.batch = v.batch.to(device, non_blocking=True)
                        if hasattr(v, 'global_attr'):
                            v.global_attr = v.global_attr.to(device, non_blocking=True)
                        processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                
                # AMP使用時は混合精度で予測
                if use_amp:
                    with autocast('cuda'):
                        output, fragment_pred = model(processed_batch)
                        loss = criterion(output, processed_batch['spec'], 
                                         fragment_pred, processed_batch['fragment_pattern'])
                else:
                    output, fragment_pred = model(processed_batch)
                    loss = criterion(output, processed_batch['spec'], 
                                     fragment_pred, processed_batch['fragment_pattern'])
                
                total_loss += loss.item()
                batch_count += 1
                
                # 類似度計算用に結果を保存
                y_true.append(processed_batch['spec'].cpu())
                y_pred.append(output.cpu())
                
            except RuntimeError as e:
                print(f"評価中にエラー発生: {str(e)}")
                continue
    
    # 結果を集計
    if batch_count > 0:
        avg_loss = total_loss / batch_count
        
        # コサイン類似度を計算
        if y_true and y_pred:
            try:
                all_true = torch.cat(y_true, dim=0)
                all_pred = torch.cat(y_pred, dim=0)
                cosine_sim = cosine_similarity_score(all_true, all_pred)
            except Exception as e:
                print(f"類似度計算エラー: {str(e)}")
                cosine_sim = 0.0
        else:
            cosine_sim = 0.0
        
        return {
            'loss': avg_loss,
            'cosine_similarity': cosine_sim
        }
    else:
        return {
            'loss': float('inf'),
            'cosine_similarity': 0.0
        }

def eval_model(model, test_loader, device, use_amp=True, transform="log10over3"):
    """テスト用の評価関数 - 元のマススペクトル形式も保存"""
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_true_original = []  # 元の形式
    y_pred_original = []  # 元の形式
    fragment_true = []
    fragment_pred = []
    mol_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="テスト中"):
            try:
                # データをGPUに転送
                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        processed_batch[k] = v.to(device, non_blocking=True)
                    elif k == 'graph':
                        # グラフデータは別途処理
                        v.x = v.x.to(device, non_blocking=True)
                        v.edge_index = v.edge_index.to(device, non_blocking=True)
                        v.edge_attr = v.edge_attr.to(device, non_blocking=True)
                        v.batch = v.batch.to(device, non_blocking=True)
                        if hasattr(v, 'global_attr'):
                            v.global_attr = v.global_attr.to(device, non_blocking=True)
                        processed_batch[k] = v
                    else:
                        processed_batch[k] = v
                
                # 予測（混合精度使用時）
                if use_amp:
                    with autocast('cuda'):
                        output, frag_pred = model(processed_batch)
                else:
                    output, frag_pred = model(processed_batch)
                
                # 結果を保存
                true_batch = processed_batch['spec'].cpu()
                pred_batch = output.cpu()
                
                # 元の形式に変換
                true_original_batch = torch.zeros_like(true_batch)
                pred_original_batch = torch.zeros_like(pred_batch)
                
                for i in range(len(true_batch)):
                    # 実測スペクトルを元の形式に変換
                    true_tensor = true_batch[i]
                    true_nonzero = (true_tensor > 0)
                    if true_nonzero.sum() > 0:
                        true_original_batch[i] = true_tensor / true_tensor.max() * 100
                    
                    # 予測スペクトルを元の形式に変換
                    pred_original_batch[i] = convert_to_original_format(
                        pred_batch[i].unsqueeze(0), transform).squeeze(0)
                
                y_true.append(true_batch)
                y_pred.append(pred_batch)
                y_true_original.append(true_original_batch)
                y_pred_original.append(pred_original_batch)
                fragment_true.append(processed_batch['fragment_pattern'].cpu())
                fragment_pred.append(frag_pred.cpu())
                mol_ids.extend(processed_batch['mol_id'])
                
                # バッチごとにメモリ解放
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"テスト中にエラー発生: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # 結果を連結
    all_true = torch.cat(y_true, dim=0)
    all_pred = torch.cat(y_pred, dim=0)
    all_true_original = torch.cat(y_true_original, dim=0)
    all_pred_original = torch.cat(y_pred_original, dim=0)
    all_fragment_true = torch.cat(fragment_true, dim=0)
    all_fragment_pred = torch.cat(fragment_pred, dim=0)
    
    # スコア計算（元の形式で計算）
    cosine_sim = cosine_similarity_score(all_true_original, all_pred_original)
    
    return {
        'cosine_similarity': cosine_sim,
        'y_true': all_true,
        'y_pred': all_pred,
        'y_true_original': all_true_original,
        'y_pred_original': all_pred_original,
        'fragment_true': all_fragment_true,
        'fragment_pred': all_fragment_pred,
        'mol_ids': mol_ids
    }

def tiered_training(model, train_ids, val_loader, criterion, optimizer, scheduler, device, 
                  mol_files_path, msp_data, transform, normalization, cache_dir, 
                  checkpoint_dir=CACHE_DIR, num_workers=0, patience=5):
    """段階的トレーニング（大規模データセット用）"""
    logger.info("段階的トレーニングを開始")
    
    # データセットサイズに基づくティア定義
    if len(train_ids) > 100000:
        train_tiers = [
            train_ids[:10000],    # 1万サンプルから開始
            train_ids[:30000],    # 次に3万
            train_ids[:60000],    # 次に6万
            train_ids[:100000],   # 次に10万
            train_ids             # 最後に全データ
        ]
        tier_epochs = [3, 3, 4, 5, 15]  # ティアごとのエポック数
    elif len(train_ids) > 50000:
        train_tiers = [
            train_ids[:10000], 
            train_ids[:30000],
            train_ids
        ]
        tier_epochs = [3, 4, 23]
    else:
        # 小さなデータセットは段階を少なく
        train_tiers = [
            train_ids[:5000] if len(train_ids) > 5000 else train_ids[:len(train_ids)//2],
            train_ids
        ]
        tier_epochs = [5, 25]
    
    best_cosine = 0.0
    all_train_losses = []
    all_val_losses = []
    all_val_cosine_similarities = []
    
    # 進行状況を表示するために各ティアにプレフィックスを追加
    tier_prefixes = [f"Tier {i+1}/{len(train_tiers)}" for i in range(len(train_tiers))]
    
    # 各ティアを処理
    for tier_idx, (tier_ids, tier_prefix) in enumerate(zip(train_tiers, tier_prefixes)):
        tier_name = f"{tier_prefix} ({len(tier_ids)} サンプル)"
        logger.info(f"=== {tier_name} のトレーニングを開始 ===")
        
        # ティア間でメモリクリーンアップ
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        # このティア用のデータセット作成
        tier_dataset = OptimizedMoleculeGraphDataset(
            tier_ids, mol_files_path, msp_data, 
            transform=transform, normalization=normalization,
            augment=True, cache_dir=cache_dir
        )
        
        # ティアサイズに基づいてバッチサイズを調整
        if len(tier_ids) <= 10000:
            tier_batch_size = 16  # 小さいティアでは大きいバッチサイズ
        elif len(tier_ids) <= 30000:
            tier_batch_size = 12  # 中間ティア
        elif len(tier_ids) <= 60000:
            tier_batch_size = 8   # 大きいティア
        else:
            tier_batch_size = 6   # 非常に大きいティア
        
        logger.info(f"ティア {tier_idx+1} のバッチサイズ: {tier_batch_size}")
        
        # このティア用のデータローダを作成
        tier_loader = DataLoader(
            tier_dataset, 
            batch_size=tier_batch_size,
            shuffle=True, 
            collate_fn=optimized_collate_fn,
            num_workers=0,  # シングルプロセス
            pin_memory=True,
            persistent_workers=False,
            drop_last=True
        )
        
        # オプティマイザの学習率を調整
        for param_group in optimizer.param_groups:
            if tier_idx == 0:
                param_group['lr'] = 0.001  # 小さいデータセット用に高い学習率
            else:
                param_group['lr'] = 0.0008 * (0.8 ** tier_idx)  # 大きいティア向けに学習率を減少
        
        # このティアの忍耐値を計算（前半のティアは早く次に進む）
        tier_patience = max(2, patience // 2) if tier_idx < len(train_tiers) - 1 else patience
        
        # このティア用のスケジューラを作成（OneCycleLR）
        steps_per_epoch = len(tier_loader)
        tier_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001 if tier_idx == 0 else 0.0008 * (0.8 ** tier_idx),
            steps_per_epoch=steps_per_epoch,
            epochs=tier_epochs[tier_idx],
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # 指定されたエポック数でこのティアをトレーニング
        train_losses, val_losses, val_cosine_similarities, tier_best_cosine = train_model(
            model=model,
            train_loader=tier_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=tier_scheduler,
            device=device,
            num_epochs=tier_epochs[tier_idx],
            eval_interval=1,  # 毎エポック評価
            patience=tier_patience,
            grad_clip=1.0,
            checkpoint_dir=os.path.join(checkpoint_dir, f"tier{tier_idx+1}")
        )
        
        # 全体の最良性能を更新
        best_cosine = max(best_cosine, tier_best_cosine)
        
        # 損失と類似度を記録
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_val_cosine_similarities.extend(val_cosine_similarities)
        
        # ティア間でキャッシュをクリア
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        del tier_dataset, tier_loader
        gc.collect()
        torch.cuda.empty_cache()
        
        # ティアチェックポイントを保存
        tier_checkpoint_path = os.path.join(checkpoint_dir, f"tier{tier_idx+1}_model.pth")
        torch.save(model.state_dict(), tier_checkpoint_path)
        logger.info(f"ティア {tier_idx+1} チェックポイント保存: {tier_checkpoint_path}")
        
        # ティア間でシステムのメモリを安定化
        logger.info(f"ティア {tier_idx+1} 完了、次のティアの前にメモリを安定化")
        time.sleep(5)  # 短い休憩を入れてシステムを安定化
    
    # 全ティアの学習曲線を保存
    try:
        final_plot_path = os.path.join(checkpoint_dir, "tiered_learning_curves.png")
        _plot_training_progress(all_train_losses, all_val_losses, all_val_cosine_similarities, 
                              best_cosine)
        logger.info(f"段階的トレーニングの学習曲線を保存: {final_plot_path}")
    except Exception as e:
        logger.error(f"プロット作成エラー: {str(e)}")
    
    return all_train_losses, all_val_losses, all_val_cosine_similarities, best_cosine

def visualize_results(test_results, num_samples=10, transform="log10over3"):
    """Prediction result visualization - 元のマススペクトル形式で表示"""
    plt.figure(figsize=(15, num_samples*4))
    
    # サンプルのインデックスをランダムに選択
    if 'mol_ids' in test_results and len(test_results['mol_ids']) > 0:
        sample_indices = np.random.choice(len(test_results['mol_ids']), 
                                         min(num_samples, len(test_results['mol_ids'])), 
                                         replace=False)
    else:
        # mol_idsがない場合は予測結果から選択
        sample_indices = np.random.choice(len(test_results['y_true']), 
                                         min(num_samples, len(test_results['y_true'])), 
                                         replace=False)
    
    for i, idx in enumerate(sample_indices):
        # 真のスペクトルと予測スペクトルを元の形式に変換
        true_spec = test_results['y_true'][idx].numpy()
        pred_spec = test_results['y_pred'][idx].numpy()
        
        # 予測を元の形式に変換
        pred_tensor = torch.FloatTensor(pred_spec)
        pred_original = convert_to_original_format(pred_tensor.unsqueeze(0), transform).squeeze(0).numpy()
        
        # 測定データを元の形式に変換（単純に最大値で正規化）
        true_tensor = torch.FloatTensor(true_spec)
        true_nonzero = (true_tensor > 0)
        if true_nonzero.sum() > 0:
            true_original = (true_tensor / true_tensor.max() * 100).numpy()
        else:
            true_original = true_tensor.numpy()
        
        # 類似度を計算 - 元の形式で計算
        true_vector = true_original.reshape(1, -1)
        pred_vector = pred_original.reshape(1, -1)
        # ゼロ除算を防ぐ
        if np.sum(true_vector**2) > 0 and np.sum(pred_vector**2) > 0:
            sim = cosine_similarity(true_vector, pred_vector)[0][0]
        else:
            sim = 0.0
        
        # 真のスペクトル - 棒グラフ表示（ステムプロット）
        plt.subplot(num_samples, 2, 2*i + 1)
        
        # 非ゼロの位置のみをプロット
        nonzero_indices = np.nonzero(true_original)[0]
        if len(nonzero_indices) > 0:
            plt.stem(nonzero_indices, true_original[nonzero_indices], markerfmt="o", basefmt=" ")
        
        # タイトルの設定
        mol_id_str = f" - ID: {test_results['mol_ids'][idx]}" if 'mol_ids' in test_results else ""
        plt.title(f"Measured Spectrum{mol_id_str}")
        plt.xlabel("m/z")
        plt.ylabel("Relative Intensity (%)")
        plt.ylim([0, 105])  # 最大値100%に少し余裕を持たせる
        
        # 予測スペクトル - 棒グラフ表示（ステムプロット）
        plt.subplot(num_samples, 2, 2*i + 2)
        
        # 非ゼロの位置のみをプロット
        nonzero_indices = np.nonzero(pred_original)[0]
        if len(nonzero_indices) > 0:
            plt.stem(nonzero_indices, pred_original[nonzero_indices], markerfmt="o", basefmt=" ", linefmt="r-")
        
        plt.title(f"Predicted Spectrum - Similarity: {sim:.4f}")
        plt.xlabel("m/z")
        plt.ylabel("Relative Intensity (%)")
        plt.ylim([0, 105])  # 最大値100%に少し余裕を持たせる
    
    plt.tight_layout()
    plt.savefig('hybrid_prediction_visualization.png')
    plt.close()

###############################
# メイン関数（最適化）
###############################

def main():
    # 開始メッセージ
    logger.info("============= 質量スペクトル予測モデルの実行開始 =============")
    
    # CUDA設定
    torch.backends.cudnn.benchmark = True  # CUDNN最適化を有効化
    
    # GPUメモリ使用状況を確認
    if torch.cuda.is_available():
        logger.info(f"使用中のGPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU総メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"利用可能メモリ: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
    
    # MSPファイルを解析（キャッシュ対応）
    logger.info("MSPファイルを解析中...")
    msp_data = parse_msp_file(MSP_FILE_PATH, cache_dir=CACHE_DIR)
    logger.info(f"MSPファイルから{len(msp_data)}個の化合物データを読み込みました")
    
    # 利用可能なMOLファイルを確認
    mol_ids = []
    mol_files = os.listdir(MOL_FILES_PATH)
    logger.info(f"MOLファイル総数: {len(mol_files)}")
    
    # キャッシュファイルのパス
    mol_id_cache_file = os.path.join(CACHE_DIR, "valid_mol_ids_full.pkl")
    
    # キャッシュが存在するか確認
    if os.path.exists(mol_id_cache_file):
        logger.info(f"キャッシュからmol_idsを読み込み中: {mol_id_cache_file}")
        with open(mol_id_cache_file, 'rb') as f:
            mol_ids = pickle.load(f)
        logger.info(f"キャッシュから{len(mol_ids)}個の有効なmol_idsを読み込みました")
    else:
        # 複数のチャンクで処理して進捗表示
        chunk_size = 5000
        for i in range(0, len(mol_files), chunk_size):
            chunk = mol_files[i:min(i+chunk_size, len(mol_files))]
            logger.info(f"MOLファイル処理中: {i+1}-{i+len(chunk)}/{len(mol_files)}")
            
            for filename in chunk:
                if filename.startswith("ID") and filename.endswith(".MOL"):
                    try:
                        mol_id = int(filename[2:-4])  # "ID300001.MOL" → 300001
                        if mol_id in msp_data:
                            mol_ids.append(mol_id)
                    except:
                        continue
                        
        # キャッシュに保存            
        logger.info(f"mol_idsをキャッシュに保存中: {mol_id_cache_file} (合計: {len(mol_ids)}件)")
        with open(mol_id_cache_file, 'wb') as f:
            pickle.dump(mol_ids, f)
    
    logger.info(f"MOLファイルとMSPデータが揃っている化合物: {len(mol_ids)}個")
    
    # データ分割 (訓練:検証:テスト = 80:10:10)
    train_ids, test_ids = train_test_split(mol_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    
    logger.info(f"訓練データ: {len(train_ids)}個")
    logger.info(f"検証データ: {len(val_ids)}個")
    logger.info(f"テストデータ: {len(test_ids)}個")
    
    # ハイパーパラメータ
    transform = "log10over3"  # スペクトル変換タイプ
    normalization = "l1"      # 正規化タイプ
    
    # データセット作成 - 段階的トレーニングを使用するため訓練データセットは後で作成
    val_dataset = OptimizedMoleculeGraphDataset(
        val_ids, MOL_FILES_PATH, msp_data,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    test_dataset = OptimizedMoleculeGraphDataset(
        test_ids, MOL_FILES_PATH, msp_data,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    logger.info(f"有効な検証データ: {len(val_dataset)}個")
    logger.info(f"有効なテストデータ: {len(test_dataset)}個")
    
    # GPU使用量を最適化するためのバッチサイズ調整
    # A4500のメモリに適したバッチサイズ - データセットサイズに応じて自動調整
    if len(train_ids) > 100000:
        batch_size = 8  # 非常に大きなデータセット用
    elif len(train_ids) > 50000:
        batch_size = 12  # 大きなデータセット用
    else:
        batch_size = 16  # 通常サイズのデータセット用
        
    num_workers = 0  # シングルプロセスのため0に設定
    
    logger.info(f"バッチサイズ: {batch_size}, ワーカー数: {num_workers}")
    
    # データローダー作成 - 段階的トレーニングのため訓練ローダーは作成しない
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=optimized_collate_fn,
        num_workers=0,  # シングルプロセス
        pin_memory=True,
        persistent_workers=False,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=optimized_collate_fn,
        num_workers=0,  # シングルプロセス
        pin_memory=True,
        drop_last=True
    )
    
    # モデルの次元を決定
    sample = val_dataset[0]
    node_features = sample['graph_data'].x.shape[1]
    edge_features = sample['graph_data'].edge_attr.shape[1]
    
    # データセットサイズに基づいて次元を調整
    if len(train_ids) > 100000:
        hidden_channels = 24  # 非常に大きなデータセット用に縮小
    else:
        hidden_channels = 32  # 通常サイズのデータセット用
        
    out_channels = MAX_MZ
    
    # 大規模データセット用にキャッシュ管理を強化
    def limit_memory_usage(percent=85):
        """GCを実行し、使用済みメモリをチェック。閾値を超えていたらキャッシュをクリア"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # GPUメモリ使用率をチェック
            gpu_memory_allocated_percent = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            if gpu_memory_allocated_percent > percent:
                logger.warning(f"GPUメモリ使用率が高い ({gpu_memory_allocated_percent:.1f}%)。メモリ管理を強化します。")
                # データセットのキャッシュをクリア
                if 'val_dataset' in globals() and hasattr(val_dataset, 'graph_cache'):
                    val_dataset.graph_cache.clear()
                if 'test_dataset' in globals() and hasattr(test_dataset, 'graph_cache'):
                    test_dataset.graph_cache.clear()
                torch.cuda.empty_cache()
                return True
        return False
    
    # モデルの初期化前にメモリ確保
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    # モデルの初期化と明示的なデバイス転送
    model = OptimizedHybridMSModel(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_fragments=NUM_FRAGS,
        prec_mass_offset=10,    # 前駆体質量オフセット
        bidirectional=True,     # 双方向予測を使用
        gate_prediction=True    # ゲート予測を使用
    ).to(device)
    
    # 総パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"総パラメータ数: {total_params:,}")
    logger.info(f"学習可能パラメータ数: {trainable_params:,}")
    
    # 損失関数、オプティマイザー、スケジューラーの設定
    criterion = combined_loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001,       # 初期学習率
        weight_decay=1e-6,  # 重み減衰
        eps=1e-8        # 数値安定性用
    )
    
    # ダミースケジューラー（段階的トレーニングでは各ティアで再定義）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # エポック数とその他のトレーニングパラメータ
    num_epochs = 30  # 30エポックに増加
    patience = 7     # 忍耐値
    
    logger.info(f"モデルトレーニング設定: エポック数={num_epochs}, 忍耐値={patience}, バッチサイズ={batch_size}")
    logger.info("段階的トレーニングを使用してモデルのトレーニングを開始します...")
    
    # CPU、GPUキャッシュをクリア
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # 段階的トレーニングを使用
    train_losses, val_losses, val_cosine_similarities, best_cosine = tiered_training(
        model=model,
        train_ids=train_ids,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mol_files_path=MOL_FILES_PATH,
        msp_data=msp_data,
        transform=transform,
        normalization=normalization,
        cache_dir=CACHE_DIR,
        checkpoint_dir=os.path.join(CACHE_DIR, "checkpoints"),
        num_workers=0,  # シングルプロセス
        patience=patience
    )
    
    logger.info(f"トレーニング完了！ 最良コサイン類似度: {best_cosine:.4f}")
    
    # キャッシュクリア
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # 最良モデルを読み込む
    try:
        best_model_path = os.path.join(CACHE_DIR, "checkpoints", 'best_model.pth')
        if not os.path.exists(best_model_path):
            # ティアごとの最良モデルを探す
            tier_models = [f for f in os.listdir(os.path.join(CACHE_DIR, "checkpoints")) 
                         if f.startswith("tier") and f.endswith("_model.pth")]
            if tier_models:
                # 最後のティアモデルを使用
                best_model_path = os.path.join(CACHE_DIR, "checkpoints", tier_models[-1])
        
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        logger.info(f"最良モデルを読み込みました: {best_model_path}")
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
    
    # テストデータでの評価
    try:
        # テスト前にメモリ解放
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        logger.info("テストデータでの評価を開始します...")
        test_results = eval_model(model, test_loader, device, use_amp=True)
        logger.info(f"テストデータ平均コサイン類似度: {test_results['cosine_similarity']:.4f}")
        
        # 予測結果の可視化
        visualize_results(test_results, num_samples=10)
        logger.info("予測結果の可視化を保存しました: hybrid_prediction_visualization.png")
    except Exception as e:
        logger.error(f"テスト評価エラー: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("解析プロセス完了！")
    
    # 追加の結果分析
    try:
        # 類似度分布のヒストグラム
        similarities = []
        for i in range(len(test_results['y_true'])):
            true_vector = test_results['y_true'][i].reshape(1, -1).cpu().numpy()
            pred_vector = test_results['y_pred'][i].reshape(1, -1).cpu().numpy()
            sim = cosine_similarity(true_vector, pred_vector)[0][0]
            similarities.append(sim)
        
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, alpha=0.7)
        plt.axvline(x=test_results['cosine_similarity'], color='r', linestyle='--', 
                    label=f'Mean: {test_results["cosine_similarity"]:.4f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Number of Samples')
        plt.title('Cosine Similarity Distribution of Test Data')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('similarity_distribution.png')
        logger.info("Similarity distribution saved: similarity_distribution.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error in additional analysis: {e}")
    
    logger.info("============= 質量スペクトル予測モデルの実行終了 =============")
    return model, train_losses, val_losses, val_cosine_similarities, test_results

if __name__ == "__main__":
    main()
