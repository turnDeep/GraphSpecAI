# -*- coding: utf-8 -*-
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
    try:
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
    except Exception as e:
        logger.warning(f"GPUメモリクリーンアップ中にエラー: {e}")
        # Continue even if cleanup fails

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

# 重要なm/z値のリスト
IMPORTANT_MZ = [18, 28, 43, 57, 71, 73, 77, 91, 105, 115, 128, 152, 165, 178, 207]

# エフェメラル値
EPS = np.finfo(np.float32).eps

# グローバル特徴量の次元数
GLOBAL_FEATURES_DIM = 16

# 原子の特徴マッピング
ATOM_FEATURES = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8,
    'Si': 9, 'B': 10, 'Na': 11, 'K': 12, 'Li': 13, 'Mg': 14, 'Ca': 15, 'Fe': 16,
    'Co': 17, 'Ni': 18, 'Cu': 19, 'Zn': 20, 'H': 21, 'OTHER': 22
}
NODE_FEATURES_DIM = len(ATOM_FEATURES) + 12 # ノード特徴量の次元数を計算

# 結合の特徴マッピング
BOND_FEATURES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}
EDGE_FEATURES_DIM = len(BOND_FEATURES) + 3 # エッジ特徴量の次元数を計算

# フラグメントパターンの数（MACCSキー使用）
NUM_FRAGS = 167  # MACCSキーのビット数

###############################
# データ処理関連の関数
###############################

def process_spec(spec, transform, normalization, eps=EPS):
    """スペクトルにトランスフォームと正規化を適用"""
    # Ensure input is a tensor
    if isinstance(spec, np.ndarray):
        spec = torch.from_numpy(spec).float()

    # スペクトルを1000までスケーリング (入力がバッチでも単一でも動作するように)
    spec_max = torch.max(spec, dim=-1, keepdim=True)[0]
    spec = torch.where(spec_max > eps, spec / (spec_max + eps) * 1000.0, torch.zeros_like(spec))

    # 信号変換
    if transform == "log10":
        spec = torch.log10(spec + 1.0) # Add 1 before log
    elif transform == "log10over3":
        spec = torch.log10(spec + 1.0) / 3.0
    elif transform == "loge":
        spec = torch.log(spec + 1.0)
    elif transform == "sqrt":
        spec = torch.sqrt(spec)
    elif transform == "none":
        pass
    else:
        raise ValueError(f"invalid transform: {transform}")

    # 正規化
    if normalization == "l1":
        spec = F.normalize(spec, p=1, dim=-1, eps=eps)
    elif normalization == "l2":
        spec = F.normalize(spec, p=2, dim=-1, eps=eps)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"invalid normalization: {normalization}")

    # Check for NaNs after processing
    if torch.isnan(spec).any():
         logger.warning("NaN detected in process_spec output. Replacing with 0.")
         spec = torch.nan_to_num(spec, nan=0.0)

    return spec

def unprocess_spec(spec, transform):
    """スペクトルの変換を元に戻す（正規化は戻さないが、スケールは変換前の最大1000に近づける）"""
    # Ensure spec is a tensor
    if isinstance(spec, np.ndarray):
        spec = torch.from_numpy(spec).float()
    elif not isinstance(spec, torch.Tensor):
        raise TypeError(f"Input spec must be a NumPy array or PyTorch tensor, got {type(spec)}")

    # Ensure float type
    spec = spec.float()

    device = spec.device # 元のデバイスを保持
    spec = spec.cpu() # CPUで計算 (安全のため)

    # Find max intensity based on transform type
    if transform == "log10":
        max_ints = float(np.log10(1000. + 1.))
        def untransform_fn(x): return torch.pow(10.0, x) - 1.0 # Use torch.pow
    elif transform == "log10over3":
        max_ints = float(np.log10(1000. + 1.) / 3.)
        def untransform_fn(x): return torch.pow(10.0, 3.0 * x) - 1.0
    elif transform == "loge":
        max_ints = float(np.log(1000. + 1.))
        def untransform_fn(x): return torch.exp(x) - 1.0
    elif transform == "sqrt":
        max_ints = float(np.sqrt(1000.))
        def untransform_fn(x): return torch.pow(x, 2) # Use torch.pow
    elif transform == "none":
        max_ints = 1000.
        def untransform_fn(x): return x
    else:
        raise ValueError(f"invalid transform: {transform}")

    # Check if max_ints is valid
    if max_ints <= 0:
        logger.warning(f"Invalid max_ints ({max_ints}) for transform '{transform}'. Using 1.0 instead.")
        max_ints = 1.0

    # --- ▼▼▼ 修正箇所 ▼▼▼ ---
    # Calculate max value for each spectrum in the batch
    max_val = torch.max(spec, dim=-1, keepdim=True)[0]

    # Condition for applying the inverse transform (element-wise)
    # Check where max_val is greater than epsilon
    condition = max_val.abs() > EPS

    # Rescale where condition is true
    # Add epsilon to max_val in denominator to avoid division by zero even if condition check somehow missed it
    spec_rescaled = torch.where(condition, spec / (max_val + EPS) * max_ints, torch.zeros_like(spec))

    # Apply the inverse transformation function element-wise
    spec_unprocessed = untransform_fn(spec_rescaled)

    # Where condition was false (max_val was near zero), output should be zero
    spec_unprocessed = torch.where(condition, spec_unprocessed, torch.zeros_like(spec))
    # --- ▲▲▲ 修正箇所 ▲▲▲ ---


    # Clamp negative values to zero (can happen due to numerical inaccuracies)
    spec_unprocessed = torch.clamp(spec_unprocessed, min=0.)

    # Ensure no NaNs
    if torch.isnan(spec_unprocessed).any():
        logger.warning("NaN detected during unprocess_spec. Replacing NaNs with 0.")
        spec_unprocessed = torch.nan_to_num(spec_unprocessed, nan=0.0)

    return spec_unprocessed.to(device) # 元のデバイスに戻す


def postprocess_prediction(pred_spec, transform):
    """モデルの予測出力を後処理し、最大強度100のスケールに戻す"""
    # Apply unprocess_spec to reverse transformations
    unprocessed = unprocess_spec(pred_spec, transform)

    # Normalize to max intensity 100 (element-wise using torch.where)
    max_intensity = torch.max(unprocessed, dim=-1, keepdim=True)[0]

    # Avoid division by zero using torch.where
    pred_spec_final = torch.where(
        max_intensity > EPS,
        unprocessed / (max_intensity + EPS) * 100.0,
        torch.zeros_like(unprocessed) # Return zeros if max intensity is near zero
    )

    # Ensure non-negative and no NaNs
    pred_spec_final = torch.clamp(pred_spec_final, min=0.)
    if torch.isnan(pred_spec_final).any():
        logger.warning("NaN detected during postprocess_prediction. Replacing NaNs with 0.")
        pred_spec_final = torch.nan_to_num(pred_spec_final, nan=0.0)

    return pred_spec_final


def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    """前駆体質量によるマスキング"""
    device = raw_prediction.device
    max_idx = raw_prediction.shape[1]

    if prec_mass_idx is None:
         logger.warning("mask_prediction_by_mass received None for prec_mass_idx. No mask applied.")
         return raw_prediction

    # Ensure prec_mass_idx is a LongTensor
    if not isinstance(prec_mass_idx, torch.Tensor):
        try:
            prec_mass_idx = torch.tensor(prec_mass_idx, dtype=torch.long, device=device)
        except Exception as e:
             logger.error(f"Failed to convert prec_mass_idx to tensor: {e}. Skipping mask.")
             return raw_prediction
    elif prec_mass_idx.dtype != torch.long:
        prec_mass_idx = prec_mass_idx.long()


    # Clamp indices to be within valid range [0, max_idx-1]
    # Ensure clamping happens on the correct device
    prec_mass_idx = torch.clamp(prec_mass_idx, min=0, max=max_idx - 1)


    idx = torch.arange(max_idx, device=device)
    # Mask includes indices <= precursor_mz + offset
    # Ensure broadcasting works correctly: idx shape (max_idx), prec_mass_idx shape (B) -> (B, 1)
    if prec_mass_idx.ndim == 0: # Handle scalar case
         prec_mass_idx = prec_mass_idx.unsqueeze(0)
    mask = (idx.unsqueeze(0) <= (prec_mass_idx.unsqueeze(1) + prec_mass_offset)).float()

    return mask * raw_prediction + (1. - mask) * mask_value

def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    """予測を反転する（双方向予測用）"""
    device = raw_prediction.device
    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]

    if prec_mass_idx is None:
         logger.warning("reverse_prediction received None for prec_mass_idx. Skipping reversal.")
         # Return zeros or original prediction? Returning zeros might be safer.
         return torch.zeros_like(raw_prediction)

    # Ensure prec_mass_idx is a LongTensor on the correct device
    if not isinstance(prec_mass_idx, torch.Tensor):
        try:
            prec_mass_idx = torch.tensor(prec_mass_idx, dtype=torch.long, device=device)
        except Exception as e:
             logger.error(f"Failed to convert prec_mass_idx to tensor in reverse: {e}. Returning zeros.")
             return torch.zeros_like(raw_prediction)
    elif prec_mass_idx.dtype != torch.long:
        prec_mass_idx = prec_mass_idx.long()
    if prec_mass_idx.device != device:
         prec_mass_idx = prec_mass_idx.to(device)


    # Clamp indices to be within valid range [0, max_idx-1]
    prec_mass_idx = torch.clamp(prec_mass_idx, min=0, max=max_idx - 1)

    # Flip the prediction along the m/z dimension
    rev_prediction = torch.flip(raw_prediction, dims=(1,))

    # Calculate the effective end index for the reversed spectrum part
    # It should be min(max_idx, precursor_mz + offset + 1)
    offset_idx = torch.minimum(
        (max_idx * torch.ones_like(prec_mass_idx)).long(), # Ensure long type for comparison
        prec_mass_idx + prec_mass_offset + 1)
    # Calculate shift amount: how many positions to shift the reversed spectrum left
    shifts = - (max_idx - offset_idx) # Negative shift amount

    # Create indices for gathering shifted elements
    gather_idx = torch.arange(max_idx, device=device).unsqueeze(0).expand(batch_size, max_idx)
    # Apply circular shift using modulo operator
    # The indices are shifted left by 'shifts' amount (which is negative)
    # Example: gather_idx = (torch.arange(10) - (-3)) % 10 = (torch.arange(10) + 3) % 10 = [3, 4, ..., 9, 0, 1, 2]
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
    offset_rev_prediction = torch.gather(rev_prediction, 1, gather_idx)

    return offset_rev_prediction

def parse_msp_file(msp_file_path, cache_dir=CACHE_DIR):
    """
    MSPファイルを解析し、ID->マススペクトルのマッピングを返す（キャッシュ対応）。
    スムージングは削除し、フィルタリング閾値を1%に変更。
    """
    cache_file = os.path.join(cache_dir, f"msp_data_cache_no_smooth_1pct_v2_{os.path.basename(msp_file_path)}.pkl") # v2 for float32 fix

    if os.path.exists(cache_file):
        logger.info(f"キャッシュからMSPデータを読み込み中: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: return pickle.load(f)
        except Exception as e: logger.warning(f"キャッシュ読み込み失敗 ({e})。再解析します。")

    logger.info(f"MSPファイルを解析中 (スムージングなし, 1%フィルタ): {msp_file_path}")
    msp_data = {}
    current_id = None
    current_peaks = []
    entry_count = 0

    try:
        with open(msp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Use tqdm for progress indication
            file_lines = f.readlines()
            logger.info(f"MSPファイル読み込み完了。{len(file_lines)} 行。エントリを解析中...")

            for line_num, line in enumerate(tqdm(file_lines, desc="MSP解析", unit="行")):
                line = line.strip()

                if line.startswith("ID:"):
                    if current_id is not None and current_peaks:
                        entry_count += 1
                        ms_vector = np.zeros(MAX_MZ, dtype=np.float32)
                        for mz, intensity in current_peaks:
                            mz_int = int(round(mz))
                            if 0 <= mz_int < MAX_MZ:
                                ms_vector[mz_int] = max(ms_vector[mz_int], float(intensity))

                        if np.sum(ms_vector) > EPS: # Use EPS check
                            max_intensity = np.max(ms_vector)
                            if max_intensity > EPS: ms_vector = (ms_vector / max_intensity) * 100.0
                            else: ms_vector.fill(0.0)

                            non_zero_intensities = ms_vector[ms_vector > EPS]
                            if len(non_zero_intensities) > 0:
                                threshold = np.percentile(non_zero_intensities, 1)
                                ms_vector[ms_vector < threshold] = 0

                            # Important MZ emphasis (optional)
                            # for mz_idx in IMPORTANT_MZ:
                            #     if 0 <= mz_idx < MAX_MZ and ms_vector[mz_idx] > EPS:
                            #         ms_vector[mz_idx] *= 1.5 # Comment out if pure spectrum needed

                            msp_data[current_id] = ms_vector # Already float32
                        else:
                            msp_data[current_id] = ms_vector # Store zero vector

                    try: current_id = int(line.split(":")[1].strip()); current_peaks = []
                    except (ValueError, IndexError): current_id = None; current_peaks = []

                elif line.startswith("Num peaks:"): pass # Ignore

                elif line == "": pass # Usually handled by ID: line trigger

                elif current_id is not None and " " in line and not any(line.startswith(prefix) for prefix in ["Name:", "Formula:", "MW:", "ExactMass:", "CASNO:", "Comment:", "Synonym:", "DB#:", "InChIKey:", "InChI:", "SMILES:"]):
                    try:
                        parts = line.split(); cleaned_parts = []
                        for part in parts:
                             if ';' in part: cleaned_parts.append(part.split(';')[0]); break
                             else: cleaned_parts.append(part)
                        if len(cleaned_parts) >= 2:
                            mz_val = float(cleaned_parts[0]); intensity_val = float(cleaned_parts[1])
                            if intensity_val >= 0: current_peaks.append((mz_val, intensity_val))
                    except ValueError: pass

    except FileNotFoundError: logger.error(f"MSPファイルが見つかりません: {msp_file_path}"); return {}
    except Exception as e: logger.error(f"MSPファイル読み込みエラー: {e}", exc_info=True)

    # Process the last entry
    if current_id is not None and current_peaks:
        entry_count += 1
        ms_vector = np.zeros(MAX_MZ, dtype=np.float32)
        for mz, intensity in current_peaks:
            mz_int = int(round(mz))
            if 0 <= mz_int < MAX_MZ: ms_vector[mz_int] = max(ms_vector[mz_int], float(intensity))
        if np.sum(ms_vector) > EPS:
            max_intensity = np.max(ms_vector)
            if max_intensity > EPS: ms_vector = (ms_vector / max_intensity) * 100.0
            else: ms_vector.fill(0.0)
            non_zero_intensities = ms_vector[ms_vector > EPS]
            if len(non_zero_intensities) > 0:
                threshold = np.percentile(non_zero_intensities, 1); ms_vector[ms_vector < threshold] = 0
            # Optional emphasis
            # for mz_idx in IMPORTANT_MZ:
            #     if 0 <= mz_idx < MAX_MZ and ms_vector[mz_idx] > EPS: ms_vector[mz_idx] *= 1.5
            msp_data[current_id] = ms_vector
        else: msp_data[current_id] = ms_vector

    logger.info(f"MSP解析完了。{entry_count} エントリ処理、{len(msp_data)} ID格納。")

    logger.info(f"MSPデータをキャッシュ保存中: {cache_file}")
    try:
        with open(cache_file, 'wb') as f: pickle.dump(msp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e: logger.error(f"キャッシュ保存失敗: {e}")
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
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Linear(in_channels, out_channels), nn.LayerNorm(out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.leaky_relu(self.ln1(self.conv1(x)))
        out = self.dropout(out)
        out = self.ln2(self.conv2(out))
        out += residual
        out = F.leaky_relu(out)
        return out

class AttentionBlock(nn.Module):
    """軽量化グローバルアテンションブロック"""
    def __init__(self, in_dim, hidden_dim, heads=2):
        super(AttentionBlock, self).__init__()
        self.heads = heads; self.head_dim = hidden_dim // heads
        self.query = nn.Linear(in_dim, hidden_dim); self.key = nn.Linear(in_dim, hidden_dim); self.value = nn.Linear(in_dim, hidden_dim)
        self.attn_combine = nn.Linear(hidden_dim, hidden_dim)
        self.gate_nn = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1))
        self.out_proj = nn.Linear(hidden_dim, in_dim); self.layer_norm = nn.LayerNorm(in_dim); self.dropout = nn.Dropout(0.1)

    def forward(self, x, batch):
        device = x.device; batch_size = torch.max(batch).item() + 1
        q = self.query(x).view(-1, self.heads, self.head_dim); k = self.key(x).view(-1, self.heads, self.head_dim); v = self.value(x).view(-1, self.heads, self.head_dim)
        global_attention = GlobalAttention(gate_nn=self.gate_nn, nn=nn.Identity())
        x_global = global_attention(x, batch)
        out = self.out_proj(x_global); out = self.layer_norm(out); out = self.dropout(out)
        return out

class TransformerEncoderLayer(nn.Module):
    """軽量化Transformerエンコーダーレイヤー"""
    def __init__(self, hidden_dim, nhead, dim_feedforward=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        if dim_feedforward is None: dim_feedforward = hidden_dim * 2
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward); self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src):
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, need_weights=False)
        src = src + self.dropout1(attn_output)
        src2 = self.norm2(src)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ffn_output)
        return src


###############################
# 軽量化ハイブリッドモデル
###############################

class OptimizedHybridMSModel(nn.Module):
    """軽量化GNN-Transformerハイブリッドモデル"""
    def __init__(self, node_features, edge_features, hidden_channels, out_channels, num_fragments=NUM_FRAGS,
                 prec_mass_offset=10, bidirectional=True, gate_prediction=True):
        super(OptimizedHybridMSModel, self).__init__()
        self.prec_mass_offset = prec_mass_offset; self.bidirectional = bidirectional
        self.gate_prediction = gate_prediction; self.global_features_dim = GLOBAL_FEATURES_DIM
        self.hidden_channels = hidden_channels; self.transformer_dim = 128

        self.gat1 = GATv2Conv(node_features, hidden_channels, edge_dim=edge_features, heads=2, concat=True)
        self.gat2 = GATv2Conv(hidden_channels*2, hidden_channels, edge_dim=edge_features, heads=2, concat=True)
        self.gat3 = GATv2Conv(hidden_channels*2, hidden_channels, edge_dim=edge_features, heads=2, concat=True)
        self.skip_connection1 = nn.Linear(hidden_channels*2, hidden_channels*2)
        self.global_proj = nn.Sequential(nn.Linear(self.global_features_dim, hidden_channels), nn.LeakyReLU(), nn.LayerNorm(hidden_channels))
        self.transformer_projection = nn.Linear(hidden_channels*3, self.transformer_dim)
        self.transformer_encoder = TransformerEncoderLayer(self.transformer_dim, nhead=4, dim_feedforward=self.transformer_dim*2, dropout=0.1)
        self.transformer_unprojection = nn.Linear(self.transformer_dim, hidden_channels*2)
        self.fc_layers = nn.ModuleList([ResidualBlock(hidden_channels*2, hidden_channels*2), ResidualBlock(hidden_channels*2, hidden_channels)])
        self.fragment_pred = nn.Sequential(nn.Linear(hidden_channels, hidden_channels//2), nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(hidden_channels//2, num_fragments))

        if bidirectional:
            self.forw_out_layer = nn.Linear(hidden_channels, out_channels); self.rev_out_layer = nn.Linear(hidden_channels, out_channels)
            self.out_gate = nn.Sequential(nn.Linear(hidden_channels, out_channels), nn.Sigmoid())
        else:
            self.out_layer = nn.Linear(hidden_channels, out_channels)
            if gate_prediction: self.out_gate = nn.Sequential(nn.Linear(hidden_channels, out_channels), nn.Sigmoid())

        self.ln1 = nn.LayerNorm(hidden_channels*2); self.ln2 = nn.LayerNorm(hidden_channels*2)
        self.ln_transformer_out = nn.LayerNorm(hidden_channels*2)
        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu');
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, GATv2Conv):
                 gain = nn.init.calculate_gain('leaky_relu', 0.01)
                 if hasattr(m, 'lin_l') and isinstance(m.lin_l, nn.Linear): nn.init.xavier_uniform_(m.lin_l.weight, gain=gain); nn.init.zeros_(m.lin_l.bias)
                 if hasattr(m, 'lin_r') and isinstance(m.lin_r, nn.Linear): nn.init.xavier_uniform_(m.lin_r.weight, gain=gain); nn.init.zeros_(m.lin_r.bias)
                 if hasattr(m, 'att_src') and isinstance(m.att_src, nn.Parameter): nn.init.xavier_uniform_(m.att_src.data, gain=gain)
                 if hasattr(m, 'att_dst') and isinstance(m.att_dst, nn.Parameter): nn.init.xavier_uniform_(m.att_dst.data, gain=gain)
                 if hasattr(m, 'att_edge') and isinstance(m.att_edge, nn.Parameter): nn.init.xavier_uniform_(m.att_edge.data, gain=gain)

    def forward(self, data):
        device = next(self.parameters()).device
        if isinstance(data, dict):
            graph = data.get('graph'); prec_mz_bin = data.get('prec_mz_bin')
            if graph is None: raise ValueError("Input dict missing 'graph'")
            if prec_mz_bin is not None: prec_mz_bin = prec_mz_bin.to(device)
        elif isinstance(data, (Data, Batch)):
            graph = data
            if hasattr(data, 'prec_mz_bin'): prec_mz_bin = data.prec_mz_bin.to(device)
            elif hasattr(data, 'mass'): prec_mz_bin = data.mass.to(device)
            else: prec_mz_bin = None
        else: raise TypeError(f"Unsupported input type: {type(data)}")

        x = graph.x.to(device).float(); edge_index = graph.edge_index.to(device)
        edge_attr = graph.edge_attr.to(device).float() if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
        batch = graph.batch.to(device) if hasattr(graph, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
        global_attr = graph.global_attr.to(device).float() if hasattr(graph, 'global_attr') and graph.global_attr is not None else None
        num_graphs = graph.num_graphs if hasattr(graph, 'num_graphs') else torch.max(batch).item() + 1

        if global_attr is not None:
            if global_attr.shape[0] != num_graphs:
                try: global_attr = global_attr.view(num_graphs, -1)
                except RuntimeError: global_attr = torch.zeros(num_graphs, self.global_features_dim, device=device, dtype=torch.float)
            if global_attr.shape[1] != self.global_features_dim:
                padded = torch.zeros(num_graphs, self.global_features_dim, device=device, dtype=torch.float)
                copy_dim = min(global_attr.shape[1], self.global_features_dim); padded[:, :copy_dim] = global_attr[:, :copy_dim]; global_attr = padded
        else: global_attr = torch.zeros(num_graphs, self.global_features_dim, device=device, dtype=torch.float)

        x1 = self.gat1(x, edge_index, edge_attr); x1 = F.leaky_relu(x1); x1 = self.dropout(x1)
        x2 = self.gat2(x1, edge_index, edge_attr); x2 = F.leaky_relu(x2); x2 = self.dropout(x2)
        x1_transformed = self.skip_connection1(x1); x2 = x2 + x1_transformed; x2 = self.ln1(x2)
        x3 = self.gat3(x2, edge_index, edge_attr); x3 = F.leaky_relu(x3); x3 = self.ln2(x3)

        x_graph = global_mean_pool(x3, batch)
        global_features_proj = self.global_proj(global_attr)
        x_combined = torch.cat([x_graph, global_features_proj], dim=1)

        x_for_transformer = self.transformer_projection(x_combined).unsqueeze(1)
        x_transformed = self.transformer_encoder(x_for_transformer).squeeze(1)
        x_unprojected = self.transformer_unprojection(x_transformed); x_unprojected = self.ln_transformer_out(x_unprojected)

        x_out = x_unprojected
        for i, fc_layer in enumerate(self.fc_layers): x_out = fc_layer(x_out)

        fragment_pred = self.fragment_pred(x_out)

        if self.bidirectional and prec_mz_bin is not None:
            ff = self.forw_out_layer(x_out); rev_input = x_out
            fr_raw = self.rev_out_layer(rev_input); fr = reverse_prediction(fr_raw, prec_mz_bin, self.prec_mass_offset)
            fg = self.out_gate(x_out); output = ff * fg + fr * (1. - fg)
            output = mask_prediction_by_mass(output, prec_mz_bin, self.prec_mass_offset)
        else:
            if hasattr(self, 'out_layer'):
                output = self.out_layer(x_out)
                if self.gate_prediction and hasattr(self, 'out_gate'): fg = self.out_gate(x_out); output = fg * output
            elif hasattr(self, 'forw_out_layer'):
                if self.bidirectional and prec_mz_bin is None: logger.debug("双方向モデル(prec_mz_binなし): 順方向のみ使用")
                output = self.forw_out_layer(x_out)
                if prec_mz_bin is not None: output = mask_prediction_by_mass(output, prec_mz_bin, self.prec_mass_offset)
            else: raise AttributeError("出力層設定無効")

        output = F.relu(output)
        return output, fragment_pred


###############################
# 最適化データセット
###############################

def process_mol_id(mol_id, mol_files_path, msp_data):
    from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
    mol_file = os.path.join(mol_files_path, f"ID{mol_id}.MOL")
    try:
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        if mol is None: return None, None
        try:
            for atom in mol.GetAtoms(): atom.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
        except Exception: pass
        try:
            maccs = MACCSkeys.GenMACCSKeys(mol); fragments = np.zeros(NUM_FRAGS, dtype=np.float32)
            for i in range(1, NUM_FRAGS + 1):
                if maccs.GetBit(i): fragments[i-1] = 1.0
        except Exception: fragments = np.zeros(NUM_FRAGS, dtype=np.float32)
        if mol_id not in msp_data: return None, None
        return mol_id, fragments
    except Exception: return None, None


class OptimizedMoleculeGraphDataset(Dataset):
    def __init__(self, mol_ids, mol_files_path, msp_data, transform="log10over3",
                normalization="l1", augment=False, cache_dir=CACHE_DIR):
        self.mol_ids = mol_ids; self.mol_files_path = mol_files_path; self.msp_data = msp_data
        self.augment = augment; self.transform = transform; self.normalization = normalization
        self.valid_mol_ids = []; self.fragment_patterns = {}; self.cache_dir = cache_dir; self.graph_cache = {}
        self._preprocess_mol_ids()

    def _preprocess_mol_ids(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = hash(str(sorted(self.mol_ids)) + "_v3_single_proc_no_smooth")
        cache_file = os.path.join(self.cache_dir, f"preprocessed_data_{cache_key}.pkl")
        if os.path.exists(cache_file):
            logger.info(f"キャッシュから前処理データ読み込み中: {cache_file}")
            try:
                with open(cache_file, 'rb') as f: cached_data = pickle.load(f)
                self.valid_mol_ids = cached_data['valid_mol_ids']; self.fragment_patterns = cached_data['fragment_patterns']
                logger.info(f"キャッシュ読み込み完了: {len(self.valid_mol_ids)} 件"); return
            except Exception as e: logger.warning(f"キャッシュ読み込み失敗 ({e})。再処理します。"); try: os.remove(cache_file) ; except OSError: pass

        logger.info("分子データ前処理開始（シングルプロセス）..."); valid_ids = []; fragment_patterns = {}
        with tqdm(total=len(self.mol_ids), desc="分子検証", unit="mol") as pbar:
            process_func = partial(process_mol_id, mol_files_path=self.mol_files_path, msp_data=self.msp_data)
            for mol_id in self.mol_ids:
                try:
                    mol_id_result, fragments = process_func(mol_id)
                    if mol_id_result is not None: valid_ids.append(mol_id_result); fragment_patterns[mol_id_result] = fragments
                    pbar.update(1);
                    if pbar.n % 500 == 0: gc.collect()
                except Exception as e: logger.error(f"分子ID {mol_id} 検証ループ中エラー: {e}", exc_info=False); pbar.update(1)
        self.valid_mol_ids = valid_ids; self.fragment_patterns = fragment_patterns
        logger.info(f"前処理結果キャッシュ保存中: {cache_file} ({len(valid_ids)} 件)")
        try:
            with open(cache_file, 'wb') as f: pickle.dump({'valid_mol_ids': valid_ids, 'fragment_patterns': fragment_patterns}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e: logger.error(f"キャッシュ保存失敗: {e}")
        logger.info(f"有効な分子: {len(valid_ids)}個 / 全体: {len(self.mol_ids)}個")

    def _mol_to_graph(self, mol_file):
        if mol_file in self.graph_cache: return self.graph_cache[mol_file]
        mol_file_basename = os.path.basename(mol_file); graph_cache_key = f"graph_cache_v3_{mol_file_basename}.pkl"
        cache_file = os.path.join(self.cache_dir, graph_cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f: graph_data = pickle.load(f); self.graph_cache[mol_file] = graph_data; return graph_data
            except Exception as e: logger.warning(f"グラフキャッシュ {cache_file} 読み込み失敗 ({e})。再生成します。"); try: os.remove(cache_file); except OSError: pass

        from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*'); mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        if mol is None: logger.error(f"Could not read molecule from {mol_file}"); return None
        try:
            for atom in mol.GetAtoms(): atom.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
        except Exception as e: logger.debug(f"Sanitization failed for {mol_file}: {e}. Proceeding.")

        num_atoms = mol.GetNumAtoms(); x = []; rings = []
        if num_atoms == 0: logger.warning(f"Molecule {mol_file} has 0 atoms."); return None
        try: ring_info = mol.GetRingInfo(); rings = ring_info.AtomRings() if ring_info.NumRings() > 0 else []
        except Exception: pass

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx(); atom_feature_vec = [0.0] * NODE_FEATURES_DIM
            atom_symbol = atom.GetSymbol(); atom_type_idx = ATOM_FEATURES.get(atom_symbol, ATOM_FEATURES['OTHER']); atom_feature_vec[atom_type_idx] = 1.0
            offset = len(ATOM_FEATURES)
            try: atom_feature_vec[offset + 0] = atom.GetDegree() / 8.0; except Exception: pass
            try: atom_feature_vec[offset + 1] = atom.GetFormalCharge() / 8.0; except Exception: pass
            try: atom_feature_vec[offset + 2] = atom.GetNumRadicalElectrons() / 4.0; except Exception: pass
            try: atom_feature_vec[offset + 3] = float(atom.GetIsAromatic()); except Exception: pass
            try: atom_feature_vec[offset + 4] = atom.GetMass() / 200.0; except Exception: pass
            try: atom_feature_vec[offset + 5] = float(atom.IsInRing()); except Exception: pass
            try: atom_feature_vec[offset + 6] = int(atom.GetHybridization()) / 8.0; except Exception: pass
            try: atom_feature_vec[offset + 7] = atom.GetExplicitValence() / 8.0; except Exception: pass
            try: atom_feature_vec[offset + 8] = atom.GetImplicitValence() / 8.0; except Exception: pass
            try: atom_feature_vec[offset + 9] = float(atom.GetIsAromatic() and atom.IsInRing()); except Exception: pass
            try:
                max_ring_size = 0
                if atom.IsInRing():
                     for ring in rings:
                          if atom_idx in ring: max_ring_size = max(max_ring_size, len(ring))
                atom_feature_vec[offset + 10] = max_ring_size / 8.0
            except Exception: pass
            try: atom_feature_vec[offset + 11] = atom.GetTotalNumHs() / 8.0; except Exception: pass
            x.append(atom_feature_vec)

        edge_indices = []; edge_attrs = []
        num_bonds = mol.GetNumBonds()
        if num_bonds > 0:
            for bond in mol.GetBonds():
                try:
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bond_feature_vec = [0.0] * EDGE_FEATURES_DIM
                    bond_type = bond.GetBondType(); bond_type_idx = BOND_FEATURES.get(bond_type, BOND_FEATURES[Chem.rdchem.BondType.SINGLE]); bond_feature_vec[bond_type_idx] = 1.0
                    offset = len(BOND_FEATURES)
                    try: bond_feature_vec[offset + 0] = float(bond.IsInRing()); except Exception: pass
                    try: bond_feature_vec[offset + 1] = float(bond.GetIsConjugated()); except Exception: pass
                    try: bond_feature_vec[offset + 2] = float(bond.GetIsAromatic()); except Exception: pass
                    edge_indices.append([i, j]); edge_attrs.append(bond_feature_vec); edge_indices.append([j, i]); edge_attrs.append(bond_feature_vec)
                except Exception as e: logger.debug(f"Error processing bond {bond.GetIdx()} in {mol_file}: {e}")
        elif num_atoms > 0: # Handle single atoms with self-loops
             logger.debug(f"No bonds for {mol_file}. Adding self-loops.")
             for i in range(num_atoms):
                 edge_indices.append([i, i]); bond_feature_vec = [0.0] * EDGE_FEATURES_DIM
                 bond_feature_vec[BOND_FEATURES[Chem.rdchem.BondType.SINGLE]] = 1.0; edge_attrs.append(bond_feature_vec)

        mol_features = [0.0] * GLOBAL_FEATURES_DIM
        try: mol_features[0] = Descriptors.MolWt(mol) / 1000.0; except Exception: pass
        try: mol_features[1] = Descriptors.NumHAcceptors(mol) / 20.0; except Exception: pass
        try: mol_features[2] = Descriptors.NumHDonors(mol) / 10.0; except Exception: pass
        try: mol_features[3] = Descriptors.TPSA(mol) / 200.0; except Exception: pass
        try: mol_features[4] = Descriptors.MolLogP(mol) / 10.0; except Exception: pass
        try: mol_features[5] = Descriptors.NumRotatableBonds(mol) / 20.0; except Exception: pass
        try: mol_features[6] = Descriptors.NumAliphaticRings(mol) / 5.0; except Exception: pass
        try: mol_features[7] = Descriptors.NumAromaticRings(mol) / 5.0; except Exception: pass

        x = torch.tensor(x, dtype=torch.float)
        if edge_indices:
             edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
             edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else: edge_index = torch.empty((2, 0), dtype=torch.long); edge_attr = torch.empty((0, EDGE_FEATURES_DIM), dtype=torch.float)
        global_attr = torch.tensor(mol_features, dtype=torch.float)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_attr=global_attr)
        self.graph_cache[mol_file] = graph_data
        try:
            with open(cache_file, 'wb') as f: pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e: logger.warning(f"グラフキャッシュ {cache_file} 保存失敗: {e}")
        return graph_data

    def _preprocess_spectrum(self, spectrum):
        spec_tensor = torch.FloatTensor(spectrum)
        if spec_tensor.dim() == 1: spec_tensor = spec_tensor.unsqueeze(0)
        processed_spec = process_spec(spec_tensor, self.transform, self.normalization)
        return processed_spec.squeeze(0).numpy()

    def __len__(self): return len(self.valid_mol_ids)

    def __getitem__(self, idx):
        mol_id = self.valid_mol_ids[idx]; mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
        try:
            graph_data = self._mol_to_graph(mol_file)
            if graph_data is None: raise ValueError(f"Graph creation failed for mol_id {mol_id}")
            raw_mass_spectrum = self.msp_data.get(mol_id, np.zeros(MAX_MZ, dtype=np.float32))
            processed_mass_spectrum = self._preprocess_spectrum(raw_mass_spectrum)
            fragment_pattern = self.fragment_patterns.get(mol_id, np.zeros(NUM_FRAGS, dtype=np.float32))
            non_zero_indices = np.nonzero(raw_mass_spectrum)[0]
            prec_mz = int(np.max(non_zero_indices)) if len(non_zero_indices) > 0 else 0; prec_mz_bin = prec_mz
            if self.augment and np.random.random() < 0.2:
                noise = 0.01; graph_data.x += torch.randn_like(graph_data.x) * noise
                if graph_data.edge_attr is not None and graph_data.edge_attr.numel() > 0: graph_data.edge_attr += torch.randn_like(graph_data.edge_attr) * noise
            return {'graph_data': graph_data, 'mass_spectrum': torch.FloatTensor(processed_mass_spectrum), 'fragment_pattern': torch.FloatTensor(fragment_pattern),
                    'mol_id': mol_id, 'prec_mz': float(prec_mz), 'prec_mz_bin': int(prec_mz_bin)}
        except Exception as e:
            logger.error(f"データ取得エラー: 分子ID {mol_id} ({mol_file}): {e}", exc_info=False)
            dummy_x = torch.zeros((1, NODE_FEATURES_DIM), dtype=torch.float); dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            dummy_edge_attr = torch.zeros((1, EDGE_FEATURES_DIM), dtype=torch.float); dummy_global_attr = torch.zeros(GLOBAL_FEATURES_DIM, dtype=torch.float) # Use constant
            dummy_graph = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, global_attr=dummy_global_attr)
            dummy_spectrum = torch.zeros(MAX_MZ, dtype=torch.float); dummy_fragments = torch.zeros(NUM_FRAGS, dtype=torch.float)
            return {'graph_data': dummy_graph, 'mass_spectrum': dummy_spectrum, 'fragment_pattern': dummy_fragments,
                    'mol_id': mol_id, 'prec_mz': 0.0, 'prec_mz_bin': 0}


def optimized_collate_fn(batch):
    valid_batch = [item for item in batch if item is not None and isinstance(item.get('graph_data'), Data)]
    if not valid_batch: logger.warning("Collate fn received empty/invalid batch."); return None
    graph_data_list = [item['graph_data'] for item in valid_batch]
    mass_spectrum = torch.stack([item['mass_spectrum'] for item in valid_batch])
    fragment_pattern = torch.stack([item['fragment_pattern'] for item in valid_batch])
    mol_ids = [item['mol_id'] for item in valid_batch]
    prec_mz = torch.tensor([item['prec_mz'] for item in valid_batch], dtype=torch.float32)
    prec_mz_bin = torch.tensor([item['prec_mz_bin'] for item in valid_batch], dtype=torch.long)
    try: batched_graphs = Batch.from_data_list(graph_data_list)
    except Exception as e: logger.error(f"バッチ作成失敗: {e}", exc_info=True); return None
    return {'graph': batched_graphs, 'spec': mass_spectrum, 'fragment_pattern': fragment_pattern, 'mol_id': mol_ids, 'prec_mz': prec_mz, 'prec_mz_bin': prec_mz_bin}


###############################
# 損失関数と類似度計算（最適化）
###############################

def cosine_similarity_loss(y_pred, y_true, important_mz=None, important_weight=3.0, eps=EPS):
    y_pred = y_pred.float(); y_true = y_true.float()
    y_pred_norm = F.normalize(y_pred, p=2, dim=1, eps=eps); y_true_norm = F.normalize(y_true, p=2, dim=1, eps=eps)
    if important_mz is not None and len(important_mz) > 0:
        weights = torch.ones_like(y_pred_norm); mz_indices = [mz for mz in important_mz if 0 <= mz < y_pred_norm.size(1)]
        if mz_indices: weights[:, mz_indices] *= important_weight
        y_pred_weighted = y_pred_norm * weights; y_true_weighted = y_true_norm * weights
        y_pred_final = F.normalize(y_pred_weighted, p=2, dim=1, eps=eps); y_true_final = F.normalize(y_true_weighted, p=2, dim=1, eps=eps)
    else: y_pred_final, y_true_final = y_pred_norm, y_true_norm
    cosine = torch.sum(y_pred_final * y_true_final, dim=1); cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    loss = 1.0 - cosine; return loss.mean()

def combined_loss(y_pred, y_true, fragment_pred=None, fragment_true=None, alpha=0.2, beta=0.6, epsilon=0.2, use_important_mz_mse=True):
    y_pred = y_pred.float(); y_true = y_true.float()
    if fragment_pred is not None: fragment_pred = fragment_pred.float()
    if fragment_true is not None: fragment_true = fragment_true.float()

    min_b = min(y_pred.shape[0], y_true.shape[0])
    if y_pred.shape[0]!=y_true.shape[0]: logger.warning(f"Loss Batch size mismatch: Using {min_b}."); y_pred=y_pred[:min_b]; y_true=y_true[:min_b]
    min_mz = min(y_pred.shape[1], y_true.shape[1])
    if y_pred.shape[1]!=y_true.shape[1]: logger.warning(f"Loss Feature size mismatch: Using {min_mz}."); y_pred=y_pred[:,:min_mz]; y_true=y_true[:,:min_mz]

    peak_mask = (y_true > EPS).float(); mse_weights = peak_mask * 10.0 + 1.0
    if use_important_mz_mse and IMPORTANT_MZ: mz_indices = [mz for mz in IMPORTANT_MZ if 0 <= mz < y_true.size(1)]; mse_weights[:, mz_indices] *= 3.0
    mse_loss = torch.mean(mse_weights * (y_pred - y_true) ** 2)
    cosine_loss = cosine_similarity_loss(y_pred, y_true, important_mz=IMPORTANT_MZ, important_weight=3.0)
    main_loss = alpha * mse_loss + beta * cosine_loss

    fragment_loss_val = 0.0
    if fragment_pred is not None and fragment_true is not None:
        frag_b = min(fragment_pred.shape[0], fragment_true.shape[0])
        if fragment_pred.shape[0]!=frag_b: fragment_pred=fragment_pred[:frag_b]
        if fragment_true.shape[0]!=frag_b: fragment_true=fragment_true[:frag_b]
        frag_d = min(fragment_pred.shape[1], fragment_true.shape[1])
        if fragment_pred.shape[1]!=frag_d: fragment_pred=fragment_pred[:,:frag_d]
        if fragment_true.shape[1]!=frag_d: fragment_true=fragment_true[:,:frag_d]
        if fragment_pred.shape == fragment_true.shape and fragment_pred.numel() > 0:
             fragment_loss = F.binary_cross_entropy_with_logits(fragment_pred, fragment_true); fragment_loss_val = fragment_loss.item(); total_loss = main_loss + epsilon * fragment_loss
        else: logger.error(f"Fragment shapes incompatible: Skipping fragment loss."); total_loss = main_loss
    else: total_loss = main_loss

    if torch.isnan(total_loss): logger.error(f"NaN loss! MSE:{mse_loss.item():.4f},Cos:{cosine_loss.item():.4f},Frag:{fragment_loss_val:.4f}"); return torch.tensor(1e6, device=total_loss.device)
    return total_loss


def cosine_similarity_score(y_true, y_pred, eps=EPS):
    min_b = min(y_true.shape[0], y_pred.shape[0])
    if y_true.shape[0] != y_pred.shape[0]: logger.warning(f"Cosine score batch mismatch: Using {min_b}.")
    y_true = y_true[:min_b]; y_pred = y_pred[:min_b]
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)
    y_true_np = y_true_np.astype(np.float64); y_pred_np = y_pred_np.astype(np.float64)
    if y_true_np.ndim > 2: y_true_np = y_true_np.reshape(y_true_np.shape[0], -1)
    if y_pred_np.ndim > 2: y_pred_np = y_pred_np.reshape(y_pred_np.shape[0], -1)
    if y_true_np.shape[0] == 0: return 0.0

    dot_products = np.sum(y_true_np * y_pred_np, axis=1)
    true_norms = np.linalg.norm(y_true_np, axis=1); pred_norms = np.linalg.norm(y_pred_np, axis=1)
    zero_norm_mask = (true_norms < eps) | (pred_norms < eps); similarities = np.zeros_like(dot_products)
    valid_mask = ~zero_norm_mask
    if np.any(valid_mask): similarities[valid_mask] = dot_products[valid_mask] / (true_norms[valid_mask] * pred_norms[valid_mask] + eps)
    similarities = np.clip(similarities, -1.0, 1.0)
    if np.isnan(similarities).any(): num_nans = np.isnan(similarities).sum(); logger.warning(f"Cosine sim resulted in {num_nans} NaNs. Replacing with 0.0."); similarities = np.nan_to_num(similarities, nan=0.0)
    return np.mean(similarities)


###############################
# トレーニングとモデル評価（最適化）
###############################

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
               eval_interval=1, patience=10, grad_clip=1.0, checkpoint_dir=CACHE_DIR, transform_param="log10over3"):
    """最適化されたモデルのトレーニング（チェックポイント機能付き）"""
    train_losses, val_losses, val_cosine_similarities = [], [], []
    best_cosine = -1.0; early_stopping_counter = 0; start_epoch = 0
    os.makedirs(checkpoint_dir, exist_ok=True); best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    latest_checkpoint_path = None; latest_epoch = -1
    try:
        ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")], key=lambda x: int(x.split("_epoch_")[1].split(".")[0]))
        if ckpt_files: latest_checkpoint_file = ckpt_files[-1]; latest_epoch = int(latest_checkpoint_file.split("_epoch_")[1].split(".")[0]); latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
    except FileNotFoundError: logger.info("チェックポイントディレクトリなし。最初から開始。")
    except (IndexError, ValueError): logger.warning("チェックポイントファイル名解析エラー。最初から開始。")

    use_amp = (device.type == 'cuda'); scaler = GradScaler(enabled=use_amp) # Define scaler early

    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        logger.info(f"チェックポイント読み込み: {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            try: model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e: logger.warning(f"モデル状態の部分読み込み/不一致: {e}. strict=False試行。"); try: model.load_state_dict(checkpoint['model_state_dict'], strict=False); except Exception as inner_e: logger.error(f"モデル状態読み込み失敗(strict=False): {inner_e}. 最初から開始。"); start_epoch = 0; latest_epoch = -1
            if start_epoch == 0 and latest_epoch != -1:
                 try: optimizer.load_state_dict(checkpoint['optimizer_state_dict']); [state.update({k: v.to(device) for k, v in state.items() if isinstance(v, torch.Tensor)}) for state in optimizer.state.values()]
                 except Exception as e_opt: logger.warning(f"オプティマイザ状態読み込み失敗: {e_opt}.")
            train_losses = checkpoint.get('train_losses', []); val_losses = checkpoint.get('val_losses', []); val_cosine_similarities = checkpoint.get('val_cosine_similarities', [])
            best_cosine = checkpoint.get('best_cosine', -1.0); early_stopping_counter = checkpoint.get('early_stopping_counter', 0); start_epoch = checkpoint.get('epoch', -1) + 1
            if scheduler is not None and 'scheduler_state_dict' in checkpoint: try: scheduler.load_state_dict(checkpoint['scheduler_state_dict']); except Exception as e_sched: logger.warning(f"スケジューラ状態読み込み失敗: {e_sched}.")
            if 'scaler_state_dict' in checkpoint: try: scaler.load_state_dict(checkpoint['scaler_state_dict']); except Exception as e_scaler: logger.warning(f"Scaler状態読み込み失敗: {e_scaler}.") # Load scaler state
            logger.info(f"チェックポイント読み込み完了。エポック {start_epoch} から再開。Best Cosine: {best_cosine:.4f}"); del checkpoint; aggressive_memory_cleanup(force_sync=False)
        except Exception as e: logger.error(f"チェックポイント読み込みエラー: {e}. 最初から開始。", exc_info=True); train_losses, val_losses, val_cosine_similarities = [], [], []; best_cosine = -1.0; early_stopping_counter = 0; start_epoch = 0
    else: logger.info("有効なチェックポイントなし。最初から開始。")

    model.to(device)
    total_steps = len(train_loader) * (num_epochs - start_epoch)
    logger.info(f"トレーニング開始: エポック {start_epoch+1}-{num_epochs} ({num_epochs - start_epoch} エポック). 総ステップ ≈ {total_steps}")
    total_batches = len(train_loader); memory_check_interval = max(1, min(total_batches // 5, 100)); logger.info(f"メモリチェック間隔: {memory_check_interval} バッチごと")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        if epoch % 4 == 0: logger.info(f"エポック {epoch+1} 開始 - 強力メモリクリーンアップ"); aggressive_memory_cleanup(force_sync=True, purge_cache=True); else: aggressive_memory_cleanup(force_sync=False)
        model.train(); epoch_loss = 0; batch_count = 0; processed_samples = 0
        train_pbar = tqdm(train_loader, desc=f"エポック {epoch+1}/{num_epochs} [訓練]", position=0, leave=True, unit="batch")
        for batch_idx, batch in enumerate(train_pbar):
            if batch is None: logger.warning(f"スキップ: バッチ {batch_idx+1} None"); continue
            if batch_idx > 0 and batch_idx % memory_check_interval == 0: aggressive_memory_cleanup(percent=85, force_sync=False)
            try: # Data to device
                processed_batch = {}; pin_memory = train_loader.pin_memory
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): processed_batch[k] = v.to(device, non_blocking=pin_memory)
                    elif k == 'graph' and isinstance(v, Batch): try: processed_batch[k] = v.to(device, non_blocking=pin_memory); except AttributeError: v.x=v.x.to(device,non_blocking=pin_memory); v.edge_index=v.edge_index.to(device,non_blocking=pin_memory); v.batch=v.batch.to(device,non_blocking=pin_memory); processed_batch[k]=v # Simplify fallback
                    else: processed_batch[k] = v
            except Exception as e_data_move: logger.error(f"バッチ {batch_idx+1} デバイス転送エラー: {e_data_move}. スキップ。", exc_info=False); continue
            try: # Forward & Loss
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): output, fragment_pred = model(processed_batch); loss = criterion(output, processed_batch['spec'], fragment_pred, processed_batch['fragment_pattern'])
                if not torch.isfinite(loss): logger.error(f"エポック {epoch+1}, バッチ {batch_idx+1}: 無効損失 ({loss.item()}). スキップ。"); continue
            except Exception as e_forward: logger.error(f"エポック {epoch+1}, バッチ {batch_idx+1}: 順伝播/損失エラー: {e_forward}.", exc_info=True); aggressive_memory_cleanup(); continue
            try: # Backward & Optimize
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip); scaler.step(optimizer); scaler.update()
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): scheduler.step()
            except Exception as e_backward: logger.error(f"エポック {epoch+1}, バッチ {batch_idx+1}: 逆伝播/最適化エラー: {e_backward}.", exc_info=True); aggressive_memory_cleanup(); continue
            current_loss = loss.item(); epoch_loss += current_loss; batch_count += 1; processed_samples += len(processed_batch['mol_id'])
            train_pbar.set_postfix({'損失': f"{current_loss:.4f}", '平均損失': f"{epoch_loss/batch_count:.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.6f}"})
            del loss, output, fragment_pred, processed_batch, batch
        train_pbar.close()

        if batch_count > 0:
            avg_train_loss = epoch_loss / batch_count; train_losses.append(avg_train_loss); epoch_duration = time.time() - epoch_start_time
            logger.info(f"エポック {epoch+1}/{num_epochs} 完了 ({epoch_duration:.2f}秒) - 平均訓練損失: {avg_train_loss:.4f}, 処理サンプル: {processed_samples}")
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            try: torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None, 'scaler_state_dict': scaler.state_dict(), 'train_losses': train_losses, 'val_losses': val_losses, 'val_cosine_similarities': val_cosine_similarities, 'best_cosine': best_cosine, 'early_stopping_counter': early_stopping_counter, 'transform_param': transform_param}, epoch_checkpoint_path); logger.info(f"エポックチェックポイント保存: {epoch_checkpoint_path}")
            except Exception as e_ckpt_save: logger.error(f"エポックチェックポイント保存失敗: {e_ckpt_save}")
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                logger.info(f"エポック {epoch+1}: 検証開始"); aggressive_memory_cleanup(force_sync=True)
                val_metrics = evaluate_model(model, val_loader, criterion, device, use_amp=use_amp, transform_param=transform_param)
                val_loss = val_metrics['loss']; cosine_sim = val_metrics['cosine_similarity']
                if val_loss == float('inf') or cosine_sim <= -1.0: logger.warning(f"エポック {epoch+1}: 検証無効メトリクス。損失: {val_loss}, 類似度: {cosine_sim}")
                else:
                    val_losses.append(val_loss); val_cosine_similarities.append(cosine_sim)
                    logger.info(f"エポック {epoch+1}/{num_epochs} [検証] - 損失: {val_loss:.4f}, コサイン類似度: {cosine_sim:.4f}")
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_loss)
                    if cosine_sim > best_cosine:
                        logger.info(f"*** 新最良モデル！類似度: {best_cosine:.4f} -> {cosine_sim:.4f} ***"); best_cosine = cosine_sim; early_stopping_counter = 0
                        try: torch.save(model.state_dict(), best_model_path); logger.info(f"最良モデル保存: {best_model_path}")
                        except Exception as e_best_save: logger.error(f"最良モデル保存失敗: {e_best_save}")
                    else:
                        early_stopping_counter += 1; logger.info(f"性能改善なし。早期停止カウンター: {early_stopping_counter}/{patience}")
                        if early_stopping_counter >= patience: logger.info(f"早期停止トリガー: エポック {epoch+1} で停止。"); try: _plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine, checkpoint_dir); except Exception as e_plot: logger.error(f"早期停止時プロットエラー: {e_plot}"); return train_losses, val_losses, val_cosine_similarities, best_cosine
            if (epoch + 1) % 5 == 0: try: _plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine, checkpoint_dir); logger.info(f"学習曲線プロット保存"); except Exception as e_plot: logger.error(f"エポック {epoch+1} プロットエラー: {e_plot}")
        else: logger.warning(f"エポック {epoch+1}: 有効バッチ処理なし。スキップ。")
    logger.info("トレーニングループ完了。"); try: _plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine, checkpoint_dir); logger.info(f"最終学習曲線保存。"); except Exception as e: logger.error(f"最終プロットエラー: {e}")
    return train_losses, val_losses, val_cosine_similarities, best_cosine

def _plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine, save_dir):
    if not train_losses: logger.warning("訓練損失データなし、プロットスキップ。"); return
    plt.figure(figsize=(12, 5)); epochs_train = range(1, len(train_losses) + 1)
    plt.subplot(1, 2, 1); plt.plot(epochs_train, train_losses, label='Training Loss', marker='o', linestyle='-', markersize=4)
    if val_losses: num_val = len(val_losses); val_epochs = np.linspace(epochs_train[0], epochs_train[-1], num_val) if num_val > 0 else []; plt.plot(val_epochs, val_losses, label='Validation Loss', marker='x', linestyle='--', markersize=5)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.legend(); plt.grid(True, alpha=0.6)
    min_loss = min(min(train_losses,default=0), min(val_losses,default=0)); max_loss = max(max(train_losses,default=1), max(val_losses,default=1)); plt.ylim(max(0, min_loss - 0.1 * abs(min_loss)), max_loss + 0.1 * abs(max_loss))
    plt.subplot(1, 2, 2)
    if val_cosine_similarities: num_val = len(val_cosine_similarities); val_epochs = np.linspace(epochs_train[0], epochs_train[-1], num_val) if num_val > 0 else []; plt.plot(val_epochs, val_cosine_similarities, label='Validation Cosine Similarity', marker='x', linestyle='--', markersize=5, color='green');
    if best_cosine > -1.0: plt.axhline(y=best_cosine, color='red', linestyle=':', linewidth=2, label=f'Best: {best_cosine:.4f}')
    plt.xlabel('Epoch'); plt.ylabel('Cosine Similarity'); plt.title('Validation Cosine Similarity'); plt.legend(); plt.grid(True, alpha=0.6)
    min_sim = min(val_cosine_similarities, default=0); max_sim = max(val_cosine_similarities, default=1); plt.ylim(max(-0.1, min_sim - 0.1), min(1.1, max_sim + 0.1))
    plt.tight_layout(); save_path = os.path.join(save_dir, 'hybrid_learning_curves.png')
    try: plt.savefig(save_path, dpi=150); except Exception as e: logger.error(f"プロット保存失敗: {save_path}, エラー: {e}"); plt.close()

def evaluate_model(model, data_loader, criterion, device, use_amp=False, transform_param="log10over3"):
    model.eval(); total_loss = 0.0; batch_count = 0; y_true_list, y_pred_list = [], []
    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc="評価中", leave=False, unit="batch")
        for batch_idx, batch in enumerate(eval_pbar):
            if batch is None: logger.warning(f"評価スキップ: バッチ {batch_idx+1} None"); continue
            try:
                processed_batch = {}; pin_memory = data_loader.pin_memory
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): processed_batch[k] = v.to(device, non_blocking=pin_memory)
                    elif k == 'graph' and isinstance(v, Batch): try: processed_batch[k] = v.to(device, non_blocking=pin_memory); except AttributeError: v.x=v.x.to(device,non_blocking=pin_memory);v.edge_index=v.edge_index.to(device,non_blocking=pin_memory);v.batch=v.batch.to(device,non_blocking=pin_memory);processed_batch[k]=v
                    else: processed_batch[k] = v
            except Exception as e_data_move: logger.error(f"評価バッチ {batch_idx+1} デバイス転送エラー: {e_data_move}. スキップ。", exc_info=False); continue
            try:
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    output, fragment_pred = model(processed_batch); loss = criterion(output, processed_batch['spec'], fragment_pred, processed_batch['fragment_pattern']) if criterion else torch.tensor(0.0)
                loss_item = loss.item() if criterion and torch.isfinite(loss) else 0.0
                if criterion and not torch.isfinite(loss): logger.warning(f"評価バッチ {batch_idx+1}: 無効損失 ({loss.item()}).")
                total_loss += loss_item; batch_count += 1; y_true_list.append(processed_batch['spec'].cpu()); y_pred_list.append(output.cpu())
            except Exception as e_eval_forward: logger.error(f"評価バッチ {batch_idx+1}: 順伝播/損失エラー: {e_eval_forward}.", exc_info=True); continue
            finally: del processed_batch, batch, output, fragment_pred, loss
    if batch_count > 0:
        avg_loss = total_loss / batch_count if criterion else 0.0; cosine_sim = 0.0
        if y_true_list and y_pred_list:
            try:
                all_true = torch.cat(y_true_list, dim=0); all_pred = torch.cat(y_pred_list, dim=0)
                all_true_post = postprocess_prediction(all_true, transform_param); all_pred_post = postprocess_prediction(all_pred, transform_param)
                cosine_sim = cosine_similarity_score(all_true_post, all_pred_post)
            except Exception as e_sim_calc: logger.error(f"評価中類似度計算エラー: {e_sim_calc}", exc_info=True); cosine_sim = 0.0
        else: logger.warning("評価中有効予測/真値収集できず。")
        return {'loss': avg_loss, 'cosine_similarity': cosine_sim}
    else: logger.error("評価中有効バッチなし。"); return {'loss': float('inf'), 'cosine_similarity': 0.0}

def eval_model(model, test_loader, device, use_amp=True, transform_param="log10over3"):
    model.to(device); model.eval(); y_true_raw_list, y_pred_raw_list = [], []; fragment_true_list, fragment_pred_list = [], []; mol_ids_list = []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="テスト中", leave=False, unit="batch")
        for batch_idx, batch in enumerate(test_pbar):
            if batch is None: logger.warning(f"テストスキップ: バッチ {batch_idx+1} None"); continue
            try:
                processed_batch = {}; pin_memory = test_loader.pin_memory
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): processed_batch[k] = v.to(device, non_blocking=pin_memory)
                    elif k == 'graph' and isinstance(v, Batch): try: processed_batch[k] = v.to(device, non_blocking=pin_memory); except AttributeError: v.x=v.x.to(device,non_blocking=pin_memory);v.edge_index=v.edge_index.to(device,non_blocking=pin_memory);v.batch=v.batch.to(device,non_blocking=pin_memory);processed_batch[k]=v
                    else: processed_batch[k] = v
            except Exception as e_data_move: logger.error(f"テストバッチ {batch_idx+1} デバイス転送エラー: {e_data_move}. スキップ。", exc_info=False); continue
            try:
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): output, frag_pred = model(processed_batch)
                y_true_raw_list.append(processed_batch['spec'].cpu()); y_pred_raw_list.append(output.cpu())
                if 'fragment_pattern' in processed_batch and processed_batch['fragment_pattern'] is not None: fragment_true_list.append(processed_batch['fragment_pattern'].cpu()); fragment_pred_list.append(frag_pred.cpu())
                mol_ids_list.extend(processed_batch['mol_id'])
            except Exception as e_test_forward: logger.error(f"テストバッチ {batch_idx+1}: 順伝播エラー: {e_test_forward}.", exc_info=True); continue
            finally: del processed_batch, batch, output, frag_pred
    if not y_true_raw_list or not y_pred_raw_list: logger.error("テスト中有効結果なし。"); return {'cosine_similarity': 0.0, 'y_true_postprocessed': torch.empty(0), 'y_pred_postprocessed': torch.empty(0), 'y_true_raw': torch.empty(0), 'y_pred_raw': torch.empty(0), 'fragment_true': torch.empty(0), 'fragment_pred': torch.empty(0), 'mol_ids': []}
    all_true_raw = torch.cat(y_true_raw_list, dim=0); all_pred_raw = torch.cat(y_pred_raw_list, dim=0)
    all_fragment_true = torch.cat(fragment_true_list, dim=0) if fragment_true_list else torch.empty(0); all_fragment_pred = torch.cat(fragment_pred_list, dim=0) if fragment_pred_list else torch.empty(0)
    logger.info("テスト結果後処理・メトリクス計算..."); cosine_sim = 0.0; all_true_postprocessed = torch.empty(0); all_pred_postprocessed = torch.empty(0)
    try:
        all_true_postprocessed = postprocess_prediction(all_true_raw, transform_param); all_pred_postprocessed = postprocess_prediction(all_pred_raw, transform_param)
        cosine_sim = cosine_similarity_score(all_true_postprocessed, all_pred_postprocessed); logger.info(f"後処理済スペクトルでのコサイン類似度: {cosine_sim:.4f}")
    except Exception as e_postprocess:
         logger.error(f"テスト結果後処理/類似度計算エラー: {e_postprocess}", exc_info=True)
         try: cosine_sim_raw = cosine_similarity_score(all_true_raw, all_pred_raw); logger.warning(f"生出力での類似度(参考): {cosine_sim_raw:.4f}"); cosine_sim = cosine_sim_raw; except Exception as e_raw_sim: logger.error(f"生出力類似度計算も失敗: {e_raw_sim}")
    return {'cosine_similarity': cosine_sim, 'y_true_postprocessed': all_true_postprocessed, 'y_pred_postprocessed': all_pred_postprocessed, 'y_true_raw': all_true_raw, 'y_pred_raw': all_pred_raw, 'fragment_true': all_fragment_true, 'fragment_pred': all_fragment_pred, 'mol_ids': mol_ids_list}

def tiered_training(model, train_ids, val_loader, criterion, optimizer, scheduler, device, mol_files_path, msp_data, transform, normalization, cache_dir, checkpoint_dir=CACHE_DIR, num_workers=0, patience=5, transform_param="log10over3"):
    logger.info("段階的トレーニング開始"); base_checkpoint_dir = checkpoint_dir
    n_train = len(train_ids)
    if n_train > 100000: train_tiers, tier_epochs, base_lr, initial_lr_factor = [train_ids[:10000], train_ids[:30000], train_ids[:60000], train_ids[:100000], train_ids], [3, 3, 4, 5, 15], 0.0008, 1.25
    elif n_train > 50000: train_tiers, tier_epochs, base_lr, initial_lr_factor = [train_ids[:10000], train_ids[:30000], train_ids], [4, 5, 21], 0.0009, 1.1
    else: train_tiers, tier_epochs, base_lr, initial_lr_factor = [train_ids[:max(5000, n_train // 2)], train_ids], [8, 22], 0.001, 1.0
    logger.info(f"ティア数: {len(train_tiers)}, 各ティアエポック: {tier_epochs}"); if len(train_tiers) != len(tier_epochs): raise ValueError("train_tiersとtier_epochs長さ不一致")
    best_overall_cosine = -1.0; all_train_losses, all_val_losses, all_val_cosine_similarities = [], [], []; current_epoch_offset = 0
    for tier_idx, tier_ids in enumerate(train_tiers):
        tier_name = f"ティア {tier_idx+1}/{len(train_tiers)} ({len(tier_ids)} サンプル)"; logger.info(f"======== {tier_name} トレーニング開始 ========"); tier_checkpoint_dir = os.path.join(base_checkpoint_dir, f"tier_{tier_idx+1}"); os.makedirs(tier_checkpoint_dir, exist_ok=True)
        logger.info("ティア間メモリクリーンアップ..."); [globals()[ds].graph_cache.clear() for ds in ['train_dataset', 'val_dataset'] if ds in globals() and hasattr(globals()[ds], 'graph_cache')]; aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        logger.info("ティアデータセット・ローダー作成中..."); tier_dataset = OptimizedMoleculeGraphDataset(tier_ids, mol_files_path, msp_data, transform=transform, normalization=normalization, augment=True, cache_dir=cache_dir)
        if len(tier_ids) <= 15000: tier_batch_size = 16; elif len(tier_ids) <= 40000: tier_batch_size = 12; elif len(tier_ids) <= 80000: tier_batch_size = 8; else: tier_batch_size = 6
        logger.info(f"{tier_name} バッチサイズ: {tier_batch_size}"); tier_loader = DataLoader(tier_dataset, batch_size=tier_batch_size, shuffle=True, collate_fn=optimized_collate_fn, num_workers=num_workers, pin_memory=(device.type == 'cuda'), persistent_workers=(num_workers > 0), drop_last=True)
        current_lr = base_lr * initial_lr_factor if tier_idx == 0 else base_lr * (0.9 ** tier_idx); logger.info(f"{tier_name} 学習率設定: {current_lr:.6f}"); [pg.update({'lr': current_lr}) for pg in optimizer.param_groups]
        steps_per_epoch_tier = len(tier_loader); epochs_this_tier = tier_epochs[tier_idx]; logger.info(f"{tier_name} エポック数: {epochs_this_tier}, ステップ/エポック: {steps_per_epoch_tier}")
        tier_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=current_lr, steps_per_epoch=steps_per_epoch_tier, epochs=epochs_this_tier, pct_start=0.3, div_factor=10.0, final_div_factor=100.0)
        tier_patience = patience; logger.info(f"{tier_name} 早期停止ペイシェンス: {tier_patience}")
        logger.info(f"{tier_name} モデルトレーニング開始..."); tier_train_losses, tier_val_losses, tier_val_similarities, tier_best_cosine = train_model(model=model, train_loader=tier_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=tier_scheduler, device=device, num_epochs=epochs_this_tier, eval_interval=1, patience=tier_patience, grad_clip=1.0, checkpoint_dir=tier_checkpoint_dir, transform_param=transform_param)
        if tier_best_cosine > best_overall_cosine: logger.info(f"ティア {tier_idx+1} で全体最良性能更新: {best_overall_cosine:.4f} -> {tier_best_cosine:.4f}"); best_overall_cosine = tier_best_cosine; tier_best_model_path = os.path.join(tier_checkpoint_dir, 'best_model.pth'); overall_best_model_path = os.path.join(base_checkpoint_dir, 'best_overall_model.pth');
        if os.path.exists(tier_best_model_path): try: import shutil; shutil.copyfile(tier_best_model_path, overall_best_model_path); logger.info(f"全体最良モデル更新・保存: {overall_best_model_path}"); except Exception as e_copy: logger.error(f"最良モデルコピー失敗: {e_copy}")
        all_train_losses.extend([(loss, current_epoch_offset + i + 1) for i, loss in enumerate(tier_train_losses)]); all_val_losses.extend([(loss, current_epoch_offset + i + 1) for i, loss in enumerate(tier_val_losses)]); all_val_cosine_similarities.extend([(sim, current_epoch_offset + i + 1) for i, sim in enumerate(tier_val_similarities)]); current_epoch_offset += len(tier_train_losses)
        logger.info(f"{tier_name} 完了。クリーンアップ中..."); del tier_dataset, tier_loader, tier_scheduler; aggressive_memory_cleanup(force_sync=True, purge_cache=True); logger.info("次のティアまで5秒待機..."); time.sleep(5)
    logger.info("======== 段階的トレーニング完了 ========"); logger.info(f"全体最良コサイン類似度: {best_overall_cosine:.4f}")
    plot_train_losses = [item[0] for item in all_train_losses]; plot_val_losses = [item[0] for item in all_val_losses]; plot_val_similarities = [item[0] for item in all_val_cosine_similarities]
    try: _plot_training_progress(plot_train_losses, plot_val_losses, plot_val_similarities, best_overall_cosine, base_checkpoint_dir); final_plot_path = os.path.join(base_checkpoint_dir, "tiered_learning_curves.png"); logger.info(f"段階的トレーニング全体学習曲線保存: {final_plot_path}"); except Exception as e: logger.error(f"最終段階的トレーニングプロットエラー: {e}")
    return plot_train_losses, plot_val_losses, plot_val_similarities, best_overall_cosine

def visualize_results(test_results, transform_param, num_samples=10, save_dir="."):
    logger.info(f"予測結果可視化生成中 (サンプル数: {num_samples})...")
    if not test_results or 'y_true_postprocessed' not in test_results or 'y_pred_postprocessed' not in test_results: logger.error("可視化用有効テスト結果(後処理済)なし。"); return
    true_spectra = test_results['y_true_postprocessed']; pred_spectra = test_results['y_pred_postprocessed']; mol_ids = test_results.get('mol_ids', [])
    num_total_samples = len(true_spectra)
    if num_total_samples == 0: logger.warning("可視化サンプルなし。"); return
    num_samples = min(num_samples, num_total_samples); sample_indices = np.random.choice(num_total_samples, num_samples, replace=False)
    plt.figure(figsize=(15, num_samples * 4))
    for i, idx in enumerate(sample_indices):
        true_spec_vis = true_spectra[idx].cpu().numpy(); pred_spec_vis = pred_spectra[idx].cpu().numpy()
        sim = cosine_similarity_score(true_spec_vis.reshape(1, -1), pred_spec_vis.reshape(1, -1))
        plt.subplot(num_samples, 2, 2*i + 1); mz_values = np.arange(len(true_spec_vis)); markerline, stemlines, baseline = plt.stem(mz_values, true_spec_vis, linefmt='b-', markerfmt=' ', basefmt='grey'); plt.setp(stemlines, 'linewidth', 1); plt.setp(baseline, 'linewidth', 0.5)
        mol_id_str = f" - ID: {mol_ids[idx]}" if mol_ids and idx < len(mol_ids) else ""; plt.title(f"Measured Spectrum{mol_id_str} (Post-processed, Max 100)"); plt.xlabel("m/z"); plt.ylabel("Relative Intensity"); plt.xlim(0, MAX_MZ); plt.ylim(bottom=0)
        plt.subplot(num_samples, 2, 2*i + 2); mz_values_pred = np.arange(len(pred_spec_vis)); markerline_p, stemlines_p, baseline_p = plt.stem(mz_values_pred, pred_spec_vis, linefmt='r-', markerfmt=' ', basefmt='grey'); plt.setp(stemlines_p, 'linewidth', 1, 'color','red'); plt.setp(baseline_p, 'linewidth', 0.5)
        plt.title(f"Predicted Spectrum - Sim: {sim:.4f} (Post-processed)"); plt.xlabel("m/z"); plt.ylabel("Relative Intensity"); plt.xlim(0, MAX_MZ); plt.ylim(bottom=0)
    plt.tight_layout(pad=2.0); save_path = os.path.join(save_dir, 'hybrid_prediction_visualization.png')
    try: plt.savefig(save_path, dpi=150); logger.info(f"予測結果可視化保存: {save_path}"); except Exception as e: logger.error(f"可視化プロット保存失敗: {save_path}, エラー: {e}"); plt.close()


###############################
# メイン関数（最適化）
###############################

def main():
    start_time = time.time(); logger.info("============= 質量スペクトル予測モデル実行開始 ============="); timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available(): logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}"); try: total_mem_gb=torch.cuda.get_device_properties(0).total_memory/(1024**3); free_mem_gb,_=torch.cuda.mem_get_info(); free_mem_gb/=(1024**3); logger.info(f"GPUメモリ: Total={total_mem_gb:.2f}GB, Free={free_mem_gb:.2f}GB"); except Exception as e: logger.warning(f"GPUメモリ情報取得エラー: {e}")
    else: logger.info("CUDAデバイスなし。CPU使用。")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); logger.info(f"使用デバイス: {device}")

    transform = "log10over3"; normalization = "l1"; num_epochs = 30; patience = 7; num_workers = 0

    logger.info("MSPファイル解析中...");
    try: msp_data = parse_msp_file(MSP_FILE_PATH, cache_dir=CACHE_DIR); assert msp_data, "MSPデータ解析失敗または空。"; logger.info(f"MSPファイルから {len(msp_data)} 化合物データ読み込み完了")
    except FileNotFoundError: logger.error(f"MSPファイル未検出: {MSP_FILE_PATH}"); return
    except AssertionError as e: logger.error(e); return
    except Exception as e: logger.error(f"MSPファイル解析エラー: {e}", exc_info=True); return

    logger.info("有効分子IDリスト作成中..."); mol_ids = []
    mol_id_cache_file = os.path.join(CACHE_DIR, "valid_mol_ids_list_cache_v2.pkl") # Cache filename update
    if os.path.exists(mol_id_cache_file): logger.info(f"キャッシュからmol_ids読み込み: {mol_id_cache_file}"); try: with open(mol_id_cache_file, 'rb') as f: mol_ids = pickle.load(f); logger.info(f"キャッシュから {len(mol_ids)} mol_ids読み込み完了"); except Exception as e: logger.warning(f"mol_idsキャッシュ読み込み失敗 ({e})。再生成。"); mol_ids = []
    if not mol_ids:
        logger.info("MOLファイルディレクトリをスキャンして有効ID照合中...")
        try: available_files = os.listdir(MOL_FILES_PATH); logger.info(f"MOLファイルディレクトリ内ファイル数: {len(available_files)}")
        for filename in tqdm(available_files, desc="MOLファイル照合"):
             if filename.startswith("ID") and filename.endswith(".MOL"): try: mol_id = int(filename[2:-4]); if mol_id in msp_data: mol_ids.append(mol_id); except ValueError: continue
        logger.info(f"照合結果、有効分子ID数: {len(mol_ids)}")
        if mol_ids: logger.info(f"mol_idsキャッシュ保存中: {mol_id_cache_file}"); try: with open(mol_id_cache_file, 'wb') as f: pickle.dump(mol_ids, f, protocol=pickle.HIGHEST_PROTOCOL); except Exception as e: logger.error(f"mol_idsキャッシュ保存失敗: {e}")
        except FileNotFoundError: logger.error(f"MOLファイルディレクトリ未検出: {MOL_FILES_PATH}"); return
        except Exception as e: logger.error(f"MOLファイルスキャンエラー: {e}", exc_info=True); return
    if not mol_ids: logger.error("有効分子ID見つからず。"); return

    logger.info(f"データ分割 (80:10:10)...")
    try: train_ids, test_val_ids = train_test_split(mol_ids, test_size=0.2, random_state=42, shuffle=True); val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42, shuffle=True)
    except Exception as e: logger.error(f"データ分割エラー: {e}."); train_ids=mol_ids[:int(0.8*len(mol_ids))]; val_ids=mol_ids[int(0.8*len(mol_ids)):int(0.9*len(mol_ids))]; test_ids=mol_ids[int(0.9*len(mol_ids)):] if len(mol_ids)>20 else (logger.error("データ少すぎ分割不可。"), None, None, None); if train_ids is None: return
    logger.info(f"訓練: {len(train_ids)}, 検証: {len(val_ids)}, テスト: {len(test_ids)}")

    logger.info("検証・テスト用データセット作成中...")
    try:
        dataset_cache_dir = os.path.join(CACHE_DIR, "dataset_cache"); os.makedirs(dataset_cache_dir, exist_ok=True)
        val_dataset = OptimizedMoleculeGraphDataset(val_ids, MOL_FILES_PATH, msp_data, transform=transform, normalization=normalization, augment=False, cache_dir=dataset_cache_dir)
        test_dataset = OptimizedMoleculeGraphDataset(test_ids, MOL_FILES_PATH, msp_data, transform=transform, normalization=normalization, augment=False, cache_dir=dataset_cache_dir)
        logger.info(f"有効検証サンプル数: {len(val_dataset)}, 有効テストサンプル数: {len(test_dataset)}"); assert len(val_dataset) > 0 and len(test_dataset) > 0, "検証/テストデータセット作成失敗。"
    except AssertionError as e: logger.error(e); return
    except Exception as e: logger.error(f"データセット作成エラー: {e}", exc_info=True); return

    if device.type == 'cuda': gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3); batch_size = 32 if gpu_mem_gb > 30 else 16 if gpu_mem_gb > 15 else 8
    else: batch_size = 16
    logger.info(f"検証/テスト用バッチサイズ: {batch_size}, ワーカー数: {num_workers}")
    try: val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=optimized_collate_fn, num_workers=num_workers, pin_memory=(device.type == 'cuda'), persistent_workers=(num_workers > 0), drop_last=False); test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=optimized_collate_fn, num_workers=num_workers, pin_memory=(device.type == 'cuda'), persistent_workers=(num_workers > 0), drop_last=False)
    except Exception as e: logger.error(f"データローダー作成エラー: {e}", exc_info=True); return

    logger.info("モデル初期化中...")
    try:
        node_features = NODE_FEATURES_DIM; edge_features = EDGE_FEATURES_DIM
        hidden_channels = 32 if len(train_ids) <= 100000 else 24; out_channels = MAX_MZ
        logger.info(f"モデルパラメータ: node={node_features}, edge={edge_features}, global={GLOBAL_FEATURES_DIM}, hidden={hidden_channels}")
        aggressive_memory_cleanup(force_sync=True, purge_cache=False)
        model = OptimizedHybridMSModel(node_features=node_features, edge_features=edge_features, hidden_channels=hidden_channels, out_channels=out_channels, num_fragments=NUM_FRAGS, prec_mass_offset=10, bidirectional=True, gate_prediction=True).to(device)
        total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"モデル初期化完了。総パラメータ: {total_params:,}, 学習可能: {trainable_params:,}")
    except Exception as e: logger.error(f"モデル初期化エラー: {e}", exc_info=True); return

    criterion = combined_loss; optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6, eps=1e-8); scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # Placeholder

    logger.info(f"モデルトレーニング設定: エポック={num_epochs} (段階的), 忍耐={patience}"); logger.info("段階的トレーニング開始...")
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    train_results = None # Initialize
    try:
        train_losses, val_losses, val_cosine_similarities, best_cosine = tiered_training(model=model, train_ids=train_ids, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, mol_files_path=MOL_FILES_PATH, msp_data=msp_data, transform=transform, normalization=normalization, cache_dir=dataset_cache_dir, checkpoint_dir=CHECKPOINT_DIR, num_workers=num_workers, patience=patience, transform_param=transform)
        logger.info(f"トレーニング完了！ 全体最良コサイン類似度(検証): {best_cosine:.4f}"); train_results = {'train_losses': train_losses, 'val_losses': val_losses, 'val_cosine_similarities': val_cosine_similarities}
    except Exception as e:
        logger.error(f"トレーニングプロセスエラー: {e}", exc_info=True)
        best_model_path = os.path.join(CHECKPOINT_DIR, 'best_overall_model.pth') or os.path.join(CHECKPOINT_DIR, 'best_model.pth') # Simplified
        if os.path.exists(best_model_path): logger.warning("トレーニングエラー発生も、最良モデルでテスト試行。"); try: model.load_state_dict(torch.load(best_model_path, map_location=device)); logger.info(f"モデル読み込み成功: {best_model_path}"); except Exception as e_lf: logger.error(f"最良モデル読み込み失敗: {e_lf}。テストスキップ。"); return
        else: logger.error("トレーニング失敗、最良モデル見つからず。テストスキップ。"); return

    logger.info("テストデータ最終評価開始...")
    test_results = None # Initialize
    try:
        best_model_path = os.path.join(CHECKPOINT_DIR, 'best_overall_model.pth') or os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        if os.path.exists(best_model_path): logger.info(f"最良モデル読み込み中: {best_model_path}"); try: model.load_state_dict(torch.load(best_model_path, map_location=device)); except Exception as e_l: logger.error(f"最良モデル読み込みエラー: {e_l}. 現在状態でテスト続行。", exc_info=True)
        else: logger.warning("最良モデルファイル未検出。最終状態でテスト。")
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        test_results = eval_model(model, test_loader, device, use_amp=(device.type == 'cuda'), transform_param=transform)
        if test_results and 'cosine_similarity' in test_results:
            logger.info(f"テスト完了。平均コサイン類似度(後処理済): {test_results['cosine_similarity']:.4f}")
            visualize_results(test_results, transform_param=transform, num_samples=10, save_dir=".")
            try: # Similarity Distribution
                similarities = []; num_test = len(test_results.get('y_true_postprocessed', []))
                if num_test > 0:
                    logger.info("テストデータ類似度分布計算中...")
                    for i in tqdm(range(num_test), desc="類似度計算"): sim = cosine_similarity_score(test_results['y_true_postprocessed'][i].cpu().numpy().reshape(1, -1), test_results['y_pred_postprocessed'][i].cpu().numpy().reshape(1, -1)); similarities.append(sim)
                    plt.figure(figsize=(10, 6)); plt.hist(similarities, bins=30, alpha=0.75, color='skyblue', edgecolor='black')
                    mean_sim = np.mean(similarities); median_sim = np.median(similarities); plt.axvline(mean_sim, color='red', linestyle='--', linewidth=1.5, label=f'平均: {mean_sim:.4f}'); plt.axvline(median_sim, color='green', linestyle=':', linewidth=1.5, label=f'中央値: {median_sim:.4f}')
                    plt.xlabel('Cosine Similarity (Post-processed)'); plt.ylabel('サンプル数'); plt.title('Test Set Cosine Similarity Distribution'); plt.legend(); plt.grid(axis='y', alpha=0.5); sim_dist_path = os.path.join(".", 'similarity_distribution.png'); plt.savefig(sim_dist_path, dpi=150); logger.info(f"類似度分布ヒストグラム保存: {sim_dist_path}"); plt.close()
                else: logger.warning("類似度分布計算スキップ: テストサンプルなし。")
            except Exception as e_analysis: logger.error(f"追加分析(類似度分布)エラー: {e_analysis}", exc_info=True)
        else: logger.error("テスト評価から有効結果得られず。")
    except Exception as e_test: logger.error(f"テストプロセスエラー: {e_test}", exc_info=True)

    end_time = time.time(); total_duration = end_time - start_time; logger.info(f"============= 質量スペクトル予測モデル実行終了 ({total_duration:.2f}秒) =============")
    # return model, train_results, test_results # Optionally return

if __name__ == "__main__":
    seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False # For debugging
    main()