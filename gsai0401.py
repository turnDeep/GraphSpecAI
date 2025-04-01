# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch # Transformer用
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
# RDKitの警告を抑制
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tqdm import tqdm
import logging
import copy
import random
import math
import gc
import pickle
from functools import partial
from torch.amp import autocast, GradScaler
import time
import datetime
# Peak matching loss (Wasserstein distance) 用のライブラリ
try:
    import ot # POT (Python Optimal Transport) library
    POT_AVAILABLE = True
except ImportError:
    print("Warning: POT library not found. Wasserstein loss will use a fallback (weighted MSE or Cosine). Install with: pip install POT")
    POT_AVAILABLE = False
    ot = None # otオブジェクトが存在しないことを示す

# ===== メモリ管理関連の関数 (既存コード流用) =====
def aggressive_memory_cleanup(force_sync=True, percent=70, purge_cache=False):
    """強化版メモリクリーンアップ関数"""
    gc.collect()

    if not torch.cuda.is_available():
        return False

    if force_sync:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()

    gpu_memory_allocated = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_percent = gpu_memory_allocated / total_memory * 100

    if gpu_memory_percent > percent:
        logger.warning(f"高いGPUメモリ使用率 ({gpu_memory_percent:.1f}%)。キャッシュをクリアします。")
        if purge_cache:
            for obj_name in ['train_dataset', 'val_dataset', 'test_dataset']:
                if obj_name in globals():
                    obj = globals()[obj_name]
                    if hasattr(obj, 'feature_cache') and isinstance(obj.feature_cache, dict):
                        obj.feature_cache.clear()
                        logger.info(f"{obj_name}の特徴量キャッシュをクリア")
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        return True
    return False

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# パス設定
DATA_PATH = "data/"
MOL_FILES_PATH = os.path.join(DATA_PATH, "mol_files/")
MSP_FILE_PATH = os.path.join(DATA_PATH, "NIST17.MSP")
CACHE_DIR = os.path.join(DATA_PATH, "cache/")
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints/") # チェックポイント保存先

# ディレクトリの作成
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # チェックポイント用ディレクトリも作成

# 最大m/z値の設定
MAX_MZ = 2000
MZ_DIM = MAX_MZ # 出力次元

# 重要なm/z値のリスト
IMPORTANT_MZ = [18, 28, 43, 57, 71, 73, 77, 91, 105, 115, 128, 152, 165, 178, 207]

# エフェメラル値
EPS = np.finfo(np.float32).eps

# --- 特徴量定義の改善 ---
ATOM_FEATURES = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8,
    'Si': 9, 'B': 10, 'Na': 11, 'K': 12, 'Li': 13, 'Mg': 14, 'Ca': 15, 'Fe': 16,
    'Co': 17, 'Ni': 18, 'Cu': 19, 'Zn': 20, 'H': 21, 'OTHER': 22
}
NUM_ATOM_TYPES = len(ATOM_FEATURES)
ADDITIONAL_ATOM_FEATURES = 15 # 次数,電荷,ラジカル,芳香族,質量,環内,ハイブリ,価電子,隠れ価電子,芳香環内,環サイズ,H数,Gasteiger電荷,環フラグ,キラル
TOTAL_ATOM_FEATURES = NUM_ATOM_TYPES + ADDITIONAL_ATOM_FEATURES

BOND_FEATURES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
    Chem.rdchem.BondType.UNSPECIFIED: 4,
    Chem.rdchem.BondType.OTHER: 5
}
NUM_BOND_TYPES = len(BOND_FEATURES)
ADDITIONAL_BOND_FEATURES = 5 # 環内,共役,芳香族,最小環サイズ,推定BDE(簡易)
TOTAL_BOND_FEATURES = NUM_BOND_TYPES + ADDITIONAL_BOND_FEATURES

NUM_FRAGS = 167 # MACCSキー

# 結合切断予測用の次元
BOND_BREAK_DIM = 1

###############################
# データ処理関連の関数 (既存および改修)
###############################

# process_spec, unprocess_spec は既存のものを流用
def process_spec(spec, transform, normalization, eps=EPS):
    """スペクトルにトランスフォームと正規化を適用"""
    spec = spec / (torch.max(spec, dim=-1, keepdim=True)[0] + eps) * 1000.
    if transform == "log10": spec = torch.log10(spec + 1)
    elif transform == "log10over3": spec = torch.log10(spec + 1) / 3
    elif transform == "loge": spec = torch.log(spec + 1)
    elif transform == "sqrt": spec = torch.sqrt(spec)
    elif transform != "none": raise ValueError("invalid transform")
    if normalization == "l1": spec = F.normalize(spec, p=1, dim=-1, eps=eps)
    elif normalization == "l2": spec = F.normalize(spec, p=2, dim=-1, eps=eps)
    elif normalization != "none": raise ValueError("invalid normalization")
    assert not torch.isnan(spec).any()
    return spec

def unprocess_spec(spec, transform):
    """スペクトルの変換を元に戻す (強度スケールに戻す)"""
    if transform == "log10":
        max_ints = float(np.log10(1000. + 1.))
        untransform_fn = lambda x: 10**x - 1.
    elif transform == "log10over3":
        max_ints = float(np.log10(1000. + 1.) / 3.)
        untransform_fn = lambda x: 10**(3 * x) - 1.
    elif transform == "loge":
        max_ints = float(np.log(1000. + 1.))
        untransform_fn = lambda x: torch.exp(x) - 1.
    elif transform == "sqrt":
        max_ints = float(np.sqrt(1000.))
        untransform_fn = lambda x: x**2
    elif transform == "none":
        max_ints = 1000.
        untransform_fn = lambda x: x
    else: raise ValueError("invalid transform")
    # 正規化を元に戻すのは難しいので、相対的な形状を復元
    spec = spec / (torch.max(spec, dim=-1, keepdim=True)[0] + EPS) * max_ints
    spec = untransform_fn(spec)
    spec = torch.clamp(spec, min=0.)
    assert not torch.isnan(spec).any()
    return spec

# 離散化関数の改善版
def improved_hybrid_spectrum_conversion(pred_intensities_processed, pred_probs, transform="log10over3",
                                       prob_threshold=0.1, top_k=200, relative_intensity_threshold=0.1):
    """
    モデルが出力する確率と強度(process_spec適用済み)に基づき、離散スペクトルへ変換
    Args:
        pred_intensities_processed: モデル出力の強度 (process_spec適用済み, numpy array)
        pred_probs: モデル出力のピーク存在確率 (numpy array)
        transform: process_specで使われた変換方法
        prob_threshold: ピークとみなす最小確率
        top_k: 保持するピークの最大数
        relative_intensity_threshold: 保持するピークの最小相対強度 (%)
    Returns:
        離散スペクトル (0-100スケール, numpy array)
    """
    # 1. 強度を元のスケールに（近似的に）戻す
    try:
        # unprocess_specはtorch tensorを入力とする
        intensities_unprocessed = unprocess_spec(torch.from_numpy(pred_intensities_processed).unsqueeze(0), transform)
        intensities_unprocessed = intensities_unprocessed.squeeze(0).numpy()
    except Exception as e:
        # logger.warning(f"Unprocessing failed during conversion: {e}. Using processed intensities.")
        # フォールバック：processされた強度をそのまま使う（スケールは異なる可能性がある）
        intensities_unprocessed = pred_intensities_processed

    intensities_unprocessed = np.maximum(0, intensities_unprocessed)
    max_intensity_unprocessed = np.max(intensities_unprocessed) if np.max(intensities_unprocessed) > 0 else 1.0

    discrete_spectrum = np.zeros_like(intensities_unprocessed)

    # 2. 確率と強度に基づいてピーク候補を選択
    potential_indices = np.where(pred_probs > prob_threshold)[0]

    # 候補がない場合でも、強度が非常に高いピークは残すことを検討
    if len(potential_indices) == 0 and max_intensity_unprocessed > 0:
        # 例：強度が上位1%のピークを候補に追加
        high_intensity_indices = np.argsort(-intensities_unprocessed)[:max(1, int(len(intensities_unprocessed)*0.01))]
        potential_indices = np.unique(np.concatenate([potential_indices, high_intensity_indices]))

    if len(potential_indices) == 0:
        return discrete_spectrum

    # 3. 候補から強度としきい値でフィルタリング
    filtered_indices = []
    min_abs_intensity = max_intensity_unprocessed * (relative_intensity_threshold / 100.0)
    for idx in potential_indices:
        if intensities_unprocessed[idx] >= min_abs_intensity:
            filtered_indices.append(idx)

    if not filtered_indices:
        # フィルタリングで全滅した場合、確率が最も高いピークだけでも残す
        if len(potential_indices) > 0:
             best_prob_idx = potential_indices[np.argmax(pred_probs[potential_indices])]
             if intensities_unprocessed[best_prob_idx] > 0: # 強度がゼロでないことを確認
                  filtered_indices = [best_prob_idx]
        if not filtered_indices: # それでもダメなら空スペクトル
             return discrete_spectrum

    # 4. 強度に基づいて上位K個を選択
    filtered_intensities = intensities_unprocessed[filtered_indices]
    if len(filtered_indices) > top_k:
        sorted_idx_indices = np.argsort(-filtered_intensities)[:top_k]
        final_indices = np.array(filtered_indices)[sorted_idx_indices]
    else:
        final_indices = np.array(filtered_indices)

    # 5. 離散スペクトルに強度を代入
    for idx in final_indices:
        discrete_spectrum[idx] = intensities_unprocessed[idx]

    # 6. 最大値で正規化 (0-100スケール)
    max_discrete_intensity = np.max(discrete_spectrum)
    if max_discrete_intensity > 0:
        discrete_spectrum = discrete_spectrum / max_discrete_intensity * 100.0

    return discrete_spectrum


# mask_prediction_by_mass は変更なし
def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    """前駆体質量によるマスキング"""
    device = raw_prediction.device
    max_idx = raw_prediction.shape[1]
    if prec_mass_idx.dtype != torch.long: prec_mass_idx = prec_mass_idx.long()
    # 範囲外アクセスを防ぐためクリップ
    prec_mass_idx = torch.clamp(prec_mass_idx, max=max_idx-1, min=0) # min=0も追加
    idx = torch.arange(max_idx, device=device)
    mask = (idx.unsqueeze(0) <= (prec_mass_idx.unsqueeze(1) + prec_mass_offset)).float()
    # マスク外を mask_value に設定 (logitの場合は-inf相当の値が良い場合も)
    if mask_value == 0.:
        return mask * raw_prediction
    else:
        return mask * raw_prediction + (1. - mask) * mask_value


# MSPファイルのパース関数 (生強度を保持するように変更)
def parse_msp_file_raw(msp_file_path, cache_dir=CACHE_DIR):
    """MSPファイルを解析し、ID->生強度マススペクトルのマッピングを返す"""
    cache_file = os.path.join(cache_dir, f"msp_data_cache_raw_{os.path.basename(msp_file_path)}.pkl")
    if os.path.exists(cache_file):
        logger.info(f"キャッシュから生MSPデータを読み込み中: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: return pickle.load(f)
        except Exception as e:
            logger.warning(f"生MSPキャッシュ読み込み失敗 ({e})。再解析します。")
            try: os.remove(cache_file)
            except OSError: pass

    logger.info(f"MSPファイルを解析中 (生データ): {msp_file_path}")
    msp_data = {}
    current_id = None
    current_peaks = []
    with open(msp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f): # 行番号を追加してデバッグしやすく
            line = line.strip()
            try:
                if line.startswith("ID:"):
                    # ID行の処理前に前の化合物を保存
                    if current_id is not None and current_peaks:
                         ms_vector = np.zeros(MAX_MZ, dtype=np.float32) # メモリ効率のためfloat32
                         for mz, intensity in current_peaks:
                             mz_int = int(round(mz))
                             if 0 <= mz_int < MAX_MZ:
                                 ms_vector[mz_int] = max(ms_vector[mz_int], intensity) # 強度が最大の値を取る
                         msp_data[current_id] = ms_vector
                    # 新しいIDとピークリストを初期化
                    current_id = int(line.split(":")[-1].strip())
                    current_peaks = []
                elif line.startswith("Num peaks:"):
                    # ピーク数情報は特に使わないが、ピーク開始の目印
                    pass
                elif current_id is not None and ";" in line: # ピーク行の形式 (e.g., "15 345; 16 521;")
                    peak_pairs = line.split(';')
                    for pair in peak_pairs:
                        pair = pair.strip()
                        if not pair: continue
                        parts = pair.split()
                        if len(parts) >= 2:
                             mz = float(parts[0])
                             intensity = float(parts[1])
                             if mz >= 0 and intensity >= 0:
                                 current_peaks.append((mz, intensity))
                elif current_id is not None and len(line.split()) == 2 and line[0].isdigit(): # シンプルな "mz intensity" 形式
                     parts = line.split()
                     mz = float(parts[0])
                     intensity = float(parts[1])
                     if mz >= 0 and intensity >= 0:
                         current_peaks.append((mz, intensity))
                elif line == "" and current_id is not None: # 化合物の終わり
                     if current_peaks: # ピークがあれば保存
                         ms_vector = np.zeros(MAX_MZ, dtype=np.float32)
                         for mz, intensity in current_peaks:
                             mz_int = int(round(mz))
                             if 0 <= mz_int < MAX_MZ:
                                 ms_vector[mz_int] = max(ms_vector[mz_int], intensity)
                         msp_data[current_id] = ms_vector
                     # IDとピークリストをリセット
                     current_id = None
                     current_peaks = []
            except Exception as e:
                 logger.error(f"MSPファイル解析エラー (行 {line_num + 1}): {e} - Line: '{line}'")
                 # エラーが発生しても、次の化合物から処理を試みる
                 current_id = None
                 current_peaks = []

    # ファイル末尾に残っているデータを処理
    if current_id is not None and current_peaks:
        ms_vector = np.zeros(MAX_MZ, dtype=np.float32)
        for mz, intensity in current_peaks:
            mz_int = int(round(mz))
            if 0 <= mz_int < MAX_MZ:
                ms_vector[mz_int] = max(ms_vector[mz_int], intensity)
        msp_data[current_id] = ms_vector


    logger.info(f"生MSPデータをキャッシュに保存中: {cache_file}")
    try:
        with open(cache_file, 'wb') as f: pickle.dump(msp_data, f)
    except Exception as e: logger.error(f"生MSPキャッシュ保存失敗: {e}")
    return msp_data

###############################
# モデル: RadicalNetMS (変更なし)
###############################
class RadicalNetMS(nn.Module):
    """
    抜本的改修版: フラグメンテーションメカニズムに着想を得たモデル
    - 階層的GNNによる特徴抽出
    - 結合切断確率の予測 (潜在的)
    - ピーク存在確率と強度の分離予測
    - Transformerによるグローバルコンテキストの統合
    """
    def __init__(self, node_features, edge_features, hidden_channels, out_channels, num_fragments=NUM_FRAGS,
                 prec_mass_offset=10, dropout=0.2, n_gnn_layers=4, n_transformer_layers=2, heads=4):
        super(RadicalNetMS, self).__init__()

        self.prec_mass_offset = prec_mass_offset
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels # = MAX_MZ
        self.dropout_rate = dropout

        # Node/Edge Embedding
        self.node_emb = nn.Linear(node_features, hidden_channels)
        self.edge_emb = nn.Linear(edge_features, hidden_channels)

        # GNN Layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        self.gnn_dropout = nn.ModuleList()
        in_channels = hidden_channels
        for _ in range(n_gnn_layers):
            conv = GATv2Conv(in_channels, hidden_channels // heads, heads=heads,
                             edge_dim=hidden_channels, dropout=dropout, concat=True) # concat=Trueがデフォルト
            self.gnn_layers.append(conv)
            self.gnn_norms.append(nn.LayerNorm(hidden_channels)) # GATv2Convは(heads * out_channels)を出力
            self.gnn_dropout.append(nn.Dropout(dropout))
            in_channels = hidden_channels

        # Bond Break Prediction MLP (Optional - Currently not used in loss)
        self.bond_break_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels), # src, dst, edge
            nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, BOND_BREAK_DIM), nn.Sigmoid()
        )

        # GNN to Transformer Projection
        self.gnn_to_transformer_proj = nn.Linear(hidden_channels, hidden_channels)

        # Transformer Encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels, nhead=heads, dim_feedforward=hidden_channels * 2,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # Global Pooling & Feature Integration
        self.global_attn_pool = GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid()),
            nn=nn.Linear(hidden_channels, hidden_channels) # Optional MLP after pooling
        )
        self.global_features_dim = 16 # Expect 16 global features
        self.global_proj = nn.Linear(self.global_features_dim, hidden_channels) if self.global_features_dim > 0 else None

        # Spectrum Prediction Head
        combined_dim = hidden_channels * 2 # Pooled GNN/Transformer + Global Features
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels * 2), nn.LeakyReLU(),
            nn.LayerNorm(hidden_channels * 2), nn.Dropout(dropout)
        )
        self.prob_head = nn.Linear(hidden_channels * 2, out_channels) # Logits for probability
        self.intensity_head = nn.Linear(hidden_channels * 2, out_channels) # Intensity values

        # Fragment Prediction Head (Optional)
        self.fragment_pred_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels), nn.LeakyReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_channels, num_fragments)
        ) # Logits for fragments

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, data):
        device = next(self.parameters()).device

        # Data Preparation (Handles both dict and Data object)
        if isinstance(data, dict):
            g = data['graph']
            prec_mz_bin = data.get('prec_mz_bin')
        else: # Assuming torch_geometric.data.Data or Batch
            g = data
            prec_mz_bin = g.prec_mz_bin if hasattr(g, 'prec_mz_bin') else None

        x = g.x.to(device, non_blocking=True).float()
        edge_index = g.edge_index.to(device, non_blocking=True)
        edge_attr = g.edge_attr.to(device, non_blocking=True).float()
        batch = g.batch.to(device, non_blocking=True) if hasattr(g, 'batch') else None
        global_attr = g.global_attr.to(device, non_blocking=True).float() if hasattr(g, 'global_attr') and g.global_attr is not None else None

        if prec_mz_bin is not None:
            prec_mz_bin = prec_mz_bin.to(device, non_blocking=True).long()

        # --- GNN Encoder ---
        node_feat = self.node_emb(x)
        edge_feat_emb = self.edge_emb(edge_attr)

        for i in range(len(self.gnn_layers)):
            node_feat_res = node_feat
            # Pass embedded edge features to edge_dim
            node_feat = self.gnn_layers[i](node_feat, edge_index, edge_dim=edge_feat_emb)
            node_feat = self.gnn_dropout[i](F.leaky_relu(self.gnn_norms[i](node_feat)))
            node_feat = node_feat + node_feat_res # Residual connection

        # --- Bond Break Prediction (Calculate but not used in loss yet) ---
        row, col = edge_index
        final_edge_repr = torch.cat([node_feat[row], node_feat[col], edge_feat_emb], dim=-1)
        bond_break_prob = self.bond_break_mlp(final_edge_repr)

        # --- Transformer Encoder ---
        node_feat_proj = self.gnn_to_transformer_proj(node_feat)
        transformer_input, mask = to_dense_batch(node_feat_proj, batch)
        transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=~mask)

        # --- Global Pooling & Feature Integration ---
        # Use attention pooling on final GNN features
        graph_pooled_emb = self.global_attn_pool(node_feat, batch)

        # Global features projection
        global_emb = torch.zeros_like(graph_pooled_emb) # Default zero vector
        if global_attr is not None and self.global_proj is not None:
             num_graphs = batch.max().item() + 1
             # Reshape and pad/truncate global_attr if necessary (robustness)
             if global_attr.shape[0] != num_graphs:
                 if global_attr.shape[0] > num_graphs: global_attr = global_attr[:num_graphs]
                 else: padding = torch.zeros(num_graphs - global_attr.shape[0], global_attr.shape[1], device=device); global_attr = torch.cat([global_attr, padding], dim=0)
             if global_attr.shape[1] != self.global_features_dim:
                 padded = torch.zeros(num_graphs, self.global_features_dim, device=device); copy_size = min(global_attr.shape[1], self.global_features_dim); padded[:, :copy_size] = global_attr[:, :copy_size]; global_attr = padded
             global_emb = self.global_proj(global_attr)

        combined_features = torch.cat([graph_pooled_emb, global_emb], dim=1)

        # --- Spectrum Prediction Head ---
        output_features = self.output_mlp(combined_features)
        pred_probs_logits = self.prob_head(output_features)
        pred_intensities = F.relu(self.intensity_head(output_features)) # Ensure non-negative intensity

        # Apply precursor mass masking
        if prec_mz_bin is not None:
            pred_intensities = mask_prediction_by_mass(pred_intensities, prec_mz_bin, self.prec_mass_offset, mask_value=0.0)
            # Also mask probabilities (set logits to large negative value)
            prob_mask = (torch.arange(self.out_channels, device=device).unsqueeze(0) <= (prec_mz_bin.unsqueeze(1) + self.prec_mass_offset)).float()
            pred_probs_logits = pred_probs_logits * prob_mask + (1. - prob_mask) * (-1e9) # Masked = large negative logit

        # --- Fragment Prediction Head ---
        pred_fragments_logits = self.fragment_pred_head(combined_features)

        # --- Output Dictionary ---
        output = {
            "pred_intensities": pred_intensities,
            "pred_probs_logits": pred_probs_logits,
            "pred_fragments_logits": pred_fragments_logits,
            "bond_break_prob": bond_break_prob, # Keep for potential future use
            "edge_index": edge_index,
            "batch": batch
        }
        return output

###############################
# データセット & データローダー (特徴量強化版)
###############################

# Helper functions for feature extraction
def get_bond_features(bond):
    """RDKitのBondオブジェクトから特徴量ベクトルを生成"""
    bond_feature = [0] * NUM_BOND_TYPES
    bond_type = bond.GetBondType()
    bond_feature[BOND_FEATURES.get(bond_type, BOND_FEATURES[Chem.rdchem.BondType.OTHER])] = 1

    additional_features = [0.0] * ADDITIONAL_BOND_FEATURES
    try: additional_features[0] = bond.IsInRing() * 1.0
    except: pass
    try: additional_features[1] = bond.GetIsConjugated() * 1.0
    except: pass
    try: additional_features[2] = bond.GetIsAromatic() * 1.0
    except: pass
    try: # Min ring size
        if bond.IsInRing():
            ring_info = bond.GetOwningMol().GetRingInfo()
            min_ring_size = float('inf')
            for ring in ring_info.BondRings():
                if bond.GetIdx() in ring: min_ring_size = min(min_ring_size, len(ring))
            additional_features[3] = min_ring_size / 10.0 if min_ring_size != float('inf') else 0.0
    except: pass
    try: # Simple BDE proxy
        if bond_type == Chem.rdchem.BondType.SINGLE: additional_features[4] = 0.1
        elif bond_type == Chem.rdchem.BondType.DOUBLE: additional_features[4] = 0.2
        elif bond_type == Chem.rdchem.BondType.TRIPLE: additional_features[4] = 0.3
        elif bond_type == Chem.rdchem.BondType.AROMATIC: additional_features[4] = 0.15
    except: pass
    bond_feature.extend(additional_features)
    return bond_feature

def get_atom_features(atom, mol=None):
    """RDKitのAtomオブジェクトから特徴量ベクトルを生成"""
    if mol is None: mol = atom.GetOwningMol()
    atom_feature = [0] * NUM_ATOM_TYPES
    atom_symbol = atom.GetSymbol()
    atom_feature[ATOM_FEATURES.get(atom_symbol, ATOM_FEATURES['OTHER'])] = 1

    additional_features = [0.0] * ADDITIONAL_ATOM_FEATURES
    try: additional_features[0] = atom.GetDegree() / 8.0
    except: pass
    try: additional_features[1] = atom.GetFormalCharge() / 8.0
    except: pass
    try: additional_features[2] = atom.GetNumRadicalElectrons() / 4.0
    except: pass
    try: additional_features[3] = atom.GetIsAromatic() * 1.0
    except: pass
    try: additional_features[4] = atom.GetMass() / 200.0
    except: pass
    try: additional_features[5] = atom.IsInRing() * 1.0
    except: pass
    try: additional_features[6] = int(atom.GetHybridization()) / 8.0
    except: pass
    try: additional_features[7] = atom.GetExplicitValence() / 8.0
    except: pass
    try: additional_features[8] = atom.GetImplicitValence() / 8.0
    except: pass
    try: additional_features[9] = (atom.GetIsAromatic() and atom.IsInRing()) * 1.0
    except: pass
    try: # Ring size
        ring_size = 0
        if atom.IsInRing():
            rings = mol.GetRingInfo().AtomRings()
            for ring in rings:
                if atom.GetIdx() in ring: ring_size = max(ring_size, len(ring))
        additional_features[10] = ring_size / 8.0
    except: pass
    try: additional_features[11] = atom.GetTotalNumHs() / 8.0
    except: pass
    try: # Gasteiger Charge (Needs pre-computation)
        charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
        additional_features[12] = np.clip(charge / 5.0, -1.0, 1.0) # Clip charge
    except: pass
    additional_features[13] = additional_features[5] # IsInRing again (redundant but keeps index)
    try: additional_features[14] = (atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED) * 1.0
    except: pass

    atom_feature.extend(additional_features)
    return atom_feature

# Dataset Class
class RadicalNetMoleculeDataset(Dataset):
    def __init__(self, mol_ids, mol_files_path, msp_data, transform="log10over3",
                 normalization="l1", augment=False, cache_dir=CACHE_DIR, use_3d=False):
        self.mol_ids = list(mol_ids) # Ensure it's a list
        self.mol_files_path = mol_files_path
        self.msp_data = msp_data
        self.augment = augment
        self.transform = transform
        self.normalization = normalization
        self.cache_dir = cache_dir
        self.use_3d = use_3d
        self.feature_cache = {} # In-memory cache for graph features
        self.valid_mol_ids = []
        self.fragment_patterns = {} # For compatibility

        self._preprocess_mol_ids()

    def _preprocess_mol_ids(self):
        """有効な分子IDとMACCSキーを前処理（キャッシュ利用）"""
        # Generate a hash based on the list of IDs for unique caching
        ids_hash = str(hash(tuple(sorted(self.mol_ids))))
        cache_file = os.path.join(self.cache_dir, f"radicalnet_preprocessed_data_{ids_hash}.pkl")

        if os.path.exists(cache_file):
            logger.info(f"キャッシュから前処理データを読み込み中: {cache_file}")
            try:
                with open(cache_file, 'rb') as f: cached_data = pickle.load(f)
                self.valid_mol_ids = cached_data['valid_mol_ids']
                self.fragment_patterns = cached_data['fragment_patterns']
                if not isinstance(self.valid_mol_ids, list) or not isinstance(self.fragment_patterns, dict): raise ValueError("Invalid cache format")
                logger.info(f"キャッシュ読み込み完了。有効ID数: {len(self.valid_mol_ids)}")
                return
            except Exception as e:
                 logger.warning(f"キャッシュ読み込み失敗 ({e})。再計算します。")
                 try: os.remove(cache_file)
                 except OSError: pass

        logger.info("分子データの前処理を開始します（シングルプロセス）...")
        valid_ids_temp = []
        fragment_patterns_temp = {}
        mol_count = len(self.mol_ids)

        with tqdm(total=mol_count, desc="分子検証 & MACCSキー計算") as pbar:
            for mol_id in self.mol_ids:
                mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
                mol = None
                fragments = np.zeros(NUM_FRAGS, dtype=np.float32)
                valid = False
                if mol_id not in self.msp_data: # Skip if no spectrum data
                    pbar.update(1)
                    continue
                try:
                    mol = Chem.MolFromMolFile(mol_file, sanitize=False)
                    if mol is not None:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
                        # Compute Gasteiger charges before MACCS keys
                        try: AllChem.ComputeGasteigerCharges(mol)
                        except: pass # Ignore charge calculation errors
                        maccs = MACCSkeys.GenMACCSKeys(mol)
                        for i in range(NUM_FRAGS):
                            if maccs.GetBit(i): fragments[i] = 1.0
                        valid = True
                except Exception: pass # Ignore errors in sanitization/MACCS
                finally: pbar.update(1)

                if valid:
                    valid_ids_temp.append(mol_id)
                    fragment_patterns_temp[mol_id] = fragments
                if pbar.n % 1000 == 0: gc.collect()

        self.valid_mol_ids = valid_ids_temp
        self.fragment_patterns = fragment_patterns_temp

        logger.info(f"前処理結果をキャッシュに保存中: {cache_file}")
        try:
            with open(cache_file, 'wb') as f: pickle.dump({'valid_mol_ids': self.valid_mol_ids, 'fragment_patterns': self.fragment_patterns}, f)
        except Exception as e: logger.error(f"キャッシュ保存失敗: {e}")
        logger.info(f"有効な分子: {len(self.valid_mol_ids)}個 / 全体: {mol_count}個")

    def _mol_to_graph_features(self, mol_id):
        """分子IDからグラフ特徴量を生成（キャッシュ対応）"""
        if mol_id in self.feature_cache: return self.feature_cache[mol_id]

        cache_file = os.path.join(self.cache_dir, f"graph_feature_cache_ID{mol_id}_v2.pkl") # Cache versioning
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f: cached_graph = pickle.load(f)
                if isinstance(cached_graph, Data) and hasattr(cached_graph, 'x'):
                    self.feature_cache[mol_id] = cached_graph
                    return cached_graph
                else: raise ValueError("Invalid graph cache format")
            except Exception as e:
                logger.warning(f"グラフキャッシュ読み込み失敗 (ID:{mol_id}): {e}")
                try: os.remove(cache_file)
                except OSError: pass

        mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
        try:
            mol = Chem.MolFromMolFile(mol_file, sanitize=False)
            if mol is None: raise ValueError("MOLファイル読み込み失敗")
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
            try: mol = Chem.AddHs(mol, addCoords=self.use_3d)
            except: pass
            try: AllChem.ComputeGasteigerCharges(mol) # Compute charges for features
            except: pass # Ignore charge errors

            pos = None
            if self.use_3d:
                try:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()); AllChem.UFFOptimizeMolecule(mol)
                    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
                except: pos = None

            x = [get_atom_features(atom, mol) for atom in mol.GetAtoms()]
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_feat = get_bond_features(bond)
                edge_indices.extend([[i, j], [j, i]]); edge_attrs.extend([bond_feat, bond_feat])

            # Global features (ensure 16 dim)
            global_attr_list = [0.0] * 16
            try: global_attr_list[0] = Descriptors.MolWt(mol) / 1000.0
            except: pass
            try: global_attr_list[1] = Descriptors.NumHAcceptors(mol) / 20.0
            except: pass
            try: global_attr_list[2] = Descriptors.NumHDonors(mol) / 10.0
            except: pass
            try: global_attr_list[3] = Descriptors.TPSA(mol) / 200.0
            except: pass
            try: global_attr_list[4] = rdMolDescriptors.CalcNumRotatableBonds(mol) / 10.0
            except: pass
            try: global_attr_list[5] = Descriptors.NumRings(mol) / 5.0
            except: pass
            try: global_attr_list[6] = Descriptors.NumAromaticRings(mol) / 5.0
            except: pass
            try: global_attr_list[7] = Descriptors.NumAliphaticRings(mol) / 5.0
            except: pass
            # Fill remaining with zeros

            graph_data = Data(
                x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
                global_attr=torch.tensor(global_attr_list, dtype=torch.float),
                pos=pos if pos is not None else None
            )

            self.feature_cache[mol_id] = graph_data
            try:
                with open(cache_file, 'wb') as f: pickle.dump(graph_data, f)
            except Exception as e: logger.error(f"グラフキャッシュ書き込み失敗 (ID:{mol_id}): {e}")
            return graph_data

        except Exception as e:
            logger.warning(f"分子グラフ生成エラー (ID:{mol_id}): {e}")
            # Return dummy data on error
            return Data(x=torch.zeros((1, TOTAL_ATOM_FEATURES), dtype=torch.float),
                      edge_index=torch.zeros((2, 0), dtype=torch.long),
                      edge_attr=torch.zeros((0, TOTAL_BOND_FEATURES), dtype=torch.float),
                      global_attr=torch.zeros(16, dtype=torch.float))

    def _preprocess_spectrum(self, spectrum_array):
        """スペクトルを前処理し、強度とターゲット確率を生成"""
        spec_tensor = torch.FloatTensor(spectrum_array).unsqueeze(0)
        # 強度は process_spec で変換・正規化
        processed_intensity = process_spec(spec_tensor.clone(), self.transform, self.normalization) # Use clone
        # ターゲット確率: 生強度が閾値(e.g., 0.1% of max or abs 1.0)より大きい場合に1
        max_raw_val = torch.max(spec_tensor)
        threshold = torch.maximum(max_raw_val * 0.001, torch.tensor(1.0)) if max_raw_val > 0 else torch.tensor(1.0)
        target_prob = (spec_tensor > threshold).float()
        return processed_intensity.squeeze(0), target_prob.squeeze(0)

    def __len__(self):
        return len(self.valid_mol_ids)

    def __getitem__(self, idx):
        if idx >= len(self.valid_mol_ids): raise IndexError("Index out of range")
        mol_id = self.valid_mol_ids[idx]
        graph_data = self._mol_to_graph_features(mol_id)
        raw_spectrum = self.msp_data.get(mol_id, np.zeros(MAX_MZ))
        processed_intensity, target_prob = self._preprocess_spectrum(raw_spectrum)
        fragment_pattern = torch.FloatTensor(self.fragment_patterns.get(mol_id, np.zeros(NUM_FRAGS)))

        peaks = np.nonzero(raw_spectrum)[0]
        prec_mz = float(np.max(peaks)) if len(peaks) > 0 else 0.0
        prec_mz_bin = int(round(prec_mz))

        # Augmentation (simple noise)
        if self.augment and graph_data.x.shape[0] > 0 and np.random.random() < 0.1:
            graph_data.x += torch.randn_like(graph_data.x) * 0.02

        # Add targets and metadata to the Data object
        graph_data.y_intensity = processed_intensity
        graph_data.y_prob = target_prob
        graph_data.y_fragment = fragment_pattern
        graph_data.mol_id = mol_id
        graph_data.prec_mz = prec_mz
        graph_data.prec_mz_bin = prec_mz_bin

        return graph_data

# Collate Function
def radicalnet_collate_fn(batch_list):
    """RadicalNet用のカスタムCollate関数"""
    # Filter out None entries if any error occurred in __getitem__
    batch_list = [item for item in batch_list if item is not None]
    if not batch_list: return None # Return None if batch is empty

    batch_graph = Batch.from_data_list(batch_list)
    # Prepare dict format expected by the model/loss
    output_dict = {
        'graph': batch_graph,
        'spec_intensity': batch_graph.y_intensity,
        'spec_prob': batch_graph.y_prob,
        'fragment_pattern': batch_graph.y_fragment,
        'mol_id': batch_graph.mol_id, # This will be a list
        'prec_mz': batch_graph.prec_mz, # This will be a tensor
        'prec_mz_bin': batch_graph.prec_mz_bin # This will be a tensor
    }
    return output_dict

###############################
# 損失関数と類似度計算 (Wasserstein導入)
###############################

# Wasserstein Loss
def wasserstein_loss(y_pred_intensity, y_pred_prob_logits, y_true_intensity, y_true_prob, mz_bins, reg=0.05, p=1):
    """Wasserstein距離に基づく損失 (POT使用、p=1版)"""
    if not POT_AVAILABLE:
        # Fallback: Weighted Cosine Similarity (more sensitive to peak position than simple MSE)
        pred_prob = torch.sigmoid(y_pred_prob_logits)
        expected_pred = y_pred_intensity * pred_prob
        expected_true = y_true_intensity # Assume y_true_intensity already reflects probability
        # Weight by true probability to focus on actual peaks
        weights = (y_true_prob > 0).float() * 10.0 + 1.0
        # L1 normalize for cosine similarity
        expected_pred_norm = F.normalize(expected_pred * weights, p=1, dim=1)
        expected_true_norm = F.normalize(expected_true * weights, p=1, dim=1)
        loss = (1.0 - F.cosine_similarity(expected_pred_norm, expected_true_norm, dim=1)).mean()
        return loss

    device = y_pred_intensity.device
    batch_size = y_pred_intensity.shape[0]
    max_mz = y_pred_intensity.shape[1]

    mz_coords = mz_bins.to(device).float().reshape(1, -1)
    # Cost matrix M: Absolute difference for p=1
    M = torch.abs(mz_coords.t() - mz_coords)
    M /= M.max() + EPS # Normalize

    loss_total = 0.0
    valid_samples = 0

    pred_prob = torch.sigmoid(y_pred_prob_logits)
    pred_dist = F.relu(y_pred_intensity) * pred_prob # Use relu intensity
    # L1 Normalize distributions
    pred_dist = pred_dist / (pred_dist.sum(dim=1, keepdim=True) + EPS)

    # Use raw intensity for true distribution, masked by true_prob
    true_dist_unnorm = y_true_intensity * y_true_prob
    true_dist = true_dist_unnorm / (true_dist_unnorm.sum(dim=1, keepdim=True) + EPS)

    M_np = M.cpu().numpy().astype(np.float64) # Ensure float64 for POT

    for i in range(batch_size):
        pred_sample = pred_dist[i].detach().cpu().numpy().astype(np.float64)
        true_sample = true_dist[i].detach().cpu().numpy().astype(np.float64)

        # Ensure non-negative and sum slightly > 0
        pred_sample = np.maximum(pred_sample, 0)
        true_sample = np.maximum(true_sample, 0)
        pred_sample /= (pred_sample.sum() + EPS)
        true_sample /= (true_sample.sum() + EPS)

        if np.sum(pred_sample) > EPS and np.sum(true_sample) > EPS:
             try:
                 # Use emd2 for exact EMD (Wasserstein-1 distance)
                 # Sinkhorn provides an approximation, emd2 is often better for 1D
                 # transport_plan = ot.emd(pred_sample, true_sample, M_np) # Gives plan
                 W_dist = ot.emd2(pred_sample, true_sample, M_np) # Gives distance
                 loss_total += W_dist
                 valid_samples += 1
             except Exception as e:
                 # logger.warning(f"EMD calculation failed for sample {i}: {e}")
                 # Fallback: cosine loss on this sample
                 cos_loss = (1.0 - F.cosine_similarity(pred_dist[i:i+1], true_dist[i:i+1])).item()
                 loss_total += cos_loss
                 valid_samples += 1
        elif np.sum(true_sample) > EPS: # Penalty if prediction is empty but target is not
             loss_total += 1.0
             valid_samples += 1

    return loss_total / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=device)

# Combined Loss Function
class RadicalNetLoss(nn.Module):
    def __init__(self, mz_dim=MAX_MZ, num_frags=NUM_FRAGS,
                 w_intensity=0.1, w_prob=0.3, w_wasserstein=0.5, w_fragment=0.1,
                 wasserstein_reg=0.05, important_mz_weight=3.0, prob_pos_weight=5.0):
        super().__init__()
        self.mz_dim = mz_dim; self.num_frags = num_frags
        self.w_intensity = w_intensity; self.w_prob = w_prob; self.w_wasserstein = w_wasserstein
        self.w_fragment = w_fragment; self.wasserstein_reg = wasserstein_reg
        self.important_mz = IMPORTANT_MZ; self.important_mz_weight = important_mz_weight
        self.prob_pos_weight = prob_pos_weight

        # Use pos_weight in BCEWithLogitsLoss for probability
        self.bce_prob = nn.BCEWithLogitsLoss(reduction='none') # Apply weights manually
        self.mse_intensity = nn.MSELoss(reduction='none') # Apply weights manually
        self.bce_fragment = nn.BCEWithLogitsLoss()
        self.mz_bins = torch.arange(self.mz_dim, dtype=torch.float32)

    def forward(self, pred_output, batch_data):
        pred_intensities = pred_output['pred_intensities']
        pred_probs_logits = pred_output['pred_probs_logits']
        pred_fragments_logits = pred_output['pred_fragments_logits']
        true_intensities = batch_data['spec_intensity'] # Processed intensity
        true_probs = batch_data['spec_prob']           # Probability target (0/1)
        true_fragments = batch_data['fragment_pattern']
        B, M = pred_intensities.shape
        device = pred_intensities.device

        total_loss = 0.0
        loss_dict = {}

        # 1. Probability Loss (Weighted BCE)
        prob_loss_unweighted = self.bce_prob(pred_probs_logits, true_probs)
        pos_weight_tensor = torch.tensor([self.prob_pos_weight], device=device)
        prob_weights = torch.where(true_probs > 0.5, pos_weight_tensor, torch.tensor([1.0], device=device))
        important_mask = torch.zeros(M, device=device)
        valid_mz = [mz for mz in self.important_mz if mz < M]
        if valid_mz: important_mask[valid_mz] = self.important_mz_weight - 1.0 # Weight is 1 + mask
        prob_weights = prob_weights * (1.0 + important_mask.unsqueeze(0))
        prob_loss = (prob_loss_unweighted * prob_weights).mean()
        total_loss += self.w_prob * prob_loss
        loss_dict['prob_loss'] = prob_loss.item()

        # 2. Intensity Loss (Weighted MSE - only on true peaks)
        intensity_loss_unweighted = self.mse_intensity(pred_intensities, true_intensities)
        # Weight by true probability and important MZ
        intensity_weights = (true_probs > 0.5).float() # Only consider loss where true peak exists
        intensity_weights = intensity_weights * (1.0 + important_mask.unsqueeze(0)) # Boost important mz
        intensity_loss = (intensity_loss_unweighted * intensity_weights).sum() / (intensity_weights.sum() + EPS) # Mean over weighted elements
        total_loss += self.w_intensity * intensity_loss
        loss_dict['intensity_loss'] = intensity_loss.item()

        # 3. Wasserstein Loss
        if self.w_wasserstein > 0:
            # Intensity inputs should ideally be L1 normalized for Wasserstein
            # Here we pass the processed intensities directly, assuming process_spec handled normalization
             ws_loss = wasserstein_loss(pred_intensities, pred_probs_logits,
                                        true_intensities, true_probs, # Pass processed true intensity
                                        self.mz_bins, reg=self.wasserstein_reg, p=1) # Use p=1 (EMD)
             total_loss += self.w_wasserstein * ws_loss
             loss_dict['wasserstein_loss'] = ws_loss.item()

        # 4. Fragment Loss
        if self.w_fragment > 0:
            # Ensure target is float
            fragment_loss = self.bce_fragment(pred_fragments_logits, true_fragments.float())
            total_loss += self.w_fragment * fragment_loss
            loss_dict['fragment_loss'] = fragment_loss.item()

        # Handle potential NaN/Inf in total_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"NaN or Inf detected in total loss! Loss components: {loss_dict}")
            # Set loss to a large finite number to prevent crash, but signal error
            total_loss = torch.tensor(1e5, device=device, requires_grad=True) # Needs grad for backward
            loss_dict['total_loss'] = total_loss.item() # Log the error value
        else:
            loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

# Evaluation Metrics
def radicalnet_cosine_similarity_score(y_true_intensity, y_pred_intensity, y_pred_prob=None):
    """コサイン類似度 (予測確率を考慮可能)"""
    y_true_np = y_true_intensity.cpu().numpy()
    y_pred_np = y_pred_intensity.cpu().numpy()
    if y_pred_prob is not None:
        y_pred_np = y_pred_np * y_pred_prob.cpu().numpy() # Weight by probability

    y_true_np = np.nan_to_num(y_true_np)
    y_pred_np = np.nan_to_num(y_pred_np)
    valid_idx = np.where((np.linalg.norm(y_true_np, axis=1) > EPS) & (np.linalg.norm(y_pred_np, axis=1) > EPS))[0]
    if len(valid_idx) == 0: return 0.0
    sim = cosine_similarity(y_true_np[valid_idx], y_pred_np[valid_idx])
    return float(np.mean(np.diag(sim)))

def peak_matching_metrics(y_true_prob, y_pred_prob_logits, prob_threshold=0.5):
    """ピーク位置のリコール、プレシジョン、F1を計算"""
    y_true_peaks = (y_true_prob > 0.5).float()
    y_pred_prob = torch.sigmoid(y_pred_prob_logits)
    y_pred_peaks = (y_pred_prob > prob_threshold).float()

    true_positives = torch.sum(y_pred_peaks * y_true_peaks, dim=1)
    predicted_positives = torch.sum(y_pred_peaks, dim=1)
    actual_positives = torch.sum(y_true_peaks, dim=1)

    precision = (true_positives / (predicted_positives + EPS)).mean().item()
    recall = (true_positives / (actual_positives + EPS)).mean().item()
    f1 = 2 * (precision * recall) / (precision + recall + EPS)

    return {'peak_precision': precision, 'peak_recall': recall, 'peak_f1': f1}

###############################
# トレーニングと評価 (RadicalNet用)
###############################

def train_radicalnet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
                     eval_interval=1, patience=10, grad_clip=1.0, checkpoint_dir=CHECKPOINT_DIR):
    """RadicalNetモデルのトレーニングループ"""
    # --- Initialization ---
    start_epoch = 0; best_val_metric = -1.0; best_val_loss = float('inf'); early_stopping_counter = 0
    train_losses_hist, val_losses_hist = [], []
    val_metrics_hist = {'cosine_similarity': [], 'peak_f1': [], 'peak_precision': [], 'peak_recall': []} # Store more metrics

    # Checkpoint directory
    rad_checkpoint_dir = os.path.join(checkpoint_dir, "radicalnet_checkpoints")
    os.makedirs(rad_checkpoint_dir, exist_ok=True)

    # --- Load Checkpoint ---
    latest_checkpoint = None
    checkpoint_prefix = "radicalnet_checkpoint_epoch_"
    if os.path.exists(rad_checkpoint_dir):
        for file in os.listdir(rad_checkpoint_dir):
            if file.startswith(checkpoint_prefix) and file.endswith(".pth"):
                try:
                    epoch_num = int(file.split("_")[-1].split(".")[0])
                    current_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0]) if latest_checkpoint else -1
                    if epoch_num > current_epoch: latest_checkpoint = file
                except ValueError: continue

    if latest_checkpoint:
        checkpoint_path = os.path.join(rad_checkpoint_dir, latest_checkpoint)
        logger.info(f"チェックポイントを読み込み: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys: logger.warning(f"モデル読み込み時、不足しているキー: {missing_keys}")
            if unexpected_keys: logger.warning(f"モデル読み込み時、予期しないキー: {unexpected_keys}")

            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 for state in optimizer.state.values(): # Move optimizer state to device
                     for k, v in state.items():
                         if isinstance(v, torch.Tensor): state[k] = v.to(device)
            else: logger.warning("チェックポイントにオプティマイザ状態なし。初期状態から開始。")

            if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                 try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 except Exception as e: logger.warning(f"スケジューラ状態復元失敗: {e}")

            start_epoch = checkpoint.get('epoch', 0) + 1
            train_losses_hist = checkpoint.get('train_losses_hist', [])
            val_losses_hist = checkpoint.get('val_losses_hist', [])
            val_metrics_hist = checkpoint.get('val_metrics_hist', val_metrics_hist) # Use default if not found
            best_val_metric = checkpoint.get('best_val_metric', -1.0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            del checkpoint; aggressive_memory_cleanup()
            logger.info(f"エポック {start_epoch} からトレーニングを再開します。")
        except Exception as e:
            logger.error(f"チェックポイント読み込みエラー: {e}。最初からトレーニングを開始します。")
            start_epoch = 0 # Reset state

    scaler = GradScaler(enabled=torch.cuda.is_available())
    model = model.to(device)

    logger.info(f"トレーニング開始: 総エポック数 = {num_epochs}, 開始エポック = {start_epoch + 1}")
    memory_check_interval = max(1, len(train_loader) // 10) # Check memory 10 times per epoch

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0; batch_count = 0; epoch_loss_details = {}

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", position=0, leave=True)
        for batch_idx, batch_data in enumerate(train_pbar):
            if batch_data is None: continue # Skip empty batches from collate_fn
            if batch_idx % memory_check_interval == 0: aggressive_memory_cleanup(percent=85)

            # Move data to GPU within the loop
            batch_data_gpu = {}
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor): batch_data_gpu[k] = v.to(device, non_blocking=True)
                elif k == 'graph': batch_data_gpu[k] = v.to(device)
                else: batch_data_gpu[k] = v

            try:
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    pred_output = model(batch_data_gpu)
                    loss, loss_detail = criterion(pred_output, batch_data_gpu)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx}: NaN/Inf loss detected! Skipping.")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()

                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): scheduler.step()

                current_loss = loss.item()
                epoch_train_loss += current_loss
                batch_count += 1
                for k, v in loss_detail.items(): epoch_loss_details[k] = epoch_loss_details.get(k, 0.0) + v

                train_pbar.set_postfix({'loss': f"{current_loss:.4f}", 'avg': f"{epoch_train_loss/batch_count:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.1E}"})

                # Cleanup batch data from GPU
                del loss, pred_output, batch_data_gpu
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"トレーニングバッチ {batch_idx} でエラー発生: {e}")
                import traceback; traceback.print_exc()
                aggressive_memory_cleanup(force_sync=True, purge_cache=True)
                continue

        # --- Epoch End ---
        if batch_count > 0:
            avg_train_loss = epoch_train_loss / batch_count
            train_losses_hist.append(avg_train_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - 平均訓練損失: {avg_train_loss:.4f}")
            avg_loss_details = {k: v / batch_count for k, v in epoch_loss_details.items()}
            logger.info(f"  Loss Details: { {k: f'{v:.4f}' for k, v in avg_loss_details.items()} }")

            # --- Validation ---
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                aggressive_memory_cleanup()
                val_results = evaluate_radicalnet(model, val_loader, criterion, device, use_amp=torch.cuda.is_available())
                val_loss = val_results['loss']
                val_losses_hist.append(val_loss)
                # Store all validation metrics
                for key in val_metrics_hist.keys():
                    if key in val_results:
                        val_metrics_hist[key].append(val_results[key])
                    else: # Append default if metric missing
                        val_metrics_hist[key].append(0.0)

                logger.info(f"Epoch {epoch+1}/{num_epochs} - 検証損失: {val_loss:.4f}, Cosine: {val_results['cosine_similarity']:.4f}, Peak F1: {val_results['peak_f1']:.4f}")
                # logger.info(f"  Validation Metrics: {val_results}") # Display all metrics

                if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_results['peak_f1']) # Step using Peak F1

                # --- Check for Improvement (using Peak F1) ---
                current_metric = val_results['peak_f1']
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_model_path = os.path.join(rad_checkpoint_dir, 'radicalnet_best_model.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"*** 新しい最良モデル保存 (Epoch {epoch+1}): Peak F1 = {current_metric:.4f} ***")
                else:
                    early_stopping_counter += 1
                    logger.info(f"早期停止カウンター: {early_stopping_counter}/{patience}")

                if early_stopping_counter >= patience:
                    logger.info(f"早期停止: 検証メトリクスが {patience} 回連続で改善しなかったため、Epoch {epoch+1} で停止します。")
                    break

            # --- Save Epoch Checkpoint ---
            checkpoint_path = os.path.join(rad_checkpoint_dir, f"radicalnet_checkpoint_epoch_{epoch+1}.pth")
            save_dict = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_losses_hist': train_losses_hist, 'val_losses_hist': val_losses_hist,
                'val_metrics_hist': val_metrics_hist, 'best_val_metric': best_val_metric,
                'best_val_loss': best_val_loss, 'early_stopping_counter': early_stopping_counter
            }
            torch.save(save_dict, checkpoint_path)

            # Plot progress periodically
            if (epoch + 1) % 5 == 0:
                 plot_radicalnet_training_progress(train_losses_hist, val_losses_hist, val_metrics_hist, best_val_metric, rad_checkpoint_dir)
        else:
            logger.warning(f"Epoch {epoch+1}: トレーニング中に有効なバッチがありませんでした。")
            train_losses_hist.append(float('inf'))
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1: # Keep hists aligned
                 val_losses_hist.append(float('inf'))
                 for key in val_metrics_hist.keys(): val_metrics_hist[key].append(0.0)

    # Final plot
    plot_radicalnet_training_progress(train_losses_hist, val_losses_hist, val_metrics_hist, best_val_metric, rad_checkpoint_dir)
    return train_losses_hist, val_losses_hist, val_metrics_hist, best_val_metric

# Evaluation Function
def evaluate_radicalnet(model, data_loader, criterion, device, use_amp=False):
    """RadicalNetモデルの評価 (検証/テスト用)"""
    model.eval()
    total_loss = 0; batch_count = 0
    all_true_intensities, all_pred_intensities = [], []
    all_true_probs, all_pred_probs_logits = [], [] # Store logits for peak metrics
    all_loss_details = {}

    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc="評価中", leave=False)
        for batch_data in eval_pbar:
            if batch_data is None: continue
            batch_data_gpu = {}
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor): batch_data_gpu[k] = v.to(device, non_blocking=True)
                elif k == 'graph': batch_data_gpu[k] = v.to(device)
                else: batch_data_gpu[k] = v

            try:
                with autocast(device_type=device.type, enabled=use_amp):
                    pred_output = model(batch_data_gpu)
                    # Note: Don't calculate loss if criterion involves randomness or non-deterministic ops during eval
                    # loss, loss_detail = criterion(pred_output, batch_data_gpu)
                    # Simulate loss calculation if needed for logging, but don't use for backprop
                    loss_val, loss_detail_val = criterion(pred_output, batch_data_gpu) # Calculate loss for logging
                    loss = loss_val.item() # Get scalar value

                total_loss += loss
                batch_count += 1
                for k, v in loss_detail_val.items(): all_loss_details[k] = all_loss_details.get(k, 0.0) + v

                all_true_intensities.append(batch_data_gpu['spec_intensity'].cpu())
                all_pred_intensities.append(pred_output['pred_intensities'].cpu())
                all_true_probs.append(batch_data_gpu['spec_prob'].cpu())
                all_pred_probs_logits.append(pred_output['pred_probs_logits'].cpu()) # Store logits

                del pred_output, batch_data_gpu, loss_val, loss_detail_val
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"評価中にエラー発生: {e}")
                import traceback; traceback.print_exc()
                continue

    if batch_count == 0:
        logger.warning("評価中に有効なバッチがありませんでした。")
        return {'loss': float('inf'), 'cosine_similarity': 0.0, 'peak_precision': 0.0, 'peak_recall': 0.0, 'peak_f1': 0.0, 'loss_details': {}}

    avg_loss = total_loss / batch_count
    avg_loss_details = {k: v / batch_count for k, v in all_loss_details.items()}

    y_true_intensity_all = torch.cat(all_true_intensities, dim=0)
    y_pred_intensity_all = torch.cat(all_pred_intensities, dim=0)
    y_true_prob_all = torch.cat(all_true_probs, dim=0)
    y_pred_prob_logits_all = torch.cat(all_pred_probs_logits, dim=0)

    cosine_sim = radicalnet_cosine_similarity_score(y_true_intensity_all, y_pred_intensity_all, torch.sigmoid(y_pred_prob_logits_all))
    peak_metrics = peak_matching_metrics(y_true_prob_all, y_pred_prob_logits_all)

    results = {'loss': avg_loss, 'cosine_similarity': cosine_sim, **peak_metrics, 'loss_details': avg_loss_details}
    return results

# Test Evaluation Function (includes discrete conversion)
def eval_radicalnet_test(model, test_loader, device, use_amp=True, transform="log10over3"):
    """RadicalNetテスト評価 (離散化処理含む)"""
    model.to(device); model.eval()
    all_true_intensities_proc, all_pred_intensities_proc = [], []
    all_pred_probs, all_pred_intensities_discrete = [], []
    all_mol_ids = []
    all_true_probs = [] # For peak metrics

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="テスト中")
        for batch_data in test_pbar:
            if batch_data is None: continue
            batch_data_gpu = {}
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor): batch_data_gpu[k] = v.to(device, non_blocking=True)
                elif k == 'graph': batch_data_gpu[k] = v.to(device)
                else: batch_data_gpu[k] = v

            try:
                with autocast(device_type=device.type, enabled=use_amp):
                    pred_output = model(batch_data_gpu)

                pred_intensities_proc_cpu = pred_output['pred_intensities'].cpu()
                pred_probs_logits_cpu = pred_output['pred_probs_logits'].cpu()
                pred_probs_cpu = torch.sigmoid(pred_probs_logits_cpu)
                true_intensities_proc_cpu = batch_data_gpu['spec_intensity'].cpu()
                true_probs_cpu = batch_data_gpu['spec_prob'].cpu() # Get true probs

                all_true_intensities_proc.append(true_intensities_proc_cpu)
                all_pred_intensities_proc.append(pred_intensities_proc_cpu)
                all_pred_probs.append(pred_probs_cpu)
                all_true_probs.append(true_probs_cpu) # Store true probs
                if isinstance(batch_data['mol_id'], list): all_mol_ids.extend(batch_data['mol_id'])
                else: all_mol_ids.append(batch_data['mol_id']) # Handle single item case

                # Discrete conversion
                for i in range(pred_intensities_proc_cpu.shape[0]):
                    intensity_np = pred_intensities_proc_cpu[i].numpy()
                    prob_np = pred_probs_cpu[i].numpy()
                    discrete_pred = improved_hybrid_spectrum_conversion(intensity_np, prob_np, transform=transform)
                    all_pred_intensities_discrete.append(torch.from_numpy(discrete_pred).float())

                del pred_output, batch_data_gpu
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"テスト中にエラー発生: {e}")
                import traceback; traceback.print_exc()
                continue

    if not all_true_intensities_proc:
        logger.error("テスト結果がありません。")
        return None

    y_true_proc_all = torch.cat(all_true_intensities_proc, dim=0)
    y_pred_proc_all = torch.cat(all_pred_intensities_proc, dim=0)
    y_prob_all = torch.cat(all_pred_probs, dim=0)
    y_true_prob_all = torch.cat(all_true_probs, dim=0) # Concatenate true probs
    y_pred_discrete_all = torch.stack(all_pred_intensities_discrete) if all_pred_intensities_discrete else torch.empty((0,MZ_DIM))

    # --- Calculate Metrics ---
    # Raw Cosine Sim (using processed intensities and predicted probs)
    raw_cosine_sim = radicalnet_cosine_similarity_score(y_true_proc_all, y_pred_proc_all, y_prob_all)

    # Discrete Cosine Sim (needs careful normalization)
    # 1. Unprocess true intensities
    y_true_unproc = unprocess_spec(y_true_proc_all, transform)
    # 2. Normalize both to relative intensity (0-100)
    max_true = torch.max(y_true_unproc, dim=1, keepdim=True)[0]
    y_true_rel = y_true_unproc / (max_true + EPS) * 100.0
    # Discrete prediction is already 0-100
    # 3. L2 normalize for cosine similarity calculation
    y_true_norm = F.normalize(y_true_rel, p=2, dim=1)
    y_pred_discrete_norm = F.normalize(y_pred_discrete_all, p=2, dim=1)
    discrete_cosine_sim = radicalnet_cosine_similarity_score(y_true_norm, y_pred_discrete_norm) # No prob needed here

    # Peak Metrics (use true probs and predicted logits)
    pred_prob_logits_all = torch.logit(y_prob_all + EPS) # Get logits back from probs
    peak_metrics = peak_matching_metrics(y_true_prob_all, pred_prob_logits_all)

    return {
        'cosine_similarity_raw': raw_cosine_sim,
        'cosine_similarity_discrete': discrete_cosine_sim,
        **peak_metrics,
        'y_true_processed': y_true_proc_all, # Processed true intensities
        'y_pred_processed': y_pred_proc_all, # Processed predicted intensities
        'y_pred_prob': y_prob_all,           # Predicted probabilities
        'y_pred_discrete': y_pred_discrete_all, # Discrete prediction (0-100)
        'mol_ids': all_mol_ids
    }

# Plotting Functions
def plot_radicalnet_training_progress(train_losses, val_losses, val_metrics, best_metric, save_dir):
    """RadicalNetの学習進捗を可視化"""
    if not train_losses: return
    epochs = range(1, len(train_losses) + 1)
    # Adjust val_epochs based on actual length of val data assuming eval_interval=1
    val_epochs = range(1, len(val_losses) + 1) if val_losses else []

    plt.figure(figsize=(18, 6))
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='.', alpha=0.8)
    if val_epochs: plt.plot(val_epochs, val_losses, label='Validation Loss', marker='.', alpha=0.8)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.legend(); plt.grid(True, alpha=0.5)
    # Cosine Similarity
    plt.subplot(1, 3, 2)
    if val_epochs and 'cosine_similarity' in val_metrics and len(val_metrics['cosine_similarity']) == len(val_epochs):
        plt.plot(val_epochs, val_metrics['cosine_similarity'], label='Validation Cosine Sim', marker='.', color='green', alpha=0.8)
        best_cos = max(val_metrics['cosine_similarity']) if val_metrics['cosine_similarity'] else 0
        plt.axhline(y=best_cos, color='r', linestyle='--', label=f'Best Cos: {best_cos:.4f}')
    plt.xlabel('Epoch'); plt.ylabel('Cosine Similarity'); plt.title('Validation Cosine Similarity'); plt.legend(); plt.grid(True, alpha=0.5); plt.ylim(bottom=0)
    # Peak F1
    plt.subplot(1, 3, 3)
    if val_epochs and 'peak_f1' in val_metrics and len(val_metrics['peak_f1']) == len(val_epochs):
        plt.plot(val_epochs, val_metrics['peak_f1'], label='Validation Peak F1', marker='.', color='purple', alpha=0.8)
        plt.axhline(y=best_metric, color='r', linestyle='--', label=f'Best F1: {best_metric:.4f}') # best_metric from training
    plt.xlabel('Epoch'); plt.ylabel('Peak F1 Score'); plt.title('Validation Peak F1 Score'); plt.legend(); plt.grid(True, alpha=0.5); plt.ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'radicalnet_learning_curves.png')
    plt.savefig(save_path); plt.close()
    # logger.info(f"学習曲線を保存しました: {save_path}") # Suppress log spam

def visualize_radicalnet_results(test_results, num_samples=10, transform="log10over3", save_dir="."):
    """RadicalNetの予測結果を可視化 (真値はunprocess, 予測はdiscrete)"""
    if not test_results: logger.error("可視化するテスト結果がありません。"); return
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(16, num_samples * 4))
    indices = np.random.choice(len(test_results['mol_ids']), min(num_samples, len(test_results['mol_ids'])), replace=False)
    y_true_proc_all = test_results['y_true_processed']
    y_pred_discrete_all = test_results['y_pred_discrete']

    for i, idx in enumerate(indices):
        mol_id = test_results['mol_ids'][idx]
        true_spec_proc = y_true_proc_all[idx]
        pred_discrete_spec = y_pred_discrete_all[idx].numpy()

        # Unprocess true spectrum for display
        try:
            true_spec_unproc = unprocess_spec(true_spec_proc.unsqueeze(0), transform).squeeze(0).numpy()
            max_true = np.max(true_spec_unproc)
            true_spec_display = true_spec_unproc / (max_true + EPS) * 100.0 if max_true > 0 else np.zeros_like(true_spec_unproc)
        except Exception as e:
            logger.warning(f"True spectrum unprocessing failed for ID {mol_id}: {e}")
            true_spec_display = np.zeros(MZ_DIM) # Error case

        # Calculate similarity for display (between relative 0-100 scales)
        sim_score = cosine_similarity(np.nan_to_num(true_spec_display.reshape(1, -1)),
                                      np.nan_to_num(pred_discrete_spec.reshape(1, -1)))[0, 0]

        # Plot Measured Spectrum (0-100 Relative Intensity)
        ax1 = plt.subplot(num_samples, 2, 2 * i + 1)
        mz_axis = np.arange(len(true_spec_display))
        peaks_true = mz_axis[true_spec_display > 0.1]
        intensities_true = true_spec_display[peaks_true]
        if len(peaks_true) > 0: ax1.vlines(peaks_true, 0, intensities_true, color='blue', linewidth=1)
        ax1.set_title(f"Measured Spectrum - ID: {mol_id}"); ax1.set_xlabel("m/z"); ax1.set_ylabel("Relative Intensity (%)")
        ax1.set_ylim(0, 110); ax1.grid(True, alpha=0.5)

        # Plot Predicted Discrete Spectrum (0-100 Relative Intensity)
        ax2 = plt.subplot(num_samples, 2, 2 * i + 2)
        peaks_pred = mz_axis[pred_discrete_spec > 0.1]
        intensities_pred = pred_discrete_spec[peaks_pred]
        if len(peaks_pred) > 0: ax2.vlines(peaks_pred, 0, intensities_pred, color='green', linewidth=1)
        ax2.set_title(f"RadicalNet Predicted (Discrete) - Sim: {sim_score:.4f}"); ax2.set_xlabel("m/z"); ax2.set_ylabel("Relative Intensity (%)")
        ax2.set_ylim(0, 110); ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'radicalnet_spectrum_comparison.png')
    plt.savefig(save_path); plt.close()
    logger.info(f"予測結果の可視化を保存しました: {save_path}")


###############################
# メイン関数 (RadicalNet用)
###############################
def main_radicalnet():
    logger.info("============= RadicalNet 質量スペクトル予測モデルの実行開始 =============")
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available(): logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else: logger.warning("CUDA 利用不可。CPUで実行します。")

    # --- データ準備 ---
    logger.info("MSPファイル(生強度)を解析中...")
    msp_data = parse_msp_file_raw(MSP_FILE_PATH, cache_dir=CACHE_DIR)
    logger.info(f"MSPファイルから{len(msp_data)}個の化合物データ読み込み完了")

    # 利用可能なMOLファイルID取得 (キャッシュ利用)
    mol_id_cache_file = os.path.join(CACHE_DIR, "valid_mol_ids_all.pkl") # Use a general name
    if os.path.exists(mol_id_cache_file):
        logger.info(f"キャッシュから全mol_idsを読み込み中: {mol_id_cache_file}")
        with open(mol_id_cache_file, 'rb') as f: mol_ids_all = pickle.load(f)
    else:
        mol_ids_all = []
        logger.info("MOLファイルリストをスキャン中...")
        for filename in tqdm(os.listdir(MOL_FILES_PATH), desc="MOLファイルスキャン"):
             if filename.startswith("ID") and filename.endswith(".MOL"):
                try: mol_ids_all.append(int(filename[2:-4]))
                except: continue
        logger.info(f"全mol_idsをキャッシュに保存中: {mol_id_cache_file}")
        with open(mol_id_cache_file, 'wb') as f: pickle.dump(mol_ids_all, f)

    mol_ids = [mid for mid in mol_ids_all if mid in msp_data] # Filter by spectrum availability
    logger.info(f"MOL/MSPデータが利用可能な化合物: {len(mol_ids)}個")
    if not mol_ids: logger.error("処理可能なデータなし。終了します。"); return

    # --- データ分割 ---
    train_ids, test_ids = train_test_split(mol_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    logger.info(f"データ分割: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # --- ハイパーパラメータ ---
    transform = "log10over3"
    normalization = "l1"
    use_3d_coords = False
    batch_size = 16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 20e9 else 8
    num_workers = 0 # Set to 0 for stability with RDKit and caching
    hidden_channels = 128; n_gnn_layers = 4; n_transformer_layers = 2; heads = 4; dropout = 0.2
    learning_rate = 1e-4; weight_decay = 1e-6
    num_epochs = 30; patience = 7; eval_interval = 1; grad_clip = 1.0
    # Loss weights
    loss_weights = {'w_intensity': 0.1, 'w_prob': 0.3, 'w_wasserstein': 0.5, 'w_fragment': 0.1}

    # --- データセット & データローダー ---
    logger.info("データセットを作成中...")
    train_dataset = RadicalNetMoleculeDataset(train_ids, MOL_FILES_PATH, msp_data, transform, normalization, augment=True, cache_dir=CACHE_DIR, use_3d=use_3d_coords)
    val_dataset = RadicalNetMoleculeDataset(val_ids, MOL_FILES_PATH, msp_data, transform, normalization, augment=False, cache_dir=CACHE_DIR, use_3d=use_3d_coords)
    test_dataset = RadicalNetMoleculeDataset(test_ids, MOL_FILES_PATH, msp_data, transform, normalization, augment=False, cache_dir=CACHE_DIR, use_3d=use_3d_coords)
    logger.info(f"データセット作成完了: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    if len(train_dataset) == 0: logger.error("訓練データがありません。"); return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=radicalnet_collate_fn, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, collate_fn=radicalnet_collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, collate_fn=radicalnet_collate_fn, num_workers=num_workers, pin_memory=True)

    # --- モデル初期化 ---
    # Get feature dims from dataset
    try:
        sample_graph = train_dataset[0] # Get a sample graph
        node_features = sample_graph.x.shape[1]
        edge_features = sample_graph.edge_attr.shape[1]
        logger.info(f"特徴量次元: Node={node_features}, Edge={edge_features}")
    except Exception as e:
        logger.error(f"データセットからの特徴量次元取得失敗: {e}. デフォルト値を使用します。")
        node_features = TOTAL_ATOM_FEATURES; edge_features = TOTAL_BOND_FEATURES

    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RadicalNetMS(node_features, edge_features, hidden_channels, MZ_DIM, NUM_FRAGS,
                         prec_mass_offset=10, dropout=dropout, n_gnn_layers=n_gnn_layers,
                         n_transformer_layers=n_transformer_layers, heads=heads).to(device)
    logger.info(f"RadicalNet モデル初期化完了 (パラメータ数: {sum(p.numel() for p in model.parameters()):,})")

    # --- 損失関数、オプティマイザー、スケジューラー ---
    criterion = RadicalNetLoss(**loss_weights) # Pass weights dictionary
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Scheduler: OneCycleLR or ReduceLROnPlateau
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate*10, steps_per_epoch=len(train_loader), epochs=num_epochs, pct_start=0.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience-2, verbose=True) # Monitor peak F1

    # --- トレーニング実行 ---
    logger.info("RadicalNet モデルのトレーニングを開始します...")
    rad_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "radicalnet_checkpoints") # Specific dir for this model
    train_losses, val_losses, val_metrics, best_val_metric = train_radicalnet(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
        eval_interval, patience, grad_clip, rad_checkpoint_dir
    )
    logger.info(f"トレーニング完了！ 最良検証メトリクス (Peak F1): {best_val_metric:.4f}")

    # --- テスト評価 ---
    logger.info("最良モデルを読み込んでテストデータで評価します...")
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    try:
        best_model_path = os.path.join(rad_checkpoint_dir, 'radicalnet_best_model.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"最良モデルを読み込みました: {best_model_path}")

        test_results = eval_radicalnet_test(model, test_loader, device, use_amp=torch.cuda.is_available(), transform=transform)
        if test_results:
            logger.info(f"テスト結果:")
            logger.info(f"  Cosine Sim (Raw):   {test_results['cosine_similarity_raw']:.4f}")
            logger.info(f"  Cosine Sim (Discrete):{test_results['cosine_similarity_discrete']:.4f}")
            logger.info(f"  Peak Precision:     {test_results['peak_precision']:.4f}")
            logger.info(f"  Peak Recall:        {test_results['peak_recall']:.4f}")
            logger.info(f"  Peak F1 Score:      {test_results['peak_f1']:.4f}")
            visualize_radicalnet_results(test_results, num_samples=10, transform=transform, save_dir=".")
        else: logger.error("テスト評価中にエラーが発生しました。")
    except FileNotFoundError: logger.error(f"最良モデルファイルが見つかりません: {best_model_path}")
    except Exception as e: logger.error(f"テスト評価または結果処理中にエラー: {e}"); import traceback; traceback.print_exc()

    logger.info("============= RadicalNet 質量スペクトル予測モデルの実行終了 =============")

if __name__ == "__main__":
    main_radicalnet()