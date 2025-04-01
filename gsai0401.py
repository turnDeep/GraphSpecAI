# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
# RDKitの警告を抑制
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Transformersライブラリのインポート (ChemBERTa用)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

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
    print("Warning: POT library not found. Wasserstein loss will use a fallback (weighted MSE). Install with: pip install POT")
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
                    if hasattr(obj, 'smiles_cache') and isinstance(obj.smiles_cache, dict):
                        obj.smiles_cache.clear()
                        logger.info(f"{obj_name}のSMILESキャッシュをクリア")
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
CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1" # ChemBERTaモデル名

# ディレクトリの作成
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # チェックポイント用ディレクトリも作成

# 最大m/z値の設定
MAX_MZ = 2000
MZ_DIM = MAX_MZ # 出力次元

# MorganフィンガープリントのパラメータとMORGAN_DIMを定義
MORGAN_RADIUS = 2
MORGAN_DIM = 2048  # 一般的に使用される次元数

# エフェメラル値
EPS = np.finfo(np.float32).eps

###############################
# データ処理関連の関数
###############################

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

# 前駆体質量によるマスキング（双方向予測用）
def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    """前駆体質量によるマスキング"""
    device = raw_prediction.device
    max_idx = raw_prediction.shape[1]
    if prec_mass_idx.dtype != torch.long: prec_mass_idx = prec_mass_idx.long()
    # 範囲外アクセスを防ぐためクリップ
    prec_mass_idx = torch.clamp(prec_mass_idx, max=max_idx-1, min=0)
    idx = torch.arange(max_idx, device=device)
    mask = (idx.unsqueeze(0) <= (prec_mass_idx.unsqueeze(1) + prec_mass_offset)).float()
    return mask * raw_prediction + (1. - mask) * mask_value

# 双方向予測用の関数
def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    """予測を反転する（双方向予測用）"""
    device = raw_prediction.device
    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    
    # prec_mass_idxのデータ型を確認し調整
    if prec_mass_idx.dtype != torch.long:
        prec_mass_idx = prec_mass_idx.long()
    
    # 範囲外の値をクリップ
    prec_mass_idx = torch.clamp(prec_mass_idx, max=max_idx-1, min=0)
    
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

# SMILES取得関数
def get_smiles_from_mol_file(mol_file):
    """MOLファイルからSMILES文字列を取得"""
    try:
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        if mol is None:
            return None
        
        # 基本的なサニタイズを試みる
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
        except:
            # 部分的なサニタイズを試みる
            for atom in mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol, 
                          sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                    Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                    Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                    Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                          catchErrors=True)
        
        # SMILESに変換
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        # logger.warning(f"SMILES変換エラー: {e}")
        return None

# Morganフィンガープリント生成関数
def get_morgan_fingerprint(mol, radius=MORGAN_RADIUS, n_bits=MORGAN_DIM):
    """RDKitのMolオブジェクトからMorganフィンガープリントを計算"""
    try:
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        # numpyアレイに変換
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(morgan_fp, arr)
        return arr.astype(np.float32)
    except Exception as e:
        # logger.warning(f"Morganフィンガープリント計算エラー: {e}")
        return np.zeros(n_bits, dtype=np.float32)

###############################
# ChemBERTa-MS モデル (ChemBERTaベース)
###############################

class ChemBERTaForMassSpec(nn.Module):
    """
    ChemBERTaをベースにした質量スペクトル予測モデル
    - SMILES文字列からChemBERTaで特徴抽出
    - ピーク存在確率と強度の分離予測
    - 双方向予測機能
    """
    def __init__(self, out_channels, num_fragments=MORGAN_DIM,
                 pretrained_model_name=CHEMBERTA_MODEL_NAME, hidden_dim=768, dropout=0.2,
                 prec_mass_offset=10, bidirectional=True):
        super(ChemBERTaForMassSpec, self).__init__()
        
        self.prec_mass_offset = prec_mass_offset
        self.out_channels = out_channels # = MAX_MZ
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # ChemBERTaモデルの読み込み
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)
            self.bert_model = RobertaModel.from_pretrained(pretrained_model_name)
            logger.info(f"事前学習済みChemBERTaモデルを読み込みました: {pretrained_model_name}")
        except Exception as e:
            logger.warning(f"事前学習済みモデルの読み込みに失敗: {e}. 初期化モデルを使用します。")
            # 初期化モードでのフォールバック
            config = RobertaConfig.from_pretrained(pretrained_model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)
            self.bert_model = RobertaModel(config)
            
        # モデル次元の取得
        self.hidden_dim = self.bert_model.config.hidden_size
        
        # Spectrum Prediction Head
        combined_dim = self.hidden_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim * 2), 
            nn.LayerNorm(self.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 双方向予測用
        if bidirectional:
            self.forw_prob_head = nn.Linear(self.hidden_dim * 2, out_channels)  # ピーク確率ロジット (順方向)
            self.forw_intensity_head = nn.Linear(self.hidden_dim * 2, out_channels)  # ピーク強度 (順方向)
            
            self.rev_prob_head = nn.Linear(self.hidden_dim * 2, out_channels)   # ピーク確率ロジット (逆方向)
            self.rev_intensity_head = nn.Linear(self.hidden_dim * 2, out_channels)   # ピーク強度 (逆方向)
            
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, out_channels),
                nn.Sigmoid()
            )
        else:
            # 通常の出力ヘッド
            self.prob_head = nn.Linear(self.hidden_dim * 2, out_channels)       # ピーク確率ロジット
            self.intensity_head = nn.Linear(self.hidden_dim * 2, out_channels)  # ピーク強度

        # MorganフィンガープリントHead (フラグメントパターン)
        self.morgan_pred_head = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim // 2), 
            nn.LeakyReLU(),
            nn.Dropout(0.2), 
            nn.Linear(self.hidden_dim // 2, num_fragments)
        )
        
        self._init_weights()

    def _init_weights(self):
        """出力層の重みを初期化"""
        for module in [self.output_mlp, self.morgan_pred_head]:
            if isinstance(module, nn.Sequential):
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None: 
                            nn.init.zeros_(m.bias)
        
        # 双方向予測用ヘッドの初期化
        if self.bidirectional:
            for module in [self.forw_prob_head, self.forw_intensity_head, 
                          self.rev_prob_head, self.rev_intensity_head, self.gate]:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            # 通常出力ヘッドの初期化
            for module in [self.prob_head, self.intensity_head]:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
    def forward(self, batch_data):
        device = next(self.parameters()).device
        
        # データの取り出し
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        prec_mz_bin = batch_data.get('prec_mz_bin')
        
        if prec_mz_bin is not None:
            prec_mz_bin = prec_mz_bin.to(device)
            
        # ChemBERTaによる特徴抽出
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [CLS]トークンの出力を使用 (分子全体の表現)
        molecule_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 出力層
        output_features = self.output_mlp(molecule_embedding)
        
        # 双方向予測
        if self.bidirectional and prec_mz_bin is not None:
            # 順方向の予測
            forw_prob_logits = self.forw_prob_head(output_features)
            forw_intensities = F.relu(self.forw_intensity_head(output_features))
            
            # 逆方向の予測とシフト
            rev_prob_logits = self.rev_prob_head(output_features)
            rev_intensities = F.relu(self.rev_intensity_head(output_features))
            
            # 逆方向予測のマススペクトル全体の反転と前駆体質量考慮
            rev_prob_logits_reversed = reverse_prediction(
                rev_prob_logits, prec_mz_bin, self.prec_mass_offset)
            rev_intensities_reversed = reverse_prediction(
                rev_intensities, prec_mz_bin, self.prec_mass_offset)
            
            # ゲート機構で重み付け
            gate_weights = self.gate(output_features)
            
            # 最終出力
            pred_probs_logits = forw_prob_logits * gate_weights + rev_prob_logits_reversed * (1 - gate_weights)
            pred_intensities = forw_intensities * gate_weights + rev_intensities_reversed * (1 - gate_weights)
            
        else:
            # 通常の単方向予測
            pred_probs_logits = self.prob_head(output_features)
            pred_intensities = F.relu(self.intensity_head(output_features))
            
        # 前駆体質量でマスキング
        if prec_mz_bin is not None:
            pred_intensities = mask_prediction_by_mass(
                pred_intensities, prec_mz_bin, self.prec_mass_offset, mask_value=0.0)
            
            # 確率もマスク (大きな負の値で確実に0に近くなるようにする)
            prob_mask = (torch.arange(self.out_channels, device=device).unsqueeze(0) <= 
                         (prec_mz_bin.unsqueeze(1) + self.prec_mass_offset)).float()
            pred_probs_logits = pred_probs_logits * prob_mask + (1. - prob_mask) * (-1e9)
        
        # Morganフィンガープリント予測
        pred_morgan_logits = self.morgan_pred_head(molecule_embedding)
        
        # 出力辞書
        output = {
            "pred_intensities": pred_intensities,
            "pred_probs_logits": pred_probs_logits,
            "pred_morgan_logits": pred_morgan_logits  # MACCSからMorganへ変更
        }
        
        if self.bidirectional and prec_mz_bin is not None:
            # デバッグや分析用に双方向予測の詳細も含める
            output.update({
                "forw_intensities": forw_intensities,
                "forw_prob_logits": forw_prob_logits,
                "rev_intensities_reversed": rev_intensities_reversed,
                "rev_prob_logits_reversed": rev_prob_logits_reversed,
                "gate_weights": gate_weights
            })
            
        return output

###############################
# データセット & データローダー (ChemBERTa用に修正)
###############################

class ChemBERTaMoleculeDataset(Dataset):
    def __init__(self, mol_ids, mol_files_path, msp_data, tokenizer,
                transform="log10over3", normalization="l1", max_length=512,
                augment=False, cache_dir=CACHE_DIR):
        self.mol_ids = list(mol_ids)  # Ensure it's a list
        self.mol_files_path = mol_files_path
        self.msp_data = msp_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.normalization = normalization
        self.max_length = max_length
        self.augment = augment
        self.cache_dir = cache_dir
        self.smiles_cache = {}  # キャッシュ
        self.morgan_fingerprints = {}  # Morganフィンガープリント
        self.valid_mol_ids = []
        
        self._preprocess_mol_ids()

    def _preprocess_mol_ids(self):
        """有効な分子IDとMorganフィンガープリントを前処理（キャッシュ利用）"""
        # Generate a hash based on the list of IDs for unique caching
        ids_hash = str(hash(tuple(sorted(self.mol_ids))))
        cache_file = os.path.join(self.cache_dir, f"chemberta_morgan_data_{ids_hash}.pkl")

        if os.path.exists(cache_file):
            logger.info(f"キャッシュから前処理データを読み込み中: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.valid_mol_ids = cached_data['valid_mol_ids']
                    self.morgan_fingerprints = cached_data['morgan_fingerprints']
                    # SMILESキャッシュも読み込む
                    if 'smiles_cache' in cached_data:
                        self.smiles_cache = cached_data['smiles_cache']
                logger.info(f"キャッシュ読み込み完了。有効ID数: {len(self.valid_mol_ids)}")
                return
            except Exception as e:
                logger.warning(f"キャッシュ読み込み失敗 ({e})。再計算します。")
                try:
                    os.remove(cache_file)
                except OSError:
                    pass

        logger.info("分子データの前処理を開始します...")
        valid_ids_temp = []
        morgan_fingerprints_temp = {}
        smiles_cache_temp = {}
        mol_count = len(self.mol_ids)

        with tqdm(total=mol_count, desc="分子検証 & Morgan FP計算") as pbar:
            for mol_id in self.mol_ids:
                mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
                
                # スペクトルデータが無い場合はスキップ
                if mol_id not in self.msp_data:
                    pbar.update(1)
                    continue
                
                try:
                    # SMILESを取得
                    smiles = get_smiles_from_mol_file(mol_file)
                    if smiles is None or smiles == "":
                        pbar.update(1)
                        continue
                    
                    # RDKitのMolオブジェクトを生成
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        pbar.update(1)
                        continue
                    
                    # Morganフィンガープリントの計算
                    from rdkit.DataStructs import cDataStructs as DataStructs  # RDKitのDataStructsをインポート
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_DIM)
                    # numpyアレイに変換
                    morgan_array = np.zeros(MORGAN_DIM, dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
                    
                    # 有効なデータとしてキャッシュに保存
                    valid_ids_temp.append(mol_id)
                    morgan_fingerprints_temp[mol_id] = morgan_array
                    smiles_cache_temp[mol_id] = smiles
                
                except Exception as e:
                    # logger.warning(f"分子ID {mol_id} の処理中にエラー: {str(e)}")
                    pass
                
                finally:
                    pbar.update(1)
                    if pbar.n % 1000 == 0:
                        gc.collect()

        self.valid_mol_ids = valid_ids_temp
        self.morgan_fingerprints = morgan_fingerprints_temp
        self.smiles_cache = smiles_cache_temp

        logger.info(f"前処理結果をキャッシュに保存中: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'valid_mol_ids': self.valid_mol_ids,
                    'morgan_fingerprints': self.morgan_fingerprints,
                    'smiles_cache': self.smiles_cache
                }, f)
        except Exception as e:
            logger.error(f"キャッシュ保存失敗: {e}")
        
        logger.info(f"有効な分子: {len(self.valid_mol_ids)}個 / 全体: {mol_count}個")

    def _preprocess_spectrum(self, spectrum_array):
        """スペクトルを前処理し、強度とターゲット確率を生成"""
        spec_tensor = torch.FloatTensor(spectrum_array).unsqueeze(0)
        # 強度は process_spec で変換・正規化
        processed_intensity = process_spec(spec_tensor.clone(), self.transform, self.normalization)
        # ターゲット確率: 生強度が閾値(0.1% of max or abs 1.0)より大きい場合に1
        max_raw_val = torch.max(spec_tensor)
        threshold = torch.maximum(max_raw_val * 0.001, torch.tensor(1.0)) if max_raw_val > 0 else torch.tensor(1.0)
        target_prob = (spec_tensor > threshold).float()
        return processed_intensity.squeeze(0), target_prob.squeeze(0)

    def __len__(self):
        return len(self.valid_mol_ids)

    def __getitem__(self, idx):
        mol_id = self.valid_mol_ids[idx]
        
        # SMILESの取得（キャッシュから）
        smiles = self.smiles_cache.get(mol_id)
        if smiles is None:
            # キャッシュにない場合は計算 (通常ここには来ない)
            mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
            smiles = get_smiles_from_mol_file(mol_file)
            if smiles is not None:
                self.smiles_cache[mol_id] = smiles
                
        # トークン化
        encoding = self.tokenizer(
            smiles,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        # バッチ次元を削除
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # スペクトルの前処理
        raw_spectrum = self.msp_data.get(mol_id, np.zeros(MAX_MZ))
        processed_intensity, target_prob = self._preprocess_spectrum(raw_spectrum)
        
        # Morganフィンガープリント
        morgan_fp = torch.FloatTensor(self.morgan_fingerprints.get(mol_id, np.zeros(MORGAN_DIM)))
        
        # 前駆体 m/z 計算
        peaks = np.nonzero(raw_spectrum)[0]
        prec_mz = float(np.max(peaks)) if len(peaks) > 0 else 0.0
        prec_mz_bin = int(round(prec_mz))
        
        # Data Augmentation
        if self.augment and random.random() < 0.1:
            # Add some noise to input_ids (randomly mask a few tokens)
            mask_token_id = self.tokenizer.mask_token_id
            prob_mask = torch.full_like(input_ids, 0.05, dtype=torch.float)
            mask_indices = torch.bernoulli(prob_mask).bool()
            # Don't mask special tokens
            special_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, 
                             self.tokenizer.pad_token_id]
            for token_id in special_tokens:
                mask_indices = mask_indices & (input_ids != token_id)
            input_ids = input_ids.clone()
            input_ids[mask_indices] = mask_token_id
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'spec_intensity': processed_intensity,
            'spec_prob': target_prob,
            'morgan_fingerprint': morgan_fp,  # MACCSからMorganに変更
            'mol_id': mol_id,
            'prec_mz': prec_mz,
            'prec_mz_bin': prec_mz_bin,
            'smiles': smiles
        }

def chemberta_collate_fn(batch):
    """ChemBERTa用のカスタムCollate関数"""
    if not batch:
        return None
    
    # バッチ内のキーを取得
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key in ['input_ids', 'attention_mask', 'spec_intensity', 'spec_prob', 
                  'morgan_fingerprint', 'prec_mz', 'prec_mz_bin']:  # fragmentをmorganに変更
            # テンソル型のデータはスタック
            result[key] = torch.stack([item[key] for item in batch])
        elif key in ['mol_id', 'smiles']:
            # リスト型のデータはリストに集約
            result[key] = [item[key] for item in batch]
        else:
            # その他の型は最初の要素のみ使用
            result[key] = batch[0][key]
    
    return result

###############################
# 損失関数と類似度計算 (Wasserstein導入)
###############################

# Wasserstein Loss (既存コード流用)
def wasserstein_loss(y_pred_intensity, y_pred_prob_logits, y_true_intensity, y_true_prob, mz_bins, reg=0.05, p=1):
    """Wasserstein距離に基づく損失 (POT使用、p=1版)"""
    if not POT_AVAILABLE:
        # Fallback: Weighted MSE (確率で重み付けされたMSE)
        pred_prob = torch.sigmoid(y_pred_prob_logits)
        expected_pred = y_pred_intensity * pred_prob
        expected_true = y_true_intensity # y_true_intensity already reflects probability
        
        # Weight by true probability to focus on actual peaks
        weights = (y_true_prob > 0).float() * 10.0 + 1.0
        
        # Weighted MSE
        mse_loss = torch.mean(weights * (expected_pred - expected_true) ** 2)
        return mse_loss

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
                 W_dist = ot.emd2(pred_sample, true_sample, M_np) # Gives distance
                 loss_total += W_dist
                 valid_samples += 1
             except Exception as e:
                 # Fallback: MSE loss on this sample
                 mse_loss = torch.mean((pred_dist[i] - true_dist[i])**2).item()
                 loss_total += mse_loss
                 valid_samples += 1
        elif np.sum(true_sample) > EPS: # Penalty if prediction is empty but target is not
             loss_total += 1.0
             valid_samples += 1

    return loss_total / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=device)

# Combined Loss Function (重要なm/zのリストを使用しないように修正)
class ChemBERTaMSLoss(nn.Module):
    def __init__(self, mz_dim=MAX_MZ, num_fragments=MORGAN_DIM,
                 w_intensity=0.1, w_prob=0.3, w_wasserstein=0.5, w_morgan=0.1,
                 wasserstein_reg=0.05, prob_pos_weight=5.0):
        super().__init__()
        self.mz_dim = mz_dim
        self.num_fragments = num_fragments
        self.w_intensity = w_intensity
        self.w_prob = w_prob
        self.w_wasserstein = w_wasserstein
        self.w_morgan = w_morgan
        self.wasserstein_reg = wasserstein_reg
        self.prob_pos_weight = prob_pos_weight

        # Use pos_weight in BCEWithLogitsLoss for probability
        self.bce_prob = nn.BCEWithLogitsLoss(reduction='none') # Apply weights manually
        self.mse_intensity = nn.MSELoss(reduction='none') # Apply weights manually
        self.bce_morgan = nn.BCEWithLogitsLoss()
        self.mz_bins = torch.arange(self.mz_dim, dtype=torch.float32)

    def forward(self, pred_output, batch_data):
        pred_intensities = pred_output['pred_intensities']
        pred_probs_logits = pred_output['pred_probs_logits']
        pred_morgan_logits = pred_output['pred_morgan_logits']  # MACCSからMorganへ変更
        true_intensities = batch_data['spec_intensity'] # Processed intensity
        true_probs = batch_data['spec_prob']           # Probability target (0/1)
        true_morgan = batch_data['morgan_fingerprint']  # MACCSからMorganへ変更
        B, M = pred_intensities.shape
        device = pred_intensities.device

        total_loss = 0.0
        loss_dict = {}

        # 1. Probability Loss (Weighted BCE)
        prob_loss_unweighted = self.bce_prob(pred_probs_logits, true_probs)
        # 重要なm/zのリストを使用せず、単純に正例に高い重みを付ける
        prob_weights = torch.where(true_probs > 0.5, 
                                   torch.tensor([self.prob_pos_weight], device=device),
                                   torch.tensor([1.0], device=device))
        prob_loss = (prob_loss_unweighted * prob_weights).mean()
        total_loss += self.w_prob * prob_loss
        loss_dict['prob_loss'] = prob_loss.item()

        # 2. Intensity Loss (Weighted MSE - only on true peaks)
        intensity_loss_unweighted = self.mse_intensity(pred_intensities, true_intensities)
        # 真のピークの位置だけ重みを付ける (重要なm/zのリストは使用しない)
        intensity_weights = (true_probs > 0.5).float() # Only consider loss where true peak exists
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

        # 4. Morganフィンガープリント Loss (MACCSからMorganへ変更)
        if self.w_morgan > 0:
            # Ensure target is float
            morgan_loss = self.bce_morgan(pred_morgan_logits, true_morgan.float())
            total_loss += self.w_morgan * morgan_loss
            loss_dict['morgan_loss'] = morgan_loss.item()

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

def wasserstein_distance_metric(y_true_intensity, y_true_prob, y_pred_intensity, y_pred_prob, mz_dim=MAX_MZ):
    """Wasserstein距離を評価メトリクスとして計算"""
    if not POT_AVAILABLE:
        # POTライブラリが無い場合はダミー値を返す
        return 0.0
    
    mz_bins = torch.arange(mz_dim, dtype=torch.float32)
    device = y_true_intensity.device
    
    mz_coords = mz_bins.to(device).float().reshape(1, -1)
    # コスト行列: 絶対差分
    M = torch.abs(mz_coords.t() - mz_coords)
    M /= M.max() + EPS  # 正規化
    
    # 予測分布の作成（確率でウェイト）
    pred_dist = F.relu(y_pred_intensity) * torch.sigmoid(y_pred_prob)
    pred_dist = pred_dist / (pred_dist.sum(dim=1, keepdim=True) + EPS)
    
    # 真の分布
    true_dist = y_true_intensity * y_true_prob
    true_dist = true_dist / (true_dist.sum(dim=1, keepdim=True) + EPS)
    
    batch_size = y_true_intensity.shape[0]
    M_np = M.cpu().numpy().astype(np.float64)
    
    total_dist = 0.0
    valid_samples = 0
    
    for i in range(batch_size):
        pred_sample = pred_dist[i].detach().cpu().numpy().astype(np.float64)
        true_sample = true_dist[i].detach().cpu().numpy().astype(np.float64)
        
        pred_sample = np.maximum(pred_sample, 0)
        true_sample = np.maximum(true_sample, 0)
        
        pred_sample /= (pred_sample.sum() + EPS)
        true_sample /= (true_sample.sum() + EPS)
        
        if np.sum(pred_sample) > EPS and np.sum(true_sample) > EPS:
            try:
                W_dist = ot.emd2(pred_sample, true_sample, M_np)
                total_dist += W_dist
                valid_samples += 1
            except Exception:
                pass
                
    return total_dist / valid_samples if valid_samples > 0 else 1.0  # 距離なので大きいほど悪い

###############################
# トレーニングと評価 (段階的トレーニング対応)
###############################

def evaluate_model(model, data_loader, criterion, device, use_amp=False):
    """ChemBERTa-MSモデルの評価 (検証/テスト用)"""
    model.eval()
    total_loss = 0; batch_count = 0
    all_true_probs, all_pred_probs_logits = [], [] # Store logits for peak metrics
    all_loss_details = {}

    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc="評価中", leave=False)
        for batch_data in eval_pbar:
            if batch_data is None: continue
            batch_data_gpu = {}
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor): batch_data_gpu[k] = v.to(device, non_blocking=True)
                else: batch_data_gpu[k] = v

            try:
                with autocast(device_type=device.type, enabled=use_amp):
                    pred_output = model(batch_data_gpu)
                    loss_val, loss_detail_val = criterion(pred_output, batch_data_gpu) # Calculate loss for logging
                    loss = loss_val.item() # Get scalar value

                total_loss += loss
                batch_count += 1
                for k, v in loss_detail_val.items(): all_loss_details[k] = all_loss_details.get(k, 0.0) + v

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
        return {'loss': float('inf'), 'peak_precision': 0.0, 'peak_recall': 0.0, 'peak_f1': 0.0, 'loss_details': {}}

    avg_loss = total_loss / batch_count
    avg_loss_details = {k: v / batch_count for k, v in all_loss_details.items()}

    y_true_prob_all = torch.cat(all_true_probs, dim=0)
    y_pred_prob_logits_all = torch.cat(all_pred_probs_logits, dim=0)

    peak_metrics = peak_matching_metrics(y_true_prob_all, y_pred_prob_logits_all)

    results = {'loss': avg_loss, **peak_metrics, 'loss_details': avg_loss_details}
    return results

def eval_model_test(model, test_loader, device, use_amp=True, transform="log10over3"):
    """テスト評価 (離散化処理含む)"""
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

    # Calculate Wasserstein distance metric
    wasserstein_dist = wasserstein_distance_metric(
        y_true_proc_all, y_true_prob_all, y_pred_proc_all, y_prob_all)

    # Peak Metrics (use true probs and predicted logits)
    pred_prob_logits_all = torch.logit(y_prob_all + EPS) # Get logits back from probs
    peak_metrics = peak_matching_metrics(y_true_prob_all, pred_prob_logits_all)

    return {
        'wasserstein_distance': wasserstein_dist,
        **peak_metrics,
        'y_true_processed': y_true_proc_all, # Processed true intensities
        'y_pred_processed': y_pred_proc_all, # Processed predicted intensities
        'y_pred_prob': y_prob_all,           # Predicted probabilities
        'y_pred_discrete': y_pred_discrete_all, # Discrete prediction (0-100)
        'mol_ids': all_mol_ids
    }

# 段階的トレーニング関数
def tiered_training(model, train_ids, val_loader, criterion, optimizer, scheduler, device, 
                   mol_files_path, msp_data, tokenizer, transform, normalization, cache_dir, 
                   checkpoint_dir=CHECKPOINT_DIR, num_workers=0, patience=5, max_epochs_per_tier=30):
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
        tier_epochs = [5, 5, 8, 8, 15]  # ティアごとのエポック数
    elif len(train_ids) > 50000:
        train_tiers = [
            train_ids[:10000], 
            train_ids[:30000],
            train_ids
        ]
        tier_epochs = [6, 8, 20]
    else:
        # 小さなデータセットは段階を少なく
        train_tiers = [
            train_ids[:5000] if len(train_ids) > 5000 else train_ids[:len(train_ids)//2],
            train_ids
        ]
        tier_epochs = [8, 25]
    
    best_peak_f1 = 0.0
    all_train_losses = []
    all_val_losses = []
    all_val_metrics = {'peak_f1': [], 'peak_precision': [], 'peak_recall': []}
    
    # 進行状況を表示するために各ティアにプレフィックスを追加
    tier_prefixes = [f"Tier {i+1}/{len(train_tiers)}" for i in range(len(train_tiers))]
    
    # 各ティアを処理
    for tier_idx, (tier_ids, tier_prefix) in enumerate(zip(train_tiers, tier_prefixes)):
        tier_name = f"{tier_prefix} ({len(tier_ids)} サンプル)"
        logger.info(f"=== {tier_name} のトレーニングを開始 ===")
        
        # ティア間でメモリクリーンアップ
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        # このティア用のデータセット作成
        tier_dataset = ChemBERTaMoleculeDataset(
            tier_ids, mol_files_path, msp_data, tokenizer,
            transform=transform, normalization=normalization,
            augment=True, cache_dir=cache_dir
        )
        
        # ティアサイズに基づいてバッチサイズを調整
        if len(tier_ids) <= 10000:
            tier_batch_size = 32  # 小さいティアでは大きいバッチサイズ
        elif len(tier_ids) <= 30000:
            tier_batch_size = 24  # 中間ティア
        elif len(tier_ids) <= 60000:
            tier_batch_size = 16  # 大きいティア
        else:
            tier_batch_size = 8   # 非常に大きいティア
        
        logger.info(f"ティア {tier_idx+1} のバッチサイズ: {tier_batch_size}")
        
        # このティア用のデータローダを作成
        tier_loader = DataLoader(
            tier_dataset, 
            batch_size=tier_batch_size,
            shuffle=True, 
            collate_fn=chemberta_collate_fn,
            num_workers=0,  # シングルプロセス
            pin_memory=True,
            drop_last=True
        )
        
        # オプティマイザの学習率を調整
        for param_group in optimizer.param_groups:
            if tier_idx == 0:
                param_group['lr'] = 5e-5  # 小さいデータセット用に高い学習率
            else:
                param_group['lr'] = 2e-5 * (0.8 ** tier_idx)  # 大きいティア向けに学習率を減少
        
        # このティアの忍耐値を計算（前半のティアは早く次に進む）
        tier_patience = max(2, patience // 2) if tier_idx < len(train_tiers) - 1 else patience
        
        # このティア用のスケジューラを作成（OneCycleLR）
        steps_per_epoch = len(tier_loader)
        tier_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4 if tier_idx == 0 else 5e-5 * (0.8 ** tier_idx),
            steps_per_epoch=steps_per_epoch,
            epochs=min(tier_epochs[tier_idx], max_epochs_per_tier),
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # 指定されたエポック数でこのティアをトレーニング
        current_tier_best_f1 = 0.0
        early_stopping_counter = 0
        
        # エポックループ
        for epoch in range(min(tier_epochs[tier_idx], max_epochs_per_tier)):
            # トレーニングモード
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            epoch_loss_details = {}
            
            train_pbar = tqdm(tier_loader, desc=f"Epoch {epoch+1}/{min(tier_epochs[tier_idx], max_epochs_per_tier)} [Train]", position=0, leave=True)
            
            for batch_idx, batch_data in enumerate(train_pbar):
                if batch_data is None: continue
                # GPU転送
                batch_data_gpu = {}
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor): 
                        batch_data_gpu[k] = v.to(device, non_blocking=True)
                    else: 
                        batch_data_gpu[k] = v
                
                # 訓練ステップ
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    pred_output = model(batch_data_gpu)
                    loss, loss_detail = criterion(pred_output, batch_data_gpu)
                
                # 勾配計算と最適化
                if torch.cuda.is_available():
                    scaler = GradScaler(enabled=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # スケジューラのステップ
                tier_scheduler.step()
                
                # 損失の記録
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1
                
                for k, v in loss_detail.items():
                    epoch_loss_details[k] = epoch_loss_details.get(k, 0.0) + v
                
                # プログレスバー更新
                train_pbar.set_postfix({
                    'loss': f"{current_loss:.4f}", 
                    'avg': f"{epoch_loss/batch_count:.4f}", 
                    'lr': f"{optimizer.param_groups[0]['lr']:.1E}"
                })
                
                # GPUメモリ解放
                del loss, pred_output, batch_data_gpu
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            # エポック終了：評価
            if batch_count > 0:
                avg_train_loss = epoch_loss / batch_count
                all_train_losses.append(avg_train_loss)
                
                # 検証
                aggressive_memory_cleanup()
                val_results = evaluate_model(model, val_loader, criterion, device, use_amp=torch.cuda.is_available())
                
                val_loss = val_results['loss']
                all_val_losses.append(val_loss)
                
                # 評価指標の保存
                for key in all_val_metrics.keys():
                    if key in val_results:
                        all_val_metrics[key].append(val_results[key])
                    else:
                        all_val_metrics[key].append(0.0)
                
                # ログ出力
                logger.info(f"Epoch {epoch+1}/{min(tier_epochs[tier_idx], max_epochs_per_tier)} - "
                           f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Peak F1: {val_results['peak_f1']:.4f}")
                
                # 早期停止チェック
                if val_results['peak_f1'] > current_tier_best_f1:
                    current_tier_best_f1 = val_results['peak_f1']
                    # 全体の最良性能を更新
                    if current_tier_best_f1 > best_peak_f1:
                        best_peak_f1 = current_tier_best_f1
                        # 最良モデルの保存
                        best_model_path = os.path.join(checkpoint_dir, f"best_model_tier{tier_idx+1}.pth")
                        torch.save(model.state_dict(), best_model_path)
                        logger.info(f"新しい最良モデル保存: Peak F1 = {best_peak_f1:.4f}")
                    
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    logger.info(f"早期停止カウンター: {early_stopping_counter}/{tier_patience}")
                    
                    if early_stopping_counter >= tier_patience:
                        logger.info(f"このティアの早期停止: {epoch+1}エポック後")
                        break
                
                # チェックポイント保存
                checkpoint_path = os.path.join(checkpoint_dir, f"tier{tier_idx+1}_epoch{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': tier_scheduler.state_dict(),
                    'tier_best_f1': current_tier_best_f1,
                    'global_best_f1': best_peak_f1
                }, checkpoint_path)
                
            else:
                logger.warning(f"Epoch {epoch+1}: トレーニング中に有効なバッチがありませんでした。")
        
        # ティア終了：キャッシュクリア
        logger.info(f"ティア {tier_idx+1} 完了、ベストF1: {current_tier_best_f1:.4f}")
        del tier_dataset, tier_loader
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # システムの安定化
        time.sleep(2)
    
    # 学習曲線の保存
    try:
        plot_training_progress(all_train_losses, all_val_losses, all_val_metrics, best_peak_f1, checkpoint_dir)
    except Exception as e:
        logger.error(f"学習曲線プロット中にエラー: {e}")
    
    return all_train_losses, all_val_losses, all_val_metrics, best_peak_f1

# 学習曲線のプロット関数
def plot_training_progress(train_losses, val_losses, val_metrics, best_metric, save_dir):
    """学習進捗の可視化"""
    if not train_losses: return
    
    epochs = range(1, len(train_losses) + 1)
    val_epochs = range(1, len(val_losses) + 1) if val_losses else []
    
    plt.figure(figsize=(16, 6))
    
    # Loss曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='.', alpha=0.7)
    if val_epochs: plt.plot(val_epochs, val_losses, label='Validation Loss', marker='.', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True, alpha=0.3)
    
    # Peak F1スコア曲線
    plt.subplot(1, 2, 2)
    if val_epochs and 'peak_f1' in val_metrics and len(val_metrics['peak_f1']) == len(val_epochs):
        plt.plot(val_epochs, val_metrics['peak_f1'], label='Peak F1', marker='.', color='green', alpha=0.7)
        plt.axhline(y=best_metric, color='r', linestyle='--', label=f'Best F1: {best_metric:.4f}')
        
        if 'peak_precision' in val_metrics and 'peak_recall' in val_metrics:
            plt.plot(val_epochs, val_metrics['peak_precision'], label='Precision', marker='.', color='purple', alpha=0.5)
            plt.plot(val_epochs, val_metrics['peak_recall'], label='Recall', marker='.', color='orange', alpha=0.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Peak Detection Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'chemberta_ms_learning_curves.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"学習曲線を保存しました: {save_path}")

def visualize_results(test_results, num_samples=10, transform="log10over3", save_dir="."):
    """テスト結果の可視化（真値と離散化予測）"""
    if not test_results:
        logger.error("可視化するテスト結果がありません。")
        return
    
    plt.figure(figsize=(16, num_samples * 4))
    
    # ランダムサンプル
    indices = np.random.choice(len(test_results['mol_ids']), min(num_samples, len(test_results['mol_ids'])), replace=False)
    
    y_true_proc = test_results['y_true_processed']
    y_pred_discrete = test_results['y_pred_discrete']
    
    for i, idx in enumerate(indices):
        mol_id = test_results['mol_ids'][idx]
        
        # 真のスペクトル (処理済み) を非処理化
        true_spec_proc = y_true_proc[idx]
        try:
            true_spec_unproc = unprocess_spec(true_spec_proc.unsqueeze(0), transform).squeeze(0).numpy()
            max_true = np.max(true_spec_unproc)
            if max_true > 0:
                true_spec_display = true_spec_unproc / max_true * 100.0
            else:
                true_spec_display = np.zeros_like(true_spec_unproc)
        except Exception as e:
            logger.warning(f"True spectrum unprocessing failed for ID {mol_id}: {e}")
            true_spec_display = np.zeros(MZ_DIM)
        
        # 離散化予測スペクトル
        pred_discrete_spec = y_pred_discrete[idx].numpy()
        
        # Wasserstein距離を計算 (POTが利用可能な場合)
        if POT_AVAILABLE:
            try:
                mz_coords = np.arange(len(true_spec_display)).reshape(-1, 1).astype(np.float64)
                C = ot.dist(mz_coords, mz_coords, metric='euclidean')
                C /= C.max()
                
                # 正規化
                true_norm = true_spec_display / (true_spec_display.sum() + EPS) if true_spec_display.sum() > 0 else np.ones_like(true_spec_display) / len(true_spec_display)
                pred_norm = pred_discrete_spec / (pred_discrete_spec.sum() + EPS) if pred_discrete_spec.sum() > 0 else np.ones_like(pred_discrete_spec) / len(pred_discrete_spec)
                
                # Wasserstein距離計算
                wass_dist = ot.emd2(true_norm, pred_norm, C)
                metric_str = f", W-dist: {wass_dist:.4f}"
            except Exception:
                metric_str = ""
        else:
            metric_str = ""
        
        # 真のスペクトルをプロット
        plt.subplot(num_samples, 2, 2*i + 1)
        mz_values = np.arange(len(true_spec_display))
        peaks_true = np.where(true_spec_display > 0.1)[0]
        if len(peaks_true) > 0:
            plt.vlines(peaks_true, 0, true_spec_display[peaks_true], colors='blue', linewidths=1)
        plt.title(f"Measured Spectrum - ID: {mol_id}")
        plt.xlabel("m/z")
        plt.ylabel("Relative Intensity (%)")
        plt.ylim(0, 110)
        
        # 予測スペクトルをプロット
        plt.subplot(num_samples, 2, 2*i + 2)
        peaks_pred = np.where(pred_discrete_spec > 0.1)[0]
        if len(peaks_pred) > 0:
            plt.vlines(peaks_pred, 0, pred_discrete_spec[peaks_pred], colors='green', linewidths=1)
        plt.title(f"ChemBERTa-MS Prediction{metric_str}")
        plt.xlabel("m/z")
        plt.ylabel("Relative Intensity (%)")
        plt.ylim(0, 110)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'chemberta_ms_spectrum_comparison.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"予測結果の可視化を保存しました: {save_path}")

###############################
# メイン関数
###############################

def main():
    logger.info("============= ChemBERTa-MS 質量スペクトル予測モデルの実行開始 =============")
    
    # CUDA設定
    if torch.cuda.is_available():
        logger.info(f"GPUを使用: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        logger.warning("GPUが利用できません。CPUで実行します。")
    
    # MSPファイルを解析
    logger.info("MSPファイル(生強度)を解析中...")
    msp_data = parse_msp_file_raw(MSP_FILE_PATH, cache_dir=CACHE_DIR)
    logger.info(f"MSPファイルから{len(msp_data)}個の化合物データを読み込みました")
    
    # 利用可能なMOLファイルを確認
    mol_id_cache_file = os.path.join(CACHE_DIR, "valid_mol_ids_all.pkl")
    if os.path.exists(mol_id_cache_file):
        logger.info(f"キャッシュからmol_idsを読み込み中: {mol_id_cache_file}")
        with open(mol_id_cache_file, 'rb') as f:
            mol_ids_all = pickle.load(f)
    else:
        mol_ids_all = []
        logger.info("MOLファイルリストをスキャン中...")
        for filename in tqdm(os.listdir(MOL_FILES_PATH), desc="MOLファイルスキャン"):
            if filename.startswith("ID") and filename.endswith(".MOL"):
                try:
                    mol_ids_all.append(int(filename[2:-4]))
                except:
                    continue
        logger.info(f"mol_idsをキャッシュに保存中: {mol_id_cache_file}")
        with open(mol_id_cache_file, 'wb') as f:
            pickle.dump(mol_ids_all, f)
    
    # MSPデータが利用可能なIDのみ使用
    mol_ids = [mid for mid in mol_ids_all if mid in msp_data]
    logger.info(f"MOLファイルとMSPデータが揃っている化合物: {len(mol_ids)}個")
    
    # データ分割
    train_ids, test_ids = train_test_split(mol_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    
    logger.info(f"訓練データ: {len(train_ids)}個")
    logger.info(f"検証データ: {len(val_ids)}個")
    logger.info(f"テストデータ: {len(test_ids)}個")
    
    # ハイパーパラメータ
    transform = "log10over3"
    normalization = "l1"
    num_epochs = 30
    patience = 7
    learning_rate = 2e-5
    weight_decay = 1e-6
    batch_size = 16
    
    # モデルのデバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ChemBERTaトークナイザの読み込み
    try:
        tokenizer = RobertaTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME, do_lower_case=False)
        logger.info(f"ChemBERTaトークナイザを読み込みました: {CHEMBERTA_MODEL_NAME}")
    except Exception as e:
        logger.error(f"トークナイザの読み込みに失敗: {e}")
        return
    
    # 検証データセットの作成
    val_dataset = ChemBERTaMoleculeDataset(
        val_ids, MOL_FILES_PATH, msp_data, tokenizer,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    test_dataset = ChemBERTaMoleculeDataset(
        test_ids, MOL_FILES_PATH, msp_data, tokenizer,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    logger.info(f"有効な検証データ: {len(val_dataset)}個")
    logger.info(f"有効なテストデータ: {len(test_dataset)}個")
    
    # データローダー
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=chemberta_collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=chemberta_collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # モデルの初期化 (Morganフィンガープリント対応)
    model = ChemBERTaForMassSpec(
        out_channels=MZ_DIM,
        num_fragments=MORGAN_DIM,  # MACCSからMorganへ変更
        pretrained_model_name=CHEMBERTA_MODEL_NAME,
        dropout=0.2,
        prec_mass_offset=10,
        bidirectional=True  # 双方向予測を有効化
    ).to(device)
    
    logger.info(f"ChemBERTa-MSモデル初期化完了")
    logger.info(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失関数、オプティマイザー (Morganフィンガープリント対応)
    criterion = ChemBERTaMSLoss(
        mz_dim=MZ_DIM,
        num_fragments=MORGAN_DIM,  # MACCSからMorganへ変更
        w_intensity=0.1,
        w_prob=0.3,
        w_wasserstein=0.5,
        w_morgan=0.1  # w_fragmentからw_morganへ変更
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # ダミースケジューラー（段階的トレーニングでは各ティアで再定義）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # メモリのクリーンアップ
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # 段階的トレーニング
    logger.info("段階的トレーニングを開始します...")
    
    train_losses, val_losses, val_metrics, best_peak_f1 = tiered_training(
        model=model,
        train_ids=train_ids,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mol_files_path=MOL_FILES_PATH,
        msp_data=msp_data,
        tokenizer=tokenizer,
        transform=transform,
        normalization=normalization,
        cache_dir=CACHE_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        num_workers=0,
        patience=patience
    )
    
    logger.info(f"段階的トレーニング完了！ 最良Peak F1: {best_peak_f1:.4f}")
    
    # 最良モデルの読み込み
    try:
        best_model_path = None
        
        # 最新のティアの最良モデルを検索
        for tier_idx in range(len(train_losses) + 1, 0, -1):
            path = os.path.join(CHECKPOINT_DIR, f"best_model_tier{tier_idx}.pth")
            if os.path.exists(path):
                best_model_path = path
                break
                
        if best_model_path is None:
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_tier1.pth")
            
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logger.info(f"最良モデルを読み込みました: {best_model_path}")
        else:
            logger.warning("最良モデルのパスが見つかりませんでした。現在のモデルでテストを実行します。")
    except Exception as e:
        logger.error(f"最良モデル読み込み中にエラー: {e}")
        logger.warning("現在のモデルの状態でテストを実行します。")
    
    # テスト評価
    logger.info("テストデータでの評価を開始します...")
    
    try:
        test_results = eval_model_test(model, test_loader, device, use_amp=torch.cuda.is_available(), transform=transform)
        
        if test_results:
            logger.info(f"テスト評価結果:")
            logger.info(f"  Peak F1 Score: {test_results['peak_f1']:.4f}")
            logger.info(f"  Peak Precision: {test_results['peak_precision']:.4f}")
            logger.info(f"  Peak Recall: {test_results['peak_recall']:.4f}")
            
            if 'wasserstein_distance' in test_results:
                logger.info(f"  Wasserstein Distance: {test_results['wasserstein_distance']:.4f}")
            
            # 結果の可視化
            visualize_results(test_results, num_samples=8, transform=transform, save_dir=CHECKPOINT_DIR)
        else:
            logger.error("テスト評価結果がありません。")
    except Exception as e:
        logger.error(f"テスト評価中にエラー: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("============= ChemBERTa-MS 質量スペクトル予測モデルの実行終了 =============")

if __name__ == "__main__":
    main()
