import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# AMPを使用しないため、autocastとGradScalerを削除
import time
import datetime

# ===== ロギング設定 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== パス設定 =====
DATA_PATH = "data/"
MOL_FILES_PATH = os.path.join(DATA_PATH, "mol_files/")
MSP_FILE_PATH = os.path.join(DATA_PATH, "NIST17.MSP")
CACHE_DIR = os.path.join(DATA_PATH, "cache/")
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints/")

# ディレクトリの作成
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== パラメータ設定 =====
MAX_MZ = 2000  # 最大m/z値
MORGAN_BITS = 1024  # Morgan フィンガープリントのビット数
ATOMPAIR_BITS = 1024  # AtomPair フィンガープリントのビット数
NUM_FRAGS = MORGAN_BITS + ATOMPAIR_BITS  # 合計フラグメントパターン数
# 重要なm/z値のリスト（フラグメントイオンに対応）- ただし強調は行わない
IMPORTANT_MZ = [18, 28, 43, 57, 71, 73, 77, 91, 105, 115, 128, 152, 165, 178, 207]
EPS = np.finfo(np.float32).eps  # エフェメラル値（小さな値）
MAX_PEAKS = 50  # 最大ピーク数 - 実際のNIST17データに基づくべき

# ===== 原子と結合の特徴マッピング =====
# 非金属元素のリスト（これらのみを含む分子を許可）
NON_METAL_ATOMS = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'Te', 'As'}

# 原子の特徴マッピング（非金属のみ）
ATOM_FEATURES = {
    'C': 0,    # 炭素
    'N': 1,    # 窒素
    'O': 2,    # 酸素
    'S': 3,    # 硫黄
    'F': 4,    # フッ素
    'Cl': 5,   # 塩素
    'Br': 6,   # 臭素
    'I': 7,    # ヨウ素
    'P': 8,    # リン
    'Si': 9,   # ケイ素
    'B': 10,   # ホウ素
    'H': 11,   # 水素
    'Se': 12,  # セレン
    'Te': 13,  # テルル
    'As': 14,  # ヒ素
    'OTHER': 15 # その他（エラー処理用）
}

# 結合の特徴マッピング
BOND_FEATURES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}

# ===== メモリ管理関数 =====
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

# ===== 評価指標関数 =====
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

def evaluate_model(model, data_loader, criterion, device):
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
                
                # 通常の予測（AMPなし）
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

def eval_model(model, test_loader, device, transform="log10over3"):
    """テスト用の評価関数 - 離散化処理追加"""
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_pred_discrete = []  # 離散化後の予測結果
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
                
                # 通常の予測（AMPなし）
                output, frag_pred = model(processed_batch)
                
                # 元のスムーズな予測結果を保存
                y_true.append(processed_batch['spec'].cpu())
                y_pred.append(output.cpu())
                
                # 離散化処理を適用
                for i in range(len(output)):
                    pred_np = output[i].cpu().numpy()
                    discrete_pred = hybrid_spectrum_conversion(pred_np, transform)
                    y_pred_discrete.append(torch.from_numpy(discrete_pred).float())
                
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
    all_pred_discrete = torch.stack(y_pred_discrete)
    all_fragment_true = torch.cat(fragment_true, dim=0)
    all_fragment_pred = torch.cat(fragment_pred, dim=0)
    
    # スコア計算
    smooth_cosine_sim = cosine_similarity_score(all_true, all_pred)
    discrete_cosine_sim = cosine_similarity_score(all_true, all_pred_discrete)
    
    return {
        'cosine_similarity': smooth_cosine_sim,  # 元の予測との類似度
        'discrete_cosine_similarity': discrete_cosine_sim,  # 離散化後の類似度
        'y_true': all_true,
        'y_pred': all_pred,
        'y_pred_discrete': all_pred_discrete,
        'fragment_true': all_fragment_true,
        'fragment_pred': all_fragment_pred,
        'mol_ids': mol_ids
    }

# ===== データ処理関数 =====
def process_spec(spec, transform, normalization, eps=EPS):
    """スペクトルにトランスフォームと正規化を適用"""
    # スペクトルを1000までスケーリング
    spec = spec / (torch.max(spec, dim=-1, keepdim=True)[0] + eps) * 1000.
    
    # 信号変換
    if transform == "log10":
        spec = torch.log10(spec + 1)
    elif transform == "log10over3":
        spec = torch.log10(spec + 1) / 3
    elif transform == "loge":
        spec = torch.log(spec + 1)
    elif transform == "sqrt":
        spec = torch.sqrt(spec)
    elif transform == "none":
        pass
    else:
        raise ValueError("invalid transform")
    
    # 正規化
    if normalization == "l1":
        spec = F.normalize(spec, p=1, dim=-1, eps=eps)
    elif normalization == "l2":
        spec = F.normalize(spec, p=2, dim=-1, eps=eps)
    elif normalization == "none":
        pass
    else:
        raise ValueError("invalid normalization")
    
    assert not torch.isnan(spec).any()
    return spec

def hybrid_spectrum_conversion(smoothed_prediction, transform="log10over3"):
    """より本物に近いマススペクトルへの変換（大幅改良版）"""
    # モデル出力を元のスケールに戻す
    if transform == "log10":
        untransformed = 10**smoothed_prediction - 1.
    elif transform == "log10over3":
        untransformed = 10**(3 * smoothed_prediction) - 1.
    elif transform == "loge":
        untransformed = np.exp(smoothed_prediction) - 1.
    elif transform == "sqrt":
        untransformed = smoothed_prediction**2
    else:
        untransformed = smoothed_prediction
    
    # 極小ノイズをフィルタリング
    noise_threshold = 0.0005
    untransformed[untransformed < noise_threshold] = 0
    
    # 初期化
    discrete_spectrum = np.zeros_like(untransformed)
    max_intensity = np.max(untransformed) if np.max(untransformed) > 0 else 1.0
    
    # 各m/z領域に適した閾値を設定
    spectrum_length = len(untransformed)
    low_mz_region = int(spectrum_length * 0.25)    # 最初の25%
    mid_mz_region = int(spectrum_length * 0.7)     # 70%まで
    
    # 領域ごとの閾値設定
    thresholds = {
        'low_mz': {
            'high': 0.02,    # 最大ピークの2%
            'medium': 0.005, # 0.5%
            'low': 0.002     # 0.2%
        },
        'mid_mz': {
            'high': 0.02,
            'medium': 0.006,
            'low': 0.0025
        },
        'high_mz': {
            'high': 0.02,
            'medium': 0.004,
            'low': 0.0015    # 高m/z域では低い閾値（0.15%）
        }
    }
    
    # 1. m/z領域ごとにピーク検出と保持を行う
    for region_start, region_end, region_name in [
        (0, low_mz_region, 'low_mz'),
        (low_mz_region, mid_mz_region, 'mid_mz'),
        (mid_mz_region, spectrum_length, 'high_mz')
    ]:
        region_data = untransformed[region_start:region_end]
        if np.max(region_data) == 0:
            continue
            
        # 領域内の高強度ピークを検出
        from scipy.signal import find_peaks
        high_peaks, _ = find_peaks(
            region_data, 
            height=max_intensity * thresholds[region_name]['high'],
            distance=1
        )
        
        # 領域内の中強度ピークを検出
        medium_peaks, _ = find_peaks(
            region_data, 
            height=max_intensity * thresholds[region_name]['medium'],
            distance=1
        )
        
        # 領域内の低強度ピークを検出（数を制限）
        low_peaks, low_props = find_peaks(
            region_data, 
            height=max_intensity * thresholds[region_name]['low'],
            distance=1
        )
        
        # 絶対m/z値に変換
        high_peaks = high_peaks + region_start
        medium_peaks = medium_peaks + region_start
        low_peaks = low_peaks + region_start
        
        # ピークを保持
        for peak in high_peaks:
            discrete_spectrum[peak] = untransformed[peak]
            
        for peak in medium_peaks:
            discrete_spectrum[peak] = untransformed[peak]
        
        # 低強度ピークは数を制限（各領域で最大50個）
        if len(low_peaks) > 0:
            # 強度でソート
            heights = [untransformed[p] for p in low_peaks]
            sorted_indices = np.argsort(-np.array(heights))
            count = 0
            for idx in sorted_indices:
                peak = low_peaks[idx]
                if discrete_spectrum[peak] == 0 and count < 50:  # 未追加で最大50個まで
                    discrete_spectrum[peak] = untransformed[peak]
                    count += 1
    
    # 2. クラスターパターンの検出と保持（連続するピークグループ）
    window_size = 5
    for i in range(window_size, len(untransformed) - window_size):
        window = untransformed[i-window_size:i+window_size+1]
        # ウィンドウ内に3つ以上のピークがあり、合計強度が閾値を超える場合
        non_zero_count = np.count_nonzero(window > 0)
        window_sum = np.sum(window)
        
        if non_zero_count >= 3 and window_sum > max_intensity * 0.025:
            # クラスター内のすべてのピークを保持
            for j in range(i-window_size, i+window_size+1):
                if untransformed[j] > 0:
                    discrete_spectrum[j] = untransformed[j]
    
    # 3. 同位体パターンの検出と保持
    for i in range(len(untransformed) - 2):
        # 強いピークを起点に
        if untransformed[i] > max_intensity * 0.08:
            # M+1, M+2ピークのチェック
            isotope_pattern = False
            
            # M+1, M+2があるかチェック
            if i+1 < len(untransformed) and untransformed[i+1] > 0:
                ratio_1 = untransformed[i+1] / untransformed[i]
                # 典型的な同位体比の範囲（〜30%）
                if 0.01 < ratio_1 < 0.3:
                    isotope_pattern = True
                    discrete_spectrum[i+1] = untransformed[i+1]
                    
            if isotope_pattern and i+2 < len(untransformed) and untransformed[i+2] > 0:
                ratio_2 = untransformed[i+2] / untransformed[i]
                # M+2は通常M+1より小さい
                if ratio_2 < ratio_1 and ratio_2 > 0.005:
                    discrete_spectrum[i+2] = untransformed[i+2]
    
    # 4. 分子イオンピーク付近の情報を詳細に保持（高m/z領域）
    high_mz_start = int(len(untransformed) * 0.8)  # 最後の20%
    for i in range(high_mz_start, len(untransformed)):
        if untransformed[i] > max_intensity * 0.001:  # 非常に低い閾値
            discrete_spectrum[i] = untransformed[i]
    
    # 5. 強度分布を保ちつつ、スペクトル全体の形状をより忠実に再現
    # 整数m/z間隔でピーク検出（質量分析の基本単位）
    mz_interval = 14  # CH2基の質量に相当（代表的なフラグメント間隔）
    for interval in [12, 14, 16, 18, 28]:  # 一般的なフラグメントパターン間隔
        for start in range(interval):  # 各開始点から間隔ごとにチェック
            for i in range(start, len(untransformed), interval):
                if untransformed[i] > max_intensity * 0.003:
                    discrete_spectrum[i] = untransformed[i]
    
    # 6. 孤立ピークの保存（周囲に他のピークがない場合）
    for i in range(3, len(untransformed) - 3):
        # 前後6ポイントの範囲でピークが唯一の場合
        surrounding = list(untransformed[i-3:i]) + list(untransformed[i+1:i+4])
        if untransformed[i] > max_intensity * 0.003 and max(surrounding) < max_intensity * 0.001:
            discrete_spectrum[i] = untransformed[i]
    
    # 7. IMPORTANT_MZ値のピークを低閾値で保持
    for mz in IMPORTANT_MZ:
        if mz < len(untransformed) and untransformed[mz] > max_intensity * 0.0008:
            discrete_spectrum[mz] = untransformed[mz]
    
    # 8. スペクトル内のギャップを検出し、適切な補間を行う
    for i in range(5, len(discrete_spectrum) - 5):
        # 前後のピークが存在するがギャップがある場合
        if (np.sum(discrete_spectrum[i-5:i] > 0) >= 2 and 
            np.sum(discrete_spectrum[i:i+5] > 0) >= 2 and
            discrete_spectrum[i] == 0):
            # ギャップを補間（元の強度の70%で保持）
            if untransformed[i] > max_intensity * 0.001:
                discrete_spectrum[i] = untransformed[i]
    
    # 9. 多すぎるピークを抑制（ノイズを減らす）
    # ピーク数をカウント
    peak_count = np.sum(discrete_spectrum > 0)
    if peak_count > MAX_PEAKS:  # ピークが多すぎる場合
        # 強度順にソートして上位MAX_PEAKS個だけを保持
        indices = np.argsort(-discrete_spectrum)
        mask = np.zeros_like(discrete_spectrum, dtype=bool)
        mask[indices[:MAX_PEAKS]] = True
        discrete_spectrum = discrete_spectrum * mask
    
    # 最大値で正規化（相対強度に変換）
    if np.max(discrete_spectrum) > 0:
        discrete_spectrum = discrete_spectrum / np.max(discrete_spectrum) * 100
    
    return discrete_spectrum

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

def parse_msp_file(msp_file_path, cache_dir=CACHE_DIR):
    """MSPファイルを解析し、ID->マススペクトルのマッピングを返す（キャッシュ対応）"""
    # キャッシュファイルのパス
    cache_file = os.path.join(cache_dir, f"msp_data_cache_{os.path.basename(msp_file_path)}.pkl")
    
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
                
                # 強度を正規化（相対強度に変換）- シンプルな最大値正規化のみ
                if np.sum(ms_vector) > 0:
                    ms_vector = ms_vector / np.max(ms_vector) * 100
                
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

def contains_metal(mol):
    """分子に非金属以外の原子が含まれているかチェック"""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in NON_METAL_ATOMS:
            return True
    return False

# ===== DMPNN関連のモジュール =====
class DirectedMessagePassing(nn.Module):
    """有向メッセージパッシングニューラルネットワーク(DMPNN)のメッセージ関数"""
    def __init__(self, hidden_size, edge_fdim, node_fdim, depth=3):
        super(DirectedMessagePassing, self).__init__()
        self.hidden_size = hidden_size
        self.edge_fdim = edge_fdim
        self.node_fdim = node_fdim
        self.depth = depth
        
        # 入力サイズは、エッジの特徴次元 + ノードの特徴次元 + 隠れ層次元
        input_size = edge_fdim + node_fdim + hidden_size
        
        # 各メッセージパッシングステップ用のネットワーク
        self.W_message = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 更新ネットワーク
        self.W_update = nn.GRUCell(hidden_size, hidden_size)
        
        # ノード表現を計算するネットワーク
        self.W_node = nn.Linear(node_fdim + hidden_size, hidden_size)
        
        # 読み出しネットワーク
        self.W_o = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, data):
        """メッセージパッシングを実行"""
        # データからグラフ情報を抽出
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        device = x.device
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        # すべてのテンソルをFloat32に変換して型を統一
        x = x.float()
        edge_attr = edge_attr.float()
        
        # 初期メッセージを準備：エッジ特徴で初期化
        messages = torch.zeros(num_edges, self.hidden_size, device=device)
        
        # メッセージパッシングのD回のステップを実行
        for step in range(self.depth):
            # 各方向エッジ（i->j）に対するメッセージを計算
            source_nodes = edge_index[0]  # メッセージの送信元
            target_nodes = edge_index[1]  # メッセージの送信先
            
            # メッセージ入力特徴の作成
            # [エッジ特徴, 送信元ノードの特徴, 隠れ状態]
            message_inputs = torch.cat([
                edge_attr,
                x[source_nodes],
                messages
            ], dim=1)
            
            # メッセージパッシング関数で新しいメッセージを計算
            new_messages = self.W_message(message_inputs)
            
            # ノードに集約するときにエッジインデックスをエッジID（0〜num_edges）に変換する必要がある
            # エッジをターゲットノードでグループ化し、メッセージをマージ
            # 各ノードへの入力メッセージを集約（合計）
            aggr_messages = torch.zeros(num_nodes, self.hidden_size, device=device)
            aggr_messages.index_add_(0, target_nodes, new_messages)
            
            # GRUを使用してメッセージを更新
            messages = self.W_update(
                new_messages,
                messages
            )
        
        # ノード特徴の最終集約
        # 各ノードに入るエッジからのメッセージを集約
        node_messages = torch.zeros(num_nodes, self.hidden_size, device=device)
        node_messages.index_add_(0, target_nodes, messages)
        
        # ノード表現を計算
        node_inputs = torch.cat([x, node_messages], dim=1)
        node_representations = self.W_node(node_inputs)
        
        # ノード表現の読み出し
        node_outputs = self.W_o(node_representations)
        
        return node_outputs

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
            
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.leaky_relu(self.ln1(self.conv1(x)))
        out = self.dropout(out)
        out = self.ln2(self.conv2(out))
        
        out += residual  # 残差接続
        out = F.leaky_relu(out)
        
        return out

class DMPNNMSPredictor(nn.Module):
    """DMPNNを用いたマススペクトル予測モデル"""
    def __init__(self, 
                 node_fdim, 
                 edge_fdim, 
                 hidden_size=128, 
                 depth=3, 
                 output_dim=MAX_MZ, 
                 global_features_dim=16,
                 num_fragments=NUM_FRAGS,
                 bidirectional=True,
                 gate_prediction=True,
                 prec_mass_offset=10):
        super(DMPNNMSPredictor, self).__init__()
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_dim = output_dim
        self.global_features_dim = global_features_dim
        self.num_fragments = num_fragments
        self.bidirectional = bidirectional
        self.gate_prediction = gate_prediction
        self.prec_mass_offset = prec_mass_offset
        
        # DMPNN部分
        self.dmpnn = DirectedMessagePassing(
            hidden_size=hidden_size,
            edge_fdim=edge_fdim,
            node_fdim=node_fdim,
            depth=depth
        )
        
        # グローバル特徴量処理
        self.global_proj = nn.Sequential(
            nn.Linear(global_features_dim, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # 分子表現の集約と処理（プーリング後）
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # スペクトル予測のための全結合層
        self.fc_layers = nn.ModuleList([
            ResidualBlock(hidden_size * 2, hidden_size * 2),
            ResidualBlock(hidden_size * 2, hidden_size * 2),
            ResidualBlock(hidden_size * 2, hidden_size)
        ])
        
        # マルチタスク学習: フラグメントパターン予測
        self.fragment_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, num_fragments),
        )
        
        # 双方向予測用レイヤー
        if bidirectional:
            self.forw_out_layer = nn.Linear(hidden_size, output_dim)
            self.rev_out_layer = nn.Linear(hidden_size, output_dim)
            self.out_gate = nn.Sequential(
                nn.Linear(hidden_size, output_dim),
                nn.Sigmoid()
            )
        else:
            # 通常の出力レイヤー
            self.out_layer = nn.Linear(hidden_size, output_dim)
            if gate_prediction:
                self.out_gate = nn.Sequential(
                    nn.Linear(hidden_size, output_dim),
                    nn.Sigmoid()
                )
                
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
        """順伝播計算"""
        device = next(self.parameters()).device
        
        # データ形式を標準化
        if isinstance(data, dict):
            # MassFormer形式の入力
            x = data['graph'].x.to(device)
            edge_index = data['graph'].edge_index.to(device)
            edge_attr = data['graph'].edge_attr.to(device)
            batch = data['graph'].batch.to(device)
            
            global_attr = data['graph'].global_attr.to(device) if hasattr(data['graph'], 'global_attr') else None
            prec_mz_bin = data.get('prec_mz_bin', None)
            if prec_mz_bin is not None:
                prec_mz_bin = prec_mz_bin.to(device)
        else:
            # 直接Dataオブジェクトが渡された場合
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
        
        # 全てのテンソルを確実にFloat32に変換
        x = x.float()
        edge_attr = edge_attr.float()
        
        # データオブジェクトの作成
        processed_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )
        
        # AMPを使用せずに処理
        # DMPNN処理
        node_features = self.dmpnn(processed_data)
        
        # グラフプーリング（ノードから分子特徴へ）
        # ここで次元の統一が必要
        batch_size = torch.max(batch).item() + 1
        pooled_features = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # 正規化されたプーリング
        for i in range(batch_size):
            batch_mask = (batch == i)
            if batch_mask.any():
                # 各分子のノード特徴を抽出
                mol_features = node_features[batch_mask]
                # 平均プーリングを適用
                pooled_features[i] = torch.mean(mol_features, dim=0)
        
        # 分子表現の読み出し
        mol_representation = self.readout(pooled_features)
        
        # 以降の処理は混合精度可
        # グローバル特徴量の処理
        if global_attr is not None:
            # グローバル属性のサイズ調整
            if len(global_attr.shape) == 1:
                # 一次元の場合、バッチサイズに合わせて再形成
                global_attr = global_attr.view(batch_size, -1)
                
                # 期待する次元にパディング
                if global_attr.shape[1] != self.global_features_dim:
                    padded = torch.zeros(batch_size, self.global_features_dim, device=device)
                    copy_size = min(global_attr.shape[1], self.global_features_dim)
                    padded[:, :copy_size] = global_attr[:, :copy_size]
                    global_attr = padded
                    
            global_features = self.global_proj(global_attr)
        else:
            # グローバル特徴がない場合はゼロパディング
            global_features = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # 分子表現とグローバル特徴を結合
        x_combined = torch.cat([mol_representation, global_features], dim=1)
        
        # 残差ブロックを通した特徴抽出
        for i, fc_layer in enumerate(self.fc_layers):
            x_combined = fc_layer(x_combined)
        
        # マルチタスク学習: フラグメントパターン予測
        fragment_pred = self.fragment_pred(x_combined)
        
        # 双方向予測を使用する場合
        if self.bidirectional and prec_mz_bin is not None:
            # 順方向と逆方向の予測
            ff = self.forw_out_layer(x_combined)
            
            # 逆方向の予測と前駆体質量による調整
            fr_raw = self.rev_out_layer(x_combined)
            # 逆順にしてから前駆体質量に基づいて位置を調整
            fr = torch.flip(fr_raw, dims=(1,))
            
            # 前駆体質量を考慮したマスキング
            prec_mass_offset = self.prec_mass_offset
            max_idx = fr.shape[1]
            
            # prec_mz_binのデータ型と範囲を調整
            if prec_mz_bin.dtype != torch.long:
                prec_mz_bin = prec_mz_bin.long()
            
            prec_mz_bin = torch.clamp(prec_mz_bin, max=max_idx-prec_mass_offset-1)
            
            # ゲート機構で重み付け
            fg = self.out_gate(x_combined)
            output = ff * fg + fr * (1. - fg)
            
            # 前駆体質量でマスク
            output = mask_prediction_by_mass(output, prec_mz_bin, prec_mass_offset)
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

# ===== データセットクラス =====
class DMPNNMoleculeDataset(Dataset):
    """金属原子を除外したDMPNN用データセット"""
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
        
        # 前処理で有効な分子IDを抽出（非金属のみを含むもの）
        self._preprocess_mol_ids()
        
    def _preprocess_mol_ids(self):
        """非金属のみを含む有効な分子IDのみを抽出する"""
        # キャッシュファイルのパス
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = f"dmpnn_preprocessed_data_{hash(str(sorted(self.mol_ids)))}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # キャッシュファイルが存在するか確認
        if os.path.exists(cache_file):
            logger.info(f"キャッシュから前処理データを読み込み中: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.valid_mol_ids = cached_data['valid_mol_ids']
                self.fragment_patterns = cached_data['fragment_patterns']
                return
        
        logger.info("分子データの前処理を開始します（非金属のみを含む分子を抽出）...")
        
        valid_ids = []
        fragment_patterns = {}
        
        # 進捗表示用
        with tqdm(total=len(self.mol_ids), desc="分子の検証") as pbar:
            for mol_id in self.mol_ids:
                try:
                    mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
                    
                    # MOLファイルが読み込めるか確認
                    mol = Chem.MolFromMolFile(mol_file, sanitize=False)
                    if mol is None:
                        pbar.update(1)
                        continue
                    
                    # 非金属以外の原子を含む分子を除外
                    if contains_metal(mol):
                        pbar.update(1)
                        continue
                    
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
                        pbar.update(1)
                        continue
                    
                    # フラグメントパターンを計算 - Morgan + AtomPair フィンガープリント
                    try:
                        # Morgan フィンガープリントを計算
                        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS)
                        
                        # AtomPair フィンガープリントを計算
                        atompair = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=ATOMPAIR_BITS)
                        
                        # 両方のフィンガープリントを結合
                        fragments = np.zeros(NUM_FRAGS)
                        for i in range(MORGAN_BITS):
                            if morgan.GetBit(i):
                                fragments[i] = 1.0
                        for i in range(ATOMPAIR_BITS):
                            if atompair.GetBit(i):
                                fragments[MORGAN_BITS + i] = 1.0
                                
                        fragment_patterns[mol_id] = fragments
                    except Exception:
                        fragment_patterns[mol_id] = np.zeros(NUM_FRAGS)
                    
                    # この分子のスペクトルがあるか確認
                    if mol_id in self.msp_data:
                        valid_ids.append(mol_id)
                    
                    # 定期的にガベージコレクション
                    if len(valid_ids) % 1000 == 0:
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
        """分子をDMPNN用グラフに変換（非金属チェック付き）"""
        # キャッシュをチェック
        if mol_file in self.graph_cache:
            return self.graph_cache[mol_file]
        
        # キャッシュファイルのパス
        cache_file = os.path.join(self.cache_dir, f"dmpnn_graph_cache_{os.path.basename(mol_file)}.pkl")
        
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
        
        # RDKitの警告を抑制
        RDLogger.DisableLog('rdApp.*')
        
        # RDKitでMOLファイルを読み込む
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        if mol is None:
            raise ValueError(f"Could not read molecule from {mol_file}")
        
        # 非金属以外の原子を含む分子を除外
        if contains_metal(mol):
            raise ValueError(f"Molecule {mol_file} contains non-allowed atoms")
        
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
            # 非金属リストにある原子のみを特定の特徴ベクトルとして扱う
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
                
                # DMPNNは有向グラフを使用するため、双方向のエッジを個別に作成
                # 方向i->j
                edge_indices.append([i, j])
                # 方向j->i
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
                
                # i->jとj->iの両方に同じ特徴を追加
                edge_attrs.append(bond_feature)
                edge_attrs.append(bond_feature)  # 同じ特徴を両方向に
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
        """スペクトルの前処理"""
        # スペクトルをPyTorchテンソルに変換
        spec_tensor = torch.FloatTensor(spectrum)
        
        # 信号処理を適用
        processed_spec = process_spec(spec_tensor.unsqueeze(0), self.transform, self.normalization)
        
        return processed_spec.squeeze(0).numpy()
        
    def __len__(self):
        return len(self.valid_mol_ids)
    
    def __getitem__(self, idx):
        mol_id = self.valid_mol_ids[idx]
        mol_file = os.path.join(self.mol_files_path, f"ID{mol_id}.MOL")
        
        # RDKitの警告を抑制
        RDLogger.DisableLog('rdApp.*')
        
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

def collate_batch(batch):
    """バッチデータの結合"""
    graph_data = [item['graph_data'] for item in batch]
    mass_spectrum = torch.stack([item['mass_spectrum'] for item in batch])
    fragment_pattern = torch.stack([item['fragment_pattern'] for item in batch])
    mol_id = [item['mol_id'] for item in batch]
    prec_mz = torch.tensor([item['prec_mz'] for item in batch], dtype=torch.float32)
    prec_mz_bin = torch.tensor([item['prec_mz_bin'] for item in batch], dtype=torch.long)
    
    # バッチ作成
    batched_graphs = Batch.from_data_list(graph_data)
    
    return {
        'graph': batched_graphs,
        'spec': mass_spectrum,
        'fragment_pattern': fragment_pattern,
        'mol_id': mol_id,
        'prec_mz': prec_mz,
        'prec_mz_bin': prec_mz_bin
    }

# ===== 損失関数 =====
def cosine_similarity_loss(y_pred, y_true, important_mz=None, important_weight=3.0):
    """ピークと重要なm/z値を重視したコサイン類似度損失関数"""
    # 正規化
    y_pred_norm = F.normalize(y_pred, p=2, dim=1)
    y_true_norm = F.normalize(y_true, p=2, dim=1)
    
    # 特徴的なm/zの重み付け
    if important_mz is not None:
        weights = torch.ones_like(y_pred)
        for mz in important_mz:
            if mz < y_pred.size(1):
                weights[:, mz] = important_weight
        
        # 重み付きベクトルで正規化
        y_pred_weighted = y_pred * weights
        y_true_weighted = y_true * weights
        
        y_pred_norm = F.normalize(y_pred_weighted, p=2, dim=1)
        y_true_norm = F.normalize(y_true_weighted, p=2, dim=1)
    
    # コサイン類似度（-1〜1の範囲）
    cosine = torch.sum(y_pred_norm * y_true_norm, dim=1)
    
    # 損失を1 - cosineにして、0〜2の範囲に
    loss = 1.0 - cosine
    
    return loss.mean()

def combined_loss(y_pred, y_true, fragment_pred=None, fragment_true=None, 
                 alpha=0.2, beta=0.6, epsilon=0.2):
    """高速化された損失関数"""
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
    
    # 重要なm/z値にさらに重みを付ける
    for mz in IMPORTANT_MZ:
        if mz < y_true.size(1):
            mse_weights[:, mz] *= 3.0
    
    mse_loss = torch.mean(mse_weights * (y_pred - y_true) ** 2)
    
    # 2. コサイン類似度損失
    cosine_loss = cosine_similarity_loss(y_pred, y_true, important_mz=IMPORTANT_MZ)
    
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

# ===== トレーニングと評価関数 =====
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
               eval_interval=2, patience=10, grad_clip=1.0, checkpoint_dir=CHECKPOINT_DIR):
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
                
                # 勾配をゼロに初期化
                optimizer.zero_grad(set_to_none=True)  # メモリ効率のためNoneに設定
                
                # 順伝播（AMPなし）
                output, fragment_pred = model(processed_batch)
                loss = criterion(output, processed_batch['spec'], fragment_pred, processed_batch['fragment_pattern'])
                
                # 逆伝播
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                
                # オプティマイザステップ
                optimizer.step()
                
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
                val_metrics = evaluate_model(model, val_loader, criterion, device)
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
                    plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine)
                except Exception as e:
                    logger.error(f"プロット作成エラー: {str(e)}")
        else:
            logger.warning("このエポックで成功したバッチ処理がありません。")
            train_losses.append(float('inf'))
            
    # 最終的な学習曲線の保存
    try:
        plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine)
    except Exception as e:
        logger.error(f"最終プロット作成エラー: {str(e)}")
    
    return train_losses, val_losses, val_cosine_similarities, best_cosine

def plot_training_progress(train_losses, val_losses, val_cosine_similarities, best_cosine):
    """トレーニング進捗の可視化"""
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
    plt.savefig('dmpnn_learning_curves.png')
    plt.close()

def visualize_results(test_results, num_samples=10):
    """テスト結果の可視化（離散化予測を含む）"""
    # 1つの大きな図に全サンプルをまとめる
    plt.figure(figsize=(16, num_samples*4))
    
    # サンプルのインデックスをランダムに選択
    if 'mol_ids' in test_results and len(test_results['mol_ids']) > 0:
        sample_indices = np.random.choice(len(test_results['mol_ids']), 
                                         min(num_samples, len(test_results['mol_ids'])), 
                                         replace=False)
    else:
        sample_indices = np.random.choice(len(test_results['y_true']), 
                                         min(num_samples, len(test_results['y_true'])), 
                                         replace=False)
    
    for i, idx in enumerate(sample_indices):
        # 類似度を計算
        true_vector = test_results['y_true'][idx].reshape(1, -1).cpu().numpy()
        discrete_vector = test_results['y_pred_discrete'][idx].reshape(1, -1).cpu().numpy()
        discrete_sim = cosine_similarity(true_vector, discrete_vector)[0][0]
        
        # 1. 真のスペクトル
        plt.subplot(num_samples, 2, 2*i + 1)
        true_spec = test_results['y_true'][idx].numpy()
        
        # 原スペクトルを相対強度（%）に変換
        if np.max(true_spec) > 0:
            true_spec = true_spec / np.max(true_spec) * 100
        
        # 非ゼロの位置を強調
        nonzero_indices = np.nonzero(true_spec)[0]
        if len(nonzero_indices) > 0:
            plt.vlines(nonzero_indices, [0] * len(nonzero_indices), 
                     true_spec[nonzero_indices], colors='b', linewidths=1)
            
        # タイトルの設定
        mol_id_str = f" - ID: {test_results['mol_ids'][idx]}" if 'mol_ids' in test_results else ""
        plt.title(f"測定スペクトル{mol_id_str}")
        plt.xlabel("m/z")
        plt.ylabel("相対強度 (%)")
        plt.ylim([0, 105])  # 最大値100%に少し余裕を持たせる
        
        # 2. 離散化した予測スペクトル
        plt.subplot(num_samples, 2, 2*i + 2)
        discrete_spec = test_results['y_pred_discrete'][idx].numpy()
        
        # 非ゼロの位置を強調
        nonzero_indices = np.nonzero(discrete_spec)[0]
        if len(nonzero_indices) > 0:
            plt.vlines(nonzero_indices, [0] * len(nonzero_indices), 
                     discrete_spec[nonzero_indices], colors='g', linewidths=1)
            
        plt.title(f"予測スペクトル - 類似度: {discrete_sim:.4f}")
        plt.xlabel("m/z")
        plt.ylabel("相対強度 (%)")
        plt.ylim([0, 105])  # 最大値100%に少し余裕を持たせる
    
    plt.tight_layout()
    plt.savefig('dmpnn_spectrum_comparison.png')
    plt.close()

def tiered_training(model, train_ids, val_loader, criterion, optimizer, scheduler, device, 
                  mol_files_path, msp_data, transform, normalization, cache_dir, 
                  checkpoint_dir=CHECKPOINT_DIR, batch_size=16, patience=5):
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
        tier_epochs = [3, 3, 3, 3, 3]  # ティアごとのエポック数
    elif len(train_ids) > 50000:
        train_tiers = [
            train_ids[:10000], 
            train_ids[:30000],
            train_ids
        ]
        tier_epochs = [3, 4, 8]
    else:
        # 小さなデータセットは段階を少なく
        train_tiers = [
            train_ids[:5000] if len(train_ids) > 5000 else train_ids[:len(train_ids)//2],
            train_ids
        ]
        tier_epochs = [5, 10]
    
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
        tier_dataset = DMPNNMoleculeDataset(
            tier_ids, mol_files_path, msp_data, 
            transform=transform, normalization=normalization,
            augment=True, cache_dir=cache_dir
        )
        
        # ティアサイズに基づいてバッチサイズを調整
        if len(tier_ids) <= 10000:
            tier_batch_size = batch_size  # 小さいティアでは指定のバッチサイズ
        elif len(tier_ids) <= 30000:
            tier_batch_size = max(8, batch_size // 2)  # 中間ティア
        elif len(tier_ids) <= 60000:
            tier_batch_size = max(4, batch_size // 3)  # 大きいティア
        else:
            tier_batch_size = max(2, batch_size // 4)  # 非常に大きいティア
        
        logger.info(f"ティア {tier_idx+1} のバッチサイズ: {tier_batch_size}")
        
        # このティア用のデータローダを作成
        tier_loader = DataLoader(
            tier_dataset, 
            batch_size=tier_batch_size,
            shuffle=True, 
            collate_fn=collate_batch,
            num_workers=0,  # シングルプロセス
            pin_memory=True,
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
        plot_training_progress(all_train_losses, all_val_losses, all_val_cosine_similarities, 
                              best_cosine)
        logger.info(f"段階的トレーニングの学習曲線を保存: {final_plot_path}")
    except Exception as e:
        logger.error(f"プロット作成エラー: {str(e)}")
    
    return all_train_losses, all_val_losses, all_val_cosine_similarities, best_cosine

# ===== メイン関数 =====
def main():
    """メイン関数：データ読み込み、モデル作成、トレーニング、評価を実行"""
    # 開始メッセージ
    logger.info("============= DMPNN質量スペクトル予測モデルの実行開始 =============")
    
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
    mol_id_cache_file = os.path.join(CACHE_DIR, "dmpnn_valid_mol_ids.pkl")
    
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
                            # 非金属以外の原子を含むかチェック
                            mol_file = os.path.join(MOL_FILES_PATH, filename)
                            mol = Chem.MolFromMolFile(mol_file, sanitize=False)
                            if mol is not None and not contains_metal(mol):
                                mol_ids.append(mol_id)
                    except:
                        continue
                        
        # キャッシュに保存            
        logger.info(f"mol_idsをキャッシュに保存中: {mol_id_cache_file} (合計: {len(mol_ids)}件)")
        with open(mol_id_cache_file, 'wb') as f:
            pickle.dump(mol_ids, f)
    
    logger.info(f"MOLファイルとMSPデータが揃っている非金属化合物: {len(mol_ids)}個")
    
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
    val_dataset = DMPNNMoleculeDataset(
        val_ids, MOL_FILES_PATH, msp_data,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    test_dataset = DMPNNMoleculeDataset(
        test_ids, MOL_FILES_PATH, msp_data,
        transform=transform, normalization=normalization,
        augment=False, cache_dir=CACHE_DIR
    )
    
    logger.info(f"有効な検証データ: {len(val_dataset)}個")
    logger.info(f"有効なテストデータ: {len(test_dataset)}個")
    
    # GPU使用量を最適化するためのバッチサイズ調整
    if len(train_ids) > 100000:
        batch_size = 8  # 非常に大きなデータセット用
    elif len(train_ids) > 50000:
        batch_size = 12  # 大きなデータセット用
    else:
        batch_size = 16  # 通常サイズのデータセット用
        
    logger.info(f"バッチサイズ: {batch_size}")
    
    # データローダー作成 - 段階的トレーニングのため訓練ローダーは作成しない
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_batch,
        num_workers=0,  # シングルプロセス
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_batch,
        num_workers=0,  # シングルプロセス
        pin_memory=True,
        drop_last=True
    )
    
    # モデルの次元を決定
    sample = val_dataset[0]
    node_fdim = sample['graph_data'].x.shape[1]
    edge_fdim = sample['graph_data'].edge_attr.shape[1]
    
    # データセットサイズに基づいて次元を調整
    if len(train_ids) > 100000:
        hidden_size = 64  # 非常に大きなデータセット用に縮小
    else:
        hidden_size = 128  # 通常サイズのデータセット用
        
    out_channels = MAX_MZ
    
    # モデルの初期化前にメモリ確保
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    # モデルの初期化
    model = DMPNNMSPredictor(
        node_fdim=node_fdim,
        edge_fdim=edge_fdim,
        hidden_size=hidden_size,
        depth=3,  # DMPNNの深さ
        output_dim=out_channels,
        global_features_dim=16,
        num_fragments=NUM_FRAGS,
        bidirectional=True,     # 双方向予測を使用
        gate_prediction=True,   # ゲート予測を使用
        prec_mass_offset=10     # 前駆体質量オフセット
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
    patience = 7     # 忍耐値
    
    logger.info(f"モデルトレーニング設定: 忍耐値={patience}, バッチサイズ={batch_size}")
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
        batch_size=batch_size,
        patience=patience
    )
    
    logger.info(f"トレーニング完了！ 最良コサイン類似度: {best_cosine:.4f}")
    
    # キャッシュクリア
    aggressive_memory_cleanup(force_sync=True, purge_cache=True)
    
    # 最良モデルを読み込む
    try:
        best_model_path = os.path.join(CACHE_DIR, "checkpoints", 'best_model.pth')
        if not os.path.exists(best_model_path):
            # tier5_model.pthを明示的に探す
            tier5_model_path = os.path.join(CACHE_DIR, "checkpoints", "tier5_model.pth")
            if os.path.exists(tier5_model_path):
                best_model_path = tier5_model_path
            else:
                # tier5が無い場合は利用可能な最後のティアモデルを探す
                tier_models = [f for f in os.listdir(os.path.join(CACHE_DIR, "checkpoints")) 
                            if f.startswith("tier") and f.endswith("_model.pth")]
                if tier_models:
                    # 番号で並べ替えて最大のティアを選択
                    tier_numbers = [int(m.split("tier")[1].split("_")[0]) for m in tier_models]
                    max_tier_idx = tier_numbers.index(max(tier_numbers))
                    best_model_path = os.path.join(CACHE_DIR, "checkpoints", tier_models[max_tier_idx])
        
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"最良モデルを読み込みました: {best_model_path}")
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
    
    # テストデータでの評価
    try:
        # テスト前にメモリ解放
        aggressive_memory_cleanup(force_sync=True, purge_cache=True)
        
        logger.info("テストデータでの評価を開始します...")
        test_results = eval_model(model, test_loader, device, transform=transform)
        logger.info(f"テストデータ平均コサイン類似度 (元の予測): {test_results['cosine_similarity']:.4f}")
        logger.info(f"テストデータ平均コサイン類似度 (離散化後): {test_results['discrete_cosine_similarity']:.4f}")
        
        # 予測結果の可視化
        visualize_results(test_results, num_samples=10)
        logger.info("予測結果の可視化を保存しました: dmpnn_spectrum_comparison.png")
        
        # 性能比較グラフの保存
        plt.figure(figsize=(10, 6))
        plt.bar(['元の予測', '離散化予測'], 
               [test_results['cosine_similarity'], test_results['discrete_cosine_similarity']], 
               color=['blue', 'green'])
        plt.title('コサイン類似度の比較')
        plt.ylabel('平均コサイン類似度')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('dmpnn_similarity_comparison.png')
        plt.close()
        logger.info("コサイン類似度比較グラフを保存しました: dmpnn_similarity_comparison.png")
    except Exception as e:
        logger.error(f"テスト評価エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 追加の結果分析
    try:
        # スムーズな予測と離散化した予測の類似度分布の比較
        smooth_similarities = []
        discrete_similarities = []
        
        for i in range(len(test_results['y_true'])):
            true_vector = test_results['y_true'][i].reshape(1, -1).cpu().numpy()
            
            # スムーズな予測との類似度
            pred_vector = test_results['y_pred'][i].reshape(1, -1).cpu().numpy()
            smooth_sim = cosine_similarity(true_vector, pred_vector)[0][0]
            smooth_similarities.append(smooth_sim)
            
            # 離散化した予測との類似度
            discrete_vector = test_results['y_pred_discrete'][i].reshape(1, -1).cpu().numpy()
            discrete_sim = cosine_similarity(true_vector, discrete_vector)[0][0]
            discrete_similarities.append(discrete_sim)
        
        # 類似度分布のヒストグラム
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(smooth_similarities, bins=20, alpha=0.7, color='blue')
        plt.axvline(x=test_results['cosine_similarity'], color='r', linestyle='--', 
                    label=f'平均: {test_results["cosine_similarity"]:.4f}')
        plt.xlabel('コサイン類似度')
        plt.ylabel('サンプル数')
        plt.title('元の予測の類似度分布')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(discrete_similarities, bins=20, alpha=0.7, color='green')
        plt.axvline(x=test_results['discrete_cosine_similarity'], color='r', linestyle='--', 
                    label=f'平均: {test_results["discrete_cosine_similarity"]:.4f}')
        plt.xlabel('コサイン類似度')
        plt.ylabel('サンプル数')
        plt.title('離散化予測の類似度分布')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dmpnn_similarity_distributions.png')
        logger.info("類似度分布を保存しました: dmpnn_similarity_distributions.png")
        plt.close()
    except Exception as e:
        logger.error(f"追加分析中にエラー: {str(e)}")
    
    logger.info("============= DMPNN質量スペクトル予測モデルの実行終了 =============")
    return model, train_losses, val_losses, val_cosine_similarities, test_results

if __name__ == "__main__":
    main()
