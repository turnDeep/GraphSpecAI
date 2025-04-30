# パート3: 実行コードとユーティリティ関数

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
from rdkit.Chem import AllChem, BRICS, MolToSmiles, MurckoScaffold, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import copy
import math
import time
import argparse
import json
from io import BytesIO
from PIL import Image
from typing import Dict, List, Tuple, Set, Union, Optional
from collections import defaultdict, deque
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
import itertools
import logging
import datetime
import sys
from mpl_toolkits.mplot3d import Axes3D

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
        distance = Levenshtein.distance(smiles1, smiles2)
        max_len = max(len(smiles1), len(smiles2))
        similarity = 1 - (distance / max_len) if max_len > 0 else 0
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

if __name__ == "__main__":
    # コマンドライン引数
    parser = argparse.ArgumentParser(description="Chemical Structure-Mass Spectrum Self-Growing Model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--model-path", type=str, help="Path to pre-trained model")
    parser.add_argument("--cpu", action="store_true", help="Use CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    main(args)