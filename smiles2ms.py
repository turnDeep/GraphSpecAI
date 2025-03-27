import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

# RDKitの警告を抑制
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 既存のコードからの定数と関数のインポート
MAX_MZ = 2000
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

def unprocess_spec(spec, transform):
    """スペクトルの変換を元に戻す"""
    # transform signal
    if transform == "log10":
        max_ints = float(np.log10(1000. + 1.))
        def untransform_fn(x): return 10**x - 1.
    elif transform == "log10over3":
        max_ints = float(np.log10(1000. + 1.) / 3.)
        def untransform_fn(x): return 10**(3 * x) - 1.
    elif transform == "loge":
        max_ints = float(np.log(1000. + 1.))
        def untransform_fn(x): return torch.exp(x) - 1.
    elif transform == "sqrt":
        max_ints = float(np.sqrt(1000.))
        def untransform_fn(x): return x**2
    elif transform == "linear":
        raise NotImplementedError
    elif transform == "none":
        max_ints = 1000.
        def untransform_fn(x): return x
    else:
        raise ValueError("invalid transform")
        
    spec = spec / (torch.max(spec, dim=-1, keepdim=True)[0] + EPS) * max_ints
    spec = untransform_fn(spec)
    spec = torch.clamp(spec, min=0.)
    assert not torch.isnan(spec).any()
    return spec

def smiles_to_graph(smiles):
    """SMILES文字列から分子グラフを生成"""
    # RDKitでSMILESを分子に変換
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    
    # 3D構造の生成（必須ではないが、より良い特徴量を得るため）
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    
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
        
        # 簡素化した特徴リスト
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
        mol_features[0] = Chem.Descriptors.MolWt(mol) / 1000.0  # 分子量
    except:
        pass
        
    try:
        mol_features[1] = Chem.Descriptors.NumHAcceptors(mol) / 20.0  # 水素結合アクセプター数
    except:
        pass
        
    try:
        mol_features[2] = Chem.Descriptors.NumHDonors(mol) / 10.0  # 水素結合ドナー数
    except:
        pass
        
    try:
        mol_features[3] = Chem.Descriptors.TPSA(mol) / 200.0  # トポロジカル極性表面積
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
    
    return graph_data, mol

def predict_spectrum_from_smiles(model, smiles, device='cuda', transform="log10over3", normalization="l1", threshold=1.0):
    """SMILESから質量スペクトルを予測"""
    # モデルを評価モードに設定
    model.eval()
    
    # SMILESから分子グラフを生成
    try:
        graph_data, mol = smiles_to_graph(smiles)
    except Exception as e:
        print(f"SMILESの変換中にエラーが発生しました: {e}")
        return None, None
    
    # 前駆体m/zの計算（分子量）
    mol_weight = Chem.Descriptors.ExactMolWt(mol)
    prec_mz = int(mol_weight)
    prec_mz_bin = min(prec_mz, MAX_MZ-1)  # MAXを超えないように制限
    
    # バッチを作成
    graph_data = graph_data.to(device)
    batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=device)
    
    # モデル入力の準備
    processed_batch = {
        'graph': graph_data,
        'prec_mz': torch.tensor([float(prec_mz)], device=device),
        'prec_mz_bin': torch.tensor([prec_mz_bin], dtype=torch.long, device=device)
    }
    
    # 予測
    with torch.no_grad():
        output, _ = model(processed_batch)
    
    # 予測スペクトルの後処理
    pred_spec = output[0].cpu()
    
    # 変換を元に戻す
    pred_spec_unprocessed = unprocess_spec(pred_spec.unsqueeze(0), transform)[0]
    
    # 閾値以下の値をゼロに
    pred_spec_unprocessed[pred_spec_unprocessed < threshold] = 0
    
    return pred_spec_unprocessed.numpy(), mol

def plot_predicted_spectrum(spectrum, mol, title=None, save_path=None, show=True, ylim=None):
    """予測スペクトルのプロット"""
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    # 分子の名前またはSMILESを取得
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else Chem.MolToSmiles(mol)
    
    # 非ゼロのピークのみを抽出
    mz_values = np.arange(len(spectrum))
    peak_indices = np.where(spectrum > 0)[0]
    peak_intensities = spectrum[peak_indices]
    
    # スペクトルを描画
    if len(peak_indices) > 0:
        ax.stem(peak_indices, peak_intensities, markerfmt=" ", basefmt="b-", linefmt="b-")
    else:
        ax.plot(mz_values, spectrum, 'b-')
    
    # y軸の範囲を設定
    if ylim:
        ax.set_ylim(0, ylim)
    
    # ラベルと凡例
    ax.set_xlabel("m/z", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    
    # タイトル設定
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Predicted Mass Spectrum for {mol_name}", fontsize=14)
    
    # グリッド表示
    ax.grid(alpha=0.3)
    
    # 主要なピークにラベルを付ける
    if len(peak_indices) > 0:
        # 上位10個のピークを見つける
        top_n = min(10, len(peak_indices))
        top_indices = np.argsort(peak_intensities)[-top_n:]
        
        for idx in top_indices:
            mz = peak_indices[idx]
            intensity = peak_intensities[idx]
            ax.text(mz, intensity, f"{mz}", ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def load_model(model_path, device='cuda'):
    """トレーニング済みモデルを読み込む"""
    from gsai0327 import OptimizedHybridMSModel
    
    # モデルの初期化（ハイパーパラメータは元のモデルと同じに）
    node_features = 35  # ATOM_FEATURES (23) + additional_features (12)
    edge_features = 7   # BOND_FEATURES (4) + additional_bond_features (3)
    hidden_channels = 32
    out_channels = MAX_MZ
    
    model = OptimizedHybridMSModel(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_fragments=167,  # MACCSキーのビット数
        prec_mass_offset=10,
        bidirectional=True,
        gate_prediction=True
    ).to(device)
    
    # モデルの読み込み
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model

def predict_and_plot_from_smiles(smiles, model_path='best_model.pth', device='cuda', 
                               save_path=None, show=True, transform="log10over3", 
                               normalization="l1", threshold=1.0):
    """SMILESから質量スペクトルを予測してプロット"""
    # デバイスの設定
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # モデルの読み込み
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        return None
    
    # スペクトル予測
    predicted_spectrum, mol = predict_spectrum_from_smiles(
        model, smiles, device, transform, normalization, threshold
    )
    
    if predicted_spectrum is None:
        print("スペクトルの予測に失敗しました。")
        return None
    
    # 予測結果のプロット
    fig = plot_predicted_spectrum(
        predicted_spectrum, mol, 
        title=f"Predicted Mass Spectrum for {smiles}", 
        save_path=save_path, show=show
    )
    
    return predicted_spectrum, fig

# 使用例
if __name__ == "__main__":
    # サンプルSMILES
    smiles_examples = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # アスピリン
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # カフェイン
        "CCO",  # エタノール
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # イブプロフェン
    ]
    
    # モデルのパス（実際の環境に合わせて変更）
    model_path = "data/cache/checkpoints/best_model.pth"
    
    for i, smiles in enumerate(smiles_examples):
        print(f"Processing SMILES {i+1}: {smiles}")
        save_path = f"predicted_spectrum_{i+1}.png"
        
        spectrum, fig = predict_and_plot_from_smiles(
            smiles, model_path=model_path, 
            save_path=save_path, show=False
        )
        
        if spectrum is not None:
            print(f"予測完了。画像を保存: {save_path}")
        print("-----")