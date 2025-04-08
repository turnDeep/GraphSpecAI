# -*- coding: utf-8 -*-
"""
EIマススペクトルから化学構造を予測するAIモデルのフレームワーク
データ: NIST17 (MSP形式のスペクトル、MOL形式の構造)
手法: DMPNNを用いたグラフ表現、スペクトルからの構造特徴予測
"""

import os
import re
import numpy as np
from tqdm import tqdm  # プログレスバー表示用

# --- RDKit ---
# 化学構造の読み込みと処理に必要
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _rdkit_available = True
except ImportError:
    print("警告: RDKit がインストールされていません。化学構造の処理はスキップされます。")
    print("インストールするには: conda install -c conda-forge rdkit")
    _rdkit_available = False
    Chem = None  # RDKitが利用できない場合のプレースホルダー

# --- PyTorch & PyTorch Geometric ---
# 深層学習モデルの構築とトレーニングに必要
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    # torch_geometric はDMPNNのようなグラフニューラルネットワークの実装に便利
    # from torch_geometric.data import Data
    # from torch_geometric.nn import MessagePassing # DMPNNの基底クラス
    _torch_available = True
except ImportError:
    print("警告: PyTorch または PyTorch Geometric がインストールされていません。モデル関連の処理はスキップされます。")
    print("インストールするには:")
    print("PyTorch: https://pytorch.org/get-started/locally/")
    print("PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    _torch_available = False
    # PyTorch/PyGが利用できない場合のプレースホルダー
    nn = None
    optim = None
    Dataset = object
    DataLoader = None
    # Data = object
    # MessagePassing = object


# --- 定数 ---
MSP_FILE_PATH = "data/NIST17.MSP"  # MSPファイルのパス (適宜変更)
MOL_FILES_DIR = "data/mol_files"  # MOLファイルが格納されているディレクトリ (適宜変更)
MAX_MZ = 2000  # 考慮する最大m/z値
SPECTRUM_VECTOR_SIZE = MAX_MZ + 1 # スペクトルベクトルのサイズ (m/z 0 から MAX_MZ まで)

# --- 1. データ読み込み ---

def parse_msp_file(filepath):
    """
    NIST MSPファイルを解析し、各化合物の情報とスペクトルデータを抽出する。
    matchmsライブラリの使用を推奨しますが、ここでは基本的なパーサーを実装します。

    Args:
        filepath (str): MSPファイルのパス。

    Returns:
        list: 各要素が化合物情報とスペクトルを含む辞書のリスト。
              例: [{'name': '...', 'id': '...', 'peaks': [(mz1, intensity1), ...]}, ...]
    """
    compounds = []
    current_compound = {}
    reading_peaks = False

    print(f"MSPファイルを解析中: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:  # 空行はレコードの終わりを示す場合がある
                    if current_compound and 'peaks' in current_compound:
                        compounds.append(current_compound)
                    current_compound = {}
                    reading_peaks = False
                    continue

                if ':' in line and not reading_peaks:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == 'name':
                        current_compound['name'] = value
                    elif key == 'id':
                        current_compound['id'] = value
                    elif key == 'formula':
                        current_compound['formula'] = value
                    elif key == 'mw':
                        current_compound['mw'] = value
                    elif key == 'casno':
                        current_compound['casno'] = value
                    # 他のメタデータも必要に応じて追加
                    elif key == 'num peaks':
                        reading_peaks = True
                        current_compound['peaks'] = []
                elif reading_peaks:
                    # ピークデータの形式は様々 (スペース区切り、タブ区切り、セミコロン区切りなど)
                    # ここではスペースまたはタブで区切られ、2つの数値がある場合を想定
                    parts = re.split(r'[;\s]+', line)
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            # m/zが範囲内であれば追加
                            if 0 <= mz <= MAX_MZ:
                                current_compound['peaks'].append((mz, intensity))
                        except ValueError:
                            # 数値に変換できない行は無視
                            pass

            # ファイル末尾の最後のレコードを追加
            if current_compound and 'peaks' in current_compound:
                compounds.append(current_compound)

    except FileNotFoundError:
        print(f"エラー: MSPファイルが見つかりません: {filepath}")
        return []
    except Exception as e:
        print(f"MSPファイルの解析中にエラーが発生しました: {e}")
        return []

    print(f"MSPファイルから {len(compounds)} 件の化合物を読み込みました。")
    return compounds

def load_mol_file(compound_id, mol_dir):
    """
    指定された化合物IDに対応するMOLファイルを読み込む。

    Args:
        compound_id (str): 化合物ID (MSPファイルから取得)。
        mol_dir (str): MOLファイルが格納されているディレクトリ。

    Returns:
        rdkit.Chem.Mol or None: 読み込んだMolオブジェクト、またはエラーの場合はNone。
    """
    if not _rdkit_available or not compound_id:
        return None

    mol_filename = f"ID{compound_id}.MOL" # ファイル名の形式を仮定
    mol_filepath = os.path.join(mol_dir, mol_filename)

    if not os.path.exists(mol_filepath):
        # print(f"警告: MOLファイルが見つかりません: {mol_filepath}")
        return None

    try:
        # sanitize=True で分子の妥当性をチェックし、問題を修正
        mol = Chem.MolFromMolFile(mol_filepath, sanitize=True)
        if mol is None:
            # print(f"警告: MOLファイルの読み込みに失敗しました: {mol_filepath}")
            return None
        return mol
    except Exception as e:
        print(f"MOLファイル {mol_filepath} の読み込み中にエラーが発生しました: {e}")
        return None

# --- 2. データ前処理 ---

def spectrum_to_vector(peaks, vector_size):
    """
    ピークリストを固定長のベクトルに変換する（ビン化）。

    Args:
        peaks (list): (m/z, intensity) のタプルのリスト。
        vector_size (int): 出力ベクトルのサイズ (MAX_MZ + 1)。

    Returns:
        np.ndarray: スペクトルベクトル。
    """
    vector = np.zeros(vector_size, dtype=np.float32)
    if not peaks:
        return vector

    total_intensity = sum(intensity for mz, intensity in peaks)
    if total_intensity == 0:
        return vector # 強度がゼロの場合はゼロベクトルを返す

    for mz, intensity in peaks:
        # m/z値を最も近い整数インデックスに丸める
        index = int(round(mz))
        if 0 <= index < vector_size:
            # 強度をベクトルに追加（同じm/zに複数のピークがある場合は加算）
            # 正規化（例：平方根を取ってから総強度で割る）は一般的
            vector[index] += np.sqrt(intensity)

    # ベクトル全体をL1ノルムで正規化（総和が1になるように）
    norm = np.sum(vector)
    if norm > 0:
        vector /= norm

    return vector

# --- RDKitを用いた分子グラフ表現 (DMPNN用) ---
# この部分はDMPNNの実装に合わせて調整が必要
# PyTorch Geometric を使うと、原子と結合の特徴量抽出が容易になる

def get_atom_features(atom):
    """原子の特徴量を抽出する（例）"""
    if not _rdkit_available: return []
    # 例: 原子番号、価電子数、形式電荷、ラジカル電子数、ハイブリダイゼーション、芳香族性、環に含まれるか
    features = [
        atom.GetAtomicNum(),
        atom.GetTotalValence(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetHybridization(), # RDKitの型を数値に変換する必要あり
        atom.GetIsAromatic(),
        atom.IsInRing(),
    ]
    # ハイブリダイゼーションをワンホットエンコーディングなどで数値化
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
        Chem.rdchem.HybridizationType.S: 6, # 他にもあるかもしれない
    }
    features[4] = hybridization_map.get(features[4], 0)
    return [float(f) for f in features] # すべて数値に

def get_bond_features(bond):
    """結合の特徴量を抽出する（例）"""
    if not _rdkit_available: return []
    # 例: 結合タイプ、共役しているか、環に含まれるか
    features = [
        bond.GetBondTypeAsDouble(), # 1.0, 1.5, 2.0, 3.0
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    return [float(f) for f in features]

def mol_to_graph_representation(mol):
    """
    RDKitのMolオブジェクトをグラフ表現（ノード特徴量、エッジインデックス、エッジ特徴量）に変換する。
    PyTorch Geometric の Data オブジェクト形式に変換するのが一般的。

    Args:
        mol (rdkit.Chem.Mol): RDKitのMolオブジェクト。

    Returns:
        tuple or None: (ノード特徴量行列, エッジインデックス行列, エッジ特徴量行列) のタプル、
                       またはエラーの場合はNone。PyGを使う場合は Data オブジェクトを返す。
    """
    if not _rdkit_available or mol is None:
        return None

    try:
        # ノード（原子）の特徴量
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float) if _torch_available else np.array(atom_features)

        # エッジ（結合）のインデックスと特徴量
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # 無向グラフなので両方向のエッジを追加
            edge_indices.append((i, j))
            edge_indices.append((j, i))
            # 結合特徴量も両方向に追加
            bond_feat = get_bond_features(bond)
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)

        # PyTorch Geometric 形式 (L x 2)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if _torch_available else np.array(edge_indices).T
        # エッジ特徴量
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if _torch_available else np.array(edge_features)

        # PyTorch Geometric の Data オブジェクトを作成する場合:
        # if _torch_available:
        #     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        #     return data
        # else:
        #     return x, edge_index, edge_attr # PyGがない場合はタプルで返す

        # ここでは簡単のためタプルで返す
        return x, edge_index, edge_attr

    except Exception as e:
        print(f"分子のグラフ表現への変換中にエラー: {e}")
        return None


# --- 3. データセットとデータローダー (PyTorch) ---

class MassSpecDataset(Dataset):
    """マススペクトルと対応する化学構造（グラフ表現）のデータセット"""
    def __init__(self, msp_data, mol_dir, spectrum_vector_size):
        super().__init__() # Datasetの初期化を呼び出す
        self.msp_data = msp_data
        self.mol_dir = mol_dir
        self.spectrum_vector_size = spectrum_vector_size
        self.valid_indices = self._prepare_data() # 有効なデータのみのインデックス

    def _prepare_data(self):
        """
        MSPデータとMOLデータを照合し、有効なペアのインデックスリストを作成する。
        """
        valid_indices = []
        print("データの前処理と検証を開始...")
        for idx, compound in enumerate(tqdm(self.msp_data)):
            compound_id = compound.get('id')
            if not compound_id: continue # IDがないデータはスキップ

            # MOLファイルをロード
            mol = load_mol_file(compound_id, self.mol_dir)
            if mol is None: continue # MOLファイルがない、または読めないデータはスキップ

            # スペクトルベクトルを作成
            spectrum_vec = spectrum_to_vector(compound.get('peaks', []), self.spectrum_vector_size)
            if np.sum(spectrum_vec) == 0: continue # 空のスペクトルはスキップ

            # グラフ表現を作成 (ここでは作成のみで保存はしない。__getitem__で再作成)
            graph_repr = mol_to_graph_representation(mol)
            if graph_repr is None: continue # グラフ表現が作れないものはスキップ

            # すべてOKなら、このインデックスを有効とする
            valid_indices.append(idx)

        print(f"有効なデータペア数: {len(valid_indices)} / {len(self.msp_data)}")
        return valid_indices

    def __len__(self):
        """データセットのサイズ（有効なペアの数）を返す"""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """指定されたインデックスに対応するデータペアを返す"""
        # 有効なインデックスリストから実際のMSPデータのインデックスを取得
        actual_idx = self.valid_indices[idx]
        compound = self.msp_data[actual_idx]

        # スペクトルベクトル
        spectrum_vec = spectrum_to_vector(compound.get('peaks', []), self.spectrum_vector_size)
        spectrum_tensor = torch.tensor(spectrum_vec, dtype=torch.float)

        # 化学構造グラフ表現 (再度ロード＆変換)
        # 注意: 大規模データの場合、前処理済みのグラフデータを保存しておき、
        #       ここでのロード/変換を避ける方が効率的
        compound_id = compound.get('id')
        mol = load_mol_file(compound_id, self.mol_dir)
        graph_repr = mol_to_graph_representation(mol) # (x, edge_index, edge_attr) のタプル

        # DMPNNモデルがグラフ全体を処理する場合、グラフ表現をそのまま返す
        # ここでは例として、グラフから計算される「ターゲット特徴量」を返すことを想定
        # 例えば、事前に計算した分子フィンガープリントや、別のGNNで計算したグラフ埋め込みベクトルなど
        # この例では、グラフ表現のタプルをそのまま返す (モデル側で処理)
        # PyGのDataオブジェクトを使う場合はそれを返す

        # return spectrum_tensor, graph_repr # (スペクトルベクトル, (ノード特徴, エッジindex, エッジ特徴))

        # --- 仮のターゲット ---
        # 実際の応用では、ここでグラフからターゲットとなるベクトル
        # (例: Morganフィンガープリント、事前学習済みGNNによる埋め込み) を計算する
        # ここではダミーとしてノード特徴量の平均を返す
        if graph_repr is not None and _torch_available:
            node_features, _, _ = graph_repr
            # 特徴量が存在し、空でないことを確認
            if node_features is not None and node_features.numel() > 0:
                 # 特徴量の次元が不揃いな場合があるため、平均を取る前に次元を確認・調整が必要な場合がある
                 # ここでは単純に平均を取る
                 target_vector = torch.mean(node_features, dim=0)
                 # ターゲットベクトルのサイズが固定になるように注意（パディングや次元削減が必要な場合も）
                 # 例として最初の 128 次元を使う (サイズが足りない場合は0埋め)
                 fixed_size_target = torch.zeros(128, dtype=torch.float)
                 actual_len = min(target_vector.shape[0], 128)
                 fixed_size_target[:actual_len] = target_vector[:actual_len]

            else:
                # ノード特徴量がない場合はゼロベクトル
                fixed_size_target = torch.zeros(128, dtype=torch.float)

            return spectrum_tensor, fixed_size_target
        else:
            # グラフ表現が得られない場合は、ダミーデータを返すか、エラー処理が必要
            # このデータは実際にはフィルタリングされるべき
            return spectrum_tensor, torch.zeros(128, dtype=torch.float)


# --- 4. モデル定義 (PyTorch) ---

class SpectrumToStructurePredictor(nn.Module):
    """
    マススペクトルベクトルから化学構造の特徴（例：グラフ埋め込み）を予測するモデル。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        if not _torch_available:
            print("警告: PyTorchが利用できないため、モデルを初期化できません。")
            return

        # 簡単な多層パーセプトロン(MLP)をスペクトルエンコーダーとして使用
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # 過学習防止
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim) # 出力層（ターゲット特徴量の次元に合わせる）
        )
        # 出力層の活性化関数はターゲットによる（例：埋め込みなら線形、分類ならSoftmax）

        # --- DMPNN部分の統合について ---
        # このモデルはスペクトルから「構造の特徴ベクトル」を予測する。
        # DMPNNは通常、構造（グラフ）を入力として特徴ベクトルを出力する。
        # 目的：「スペクトル」->「構造」
        # アプローチ案：
        # 1. 「スペクトル」-> MLP -> 「構造の特徴ベクトル（目標）」
        #    別途、DMPNNで「構造」->「構造の特徴ベクトル（目標）」を学習 or 事前計算。
        #    MLPの出力とDMPNNの出力が近くなるように学習（MSE損失など）。
        # 2. 「スペクトル」-> Encoder -> latent_spec, 「構造」-> DMPNN -> latent_struct
        #    latent_spec と latent_struct が近くなるように学習（Contrastive Lossなど）。
        # 3. End-to-End: スペクトルから直接グラフを生成するモデル（より高度）。
        #
        # このコード例ではアプローチ1を想定し、ターゲットベクトルをデータセットで準備。

    def forward(self, spectrum_vector):
        """順伝播"""
        if not _torch_available: return None
        predicted_features = self.encoder(spectrum_vector)
        return predicted_features

# --- 5. トレーニング ---

def train_model(model, dataloader, optimizer, criterion, device):
    """モデルのトレーニングを行う関数（1エポック分）"""
    if not _torch_available: return 0.0
    model.train() # トレーニングモード
    total_loss = 0.0

    for spectrum_batch, target_batch in tqdm(dataloader, desc="Training"):
        spectrum_batch = spectrum_batch.to(device)
        target_batch = target_batch.to(device) # ターゲットもデバイスへ

        optimizer.zero_grad() # 勾配をリセット

        # モデルによる予測
        predictions = model(spectrum_batch)

        # 損失計算
        # ターゲットがグラフ表現そのものではなく、ベクトル化されていると仮定
        loss = criterion(predictions, target_batch)

        loss.backward() # 誤差逆伝播
        optimizer.step() # パラメータ更新

        total_loss += loss.item() * spectrum_batch.size(0) # バッチサイズを考慮

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

# --- メイン処理 ---

if __name__ == "__main__":
    print("--- マススペクトル言語AI フレームワーク ---")

    # 1. データの読み込み
    print("\n[ステップ1: データ読み込み]")
    msp_compounds = parse_msp_file(MSP_FILE_PATH)

    if not msp_compounds:
        print("MSPデータの読み込みに失敗したため、処理を終了します。")
        exit()

    if not _rdkit_available:
        print("RDKitが利用できないため、化学構造の処理は行えません。")
        # RDKitなしで実行できる処理を続けるか、終了するか選択

    if not _torch_available:
        print("PyTorchが利用できないため、モデルのトレーニングは行えません。")
        # PyTorchなしで実行できる処理を続けるか、終了するか選択

    # 2. データセットの準備 (PyTorchが利用可能な場合)
    if _torch_available and _rdkit_available:
        print("\n[ステップ2: データセット準備]")
        dataset = MassSpecDataset(msp_compounds, MOL_FILES_DIR, SPECTRUM_VECTOR_SIZE)

        if len(dataset) == 0:
            print("有効な学習データが見つかりませんでした。データパスや形式を確認してください。")
            exit()

        # データローダーの作成
        batch_size = 64 # バッチサイズ (GPUメモリに応じて調整)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # num_workers > 0 でデータのロードを高速化 (環境による)
        # pin_memory=True でGPUへの転送を高速化

        # 3. モデルの準備
        print("\n[ステップ3: モデル準備]")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {device}")

        # モデルパラメータ (仮)
        input_dim = SPECTRUM_VECTOR_SIZE # スペクトルベクトルの次元
        hidden_dim = 512 # 中間層の次元
        output_dim = 128 # 予測するターゲット特徴量の次元 (データセットの __getitem__ と合わせる)

        model = SpectrumToStructurePredictor(input_dim, hidden_dim, output_dim).to(device)
        print(model)

        # 損失関数と最適化手法
        # ターゲットが連続値ベクトル（埋め込みなど）の場合、MSE損失やCosine類似度損失が一般的
        criterion = nn.MSELoss()
        # criterion = nn.CosineEmbeddingLoss() # Cosine類似度を使う場合、ターゲットは通常(batch_size)の1 or -1
        optimizer = optim.Adam(model.parameters(), lr=0.001) # 学習率

        # 4. トレーニングループ
        print("\n[ステップ4: トレーニング開始]")
        num_epochs = 10 # エポック数 (仮)

        for epoch in range(num_epochs):
            avg_loss = train_model(model, dataloader, optimizer, criterion, device)
            print(f"エポック [{epoch+1}/{num_epochs}], 平均損失: {avg_loss:.4f}")

            # TODO: 検証データセットでの評価、モデルの保存などを追加

        print("\nトレーニング完了。")

        # 5. 予測（推論）
        # トレーニング済みモデルを使って、未知のスペクトルから構造特徴を予測する
        # 例:
        # model.eval() # 評価モード
        # with torch.no_grad():
        #     test_spectrum = torch.rand(1, input_dim).to(device) # ダミーのテストスペクトル
        #     predicted_struct_features = model(test_spectrum)
        #     print("予測された構造特徴量:", predicted_struct_features)
        #     # この特徴量を使って、データベース内の既知の構造特徴量と比較し、
        #     # 最も類似する化学構造を検索するなどの応用が考えられる。

    else:
        print("\n必要なライブラリ (RDKit, PyTorch) が不足しているため、モデルの構築とトレーニングはスキップされました。")

    print("\n--- 処理終了 ---")