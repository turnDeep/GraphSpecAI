# GraphSpecAI

GraphSpecAI は分子構造からマススペクトルを予測するための深層学習フレームワークです。Graph Neural Networks (GNNs) を使用して分子をグラフとして表現し、高精度なマススペクトル予測を可能にします。

## 概要

質量分析法は化合物の同定や構造解析に広く使用される分析手法ですが、すべての分子について実験的なマススペクトルを取得するのは時間とコストがかかります。GraphSpecAI は機械学習を活用して分子構造からマススペクトルを予測することでこの課題に対応します。

主な特徴:
- 分子構造の表現学習のためのGraph Neural Networks (GNNs)
- 重要な分子サブ構造に注目するアテンションメカニズム
- フラグメントパターン予測との同時最適化のためのマルチタスク学習
- 予測精度を向上させるアンサンブル学習
- コサイン類似度ベースの評価指標

## モデルの種類

GraphSpecAI は3つの異なるモデルを提供しています：

1. **GCN_model**: 基本的なGraph Convolutional Network (GCN)を使用した初期モデル。シンプルな構造でマススペクトル予測の基礎を提供します。

2. **Generalized_model**: より高度なGraph Attention Network (GAT)と残差接続を組み合わせた汎用モデル。幅広い分子に対して高い予測精度を実現します。

3. **Specialized_model**: 特定の分子ファミリーに特化した最適化モデル。クラスタリングと転移学習を活用して、特定のターゲット分子に対する予測精度を最大化します。

## 技術的詳細

このプロジェクトは以下の技術を活用しています：

- **Graph Attention Networks (GATv2)**: 分子グラフ構造を効率的に学習
- **アテンションメカニズム**: 分子内の重要なサブ構造に焦点を当てる
- **残差接続**: 深層ネットワークの学習を安定化
- **マルチタスク学習**: マススペクトル予測とフラグメントパターン予測を同時に学習
- **アンサンブル学習**: 複数のモデルからの予測を組み合わせて精度を向上
- **コサイン類似度損失**: マススペクトル予測のための特殊な損失関数

## Docker を使った環境構築

GraphSpecAI は Docker を使って簡単に環境構築することができます。

### 前提条件

- Docker がインストールされていること
- Docker Compose (オプション)
- NVIDIA GPU + CUDA (推奨、GPU がない場合は CPU モードで動作)

### ビルドと実行

1. リポジトリをクローン:
```bash
git clone https://github.com/your-username/GraphSpecAI.git
cd GraphSpecAI
```

2. Docker イメージのビルド:
```bash
docker build -t graphspecai .
```

3. コンテナの実行:
```bash
docker run --gpus all -it -p 8888:8888 -v $(pwd)/data:/app/data graphspecai
```

起動すると、次のオプションが表示されます:
```
利用可能なモデル:
1. GCN_model.py
2. Generalized_model.py
3. Specialized_model.py
4. Jupyter Lab

選択してください (1-4):
```

### データセット構造

データは以下のディレクトリ構造で配置してください:

```
data/
├── mol_files/
│   ├── ID200001.MOL
│   ├── ID200002.MOL
│   └── ...
└── NIST17.MSP
```

- `mol_files/`: 分子構造ファイル (MOL形式)
- `NIST17.MSP`: マススペクトルデータ (MSP形式)

## 各モデルの使用方法

### 1. GCN_model

基本的なGraph Convolutional Networkを使用した初期モデルです。

```bash
# Dockerコンテナ内でオプション1を選択
python GCN_model.py
```

または、Jupyter Labで対話的に実行:
```bash
# Dockerコンテナ内でオプション4を選択し、ブラウザでGCN_model.ipynbを開く
```

### 2. Generalized_model

より高度なネットワーク構造を持つ汎用モデルです。

```bash
# Dockerコンテナ内でオプション2を選択
python Generalized_model.py
```

### 3. Specialized_model

特定の分子ファミリーに特化したモデルです。

```bash
# Dockerコンテナ内でオプション3を選択
python Specialized_model.py
```

## カスタマイズ

主要なパラメータ:

- `NUM_FRAGS`: フラグメントパターンの数
- `MAX_MZ`: 最大m/z値
- `IMPORTANT_MZ`: 重視するm/z値のリスト
- `hidden_channels`: モデルの隠れ層のサイズ
- `num_models`: アンサンブルするモデルの数
- `num_epochs`: トレーニングのエポック数

## 出力結果

各モデルの評価結果は以下のディレクトリに出力されます:

- **models/**: 学習済みモデルファイル
- **results/**: 評価スコアやトレーニング履歴のJSON
- **plots/**: 損失曲線や予測スペクトルのグラフ

## コード構造

- **データ処理**:
  - `MoleculeGraphDataset`: 分子からグラフへの変換
  - `parse_msp_file`: MSPファイルの解析

- **モデル**:
  - `HybridGNNModel`: GNN、CNN、Transformerを組み合わせたハイブリッドモデル
  - `AttentionBlock`: アテンションメカニズム
  - `ResidualBlock`: 残差ブロック
  - `ModelEnsemble`: 複数モデルのアンサンブル

- **損失関数**:
  - `peak_weighted_cosine_loss`: ピーク重み付きコサイン類似度損失
  - `combined_loss`: MSEとコサイン類似度損失の組み合わせ

- **評価**:
  - `cosine_similarity_score`: コサイン類似度によるモデル評価

## ライセンス

このプロジェクトは [MIT License](LICENSE) の下でリリースされています。

## 引用

研究でこのプロジェクトを使用する場合は、以下のように引用してください:

```
GraphSpecAI: A Deep Learning Framework for Mass Spectrum Prediction from Molecular Structures
https://github.com/DeepMassSpec/GraphSpecAI
```

## コントリビューション

バグ報告とプルリクエストを歓迎します。大きな変更を行う場合は、まず変更したい内容について議論するためにissueを開いてください。

---

最終更新: 2025年3月
