# ベースイメージの指定 (PyTorchとCUDAの最新版を含むイメージ)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 作業ディレクトリの設定
WORKDIR /app

# 必要なライブラリのインストール
RUN apt-get update && apt-get install -y     wget     unzip     git     && rm -rf /var/lib/apt/lists/*

# RDKitのインストール (完全版を使用してIPythonConsoleとDataStructsを含める)
RUN pip install rdkit

# PyTorch Geometricのインストール (CUDA対応バージョン)
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# 科学計算用ライブラリのインストール
RUN pip install scipy scikit-learn pandas

# トランスフォーマーと関連ライブラリのインストール (LLM-massformer用)
RUN pip install transformers

# プロット・可視化ライブラリのインストール
RUN pip install matplotlib seaborn

# データ処理とユーティリティのインストール
RUN pip install tqdm joblib pyyaml Pillow python-Levenshtein

# Jupyter関連ライブラリのインストール
RUN pip install jupyterlab jupytext ipywidgets nbformat

# LLM-massformerとGPTモデルの特別なライブラリ (正しいバージョンを指定)
RUN pip install tokenizers==0.13.3 tensorboard

# 3つのモデルで必要な追加パッケージのインストール
RUN pip install networkx einops

# 必要なファイルのコピー
COPY analysis.ipynb /app/
COPY data /app/data
COPY gsai0527.py /app/

# ポートの公開 (Jupyter Labを使用する場合)
# EXPOSE 8888

# スクリプトの実行
CMD ["python", "gsai0527.py"]
