# ベースイメージの指定 (PyTorchとCUDAの最新版を含むイメージ)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 作業ディレクトリの設定
WORKDIR /app

# 必要なライブラリのインストール
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# RDKitのインストール (ビルド不要のPyPIパッケージを使用)
RUN pip install rdkit

# PyTorch Geometricのインストール (CUDA 11.8に対応したバージョン)
RUN pip install torch_geometric

# Jupyter Lab, matplotlib, scikit-learnのインストール
RUN pip install jupyterlab matplotlib scikit-learn

# 必要なファイルのコピー
COPY analysis.ipynb /app/
COPY data /app/data

# ポートの公開 (Jupyter Labを使用する場合)
EXPOSE 8888

# Jupyter Labの起動
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]