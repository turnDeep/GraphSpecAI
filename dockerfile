# ベースイメージの指定 (PyTorchとCUDAの最新版を含むイメージ)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 作業ディレクトリの設定
WORKDIR /app

# 必要なライブラリのインストール
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# RDKitのインストール (ビルド不要のPyPIパッケージを使用)
RUN pip install rdkit

# PyTorch Geometricのインストール (CUDA対応バージョン)
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# 3つのモデルで必要な追加パッケージのインストール
RUN pip install jupyterlab matplotlib scikit-learn pandas tqdm seaborn ipywidgets nbformat

# .inpybファイルをPythonスクリプトに変換するためのツールをインストール
RUN pip install jupytext

# アプリケーションファイルのコピー
COPY GCN_model.inpyb /app/
COPY Generalized_model.inpyb /app/
COPY Specialized_model.inpyb /app/
COPY analysis.ipynb /app/

# データディレクトリを作成
RUN mkdir -p /app/data/mol_files

# 必要なその他のデータディレクトリを作成
RUN mkdir -p /app/models /app/results /app/plots

# inpybファイルをPythonスクリプトに変換
RUN jupytext --to py GCN_model.inpyb -o GCN_model.py
RUN jupytext --to py Generalized_model.inpyb -o Generalized_model.py
RUN jupytext --to py Specialized_model.inpyb -o Specialized_model.py

# 実行権限を付与
RUN chmod +x GCN_model.py Generalized_model.py Specialized_model.py

# inpybファイルをnotebookに変換（Jupyter Labで編集可能に）
RUN jupytext --to notebook GCN_model.inpyb
RUN jupytext --to notebook Generalized_model.inpyb
RUN jupytext --to notebook Specialized_model.inpyb

# ポートの公開 (Jupyter Labを使用する場合)
EXPOSE 8888

# 起動スクリプトの作成
RUN echo '#!/bin/bash\necho "利用可能なモデル:"\necho "1. GCN_model.py"\necho "2. Generalized_model.py"\necho "3. Specialized_model.py"\necho "4. Jupyter Lab"\n\necho "選択してください (1-4):"\nread choice\n\ncase $choice in\n  1)\n    echo "GCN_modelを実行中..."\n    python GCN_model.py\n    ;;\n  2)\n    echo "Generalized_modelを実行中..."\n    python Generalized_model.py\n    ;;\n  3)\n    echo "Specialized_modelを実行中..."\n    python Specialized_model.py\n    ;;\n  4)\n    echo "Jupyter Labを起動中..."\n    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n    ;;\n  *)\n    echo "無効な選択です。Jupyter Labを起動します。"\n    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n    ;;\nesac' > /app/start.sh && chmod +x /app/start.sh

# コンテナ起動時にスクリプトを実行
CMD ["/app/start.sh"]
