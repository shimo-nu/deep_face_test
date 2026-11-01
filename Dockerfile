# CUDA 12.2.2 + cuDNN8 (Ubuntu 20.04)
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_PATH=/usr/local/cuda
ENV DLIB_USE_CUDA=1
ENV USE_AVX_INSTRUCTIONS=1

# 基本ビルド依存（system Python は入れない）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gpg git pkg-config \
    build-essential ninja-build \
    libopenblas-dev liblapack-dev \
    libjpeg-dev libpng-dev \
    libgtk-3-dev libx11-dev \
 && rm -rf /var/lib/apt/lists/*

# uv インストール（インストーラは /root/.local/bin/uv を作る）
# 明示的に PATH へ追加（このレイヤ以降で有効）

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Python 3.10 を uv で取得し、仮想環境を作成
RUN uv python install 3.10 && \
    uv venv /opt/venv -p 3.10
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# pipツール類 + cmake(バイナリ同梱) + ビルド補助
RUN uv pip install --upgrade pip setuptools wheel packaging && \
    uv pip install "cmake>=3.26,<4" scikit-build-core ninja numpy

# dlib（CUDA を自動検出してビルド）
RUN uv pip install dlib

# face_recognition & OpenCV (headless)
RUN uv pip install face_recognition face_recognition_models opencv-python-headless

# InsightFace + ONNX Runtime（GPU。CPUなら onnxruntime に置換）
RUN uv pip install insightface onnxruntime-gpu


WORKDIR /workspace
CMD ["bash"]

