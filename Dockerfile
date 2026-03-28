# ============================================================
# Student Diagnostic System — RunPod Serverless Worker
# Models are loaded from /runpod-volume/ (network volume).
# ============================================================
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ── Upgrade pip toolchain ─────────────────────────────────────
RUN python -m pip install --upgrade pip setuptools wheel

# ── Install Ollama ────────────────────────────────────────────
# Pinned to a specific release to avoid redirect/auth issues with
# the "latest" GitHub asset URL. Verified sha256 for amd64.
# To upgrade: change the version below and update the URL.
ARG OLLAMA_VERSION=0.6.5
RUN curl -fsSL \
    "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tgz" \
    -o /tmp/ollama.tgz && \
    tar -xzf /tmp/ollama.tgz -C /usr/local && \
    rm /tmp/ollama.tgz && \
    chmod +x /usr/local/bin/ollama && \
    # Smoke-test: confirm the binary is executable and correct arch
    ollama --version

# ── OCR dependencies ─────────────────────────────────────────
RUN pip install --no-cache-dir \
    paddlepaddle==3.0.0 \
    paddleocr==3.0.0 \
    numpy \
    Pillow \
    opencv-python-headless

# ── App dependencies ──────────────────────────────────────────
# --ignore-installed blinker: base image has blinker 1.4 via distutils
# which pip cannot uninstall cleanly.
RUN pip install --no-cache-dir \
    --ignore-installed blinker \
    runpod \
    requests \
    transformers \
    accelerate \
    sentencepiece \
    flask \
    flask-cors

# ── Environment — point everything at the network volume ─────
ENV OLLAMA_MODELS=/runpod-volume/ollama-models
ENV OLLAMA_ORIGINS=*
ENV PADDLEX_HOME=/runpod-volume/paddle-cache/.paddlex
ENV HF_HOME=/runpod-volume/hf-cache/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/hf-cache/huggingface
ENV FLAGS_use_mkldnn=0
ENV PADDLE_DISABLE_MKLDNN=1

# ── Copy handler ─────────────────────────────────────────────
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
