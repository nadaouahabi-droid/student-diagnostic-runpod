# ============================================================
# Student Diagnostic System — RunPod Serverless Worker
# Models are loaded from /runpod-volume/ (network volume).
# ============================================================
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages (single apt layer) ───────────────────────
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

# ── Install Ollama manually ───────────────────────────────────
RUN curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 \
    -o /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama

# ── OCR dependencies ─────────────────────────────────────────
RUN pip install --no-cache-dir \
    paddlepaddle==3.0.0 \
    paddleocr==3.0.0 \
    numpy \
    Pillow \
    opencv-python-headless

# ── App dependencies ──────────────────────────────────────────
# --ignore-installed blinker: the base image ships blinker 1.4 via
# distutils (apt), which pip cannot safely uninstall. This flag tells
# pip to install its own copy without touching the system package.
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
