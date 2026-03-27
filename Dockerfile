# ============================================================
# Student Diagnostic System — RunPod Serverless Worker
# Image: ~4GB (code + deps only, NO models baked in)
# Models loaded at runtime from /runpod-volume/ (network volume)
# ============================================================
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ── Make python3.11 the default ──────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# ── Install Ollama ───────────────────────────────────────────
RUN curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 \
    -o /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama

# ── Install Python dependencies (including runpod) ───────────
RUN python3.11 -m pip install --no-cache-dir \
    runpod \
    requests \
    paddlepaddle==3.0.0 \
    paddleocr==3.0.0 \
    opencv-python-headless \
    Pillow \
    numpy \
    flask \
    flask-cors \
    transformers \
    torch \
    sentencepiece

ENV OLLAMA_MODELS=/runpod-volume/ollama-models
ENV OLLAMA_ORIGINS=*
ENV PADDLEX_HOME=/runpod-volume/paddle-cache/.paddlex
ENV HF_HOME=/runpod-volume/hf-cache/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/hf-cache/huggingface
ENV FLAGS_use_mkldnn=0
ENV PADDLE_DISABLE_MKLDNN=1

# ── Copy handler ─────────────────────────────────────────────
COPY handler.py /handler.py

CMD ["python3.11", "-u", "/handler.py"]
