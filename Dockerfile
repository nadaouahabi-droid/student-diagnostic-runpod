# ============================================================
# Student Diagnostic System — RunPod Serverless Worker
# Models are loaded from /runpod-volume/ (network volume).
# ============================================================
FROM --platform=linux/amd64 runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    file && \
    rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ───────────────────────────────────────────────
RUN python -m pip install --upgrade pip setuptools wheel

# ── Install Ollama ────────────────────────────────────────────
ARG OLLAMA_VERSION=0.6.5
RUN curl -fsSL \
    "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tgz" \
    -o /tmp/ollama.tgz && \
    tar -xzf /tmp/ollama.tgz -C /usr/local && \
    rm /tmp/ollama.tgz && \
    chmod +x /usr/local/bin/ollama && \
    file /usr/local/bin/ollama && \
    ollama --version

# ── App dependencies ──
RUN pip install --no-cache-dir runpod requests

RUN pip install --no-cache-dir \
    accelerate==0.30.1 \
    transformers==4.41.2 \
    sentencepiece

RUN pip install --no-cache-dir \
    runpod \
    requests \
    accelerate==0.30.1 \
    transformers==4.41.2 \
    sentencepiece \
    flask \
    flask-cors && \
    rm -rf /root/.cache /tmp/*

# ── OCR (install last) ───────────────────────────────────────
RUN pip install --no-cache-dir \
    paddlepaddle==2.6.1 \
    paddleocr==2.7.0.3 \
    numpy Pillow opencv-python-headless

# ── Environment ───────────────────────────────────────────────
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
