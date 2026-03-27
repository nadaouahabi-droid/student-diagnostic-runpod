# ============================================================
# Student Diagnostic System — RunPod Serverless Worker
# Models are loaded from /runpod-volume/ (network volume).
# ============================================================
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ── Install Ollama ───────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python dependencies ──────────────────────────────────────
RUN pip install --no-cache-dir \
    runpod \
    requests \
    paddlepaddle-gpu==3.0.0 \
    paddleocr==3.0.0 \
    opencv-python-headless \
    Pillow \
    numpy \
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
