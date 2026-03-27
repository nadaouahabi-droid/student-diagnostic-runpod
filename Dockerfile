# ============================================================
# Student Diagnostic System — RunPod Serverless (Ollama + PaddleOCR fixed)
# ============================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Install Python 3.11 explicitly (PaddleOCR requires <=3.11) ──
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    python3-pip \
    curl \
    ca-certificates \
    zstd \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ── Make python3.11 the default ─────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip3    pip3    /usr/bin/pip3.11    1 && \
    python3 --version

# ── Fix pkg_resources (setuptools regression) ───────────────
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ── Install Ollama ───────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Install Python dependencies ─────────────────────────────
# PaddleOCR needs specific versions pinned for Python 3.11
RUN pip3 install --no-cache-dir \
    runpod \
    requests \
    setuptools \
    paddlepaddle==2.6.1 \
    paddleocr==2.7.3 \
    opencv-python-headless \
    Pillow

# ── Pre-bake Ollama models ───────────────────────────────────
ENV OLLAMA_MODELS=/root/.ollama/models

RUN set -eux; \
    /usr/local/bin/ollama serve > /tmp/ollama.log 2>&1 & \
    sleep 8; \
    /usr/local/bin/ollama pull qwen2.5vl:7b-q4_K_M; \
    /usr/local/bin/ollama pull qwen2.5:7b-instruct-q4_K_M; \
    pkill -f "ollama serve" || true; \
    sleep 2; \
    echo "✓ Models baked"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
