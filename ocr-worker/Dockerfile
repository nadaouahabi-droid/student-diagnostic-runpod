# ============================================================
# OCR Worker — PaddleOCR + TrOCR
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    curl \
    wget \
    ca-certificates \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
        runpod requests accelerate==0.30.1 transformers==4.41.2 \
        sentencepiece numpy Pillow opencv-python-headless \
        paddlepaddle==3.0.0 paddleocr==3.0.0 && \
    pip cache purge && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# ── Environment ─────────────────────────
ENV PADDLEX_HOME=/runpod-volume/paddle-cache/.paddlex
ENV HF_HOME=/runpod-volume/hf-cache/huggingface
ENV FLAGS_use_mkldnn=0
ENV PADDLE_DISABLE_MKLDNN=1

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
