# ─── Base: CUDA + Ollama runtime ───────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Layer 1: System packages (changes rarely) ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        curl wget git ca-certificates \
        libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
        libgomp1 libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Symlink so 'python' works
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# ── Layer 2: Ollama binary (changes rarely) ────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Layer 3: Heavy Python deps (changes occasionally) ─────────────────────
# Install CPU-only torch first — saves ~2 GB vs CUDA torch for PaddleOCR/TrOCR
# (Ollama handles GPU inference; these CPU libs only do OCR preprocessing)
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install \
    paddlepaddle==2.6.1 \
    paddleocr==2.8.1

RUN pip install \
    transformers==4.44.2 \
    sentencepiece \
    timm

# ── Layer 4: App server deps (changes occasionally) ───────────────────────
RUN pip install \
    flask==3.0.3 \
    flask-cors==4.0.1 \
    runpod==1.7.3 \
    requests==2.32.3 \
    Pillow==10.4.0 \
    numpy==1.26.4

# ── Layer 5: Pre-download PaddleOCR model files into the image ────────────
# These are small (< 200 MB total) and avoid a 60 s download on every cold start
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)"

# ── Layer 6: Pre-download TrOCR weights into the image ────────────────────
# ~1.3 GB but worth it: eliminates 3–5 minute HuggingFace download at cold start
RUN python3 -c "\
from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten'); \
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')"

# ── Layer 7: App code (changes frequently — always last!) ─────────────────
WORKDIR /app
COPY ocr_server.py handler.py ./

CMD ["python", "-u", "handler.py"]
