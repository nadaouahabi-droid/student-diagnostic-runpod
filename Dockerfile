# ─────────────────────────────────────────────────────────────────────────────
# Student Diagnostic System v10 — RunPod Serverless on RTX 3090
# Image registry: ghcr.io (free, unlimited private repos)
#
# Build strategy:
#   - Models stored on RunPod Network Volume (/runpod-volume/ollama-models)
#     → image stays ~3 GB instead of ~12 GB
#     → image push/pull is fast
#     → models load from NVMe volume at boot (~5s), not re-downloaded
# ─────────────────────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git \
    libgl1 libglib2.0-0 libgomp1 \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Ollama ────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python packages ───────────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer unless deps change
COPY requirements_ocr.txt .
RUN pip install --no-cache-dir runpod requests \
 && pip install --no-cache-dir -r requirements_ocr.txt

# ── Application files ─────────────────────────────────────────────────────────
COPY ocr_server.py .
COPY handler.py    .

# ── Tell Ollama to load models from the Network Volume ────────────────────────
# This path is where you'll mount your RunPod Network Volume
ENV OLLAMA_MODELS=/runpod-volume/ollama-models
ENV OLLAMA_ORIGINS=*

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
