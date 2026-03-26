# ============================================================
# Student Diagnostic System — RunPod Serverless Worker
# ============================================================
# Strategy: bake Ollama + models INTO the image during build.
# At cold start, models load from local NVMe (seconds),
# not from a network volume (minutes).
#
# Resulting image size: ~12–14 GB depending on models chosen.
# RunPod caches this image on workers, so after the first pull
# the image never downloads again.
#
# Build:
#   docker build -t yourdockerhubuser/student-diagnostic:latest .
#   docker push yourdockerhubuser/student-diagnostic:latest
#
# Environment variables you can override at deploy time:
#   VISION_MODEL  (default: qwen2.5vl:7b-q4_K_M)
#   TEXT_MODEL    (default: qwen2.5:7b-instruct-q4_K_M)
# ============================================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ── System packages ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
# ── System packages ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    zstd && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 \
    -o /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama

# ── Install Python dependencies ─────────────────────────────────
RUN pip3 install --no-cache-dir runpod requests

# ── Pre-bake models into the image ──────────────────────────────────────────
# Ollama serves in CPU-only mode during build (no GPU in docker build).
# This is fine — we only need to DOWNLOAD the GGUF weight files.
# At inference time they load from NVMe into GPU VRAM normally.
#
# Models stored at: /root/.ollama/models (inside the image layer)
ENV OLLAMA_MODELS=/root/.ollama/models

RUN ollama serve >/tmp/ollama_build.log 2>&1 & \
    # Wait for Ollama API to become available
    for i in $(seq 1 30); do \
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; \
        sleep 2; \
    done && \
    # Pull the vision model (used for OCR + image understanding)
    ollama pull qwen2.5vl:7b-q4_K_M && \
    # Pull the text model (used for analysis, report, plan, worksheet)
    ollama pull qwen2.5:7b-instruct-q4_K_M && \
    # Gracefully stop Ollama
    pkill -f "ollama serve" || true && \
    sleep 2 && \
    echo "✓ Models baked into image successfully"

# ── Copy handler ─────────────────────────────────────────────────────────────
COPY handler.py /handler.py

# ── Start handler ─────────────────────────────────────────────────────────────
CMD ["python3", "-u", "/handler.py"]
