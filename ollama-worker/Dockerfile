# ============================================================
# Ollama Worker — LLM only
# ============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ── Install Ollama ─────────────────
ARG OLLAMA_VERSION=0.6.5
RUN curl -fsSL https://ollama.com/install.sh | \
    OLLAMA_VERSION=${OLLAMA_VERSION} sh

RUN pip install --no-cache-dir runpod requests

ENV OLLAMA_MODELS=/runpod-volume/ollama-models
ENV OLLAMA_ORIGINS="*"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
