#!/bin/bash
# ============================================================
# Populate RunPod Network Volume — Ollama models only
# Python packages and OCR weights are baked into the image.
# ============================================================
set -euo pipefail

echo "=== Checking volume is mounted ==="
df -h | grep runpod-volume || { echo "ERROR: /runpod-volume not mounted"; exit 1; }

mkdir -p /runpod-volume/ollama-models

# ── Install Ollama if not present ─────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "=== Installing Ollama ==="
    curl -fsSL https://ollama.com/install.sh | sh
fi

# ── Start Ollama temporarily to pull models ───────────────────
echo "=== Starting Ollama ==="
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_MODELS=/runpod-volume/ollama-models \
    ollama serve &
OLLAMA_PID=$!

echo "=== Waiting for Ollama to be ready ==="
for i in $(seq 1 30); do
    sleep 2
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null && echo "Ollama ready." && break
    echo "  waiting... ($i/30)"
done

# ── Pull models ───────────────────────────────────────────────
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_MODELS=/runpod-volume/ollama-models \
    ollama pull minicpm-v:8b

OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_MODELS=/runpod-volume/ollama-models \
    ollama pull qwen2.5:7b-instruct-q4_K_M

kill "$OLLAMA_PID" 2>/dev/null || true

# ── Verify manifests ──────────────────────────────────────────
echo "=== Verifying Ollama manifests ==="
MANIFEST_BASE="/runpod-volume/ollama-models/manifests/registry.ollama.ai/library"
ERRORS=0
for MODEL_TAG in "minicpm-v:8b" "qwen2.5:7b-instruct-q4_K_M"; do
    NAME="${MODEL_TAG%%:*}"; TAG="${MODEL_TAG##*:}"
    if [ -f "$MANIFEST_BASE/$NAME/$TAG" ]; then
        echo "✅ $MODEL_TAG manifest present"
    else
        echo "❌ $MODEL_TAG manifest MISSING at $MANIFEST_BASE/$NAME/$TAG"
        ERRORS=$((ERRORS + 1))
    fi
done

[ "$ERRORS" -gt 0 ] && { echo "❌ $ERRORS model(s) failed."; exit 1; }

echo ""
echo "=== DISK USAGE ==="
du -sh /runpod-volume/*

echo ""
echo "✅ Volume populated. Only Ollama weights live here — everything else is baked."
