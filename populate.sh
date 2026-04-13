#!/bin/bash
set -euo pipefail

echo "=== Checking volume ==="
df -h | grep runpod-volume || { echo "ERROR: volume not mounted"; exit 1; }

export OLLAMA_MODELS=/runpod-volume/ollama-models
mkdir -p "$OLLAMA_MODELS"

echo "=== Installing Ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== Starting Ollama ==="
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" ollama serve > /tmp/ollama.log 2>&1 &
PID=$!

echo "=== Waiting for Ollama API ==="
for i in $(seq 1 60); do
    sleep 2
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
        echo "Ollama API ready"
        break
    fi
    echo "waiting... ($i)"
done

echo "=== Pulling model (this can take time) ==="
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" \
ollama pull qwen2.5vl:7b-q4_K_M

echo "=== Verifying model registration ==="
OLLAMA_HOST=127.0.0.1:11434 ollama list

echo "=== Checking manifest exists ==="
ls -lh /runpod-volume/ollama-models/manifests/registry.ollama.ai/library/qwen2.5vl/ || true

echo "=== Stopping Ollama ==="
kill "$PID" || true

echo "✅ Done — model fully installed"
