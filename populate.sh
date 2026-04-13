#!/bin/bash
set -euo pipefail

echo "=== Checking volume ==="
df -h | grep runpod-volume || { echo "ERROR: volume not mounted"; exit 1; }

# Paths
export OLLAMA_MODELS=/runpod-volume/ollama-models
mkdir -p "$OLLAMA_MODELS"

# Install Ollama only
echo "=== Installing Ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama
echo "=== Starting Ollama ==="
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" ollama serve &
PID=$!

# Wait for server
for i in $(seq 1 30); do
    sleep 2
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
    echo "waiting for Ollama... ($i)"
done

# Pull ONLY Qwen model
echo "=== Pulling Qwen ==="
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" \
ollama pull qwen2.5vl:7b-q8_0

# Stop Ollama
kill "$PID" 2>/dev/null || true

# Verify
echo "=== Verify ==="
ls -lh $OLLAMA_MODELS

echo "✅ Done: Qwen only"
