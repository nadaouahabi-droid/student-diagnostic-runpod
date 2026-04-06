#!/bin/bash
# ============================================================
# Populate RunPod Network Volume
# ============================================================
set -euo pipefail

echo "=== Checking volume is mounted ==="
df -h | grep runpod-volume || { echo "ERROR: /runpod-volume not mounted"; exit 1; }

# ── Canonical paths ──────────────────────────────────────────
export PADDLE_HOME=/runpod-volume1/paddle-cache/.paddleocr
export PADDLEOCR_HOME=/runpod-volume1/paddle-cache/.paddleocr
export PPOCR_HOME=$PADDLEOCR_HOME
export PADDLEX_HOME=/runpod-volume1/paddle-cache/.paddlex
export HF_HOME=/runpod-volume1/hf-cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OLLAMA_MODELS=/runpod-volume1/ollama-models
export FLAGS_use_mkldnn=0
export PADDLE_DISABLE_MKLDNN=1

PYPACKAGES=/runpod-volume1/pypackages
mkdir -p "$PADDLEOCR_HOME" "$PADDLEX_HOME" "$HF_HOME" "$OLLAMA_MODELS" "$PYPACKAGES"

TROCR_CHECKPOINT="microsoft/trocr-base-handwritten"

# ── System deps ──────────────────────────────────────────────
echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3.10 python3.10-venv python3.10-distutils \
    curl ca-certificates \
    libglib2.0-0 libgl1

# Upgrade libstdc++ to support Paddle 
echo "=== Upgrading libstdc++ (fix GLIBCXX) ==="
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update -qq && apt-get install -y --no-install-recommends \
    libstdc++6

# ── Pin all pip/python calls to python3.10  ─
echo "=== Bootstrapping pip for Python 3.10 ==="
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
PIP="python3.10 -m pip"

# ── Stage 1: PaddlePaddle + PaddleOCR ───────────────────────
echo "=== Installing PaddleOCR stack ==="
$PIP install --target="$PYPACKAGES" --quiet \
    "numpy==1.24.4" \
    "paddlepaddle-gpu==2.6.1" \
        -f https://www.paddlepaddle.org.cn/whl/linux/cudnn8.9-cuda12.1/stable.html \
    "paddleocr==2.7.3" \
    "opencv-python-headless==4.8.1.78" \
    "Pillow"

# ── Stage 2: PyTorch + HuggingFace ───────────────────────────
echo "=== Installing PyTorch + Transformers stack ==="
$PIP install --target="$PYPACKAGES" --quiet \
    "torch==2.1.2" --index-url https://download.pytorch.org/whl/cu121

$PIP install --target="$PYPACKAGES" --quiet \
    "transformers==4.36.2" \
    "accelerate==0.25.0" \
    "sentencepiece"

# ── Stage 3: RunPod + runtime deps ───────────────────────────
echo "=== Installing RunPod + runtime helpers ==="
$PIP install --target="$PYPACKAGES" --quiet \
    "runpod" \
    "requests"

# ── Download PaddleOCR models ────────────────────────────────
echo "=== Downloading PaddleOCR models ==="

# Force HOME so ~/.paddleocr resolves to our volume path
export HOME=/runpod-volume1/paddle-cache
export PADDLE_HOME=/runpod-volume1/paddle-cache/.paddleocr
export PADDLEOCR_HOME=/runpod-volume1/paddle-cache/.paddleocr
export PPOCR_HOME=/runpod-volume1/paddle-cache/.paddleocr

PYTHONPATH="$PYPACKAGES" python3.10 - <<'EOF'
import os, sys

os.environ["HOME"]           = "/runpod-volume1/paddle-cache"
os.environ["PADDLE_HOME"]    = "/runpod-volume1/paddle-cache/.paddleocr"
os.environ["PADDLEOCR_HOME"] = "/runpod-volume1/paddle-cache/.paddleocr"
os.environ["PPOCR_HOME"]     = "/runpod-volume1/paddle-cache/.paddleocr"

sys.path.insert(0, "/runpod-volume1/pypackages")

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

print("Downloading PaddleOCR models into", os.environ["PADDLEOCR_HOME"])
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=True)
arr = np.array(Image.new("RGB", (200, 50), color="white"))
ocr.ocr(arr, cls=True)
print("PaddleOCR models ready at", os.environ["PADDLEOCR_HOME"])
EOF

echo "=== Verifying PaddleOCR cache ==="
if [ -d "$PADDLEOCR_HOME" ] && [ "$(ls -A "$PADDLEOCR_HOME")" ]; then
    echo "✅ PaddleOCR models present at $PADDLEOCR_HOME:"
    find "$PADDLEOCR_HOME" -name "*.pdmodel" -o -name "*.pdiparams" | head -20
else
    echo "❌ ERROR: PaddleOCR models missing at $PADDLEOCR_HOME"
    exit 1
fi

# ── Download TrOCR ───────────────────────────────────────────
echo "=== Downloading TrOCR: $TROCR_CHECKPOINT ==="
PYTHONPATH="$PYPACKAGES" python3.10 - <<EOF
import os, sys, torch
sys.path.insert(0, os.environ["PYTHONPATH"])
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ckpt  = "${TROCR_CHECKPOINT}"
cache = os.environ["HF_HOME"]
print(f"Downloading {ckpt} -> {cache}")
TrOCRProcessor.from_pretrained(ckpt, cache_dir=cache)
VisionEncoderDecoderModel.from_pretrained(ckpt, cache_dir=cache, torch_dtype=torch.float16)
print("TrOCR ready.")
EOF

echo "=== Verifying HuggingFace cache ==="
if [ "$(ls -A "$HF_HOME" 2>/dev/null)" ]; then
    echo "✅ HF cache ready:"
    ls "$HF_HOME"
else
    echo "❌ ERROR: HF cache empty at $HF_HOME"
    exit 1
fi

# ── Install Ollama + pull models ─────────────────────────────
echo "=== Installing Ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== Pulling Ollama models ==="
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" ollama serve &
OLLAMA_PID=$!

for i in $(seq 1 30); do
    sleep 2
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
    echo "  waiting for Ollama... ($i)"
done

OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull qwen2.5vl:7b-q4_K_M
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull qwen2.5:7b-instruct-q4_K_M
kill "$OLLAMA_PID" 2>/dev/null || true

# ── Verify Ollama manifests (tag-level) ──────────────────────
echo "=== Verifying Ollama manifests ==="
MANIFEST_BASE="$OLLAMA_MODELS/manifests/registry.ollama.ai/library"
REQUIRED_MODELS=("qwen2.5vl:7b-q4_K_M" "qwen2.5:7b-instruct-q4_K_M")
MANIFEST_ERRORS=0

for MODEL_TAG in "${REQUIRED_MODELS[@]}"; do
    NAME="${MODEL_TAG%%:*}"
    TAG="${MODEL_TAG##*:}"
    MANIFEST_FILE="$MANIFEST_BASE/$NAME/$TAG"
    if [ -f "$MANIFEST_FILE" ]; then
        echo "✅ $MODEL_TAG manifest present"
    else
        echo "❌ ERROR: $MODEL_TAG manifest missing — expected $MANIFEST_FILE"
        [ -d "$MANIFEST_BASE/$NAME" ] \
            && echo "   Tags found: $(ls "$MANIFEST_BASE/$NAME")" \
            || echo "   No directory at $MANIFEST_BASE/$NAME"
        MANIFEST_ERRORS=$((MANIFEST_ERRORS + 1))
    fi
done

[ "$MANIFEST_ERRORS" -gt 0 ] && { echo "❌ $MANIFEST_ERRORS model(s) failed verification."; exit 1; }

# ── Final summary ─────────────────────────────────────────────
echo ""
echo "=== FINAL STRUCTURE ==="
find /runpod-volume -maxdepth 3 -type d

echo ""
echo "=== DISK USAGE ==="
du -sh /runpod-volume1/*

echo ""
echo "✅ Volume fully populated."
