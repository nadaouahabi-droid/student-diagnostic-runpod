#!/bin/bash
# ============================================================
# Populate RunPod Network Volume
# ============================================================
set -euo pipefail

echo "=== Checking volume is mounted ==="
df -h | grep runpod-volume || { echo "ERROR: /runpod-volume not mounted"; exit 1; }

# ── Canonical paths ──────────────────────────────────────────
export PADDLE_HOME=/runpod-volume/paddle-cache
export PADDLEOCR_HOME=/runpod-volume/paddle-cache/.paddleocr
export PPOCR_HOME=$PADDLEOCR_HOME
export PADDLEX_HOME=/runpod-volume/paddle-cache/.paddlex
export HF_HOME=/runpod-volume/hf-cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OLLAMA_MODELS=/runpod-volume/ollama-models
export FLAGS_use_mkldnn=0
export PADDLE_DISABLE_MKLDNN=1

PYPACKAGES=/runpod-volume/pypackages
mkdir -p "$PADDLEOCR_HOME" "$PADDLEX_HOME" "$HF_HOME" "$OLLAMA_MODELS" "$PYPACKAGES"

TROCR_CHECKPOINT="microsoft/trocr-base-handwritten"

# ── System deps ──────────────────────────────────────────────
echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3.10 python3.10-venv python3.10-distutils \
    curl ca-certificates \
    libglib2.0-0 libgl1

# Upgrade libstdc++ to support Paddle (fix GLIBCXX error)
echo "=== Upgrading libstdc++ (fix GLIBCXX) ==="
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update -qq && apt-get install -y --no-install-recommends \
    libstdc++6

# ── FIX 1: pin all pip/python calls to python3.10 explicitly ─
# The RunPod base image defaults to python3.13 for bare `pip`
# and `python3`. PaddlePaddle has no 3.13 wheel — it tops out
# at 3.12. We install pip into the 3.10 interpreter and use it
# for every subsequent install.
echo "=== Bootstrapping pip for Python 3.10 ==="
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
PIP="python3.10 -m pip"

# ── Stage 1: PaddlePaddle + PaddleOCR ───────────────────────
echo "=== Installing PaddleOCR stack ==="
# FIX 2: paddlepaddle-gpu==2.6.0.post120 does not exist.
#   Correct package: paddlepaddle-gpu==2.6.1.post120
#   Correct index:   cuda12.1  (not cudnn8.6-cuda12.0)
$PIP install --target="$PYPACKAGES" --quiet \
    "numpy==1.26.4" \
    "paddlepaddle-gpu==2.6.1.post120" \
        -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
    "paddleocr==2.7.3" \
    "opencv-python-headless==4.8.1.78" \
    "Pillow"

# ── Stage 2: PyTorch + HuggingFace ───────────────────────────
echo "=== Installing PyTorch + Transformers stack ==="
$PIP install --target="$PYPACKAGES" --quiet \
    "torch==2.2.0" --index-url https://download.pytorch.org/whl/cu121
$PIP install --target="$PYPACKAGES" --quiet \
    "transformers==4.41.2" \
    "accelerate>=0.27" \
    "sentencepiece"

# ── Stage 3: RunPod + runtime deps ───────────────────────────
echo "=== Installing RunPod + runtime helpers ==="
$PIP install --target="$PYPACKAGES" --quiet \
    "runpod" \
    "requests"

# ── Download PaddleOCR models ────────────────────────────────
echo "=== Downloading PaddleOCR models ==="
PYTHONPATH="$PYPACKAGES" python3.10 - <<'EOF'
import os, sys

# Force correct cache path BEFORE imports
os.environ["PADDLEOCR_HOME"] = "/runpod-volume/paddle-cache/.paddleocr"
os.environ["PPOCR_HOME"] = os.environ["PADDLEOCR_HOME"]

sys.path.insert(0, os.environ["PYTHONPATH"])

import paddle
from paddleocr import PaddleOCR
from PIL import Image

print("Downloading PaddleOCR models into", os.environ["PADDLEOCR_HOME"])

use_gpu = False  # keep stable

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=use_gpu
)

img = Image.new("RGB", (200, 50), color="white")
img.save("/tmp/test_ocr.png")
ocr.ocr("/tmp/test_ocr.png")

print("PaddleOCR models ready.")
EOF

echo "=== Verifying PaddleOCR cache ==="
if [ -d "$PADDLEOCR_HOME" ] && [ "$(ls -A "$PADDLEOCR_HOME")" ]; then
    echo "✅ PaddleOCR models present:"
    ls -lh "$PADDLEOCR_HOME"
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

OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull qwen2.5vl:7b-q8_0
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull qwen2.5:7b-instruct-q4_K_M
kill "$OLLAMA_PID" 2>/dev/null || true

# ── Verify Ollama manifests (tag-level) ──────────────────────
echo "=== Verifying Ollama manifests ==="
MANIFEST_BASE="$OLLAMA_MODELS/manifests/registry.ollama.ai/library"
REQUIRED_MODELS=("qwen2.5vl:7b-q8_0" "qwen2.5:7b-instruct-q4_K_M")
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
du -sh /runpod-volume/*

echo ""
echo "✅ Volume fully populated."
