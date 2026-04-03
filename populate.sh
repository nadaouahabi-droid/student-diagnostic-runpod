#!/bin/bash
# ============================================================
# Populate RunPod Network Volume
# Changes vs original:
#   - Manifest check verifies name:tag files, not just directories
#     (matches handler.py verify_models_on_volume logic exactly)
#   - pip install split into two stages so a torch failure doesn't
#     silently skip paddle and vice-versa
#   - Minor: set -euo pipefail already present, kept; added ||true on
#     ollama kill to suppress "no process" noise
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

# ── Package target ───────────────────────────────────────────
PYPACKAGES=/runpod-volume/pypackages
mkdir -p "$PADDLEOCR_HOME" "$PADDLEX_HOME" "$HF_HOME" "$OLLAMA_MODELS" "$PYPACKAGES"

TROCR_CHECKPOINT="microsoft/trocr-base-handwritten"

# ── System deps ──────────────────────────────────────────────
echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-distutils curl ca-certificates \
    libglib2.0-0 libgl1

# ── Python base tools ────────────────────────────────────────
echo "=== Installing base Python tools ==="
pip install --target="$PYPACKAGES" --upgrade pip setuptools wheel --quiet

# ── Stage 1: PaddlePaddle + PaddleOCR ───────────────────────
echo "=== Installing PaddleOCR stack ==="
pip install --target="$PYPACKAGES" --quiet \
    "numpy==1.26.4" \
    "paddlepaddle-gpu==2.6.0.post120" \
        -f https://www.paddlepaddle.org.cn/whl/linux/cudnn8.6-cuda12.0/stable.html \
    "paddleocr==2.7.3" \
    "opencv-python-headless==4.8.1.78" \
    "Pillow"

# ── Stage 2: PyTorch + HuggingFace ───────────────────────────
echo "=== Installing PyTorch + Transformers stack ==="
pip install --target="$PYPACKAGES" --quiet \
    "torch==2.2.0" --index-url https://download.pytorch.org/whl/cu121
pip install --target="$PYPACKAGES" --quiet \
    "transformers==4.41.2" \
    "accelerate>=0.27" \
    "sentencepiece"

# ── Stage 3: RunPod + runtime deps ───────────────────────────
echo "=== Installing RunPod + runtime helpers ==="
pip install --target="$PYPACKAGES" --quiet \
    "runpod" \
    "requests"

# ── Download PaddleOCR models ────────────────────────────────
echo "=== Downloading PaddleOCR models ==="
PYTHONPATH="$PYPACKAGES" python3 - <<'EOF'
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
from paddleocr import PaddleOCR
from PIL import Image

print("Downloading PaddleOCR models into", os.environ["PADDLEOCR_HOME"])
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
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
PYTHONPATH="$PYPACKAGES" python3 - <<EOF
import os, sys, torch
sys.path.insert(0, os.environ["PYTHONPATH"])
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ckpt  = "${TROCR_CHECKPOINT}"
cache = os.environ["HF_HOME"]
print(f"Downloading {ckpt} → {cache}")
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

# ── Verify Ollama manifests (tag-level, matches handler.py exactly) ──
echo "=== Verifying Ollama manifests ==="
MANIFEST_BASE="$OLLAMA_MODELS/manifests/registry.ollama.ai/library"
REQUIRED_MODELS=("qwen2.5vl:7b-q8_0" "qwen2.5:7b-instruct-q4_K_M")
MANIFEST_ERRORS=0

for MODEL_TAG in "${REQUIRED_MODELS[@]}"; do
    NAME="${MODEL_TAG%%:*}"
    TAG="${MODEL_TAG##*:}"
    MANIFEST_FILE="$MANIFEST_BASE/$NAME/$TAG"
    if [ -f "$MANIFEST_FILE" ]; then
        echo "✅ $MODEL_TAG manifest present ($MANIFEST_FILE)"
    else
        echo "❌ ERROR: $MODEL_TAG manifest missing — expected $MANIFEST_FILE"
        # Show what's actually there to aid debugging
        if [ -d "$MANIFEST_BASE/$NAME" ]; then
            echo "   Tags found under $NAME: $(ls "$MANIFEST_BASE/$NAME")"
        else
            echo "   No directory found at $MANIFEST_BASE/$NAME"
        fi
        MANIFEST_ERRORS=$((MANIFEST_ERRORS + 1))
    fi
done

if [ "$MANIFEST_ERRORS" -gt 0 ]; then
    echo "❌ $MANIFEST_ERRORS model(s) failed manifest verification."
    exit 1
fi

# ── Final summary ─────────────────────────────────────────────
echo ""
echo "=== FINAL STRUCTURE ==="
find /runpod-volume -maxdepth 3 -type d

echo ""
echo "=== DISK USAGE ==="
du -sh /runpod-volume/*

echo ""
echo "✅ Volume fully populated."
