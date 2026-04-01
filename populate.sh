#!/bin/bash
# ============================================================
# Populate RunPod Network Volume (MATCH DOCKERFILE v2.7.3)
# ============================================================
set -e

echo "=== Checking volume is mounted ==="
df -h | grep runpod || { echo "ERROR: /runpod-volume not mounted"; exit 1; }

# ------------------------------------------------------------
# 0. Environment variables (MATCH DOCKERFILE)
# ------------------------------------------------------------
export HF_HOME=/runpod-volume/hf-cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OLLAMA_MODELS=/runpod-volume/ollama-models

export PADDLE_HOME=/runpod-volume/paddle-cache
export PADDLEOCR_HOME=/runpod-volume/paddle-cache/.paddleocr

mkdir -p $HF_HOME
mkdir -p $OLLAMA_MODELS
mkdir -p $PADDLEOCR_HOME

# ------------------------------------------------------------
# 1. Python 3.10 + env
# ------------------------------------------------------------
echo "=== Setting up Python ==="
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-distutils wget

python3.10 -m venv /tmp/venv
source /tmp/venv/bin/activate

pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 2. Install EXACT SAME versions as Dockerfile
# ------------------------------------------------------------
echo "=== Installing PaddleOCR stack (MATCHED) ==="

pip install \
    torch==2.2.0 \
    paddlepaddle==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
    paddleocr==2.7.3 \
    opencv-python-headless==4.8.1.78 \
    transformers==4.41.2 \
    sentencepiece \
    Pillow \
    numpy==1.26.4 \
    --quiet

# ------------------------------------------------------------
# 3. Download PaddleOCR models (PP-OCRv3)
# ------------------------------------------------------------
echo "=== Downloading PaddleOCR v2.7.3 models ==="

FLAGS_use_mkldnn=0 PADDLE_DISABLE_MKLDNN=1 python3 - <<EOF
from paddleocr import PaddleOCR
from PIL import Image

print("Downloading PaddleOCR models...")

# trigger model download
ocr = PaddleOCR(use_angle_cls=True, lang="en")

img = Image.new('RGB', (200, 50), color='white')
img.save('/tmp/test.png')

ocr.ocr('/tmp/test.png')

print("Download complete.")
EOF

# ------------------------------------------------------------
# 4. Verify PaddleOCR cache
# ------------------------------------------------------------
echo "=== Verifying PaddleOCR cache ==="

if [ -d "$PADDLEOCR_HOME" ] && [ "$(ls -A $PADDLEOCR_HOME)" ]; then
    echo "✅ PaddleOCR models present:"
    ls -lh $PADDLEOCR_HOME
else
    echo "❌ ERROR: PaddleOCR models missing"
    exit 1
fi

# ------------------------------------------------------------
# 5. TrOCR (MATCH Dockerfile)
# ------------------------------------------------------------
echo "=== Downloading TrOCR ==="

python3 - <<EOF
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ckpt = "microsoft/trocr-large-handwritten"

print("Downloading TrOCR...")

TrOCRProcessor.from_pretrained(ckpt)
VisionEncoderDecoderModel.from_pretrained(ckpt)

print("TrOCR ready.")
EOF

# ------------------------------------------------------------
# 6. Verify HF cache
# ------------------------------------------------------------
echo "=== Verifying HuggingFace cache ==="

if [ "$(ls -A $HF_HOME 2>/dev/null)" ]; then
    echo "✅ HF cache ready"
    ls $HF_HOME
else
    echo "❌ ERROR: HF cache empty"
    exit 1
fi

# ------------------------------------------------------------
# 7. Final check
# ------------------------------------------------------------
echo ""
echo "=== FINAL STRUCTURE ==="
find /runpod-volume -maxdepth 3 -type d

echo ""
echo "=== DISK USAGE ==="
du -sh /runpod-volume/*

echo ""
echo "✅ Volume ready (MATCHED with Dockerfile)"
