#!/bin/bash

# ============================================================

# Populate RunPod Network Volume with all models (FIXED VERSION)

# ============================================================

set -e

echo "=== Checking volume is mounted ==="
df -h | grep runpod || { echo "ERROR: /runpod-volume not mounted"; exit 1; }

# ------------------------------------------------------------

# 0. Environment variables (IMPORTANT)

# ------------------------------------------------------------

export HF_HOME=/runpod-volume/hf-cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OLLAMA_MODELS=/runpod-volume/ollama-models

mkdir -p $HF_HOME
mkdir -p $OLLAMA_MODELS
mkdir -p /runpod-volume/paddle-cache

# ------------------------------------------------------------

# 1. Ollama models

# ------------------------------------------------------------

echo ""
echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Pulling Ollama models ==="
ollama serve &
OLLAMA_PID=$!
sleep 5

ollama pull qwen2.5vl:7b-q8_0
ollama pull qwen2.5:7b-instruct-q4_K_M

echo "=== Verifying Ollama models ==="
ollama list

kill $OLLAMA_PID 2>/dev/null || true

# ------------------------------------------------------------

# 2. Python 3.10 environment (CRITICAL FIX)

# ------------------------------------------------------------

echo ""
echo "=== Setting up Python 3.10 ==="
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-distutils

python3.10 -m venv /tmp/venv
source /tmp/venv/bin/activate

pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------

# 3. PaddleOCR (FIXED)

# ------------------------------------------------------------

echo ""
echo "=== Installing PaddleOCR ==="
pip install paddleocr==3.0.0 paddlepaddle==3.0.0 Pillow numpy --quiet

echo "=== Downloading PaddleOCR weights ==="
FLAGS_use_mkldnn=0 PADDLE_DISABLE_MKLDNN=1 python3 -c "
from paddleocr import PaddleOCR
from PIL import Image

img = Image.new('RGB', (200, 50), color='white')
img.save('/tmp/test.png')

ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='cpu')
ocr.ocr('/tmp/test.png')

print('PaddleOCR weights downloaded successfully')
"

# ---- FORCE COPY (IMPORTANT FIX) ----

echo "=== Copying PaddleOCR weights to volume ==="
cp -r /root/.paddlex /runpod-volume/paddle-cache/

echo "PaddleOCR contents:"
ls /runpod-volume/paddle-cache/.paddlex/official_models/

# ------------------------------------------------------------

# 4. TrOCR (HuggingFace FIXED)

# ------------------------------------------------------------

echo ""
echo "=== Installing Transformers ==="
pip install transformers torch sentencepiece --quiet

echo "=== Downloading TrOCR weights ==="
HF_HOME=$HF_HOME python3 -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ckpt = 'microsoft/trocr-large-handwritten'
TrOCRProcessor.from_pretrained(ckpt)
VisionEncoderDecoderModel.from_pretrained(ckpt)

print('TrOCR weights downloaded successfully')
"

echo "TrOCR contents:"
ls $HF_HOME

# ------------------------------------------------------------

# 5. Final verification

# ------------------------------------------------------------

echo ""
echo "=== FINAL STRUCTURE ==="
find /runpod-volume -maxdepth 3 -type d

echo ""
echo "=== DISK USAGE ==="
du -sh /runpod-volume/*

echo ""
echo "=== TOTAL SIZE ==="
du -sh /runpod-volume

echo ""
echo "✅ Volume populated successfully. You can now terminate this pod."
