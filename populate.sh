#!/bin/bash
# ============================================================
# Run this script ONCE on a temporary RunPod pod
# to populate the network volume with all models.
#
# Requirements:
#   - Network volume mounted at /runpod-volume
#   - At least 30GB free on the volume
#
# Usage:
#   bash populate_volume.sh
# ============================================================

set -e

echo "=== Checking volume is mounted ==="
df -h | grep runpod || { echo "ERROR: /runpod-volume not mounted"; exit 1; }
ls /runpod-volume/ || { echo "ERROR: /runpod-volume is empty or not accessible"; exit 1; }

# ── 1. Ollama models ─────────────────────────────────────────
echo ""
echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Pulling Ollama models ==="
export OLLAMA_MODELS=/runpod-volume/ollama-models
mkdir -p /runpod-volume/ollama-models

ollama serve &
OLLAMA_PID=$!
sleep 5

ollama pull qwen2.5vl:7b-q8_0
ollama pull qwen2.5:7b-instruct-q4_K_M

echo "=== Verifying Ollama models ==="
ollama list

kill $OLLAMA_PID 2>/dev/null || true

# ── 2. PaddleOCR weights ─────────────────────────────────────
echo ""
echo "=== Setting up Python 3.10 environment ==="
apt-get update && apt-get install -y python3.10 python3.10-venv

python3.10 -m venv /tmp/venv
source /tmp/venv/bin/activate

pip install --upgrade pip setuptools wheel

echo "=== Installing PaddleOCR ==="
pip install paddleocr==3.0.0 paddlepaddle==3.0.0 Pillow numpy

echo "=== Downloading PaddleOCR weights (runs inference to trigger download) ==="
PADDLEX_HOME=/runpod-volume/paddle-cache/.paddlex \
FLAGS_use_mkldnn=0 PADDLE_DISABLE_MKLDNN=1 python3 -c "
from paddleocr import PaddleOCR
from PIL import Image

img = Image.new('RGB', (200, 50), color='white')
img.save('/tmp/test.png')

ocr = PaddleOCR(use_angle_cls=True, lang='en', device='cpu')
ocr.ocr('/tmp/test.png')
print('PaddleOCR weights downloaded successfully')
"

echo "PaddleOCR done. Contents:"
ls /runpod-volume/paddle-cache/.paddlex/official_models/

# ── 3. TrOCR weights ─────────────────────────────────────────
echo ""
echo "=== Installing Transformers ==="
pip install transformers torch sentencepiece --quiet

echo "=== Downloading TrOCR weights ==="
# BUG FIX: Point HF_HOME at the target volume path so weights are
# downloaded directly there.  Previously the script downloaded to
# the default ~/.cache/huggingface and then cp'd them across, but
# the destination path ended up as:
#   /runpod-volume/hf-cache/huggingface/huggingface/hub/...
#              ^^^ extra 'huggingface' level from `cp -r`
# which doesn't match HF_HOME=/runpod-volume/hf-cache/huggingface
# that the handler sets.  Downloading directly avoids the mismatch.
HF_HOME=/runpod-volume/hf-cache/huggingface \
python3 -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
ckpt = 'microsoft/trocr-large-handwritten'
TrOCRProcessor.from_pretrained(ckpt)
VisionEncoderDecoderModel.from_pretrained(ckpt)
print('TrOCR weights downloaded successfully')
"

echo "TrOCR done. Contents:"
ls /runpod-volume/hf-cache/huggingface/

# ── 4. Final verification ────────────────────────────────────
echo ""
echo "=== Final volume structure ==="
find /runpod-volume -maxdepth 4 -type d

echo ""
echo "=== Disk usage ==="
du -sh /runpod-volume/*

echo ""
echo "✅ Volume populated successfully. You can now terminate this pod."
