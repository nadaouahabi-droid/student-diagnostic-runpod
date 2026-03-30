#!/bin/bash
# ============================================================
# Populate RunPod Network Volume with all models (FIXED VERSION)
# ============================================================
set -e

echo "=== Checking volume is mounted ==="
df -h | grep runpod || { echo "ERROR: /runpod-volume not mounted"; exit 1; }

# ------------------------------------------------------------
# 0. Environment variables
# ------------------------------------------------------------
export HF_HOME=/runpod-volume/hf-cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OLLAMA_MODELS=/runpod-volume/ollama-models

mkdir -p $HF_HOME
mkdir -p $OLLAMA_MODELS
mkdir -p /runpod-volume/paddle-cache/fonts

# ------------------------------------------------------------
# 1. Ollama models
# ------------------------------------------------------------
echo ""
echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

# Stop any auto-started Ollama service spawned by the installer
echo "=== Stopping any auto-started Ollama service ==="
systemctl stop ollama 2>/dev/null || true
systemctl disable ollama 2>/dev/null || true
pkill ollama 2>/dev/null || true
sleep 3

echo "=== Starting Ollama with correct model path ==="
OLLAMA_MODELS=/runpod-volume/ollama-models ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is actually ready (instead of fixed sleep)
echo "=== Waiting for Ollama to be ready ==="
for i in $(seq 1 20); do
  ollama list &>/dev/null && echo "  Ollama ready after ${i}s" && break
  echo "  Waiting... ($i/20)"
  sleep 2
done

echo "=== Pulling Ollama models ==="
OLLAMA_MODELS=/runpod-volume/ollama-models ollama pull qwen2.5vl:7b-q8_0
OLLAMA_MODELS=/runpod-volume/ollama-models ollama pull qwen2.5:7b-instruct-q4_K_M

echo "=== Verifying Ollama models ==="
ollama list

echo "=== Verifying Ollama storage path ==="
if ls /runpod-volume/ollama-models/manifests/registry.ollama.ai/library/ &>/dev/null; then
  echo "✅ Ollama models confirmed on volume:"
  ls -lh /runpod-volume/ollama-models/manifests/registry.ollama.ai/library/
else
  echo "❌ ERROR: Ollama models NOT found on volume! Aborting."
  kill $OLLAMA_PID 2>/dev/null || true
  exit 1
fi

# Warn if models leaked to default path
if [ -d "/root/.ollama/models" ] && [ "$(ls -A /root/.ollama/models 2>/dev/null)" ]; then
  echo "⚠️  WARNING: Files found in default /root/.ollama/models — possible leak!"
  du -sh /root/.ollama/models
fi

kill $OLLAMA_PID 2>/dev/null || true
sleep 2

# ------------------------------------------------------------
# 2. Python 3.10 environment
# ------------------------------------------------------------
echo ""
echo "=== Setting up Python 3.10 ==="
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-distutils ccache
python3.10 -m venv /tmp/venv
source /tmp/venv/bin/activate
pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 3. PaddleOCR
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

# Pre-download fonts so they are never fetched at runtime
echo "=== Pre-downloading PaddleOCR fonts ==="
FONT_BASE="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts"
wget -q "$FONT_BASE/PingFang-SC-Regular.ttf" -O /runpod-volume/paddle-cache/fonts/PingFang-SC-Regular.ttf
wget -q "$FONT_BASE/simfang.ttf"              -O /runpod-volume/paddle-cache/fonts/simfang.ttf

echo "✅ Fonts downloaded:"
ls -lh /runpod-volume/paddle-cache/fonts/

echo "=== Copying PaddleOCR weights to volume ==="
cp -r /root/.paddlex /runpod-volume/paddle-cache/

echo "PaddleOCR contents:"
ls /runpod-volume/paddle-cache/.paddlex/official_models/

# Verify paddle copy succeeded
if ls /runpod-volume/paddle-cache/.paddlex/official_models/ &>/dev/null; then
  echo "✅ PaddleOCR weights confirmed on volume"
else
  echo "❌ ERROR: PaddleOCR weights NOT found on volume!"
  exit 1
fi

# ------------------------------------------------------------
# 4. TrOCR (HuggingFace)
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

# Verify TrOCR download
if [ "$(ls -A $HF_HOME 2>/dev/null)" ]; then
  echo "✅ TrOCR weights confirmed on volume"
else
  echo "❌ ERROR: TrOCR weights NOT found on volume!"
  exit 1
fi

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
echo "=== FINAL SANITY CHECK ==="
echo -n "Ollama models:     "; ls /runpod-volume/ollama-models/manifests/registry.ollama.ai/library/ 2>/dev/null | tr '\n' ' ' && echo ""
echo -n "PaddleOCR models:  "; ls /runpod-volume/paddle-cache/.paddlex/official_models/ 2>/dev/null | tr '\n' ' ' && echo ""
echo -n "PaddleOCR fonts:   "; ls /runpod-volume/paddle-cache/fonts/ 2>/dev/null | tr '\n' ' ' && echo ""
echo -n "HuggingFace cache: "; ls $HF_HOME 2>/dev/null | tr '\n' ' ' && echo ""

echo ""
echo "✅ Volume populated successfully. You can now terminate this pod."
