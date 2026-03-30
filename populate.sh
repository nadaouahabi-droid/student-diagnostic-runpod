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
export PADDLEX_HOME=/runpod-volume/paddle-cache/.paddlex

mkdir -p $HF_HOME
mkdir -p $OLLAMA_MODELS
mkdir -p $PADDLEX_HOME/fonts           # ✅ FIX: fonts must live inside PADDLEX_HOME

# ------------------------------------------------------------
# 1. Ollama models
# ------------------------------------------------------------
echo ""
echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Stopping any auto-started Ollama service ==="
systemctl stop ollama 2>/dev/null || true
systemctl disable ollama 2>/dev/null || true
pkill ollama 2>/dev/null || true
sleep 3

echo "=== Starting Ollama with correct model path ==="
OLLAMA_MODELS=/runpod-volume/ollama-models ollama serve &
OLLAMA_PID=$!

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
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-distutils wget ccache
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
PADDLEX_HOME=$PADDLEX_HOME FLAGS_use_mkldnn=0 PADDLE_DISABLE_MKLDNN=1 python3 -c "
from paddleocr import PaddleOCR
from PIL import Image
img = Image.new('RGB', (200, 50), color='white')
img.save('/tmp/test.png')
ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='cpu')
ocr.predict('/tmp/test.png')
print('PaddleOCR weights downloaded successfully')
"

# ✅ FIX: Fonts must go inside PADDLEX_HOME so PaddleX finds them at runtime
echo "=== Pre-downloading PaddleOCR fonts into PADDLEX_HOME ==="
FONT_BASE="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts"
wget -q "$FONT_BASE/PingFang-SC-Regular.ttf" -O $PADDLEX_HOME/fonts/PingFang-SC-Regular.ttf
wget -q "$FONT_BASE/simfang.ttf"              -O $PADDLEX_HOME/fonts/simfang.ttf

echo "✅ Fonts downloaded to PADDLEX_HOME/fonts:"
ls -lh $PADDLEX_HOME/fonts/

echo "=== Copying PaddleOCR weights to volume ==="
# Weights are already written to PADDLEX_HOME by the python call above
# (since we exported PADDLEX_HOME pointing to the volume)
echo "PaddleOCR contents:"
ls $PADDLEX_HOME/official_models/

if ls $PADDLEX_HOME/official_models/ &>/dev/null; then
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
# ✅ FIX: HF_HOME env var is set — from_pretrained without cache_dir
# will write to HF_HOME/hub/ automatically
HF_HOME=$HF_HOME python3 -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
ckpt = 'microsoft/trocr-large-handwritten'
TrOCRProcessor.from_pretrained(ckpt)
VisionEncoderDecoderModel.from_pretrained(ckpt)
print('TrOCR weights downloaded successfully')
"

echo "TrOCR contents:"
ls $HF_HOME/hub/

if [ "$(ls -A $HF_HOME/hub/ 2>/dev/null)" ]; then
  echo "✅ TrOCR weights confirmed on volume at HF_HOME/hub/"
else
  echo "❌ ERROR: TrOCR weights NOT found on volume!"
  exit 1
fi

# ------------------------------------------------------------
# 5. Final verification
# ------------------------------------------------------------
echo ""
echo "=== FINAL STRUCTURE ==="
find /runpod-volume -maxdepth 4 -type d

echo ""
echo "=== DISK USAGE ==="
du -sh /runpod-volume/*

echo ""
echo "=== TOTAL SIZE ==="
du -sh /runpod-volume

echo ""
echo "=== FINAL SANITY CHECK ==="
echo -n "Ollama models:     "; ls /runpod-volume/ollama-models/manifests/registry.ollama.ai/library/ 2>/dev/null | tr '\n' ' ' && echo ""
echo -n "PaddleOCR models:  "; ls $PADDLEX_HOME/official_models/ 2>/dev/null | tr '\n' ' ' && echo ""
echo -n "PaddleOCR fonts:   "; ls $PADDLEX_HOME/fonts/ 2>/dev/null | tr '\n' ' ' && echo ""
echo -n "HuggingFace cache: "; ls $HF_HOME/hub/ 2>/dev/null | tr '\n' ' ' && echo ""

echo ""
echo "✅ Volume populated successfully. You can now terminate this pod."
