#!/bin/bash
# ============================================================
# populate.sh — RunPod Network Volume Seeder
# ============================================================
set -euo pipefail

# ──────────────── Config ────────────────────────────────────
TROCR_CKPT="microsoft/trocr-base-handwritten"          
VISION_MODEL="${VISION_MODEL:-qwen2.5vl:7b-q8_0}"    
TEXT_MODEL="${TEXT_MODEL:-qwen2.5:7b-instruct-q4_K_M}"  
OLLAMA_VERSION="${OLLAMA_VERSION:-0.6.5}"

# ───────────────── Volume paths ─────────────────────────────
export HF_HOME=/runpod-volume/hf-cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OLLAMA_MODELS=/runpod-volume/ollama-models
export PADDLE_HOME=/runpod-volume/paddle-cache
export PADDLEOCR_HOME=/runpod-volume/paddle-cache/.paddleocr
export FLAGS_use_mkldnn=0
export PADDLE_DISABLE_MKLDNN=1
PYPACKAGES=/runpod-volume/pypackages

echo "============================================================"
echo " Network Volume Seeder"
echo " TrOCR        : $TROCR_CKPT"
echo " Vision model : $VISION_MODEL"
echo " Text model   : $TEXT_MODEL"
echo "============================================================"
echo ""

# ── 0. Verify volume ─────────────────────────────────────────
echo "=== Checking network volume ==="
df -h | grep runpod-volume || { echo "ERROR: /runpod-volume not mounted"; exit 1; }
AVAIL=$(df -BG /runpod-volume | tail -1 | awk '{print $4}' | tr -d 'G')
[[ "$AVAIL" -lt 18 ]] && echo "WARNING: Only ${AVAIL}GB free — need at least 22 GB."
echo ""

# ── 1. Directories ───────────────────────────────────────────
echo "=== Creating directories ==="
mkdir -p "$HF_HOME" "$OLLAMA_MODELS" "$PADDLEOCR_HOME" "$PYPACKAGES"
echo "Done."
echo ""

# ── 2. System packages ───────────────────────────────────────
echo "=== System packages ==="
apt-get update -qq && apt-get install -y -qq \
    python3.10 python3.10-venv python3-pip libglib2.0-0 libgl1 curl
python3.10 -m venv /tmp/venv
source /tmp/venv/bin/activate
pip install --upgrade pip setuptools wheel --quiet
echo "Done."
echo ""

# ── 3. Python packages → volume ──────────────────────────────
echo "=== Python packages → $PYPACKAGES (~4 GB) ==="
pip install \
    torch==2.2.0 \
    paddlepaddle==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
    paddleocr==2.7.3 \
    opencv-python-headless==4.8.1.78 \
    transformers==4.41.2 \
    sentencepiece \
    Pillow \
    numpy==1.26.4 \
    --target "$PYPACKAGES" \
    --quiet
echo "Done: $(du -sh $PYPACKAGES | cut -f1)"
echo ""

# ── 4. PaddleOCR models ──────────────────────────────────────
echo "=== PaddleOCR PP-OCRv3 models (~300 MB) ==="
PYTHONPATH=$PYPACKAGES FLAGS_use_mkldnn=0 PADDLE_DISABLE_MKLDNN=1 python3 - <<'PYEOF'
import sys; sys.path.insert(0, "/runpod-volume/pypackages")
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import tempfile, os

ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
img = Image.new("RGB", (200, 50), "white")
ImageDraw.Draw(img).text((10, 15), "test", fill="black")
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    img.save(f.name); ocr.ocr(f.name, cls=True); os.unlink(f.name)
print("PaddleOCR models downloaded.")
PYEOF

if [ -d "$PADDLEOCR_HOME" ] && [ "$(ls -A $PADDLEOCR_HOME 2>/dev/null)" ]; then
    echo "Verified: $(du -sh $PADDLEOCR_HOME | cut -f1)"
else
    echo "ERROR: PaddleOCR models missing"; exit 1
fi
echo ""

# ── 5. TrOCR BASE model ──────────────────────────────────────
echo "=== TrOCR ($TROCR_CKPT, ~430 MB) ==="
PYTHONPATH=$PYPACKAGES python3 - <<PYEOF
import sys; sys.path.insert(0, "/runpod-volume/pypackages")
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
ckpt = "$TROCR_CKPT"
print(f"Downloading {ckpt}...")
TrOCRProcessor.from_pretrained(ckpt)
VisionEncoderDecoderModel.from_pretrained(ckpt)
print("TrOCR ready.")
PYEOF

[ "$(ls -A $HF_HOME 2>/dev/null)" ] && echo "Verified: $(du -sh $HF_HOME | cut -f1)" \
    || { echo "ERROR: HF cache empty"; exit 1; }
echo ""

# ── 6. Ollama ────────────────────────────────────────────────
echo "=== Installing Ollama v${OLLAMA_VERSION} ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=${OLLAMA_VERSION} sh
fi
ollama --version
echo ""

# ── 7. Start Ollama daemon ───────────────────────────────────
echo "=== Starting Ollama ==="
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_ORIGINS="*"
OLLAMA_MODELS="$OLLAMA_MODELS" ollama serve &
OLLAMA_PID=$!

for i in {1..40}; do
    curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && { echo "Ollama ready."; break; }
    sleep 2
    [[ $i -eq 40 ]] && { echo "ERROR: Ollama API timeout"; kill $OLLAMA_PID; exit 1; }
done
echo ""

# ── 8. Pull Qwen models ──────────────────────────────────────
echo "=== Pulling $VISION_MODEL (~8 GB, takes 3-8 min) ==="
OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull "$VISION_MODEL"
echo "Done: $VISION_MODEL"
echo ""

echo "=== Pulling $TEXT_MODEL (~4.7 GB, takes 2-5 min) ==="
OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull "$TEXT_MODEL"
echo "Done: $TEXT_MODEL"
echo ""

# ── 9. Verify manifests ──────────────────────────────────────
echo "=== Verifying Ollama manifests ==="
MANIFEST_BASE="$OLLAMA_MODELS/manifests/registry.ollama.ai/library"
ERRORS=0
for MODEL in "$VISION_MODEL" "$TEXT_MODEL"; do
    NAME="${MODEL%%:*}"; TAG="${MODEL##*:}"
    if [ -f "$MANIFEST_BASE/$NAME/$TAG" ]; then
        echo "  $MODEL"
    else
        echo "  MISSING: $MODEL (expected at $MANIFEST_BASE/$NAME/$TAG)"
        ERRORS=$((ERRORS+1))
    fi
done
[[ $ERRORS -gt 0 ]] && { echo "ERROR: $ERRORS model(s) missing"; kill $OLLAMA_PID; exit 1; }
echo ""

# ── 10. Smoke tests ──────────────────────────────────────────
echo "=== Smoke test: $TEXT_MODEL ==="
OLLAMA_MODELS="$OLLAMA_MODELS" ollama run "$TEXT_MODEL" "Reply TEXT_OK only." --nowordwrap
echo ""

echo "=== Smoke test: $VISION_MODEL ==="
OLLAMA_MODELS="$OLLAMA_MODELS" ollama run "$VISION_MODEL" "Reply VISION_OK only." --nowordwrap
echo ""

kill $OLLAMA_PID 2>/dev/null || true

# ── 11. Final summary ─────────────────────────────────────────
echo ""
echo "=== Disk usage ==="
du -sh /runpod-volume/*/
echo "TOTAL: $(du -sh /runpod-volume | cut -f1)"
echo ""
echo "============================================================"
echo "Volume seeded. RunPod Serverless settings:"
echo "  Min workers  : 0          (zero idle cost)"
echo "  Flash Boot   : ENABLED"
echo "  Network vol  : /runpod-volume"
echo "  Idle timeout : 60s"
echo ""
echo "Endpoint env vars:"
echo "  VISION_MODEL=$VISION_MODEL"
echo "  TEXT_MODEL=$TEXT_MODEL"
echo "  OLLAMA_KEEP_ALIVE=15m"
echo "============================================================"
