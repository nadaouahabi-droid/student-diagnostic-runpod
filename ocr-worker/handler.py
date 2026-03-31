import runpod
import threading
import time
import requests
import os
import io
import base64
import logging
import traceback
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

# ── Env ─────────────────────────────────────
os.environ['PADDLEX_HOME']          = '/runpod-volume/paddle-cache/.paddlex'
os.environ['FLAGS_use_mkldnn']      = '0'
os.environ['PADDLE_DISABLE_MKLDNN'] = '1'
os.environ['HF_HOME']               = '/runpod-volume/hf-cache/huggingface'

HF_CACHE_DIR = "/runpod-volume/hf-cache/huggingface/hub"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODELS_READY = threading.Event()

_paddle          = None
_trocr_processor = None
_trocr_model     = None
_trocr_device    = None


# ── PREWARM ─────────────────────────────────
def _ocr_prewarm():
    global _paddle, _trocr_processor, _trocr_model, _trocr_device

    try:
        from paddleocr import PaddleOCR
        _paddle = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
    except Exception:
        log.error(traceback.format_exc())

    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _trocr_device = "cuda" if torch.cuda.is_available() else "cpu"

        _trocr_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-large-handwritten",
            cache_dir=HF_CACHE_DIR
        )
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-handwritten",
            cache_dir=HF_CACHE_DIR
        ).to(_trocr_device)

        _trocr_model.eval()

    except Exception:
        log.error(traceback.format_exc())

    if _paddle and _trocr_model:
        MODELS_READY.set()

threading.Thread(target=_ocr_prewarm, daemon=True).start()


# ── HELPERS ─────────────────────────────────
def get_paddle():
    if _paddle is None:
        raise RuntimeError("PaddleOCR not ready")
    return _paddle

def get_trocr():
    if _trocr_model is None:
        raise RuntimeError("TrOCR not ready")
    return _trocr_processor, _trocr_model, _trocr_device


def b64_to_pil(b64: str):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def preprocess(img):
    w, h = img.size
    img = img.resize((int(w * 2), int(h * 2)), Image.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    return img


# ── OCR CORE ─────────────────────────────────
def run_paddle(img):
    paddle = get_paddle()
    arr = np.array(img)

    result = paddle.predict(arr)
    items = []

    if isinstance(result, dict):
        result = [result]

    for page in result or []:
        if isinstance(page, dict):
            page = [page]

        for line in page or []:
            try:
                if isinstance(line, str):
                    items.append({"bbox": [], "text": line.strip(), "confidence": 1.0})
                    continue

                if isinstance(line, dict):
                    text = line.get("rec_text", "").strip()
                    conf = float(line.get("rec_score", 0))
                    bbox = line.get("bbox", [])

                elif isinstance(line, list) and len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0].strip()
                    conf = float(line[1][1])
                else:
                    continue

                if text:
                    items.append({"bbox": bbox, "text": text, "confidence": conf})

            except Exception:
                continue

    return items


def refine_with_trocr(img, bbox):
    try:
        import torch
        processor, model, device = get_trocr()

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        crop = img.crop((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))

        pv = processor(images=crop, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pv)

        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    except Exception:
        return None


def ocr_page(b64):
    img = b64_to_pil(b64)
    img_prep = preprocess(img)

    items = run_paddle(img_prep)

    lines = []
    for item in items:
        lines.append(item["text"])

    return {"text": "\n".join(lines), "lines": lines}


# ── HANDLER ─────────────────────────────────
def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action")

    if action == "health":
        return {"status": "ok", "models_ready": MODELS_READY.is_set()}

    if action == "ocr-batch":
        if not MODELS_READY.wait(timeout=300):
            return {"error": "Models not ready"}

        images = job_input.get("images", [])
        pages = [ocr_page(b64) for b64 in images]

        return {"success": True, "pages": pages}

    return {"error": "Unknown action"}


runpod.serverless.start({"handler": handler})
