"""
Student Diagnostic OCR + Reasoning API (CPU-only)
PaddleOCR + TrOCR + FLAN-T5
"""

import threading
import time
import io
import base64
import logging
import traceback
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

from fastapi import FastAPI
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TROCR_CHECKPOINT = "microsoft/trocr-base-handwritten"
TEXT_MODEL_NAME = "google/flan-t5-small"
MAX_OCR_SIDE = 3000

# ─────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────
_paddle = None
_paddle_lock = threading.Lock()

_trocr_processor = None
_trocr_model = None
_torch = None
_trocr_lock = threading.Lock()

_text_tokenizer = None
_text_model = None

PADDLE_READY = threading.Event()
TROCR_READY = threading.Event()
TEXT_READY = threading.Event()
MODELS_READY = threading.Event()

# ─────────────────────────────────────────────
# Model Warmup
# ─────────────────────────────────────────────
def _warm_paddle():
    global _paddle
    try:
        from paddleocr import PaddleOCR
        with _paddle_lock:
            if _paddle is None:
                _paddle = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=False,
                    det_limit_side_len=960
                )
        PADDLE_READY.set()
        log.info("[init] PaddleOCR ready")
        return True
    except Exception:
        log.error(traceback.format_exc())
        return False


def _warm_trocr():
    global _trocr_processor, _trocr_model, _torch
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(TROCR_CHECKPOINT)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_CHECKPOINT, torch_dtype=torch.float32)
        model.eval()

        with _trocr_lock:
            _trocr_processor = processor
            _trocr_model = model
            _torch = torch

        TROCR_READY.set()
        log.info("[init] TrOCR ready (CPU)")
        return True
    except Exception:
        log.error(traceback.format_exc())
        return False


def _warm_text():
    global _text_tokenizer, _text_model
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        _text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        _text_model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_NAME)

        TEXT_READY.set()
        log.info("[init] FLAN-T5 ready (CPU)")
        return True
    except Exception:
        log.error(traceback.format_exc())
        return False


def _prewarm():
    results = {}

    def run(name, fn):
        results[name] = fn()

    threads = [
        threading.Thread(target=run, args=("paddle", _warm_paddle)),
        threading.Thread(target=run, args=("trocr", _warm_trocr)),
        threading.Thread(target=run, args=("text", _warm_text)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if all(results.values()):
        MODELS_READY.set()
        log.info("[init] All models ready")


threading.Thread(target=_prewarm, daemon=True).start()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def preprocess(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = min(1.5, MAX_OCR_SIDE / max(w, h))
    if scale != 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = ImageEnhance.Sharpness(img).enhance(1.8)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    return img


# ─────────────────────────────────────────────
# PaddleOCR
# ─────────────────────────────────────────────
def run_paddle(img):
    result = _paddle.ocr(np.array(img), cls=True)
    items = []

    if not result:
        return items

    for line in result[0]:
        bbox = line[0]
        text = line[1][0]
        conf = float(line[1][1])
        items.append({"bbox": bbox, "text": text, "confidence": conf})

    return items


# ─────────────────────────────────────────────
# TrOCR refinement
# ─────────────────────────────────────────────
def refine_with_trocr(img, items, threshold=0.7):
    processor = _trocr_processor
    model = _trocr_model

    results = []

    for item in items:
        if item["confidence"] > threshold:
            results.append(item)
            continue

        try:
            bbox = item["bbox"]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            crop = img.crop((min(xs), min(ys), max(xs), max(ys)))

            pixel_values = processor(images=crop, return_tensors="pt").pixel_values

            with _torch.no_grad():
                ids = model.generate(pixel_values, max_new_tokens=50)

            text = processor.batch_decode(ids, skip_special_tokens=True)[0]

            item["text"] = text.strip()
            item["refined"] = True

        except:
            pass

        results.append(item)

    return results


# ─────────────────────────────────────────────
# OCR Pipeline
# ─────────────────────────────────────────────
def ocr_page(b64):
    img = b64_to_pil(b64)
    img_prep = preprocess(img)

    items = run_paddle(img_prep)
    items = refine_with_trocr(img, items)

    full_text = "\n".join(i["text"] for i in items if i["text"])

    return {
        "text": full_text,
        "lines": items,
        "count": len(items)
    }


# ─────────────────────────────────────────────
# Text model
# ─────────────────────────────────────────────
def run_text_model(prompt):
    inputs = _text_tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = _text_model.generate(
        **inputs,
        max_new_tokens=200
    )

    return _text_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
class Request(BaseModel):
    image: Optional[str] = None
    images: Optional[list[str]] = None
    prompt: Optional[str] = None


@app.get("/")
def root():
    return {"message": "OCR + AI API running"}


@app.get("/health")
def health():
    return {
        "paddle": PADDLE_READY.is_set(),
        "trocr": TROCR_READY.is_set(),
        "text": TEXT_READY.is_set(),
        "ready": MODELS_READY.is_set()
    }


@app.post("/ocr")
def ocr(req: Request):
    if not MODELS_READY.is_set():
        return {"error": "models not ready"}

    return ocr_page(req.image)


@app.post("/ocr-batch")
def ocr_batch(req: Request):
    if not MODELS_READY.is_set():
        return {"error": "models not ready"}

    results = []
    for img in req.images:
        results.append(ocr_page(img))

    return {"pages": results}


@app.post("/analyze")
def analyze(req: Request):
    if not TEXT_READY.is_set():
        return {"error": "text model not ready"}

    return {"result": run_text_model(req.prompt)}
