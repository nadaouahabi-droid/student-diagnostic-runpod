"""
RunPod Serverless Handler — Student Diagnostic System
======================================================
Actions:
  health      → service status
  ocr-batch   → PaddleOCR (print text) → TrOCR (handwriting refinement)
  ollama-chat → Ollama vision/text model
"""

import runpod
import subprocess
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

# ── Cache env vars BEFORE any paddle/hf imports ──────────────
os.environ['PADDLEX_HOME']          = '/runpod-volume/paddle-cache/.paddlex'
os.environ['FLAGS_use_mkldnn']      = '0'
os.environ['PADDLE_DISABLE_MKLDNN'] = '1'
os.environ['HF_HOME']               = '/runpod-volume/hf-cache/huggingface'
os.environ['OLLAMA_MODELS']         = '/runpod-volume/ollama-models'
os.environ['OLLAMA_ORIGINS']        = '*'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUIRED_MODELS    = ["qwen2.5vl:7b-q8_0", "qwen2.5:7b-instruct-q4_K_M"]
OLLAMA_MODELS_PATH = "/runpod-volume/ollama-models/manifests/registry.ollama.ai/library"

# ✅ Correct HF cache dir — models live under HF_HOME/hub/
HF_CACHE_DIR = "/runpod-volume/hf-cache/huggingface/hub"

def verify_models_on_volume():
    missing = []
    for model_name in REQUIRED_MODELS:
        model_dir = model_name.split(":")[0]
        path = os.path.join(OLLAMA_MODELS_PATH, model_dir)
        if not os.path.isdir(path):
            missing.append(model_name)

    if missing:
        log.error(
            f"⚠️  Models not found on volume (is network volume attached?): {missing}\n"
            f"  Expected path: {OLLAMA_MODELS_PATH}"
        )
    else:
        log.info("✅ All required Ollama models found on volume.")

verify_models_on_volume()

# ── Configuration ─────────────────────────────────────────────
OLLAMA_URL   = "http://127.0.0.1:11434"
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b-q8_0")
TEXT_MODEL   = os.environ.get("TEXT_MODEL",   "qwen2.5:7b-instruct-q4_K_M")

# ── Startup synchronisation events ───────────────────────────
OLLAMA_READY = threading.Event()
MODELS_READY = threading.Event()

# ── Model singletons ─────────────────────────────────────────
_paddle          = None
_trocr_processor = None
_trocr_model     = None
_trocr_device    = None


# ═══════════════════════════════════════════════════════════════
# OLLAMA BOOT
# ═══════════════════════════════════════════════════════════════
def _ollama_boot():
    log.info("[init] Starting Ollama server...")
    env = os.environ.copy()
    env["OLLAMA_HOST"]   = "127.0.0.1:11434"
    env["OLLAMA_MODELS"] = "/runpod-volume/ollama-models"

    proc = subprocess.Popen(["ollama", "serve"], env=env)

    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                log.info(f"[init] Ollama ready. Loaded models: {models}")
                OLLAMA_READY.set()
                return
        except Exception:
            pass
        time.sleep(2)

    proc.kill()
    log.error("[init] Ollama FAILED to start within 4 minutes.")


# ═══════════════════════════════════════════════════════════════
# OCR PRE-WARM
# ═══════════════════════════════════════════════════════════════
def _ocr_prewarm():
    global _paddle, _trocr_processor, _trocr_model, _trocr_device

    paddle_ok = False
    trocr_ok  = False

    log.info("[init] Pre-warming PaddleOCR...")
    try:
        from paddleocr import PaddleOCR
        _paddle = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
        log.info("[init] PaddleOCR ready.")
        paddle_ok = True
    except Exception:
        log.error("[init] PaddleOCR pre-warm FAILED:\n" + traceback.format_exc())

    log.info("[init] Pre-warming TrOCR...")
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = "microsoft/trocr-large-handwritten"

        # ✅ FIX: use cache_dir pointing to HF_HOME/hub where models were saved
        _trocr_processor = TrOCRProcessor.from_pretrained(
            ckpt, cache_dir=HF_CACHE_DIR
        )
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            ckpt, cache_dir=HF_CACHE_DIR
        ).to(_trocr_device)

        _trocr_model.eval()
        log.info(f"[init] TrOCR ready on {_trocr_device}.")
        trocr_ok = True

    except Exception:
        log.error("[init] TrOCR pre-warm FAILED:\n" + traceback.format_exc())

    if paddle_ok and trocr_ok:
        MODELS_READY.set()
        log.info("[init] ✅ All models warm — worker is fully ready.")
    else:
        log.error(
            f"[init] ❌ Pre-warm incomplete — PaddleOCR={'ok' if paddle_ok else 'FAILED'}, "
            f"TrOCR={'ok' if trocr_ok else 'FAILED'}"
        )


threading.Thread(target=_ollama_boot,  daemon=True, name="ollama-boot").start()
threading.Thread(target=_ocr_prewarm, daemon=True, name="ocr-prewarm").start()


# ═══════════════════════════════════════════════════════════════
# OCR MODEL ACCESSORS
# ═══════════════════════════════════════════════════════════════
def get_paddle():
    if _paddle is None:
        raise RuntimeError("PaddleOCR not loaded yet — pre-warm incomplete.")
    return _paddle


def get_trocr():
    if _trocr_model is None:
        raise RuntimeError("TrOCR not loaded yet — pre-warm incomplete.")
    return _trocr_processor, _trocr_model, _trocr_device


# ═══════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ═══════════════════════════════════════════════════════════════
def b64_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def preprocess(img: Image.Image, scale: float = 2.0) -> Image.Image:
    w, h = img.size
    img  = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img  = ImageEnhance.Sharpness(img).enhance(2.0)
    img  = ImageEnhance.Contrast(img).enhance(1.4)
    return img


# ═══════════════════════════════════════════════════════════════
# PADDLEOCR — print text detection
# ═══════════════════════════════════════════════════════════════
def run_paddle(img: Image.Image) -> list:
    paddle = get_paddle()
    arr = np.array(img)

    result = paddle.predict(arr)
    items = []

    if not result:
        return items

    # 🔥 Normalize result to iterable list
    if isinstance(result, dict):
        result = [result]

    for page in result:
        # If page itself is a string → wrap it
        if isinstance(page, str):
            items.append({
                "bbox": [],
                "text": page.strip(),
                "confidence": 1.0
            })
            continue

        if not page:
            continue

        # If page is dict → treat as single line
        if isinstance(page, dict):
            page = [page]

        for line in page:
            try:
                # 🔴 Case: string (THIS IS YOUR CRASH CASE)
                if isinstance(line, str):
                    items.append({
                        "bbox": [],
                        "text": line.strip(),
                        "confidence": 1.0
                    })
                    continue

                # ✅ Case: dict (new PaddleOCR)
                if isinstance(line, dict):
                    text = line.get("rec_text", "").strip()
                    conf = float(line.get("rec_score", 0))
                    bbox = line.get("bbox", [])

                # ✅ Case: list (old PaddleOCR)
                elif isinstance(line, list) and len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0].strip()
                    conf = float(line[1][1])

                else:
                    continue

                if text:
                    items.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": conf
                    })

            except Exception as e:
                log.warning(f"[ocr] Skipping bad line: {line} | error: {e}")
                continue

    log.info(f"[ocr] Extracted {len(items)} items")
    return items

# ═══════════════════════════════════════════════════════════════
# TROCR — handwriting refinement on low-confidence regions
# ═══════════════════════════════════════════════════════════════
def refine_with_trocr(img: Image.Image, bbox, pad: int = 6) -> Optional[str]:
    try:
        import torch
        processor, model, device = get_trocr()
        xs = [p[0] for p in bbox];  ys = [p[1] for p in bbox]
        x1 = max(0, int(min(xs)) - pad);  x2 = min(img.width,  int(max(xs)) + pad)
        y1 = max(0, int(min(ys)) - pad);  y2 = min(img.height, int(max(ys)) + pad)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = img.crop((x1, y1, x2, y2))
        if crop.width < 64 or crop.height < 16:
            s    = max(64 / crop.width, 16 / crop.height, 1)
            crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
        pv = processor(images=crop, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=80)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        log.debug(f"[ocr] TrOCR crop failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════
# FULL OCR PIPELINE FOR ONE PAGE
# ═══════════════════════════════════════════════════════════════
def ocr_page(b64: str, refine_threshold: float = 0.72) -> dict:
    t0       = time.time()
    img      = b64_to_pil(b64)
    img_prep = preprocess(img)

    items = run_paddle(img_prep)
    log.info(f"[ocr] PaddleOCR → {len(items)} regions")

    output_lines  = []
    refined_count = 0

    for item in items:
        text    = item["text"]
        conf    = item["confidence"]
        refined = False

        if conf < refine_threshold and text.strip():
            orig_bbox = [[p[0] / 2.0, p[1] / 2.0] for p in item["bbox"]]
            better    = refine_with_trocr(img, orig_bbox)
            if better and len(better) >= len(text) * 0.5:
                text    = better
                refined = True
                refined_count += 1

        output_lines.append({"text": text, "confidence": conf, "refined": refined})

    full_text = "\n".join(l["text"] for l in output_lines if l["text"])
    elapsed   = round(time.time() - t0, 2)
    log.info(f"[ocr] Page done: {len(output_lines)} lines, {refined_count} TrOCR-refined, {elapsed}s")

    return {
        "text":          full_text,
        "lines":         output_lines,
        "line_count":    len(output_lines),
        "refined_count": refined_count,
        "timing_s":      elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# OLLAMA CHAT HELPER
# ═══════════════════════════════════════════════════════════════
def ollama_chat(model, messages, options=None, timeout=120):
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  options or {"temperature": 0.05},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


# ═══════════════════════════════════════════════════════════════
# RUNPOD HANDLER
# ═══════════════════════════════════════════════════════════════
def handler(job):
    job_input = job.get("input", {})
    action    = job_input.get("action", "ollama-chat")

    if action == "health":
        try:
            r      = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            return {
                "status":        "ok",
                "ollama_ready":  OLLAMA_READY.is_set(),
                "models_ready":  MODELS_READY.is_set(),
                "models_loaded": models,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    STARTUP_TIMEOUT = 300

    if not OLLAMA_READY.wait(timeout=STARTUP_TIMEOUT):
        return {"error": "Ollama not ready — timed out. Check worker logs."}

    if action == "ocr-batch" and not MODELS_READY.wait(timeout=STARTUP_TIMEOUT):
        return {"error": "OCR models not ready — timed out. Check worker logs."}

    if action == "ocr-batch":
        images    = job_input.get("images", [])
        threshold = float(job_input.get("refine_threshold", 0.72))

        if not images:
            return {"success": False, "error": "No images provided",
                    "combined_text": "", "pages": []}

        log.info(f"[ocr] Batch: {len(images)} page(s), threshold={threshold}")
        pages = []

        for i, b64 in enumerate(images):
            log.info(f"[ocr] Processing page {i+1}/{len(images)}...")
            try:
                result = ocr_page(b64, threshold)
                pages.append({"page": i + 1, **result})
            except Exception as e:
                log.error(traceback.format_exc())
                return {"success": False,
                        "error": f"OCR failed on page {i+1}: {e}",
                        "pages_completed": i}

        combined = "\n\n".join(
            f"=== PAGE {p['page']} ===\n{p['text']}" for p in pages
        )
        return {"success": True, "pages": pages,
                "combined_text": combined, "total_pages": len(pages)}

    if action == "ollama-chat":
        model    = job_input.get("model", TEXT_MODEL)
        messages = job_input.get("messages", [])
        options  = job_input.get("options", {"temperature": 0.05})
        timeout  = int(job_input.get("timeout_seconds", 900))
        try:
            content = ollama_chat(model, messages, options, timeout)
            return {"content": content}
        except requests.exceptions.Timeout:
            return {"error": f"Ollama timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown action: '{action}'"}


log.info("🚀 Handler loaded — Ollama boot and OCR pre-warm running in background threads.")
runpod.serverless.start({"handler": handler})
