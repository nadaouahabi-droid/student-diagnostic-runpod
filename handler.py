"""
RunPod Serverless Handler — Student Diagnostic System
"""

import sys as _sys
import os as _os
_sys.path.insert(0, "/runpod-volume/pypackages")

_PADDLE_HOME = "/runpod-volume/.paddleocr"          # ← was /runpod-volume/paddle-cache/.paddleocr
_os.environ["PADDLE_HOME"]           = _PADDLE_HOME
_os.environ["PADDLEOCR_HOME"]        = _PADDLE_HOME
_os.environ["PPOCR_HOME"]            = _PADDLE_HOME
_os.environ["PADDLEX_HOME"]          = _PADDLE_HOME + "/.paddlex"
_os.environ["FLAGS_use_mkldnn"]      = "0"
_os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
_os.environ["HF_HOME"]               = "/runpod-volume/hf-cache/huggingface"
_os.environ["TRANSFORMERS_CACHE"]    = "/runpod-volume/hf-cache/huggingface"
_os.environ["OLLAMA_MODELS"]         = "/runpod-volume/ollama-models"
_os.environ["OLLAMA_ORIGINS"]        = "*"
_os.environ["PYTHONPATH"]            = "/runpod-volume/pypackages"

import runpod
import subprocess
import threading
import time
import requests
import io
import base64
import logging
import traceback
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
REQUIRED_MODELS      = ["qwen2.5vl:7b-q8_0", "qwen2.5:7b-instruct-q4_K_M"]
OLLAMA_MANIFEST_BASE = "/runpod-volume/ollama-models/manifests/registry.ollama.ai/library"
HF_CACHE_DIR         = "/runpod-volume/hf-cache/huggingface"
TROCR_CHECKPOINT     = "microsoft/trocr-base-handwritten"
OLLAMA_URL           = "http://127.0.0.1:11434"
VISION_MODEL         = _os.environ.get("VISION_MODEL", "qwen2.5vl:7b-q8_0")
TEXT_MODEL           = _os.environ.get("TEXT_MODEL",   "qwen2.5:7b-instruct-q4_K_M")
MAX_OCR_SIDE         = 3800
STARTUP_TIMEOUT      = 300

# ── Startup synchronisation events ───────────────────────────────
OLLAMA_READY  = threading.Event()
MODELS_READY  = threading.Event()   
PADDLE_READY  = threading.Event()  
TROCR_READY   = threading.Event()

# ── Model singletons ─────────────────────────────────────────────
_paddle          = None
_paddle_lock     = threading.Lock()

_trocr_processor = None
_trocr_model     = None
_trocr_device    = None
_torch           = None
_trocr_lock      = threading.Lock()

# ═══════════════════════════════════════════════════════════════
# VOLUME VERIFICATION
# ═══════════════════════════════════════════════════════════════
def verify_models_on_volume():
    """
    Check that exact model tags exist as manifest files.
    Directory presence alone is insufficient — a dir can exist from a
    different tag or a partial pull.
    """
    missing = []
    for model in REQUIRED_MODELS:
        name, tag = model.split(":", 1)
        manifest  = _os.path.join(OLLAMA_MANIFEST_BASE, name, tag)
        if not _os.path.isfile(manifest):
            missing.append(f"{model}  (expected: {manifest})")

    if missing:
        log.error(
            "Models missing on volume — did you run populate.sh?\n  "
            + "\n  ".join(missing)
        )
    else:
        log.info("All required Ollama models found on volume.")

verify_models_on_volume()

# ═══════════════════════════════════════════════════════════════
# OLLAMA BOOT + WATCHDOG
# ═══════════════════════════════════════════════════════════════
def _ollama_vram_warmup():
    """
    Fire a minimal 1-token inference on each model immediately after Ollama
    is ready so the GGUF weights are already mapped into VRAM before the
    first real job arrives.  Without this, the first job pays a ~8–20 s
    model-load tax from the network volume.
    """
    log.info("[ollama] Starting VRAM warm-up for all models...")
    for model in [VISION_MODEL, TEXT_MODEL]:
        try:
            ollama_chat(
                model,
                [{"role": "user", "content": "hi"}],
                options={"temperature": 0, "num_predict": 1},
                timeout=180,
            )
            log.info(f"[ollama] VRAM warm-up done: {model}")
        except Exception:
            log.warning(f"[ollama] VRAM warm-up failed for {model} — "
                        "model will load on first real job.\n" + traceback.format_exc())

def _ollama_boot():
    env = _os.environ.copy()
    env["OLLAMA_HOST"]       = "127.0.0.1:11434"
    env["OLLAMA_MODELS"]     = "/runpod-volume/ollama-models"
    env["OLLAMA_KEEP_ALIVE"] = _os.environ.get("OLLAMA_KEEP_ALIVE", "15m")

    while True:
        log.info("[ollama] Starting Ollama server...")
        proc = subprocess.Popen(["ollama", "serve"], env=env)

        deadline = time.time() + 240
        ready    = False
        while time.time() < deadline:
            try:
                r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    log.info(f"[ollama] Ready. Models available: {models}")
                    OLLAMA_READY.set()
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not ready:
            log.error("[ollama] Failed to start within 4 minutes — retrying...")
            proc.kill()
            time.sleep(5)
            continue

        threading.Thread(
            target=_ollama_vram_warmup, daemon=True, name="ollama-vram-warmup"
        ).start()

        proc.wait()
        log.warning("[ollama] Process exited unexpectedly — restarting...")
        OLLAMA_READY.clear()
        time.sleep(3)

# ═══════════════════════════════════════════════════════════════
# OCR PRE-WARM  (Paddle + TrOCR in parallel)
# ═══════════════════════════════════════════════════════════════
def _warm_paddle():
    global _paddle
    log.info("[init] Pre-warming PaddleOCR...")
    try:
        from paddleocr import PaddleOCR
        with _paddle_lock:
            if _paddle is None:
                _paddle = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
        PADDLE_READY.set()
        log.info("[init] PaddleOCR ready.")
        return True
    except Exception:
        log.error("[init] PaddleOCR FAILED:\n" + traceback.format_exc())
        return False

def _warm_trocr():
    global _trocr_processor, _trocr_model, _trocr_device, _torch
    log.info("[init] Pre-warming TrOCR (%s)...", TROCR_CHECKPOINT)
    snap_dir = os.path.join(HF_CACHE_DIR, "models--microsoft--trocr-base-handwritten")
    if not os.path.isdir(snap_dir):
        log.error(
            f"[init] TrOCR cache missing at {snap_dir} — "
            "did populate.sh complete successfully?"
        )
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        processor = TrOCRProcessor.from_pretrained(
            TROCR_CHECKPOINT, cache_dir=HF_CACHE_DIR, local_files_only=True, trust_remote_code=True
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            TROCR_CHECKPOINT,
            cache_dir=HF_CACHE_DIR,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)
        model.eval()

        with _trocr_lock:
            _torch           = torch
            _trocr_device    = device
            _trocr_processor = processor
            _trocr_model     = model

        TROCR_READY.set()
        log.info(f"[init] TrOCR ready on {device}.")
        return True
    except Exception:
        log.error("[init] TrOCR FAILED:\n" + traceback.format_exc())
        return False

def _ocr_prewarm():
    results = {}

    def run(name, fn):
        results[name] = fn()

    t_paddle = threading.Thread(target=run, args=("paddle", _warm_paddle))
    t_trocr  = threading.Thread(target=run, args=("trocr",  _warm_trocr))
    t_paddle.start()
    t_trocr.start()
    t_paddle.join()
    t_trocr.join()

    paddle_ok = results.get("paddle", False)
    trocr_ok  = results.get("trocr",  False)

    if paddle_ok and trocr_ok:
        MODELS_READY.set()
        log.info("[init] All OCR models warm — worker fully ready.")
    else:
        log.error(
            f"[init] Pre-warm incomplete — "
            f"PaddleOCR={'ok' if paddle_ok else 'FAILED'}, "
            f"TrOCR={'ok' if trocr_ok else 'FAILED'}"
        )

threading.Thread(target=_ollama_boot,  daemon=True, name="ollama-boot").start()
threading.Thread(target=_ocr_prewarm, daemon=True, name="ocr-prewarm").start()

# ═══════════════════════════════════════════════════════════════
# MODEL ACCESSORS
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
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def preprocess(img: Image.Image, scale: float = 2.0) -> Image.Image:
    """Upscale for OCR quality, clamped to MAX_OCR_SIDE."""
    w, h            = img.size
    effective_scale = min(scale, MAX_OCR_SIDE / max(w, h))
    if effective_scale != 1.0:
        img = img.resize(
            (int(w * effective_scale), int(h * effective_scale)), Image.LANCZOS
        )
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    return img

# ═══════════════════════════════════════════════════════════════
# PADDLEOCR
# ═══════════════════════════════════════════════════════════════
def run_paddle(img: Image.Image) -> list:
    result = get_paddle().ocr(np.array(img), cls=True)
    items  = []
    if not result:
        return items
    if isinstance(result, dict):
        result = [result]

    for page in result:
        if isinstance(page, str):
            items.append({"bbox": [], "text": page.strip(), "confidence": 1.0})
            continue
        if not page:
            continue
        if isinstance(page, dict):
            page = [page]
        for line in page:
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
            except Exception as e:
                log.warning(f"[ocr] Skipping bad line: {line} | {e}")

    log.info(f"[ocr] Extracted {len(items)} items")
    return items

# ═══════════════════════════════════════════════════════════════
# TROCR — BATCHED REFINEMENT
# ═══════════════════════════════════════════════════════════════
def _crop_for_trocr(img: Image.Image, bbox, pad: int = 6) -> Optional[Image.Image]:
    """
    Extract and minimally resize one bounding-box crop from *img*.
    Returns None if the crop is too small to be useful.
    """
    try:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1 = max(0, int(min(xs)) - pad)
        x2 = min(img.width,  int(max(xs)) + pad)
        y1 = max(0, int(min(ys)) - pad)
        y2 = min(img.height, int(max(ys)) + pad)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = img.crop((x1, y1, x2, y2))
        if crop.width < 64 or crop.height < 16:
            s    = max(64 / crop.width, 16 / crop.height, 1.0)
            crop = crop.resize(
                (int(crop.width * s), int(crop.height * s)), Image.LANCZOS
            )
        return crop
    except Exception as exc:
        log.debug(f"[ocr] Crop extraction failed: {exc}")
        return None

def refine_batch_with_trocr(
    img: Image.Image,
    candidates: list[tuple[int, dict]],   # [(original_index, item), ...]
    orig_scale: tuple[float, float],
) -> dict[int, str]:
    """
    Run TrOCR on all low-confidence candidates in a single batched
    model.generate() call.  Returns a mapping {original_index: refined_text}.

    Batching matters because N sequential generate() calls leave the GPU
    mostly idle between calls.  A single padded batch saturates it once.
    """
    if not candidates:
        return {}

    processor, model, device = get_trocr()
    sx, sy = orig_scale

    # ── Build crop list, keeping track of which indices yielded valid crops ──
    valid_indices: list[int]        = []
    crops:         list[Image.Image] = []

    for idx, item in candidates:
        if not item["bbox"]:
            continue
        orig_bbox = [[p[0] * sx, p[1] * sy] for p in item["bbox"]]
        crop      = _crop_for_trocr(img, orig_bbox)
        if crop is not None:
            valid_indices.append(idx)
            crops.append(crop)

    if not crops:
        return {}

    # ── Single batched inference ──────────────────────────────────────────────
    results: dict[int, str] = {}
    try:
        # TrOCRProcessor pads to the longest image in the batch automatically
        pixel_values = processor(
            images=crops, return_tensors="pt", padding=True
        ).pixel_values.to(device)

        with _torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=80)

        texts = processor.batch_decode(ids, skip_special_tokens=True)

        for idx, text in zip(valid_indices, texts):
            refined = text.strip()
            if refined:
                results[idx] = refined

        log.debug(f"[ocr] TrOCR batch: {len(crops)} crops → {len(results)} non-empty")
    except Exception:
        log.warning("[ocr] TrOCR batch failed:\n" + traceback.format_exc())

    return results

# ═══════════════════════════════════════════════════════════════
# FULL OCR PIPELINE
# ═══════════════════════════════════════════════════════════════
def ocr_page(b64: str, refine_threshold: float = 0.72) -> dict:
    t0       = time.time()
    img      = b64_to_pil(b64)
    img_prep = preprocess(img)

    items = run_paddle(img_prep)
    log.info(f"[ocr] PaddleOCR → {len(items)} regions")

    prep_w, prep_h = img_prep.size
    orig_w, orig_h = img.size
    scale          = (orig_w / prep_w, orig_h / prep_h)

    # ── Collect all low-confidence items for a single batched TrOCR call ──
    candidates = [
        (i, item)
        for i, item in enumerate(items)
        if item["confidence"] < refine_threshold
        and item["text"].strip()
        and item["bbox"]
    ]

    refined_map = refine_batch_with_trocr(img, candidates, scale)

    # ── Merge results ─────────────────────────────────────────────────────
    output_lines  = []
    refined_count = 0

    for i, item in enumerate(items):
        text    = item["text"]
        conf    = item["confidence"]
        refined = False

        if i in refined_map:
            better = refined_map[i]
            # Accept TrOCR result only if it's not drastically shorter
            if len(better) >= len(text) * 0.5:
                text    = better
                refined = True
                refined_count += 1

        output_lines.append({"text": text, "confidence": conf, "refined": refined})

    full_text = "\n".join(l["text"] for l in output_lines if l["text"])
    elapsed   = round(time.time() - t0, 2)
    log.info(
        f"[ocr] Page done: {len(output_lines)} lines, "
        f"{refined_count} TrOCR-refined ({len(candidates)} candidates), {elapsed}s"
    )

    return {
        "text":          full_text,
        "lines":         output_lines,
        "line_count":    len(output_lines),
        "refined_count": refined_count,
        "timing_s":      elapsed,
    }

# ═══════════════════════════════════════════════════════════════
# OLLAMA CHAT
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

    # ── health — always responds, even before models are ready ────────────
    if action == "health":
        try:
            r      = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            return {
                "status":        "ok",
                "ollama_ready":  OLLAMA_READY.is_set(),
                "paddle_ready":  PADDLE_READY.is_set(),
                "trocr_ready":   TROCR_READY.is_set(),
                "models_ready":  MODELS_READY.is_set(),
                "models_loaded": models,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    if not OLLAMA_READY.wait(timeout=STARTUP_TIMEOUT):
        return {"error": "Ollama not ready — timed out. Check worker logs."}

    # ── ocr-batch ────────────────────────────────────────────────────────
    if action == "ocr-batch":
        if not MODELS_READY.wait(timeout=STARTUP_TIMEOUT):
            paddle_ok = PADDLE_READY.is_set()
            trocr_ok  = TROCR_READY.is_set()
            return {
                "error": (
                    f"OCR models not ready — timed out. "
                    f"PaddleOCR={'ok' if paddle_ok else 'FAILED'}, "
                    f"TrOCR={'ok' if trocr_ok else 'FAILED'}. "
                    "Check worker logs."
                )
            }

        images    = job_input.get("images", [])
        threshold = float(job_input.get("refine_threshold", 0.72))

        if not images:
            return {
                "success": False, "error": "No images provided",
                "combined_text": "", "pages": [],
            }

        log.info(f"[ocr] Batch: {len(images)} page(s), threshold={threshold}")
        pages = []
        for i, b64 in enumerate(images):
            log.info(f"[ocr] Processing page {i+1}/{len(images)}...")
            try:
                result = ocr_page(b64, threshold)
                pages.append({"page": i + 1, **result})
            except Exception as e:
                log.error(traceback.format_exc())
                return {
                    "success": False,
                    "error":   f"OCR failed on page {i+1}: {e}",
                    "pages_completed": i,
                }

        combined = "\n\n".join(
            f"=== PAGE {p['page']} ===\n{p['text']}" for p in pages
        )
        return {
            "success":      True,
            "pages":        pages,
            "combined_text": combined,
            "total_pages":  len(pages),
        }

    # ── ollama-chat ──────────────────────────────────────────────────────
    if action == "ollama-chat":
        model    = job_input.get("model", TEXT_MODEL)
        messages = job_input.get("messages", [])
        options  = job_input.get("options", {"temperature": 0.05})
        timeout  = int(job_input.get("timeout_seconds", 900))
        try:
            return {"content": ollama_chat(model, messages, options, timeout)}
        except requests.exceptions.Timeout:
            return {"error": f"Ollama timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown action: '{action}'"}

log.info("Handler loaded — Ollama boot and OCR pre-warm running in background threads.")
runpod.serverless.start({"handler": handler})
