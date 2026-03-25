#!/usr/bin/env python3
"""
OCR Server for Student Diagnostic System
Pipeline: PaddleOCR (layout + print text) → TrOCR (handwriting refinement)
Serves results to the HTML frontend at http://localhost:5005
"""

import os, io, base64, sys, time, logging, traceback
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# LAZY-LOAD MODELS (initialise once on first request)
# ─────────────────────────────────────────
_paddle = None
_trocr_processor = None
_trocr_model = None
_device = None

def get_paddle():
    global _paddle
    if _paddle is None:
        log.info("Loading PaddleOCR (first call)…")
        from paddleocr import PaddleOCR
        _paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False,
                            use_gpu=False, enable_mkldnn=True)
        log.info("PaddleOCR ready.")
    return _paddle

def get_trocr():
    global _trocr_processor, _trocr_model, _device
    if _trocr_model is None:
        log.info("Loading TrOCR (first call)…")
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use 'large' for best accuracy; swap to 'base' if RAM is tight
        ckpt = "microsoft/trocr-large-handwritten"
        _trocr_processor = TrOCRProcessor.from_pretrained(ckpt)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(ckpt).to(_device)
        _trocr_model.eval()
        log.info(f"TrOCR ready on {_device}.")
    return _trocr_processor, _trocr_model, _device


# ─────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────
def b64_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def preprocess(img: Image.Image, scale: float = 2.0) -> Image.Image:
    """Upscale + sharpen for better OCR accuracy."""
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    return img


# ─────────────────────────────────────────
# PADDLE OCR ON ONE PAGE
# ─────────────────────────────────────────
def run_paddle(img: Image.Image):
    """Return list of dicts: {bbox, text, confidence}"""
    paddle = get_paddle()
    arr = np.array(img)
    result = paddle.ocr(arr, cls=True)
    items = []
    if result and result[0]:
        for line in result[0]:
            if not line:
                continue
            bbox, (text, conf) = line
            items.append({"bbox": bbox, "text": text.strip(), "confidence": float(conf)})
    return items


# ─────────────────────────────────────────
# TROCR REFINEMENT ON A SINGLE REGION
# ─────────────────────────────────────────
def refine_with_trocr(img: Image.Image, bbox, pad: int = 6) -> Optional[str]:
    """Crop the bbox from img, run TrOCR, return refined text or None on failure."""
    try:
        import torch
        processor, model, device = get_trocr()
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(img.width,  int(max(xs)) + pad)
        y2 = min(img.height, int(max(ys)) + pad)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = img.crop((x1, y1, x2, y2))
        # Upscale tiny crops so TrOCR has enough pixels
        if crop.width < 64 or crop.height < 16:
            scale = max(64 / crop.width, 16 / crop.height, 1)
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        pv = processor(images=crop, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=80)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        log.debug(f"TrOCR crop failed: {exc}")
        return None


# ─────────────────────────────────────────
# COMBINED PIPELINE FOR ONE PAGE
# ─────────────────────────────────────────
def ocr_page(b64: str, refine_threshold: float = 0.72) -> dict:
    """
    Full pipeline for one page image (base64).
    Returns: {text, lines, line_count, timing}
    """
    t0 = time.time()

    img = b64_to_pil(b64)
    img_prep = preprocess(img)

    # Stage 1 – PaddleOCR
    items = run_paddle(img_prep)
    log.info(f"  PaddleOCR → {len(items)} regions")

    # Stage 2 – TrOCR refinement for low-confidence lines
    output_lines = []
    refined_count = 0
    for item in items:
        text = item["text"]
        conf = item["confidence"]
        refined = False

        if conf < refine_threshold and len(text.strip()) > 0:
            # Scale bbox back to original image coordinates (we upscaled 2×)
            orig_bbox = [
                [p[0] / 2.0, p[1] / 2.0] for p in item["bbox"]
            ]
            better = refine_with_trocr(img, orig_bbox)
            if better and len(better) >= len(text) * 0.5:
                text = better
                refined = True
                refined_count += 1

        output_lines.append({
            "text": text,
            "confidence": conf,
            "refined": refined,
        })

    full_text = "\n".join(l["text"] for l in output_lines if l["text"])
    elapsed = round(time.time() - t0, 2)
    log.info(f"  Page done: {len(output_lines)} lines, {refined_count} TrOCR-refined, {elapsed}s")

    return {
        "text": full_text,
        "lines": output_lines,
        "line_count": len(output_lines),
        "refined_count": refined_count,
        "timing_s": elapsed,
    }


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "pipeline": ["PaddleOCR", "TrOCR"], "version": "1.0"})


@app.route("/ocr", methods=["POST"])
def ocr_single():
    """Process a single page image."""
    data = request.get_json(force=True)
    b64 = data.get("image", "")
    threshold = float(data.get("refine_threshold", 0.72))
    if not b64:
        return jsonify({"success": False, "error": "No image provided"}), 400
    try:
        result = ocr_page(b64, threshold)
        return jsonify({"success": True, **result})
    except Exception as exc:
        log.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/ocr-batch", methods=["POST"])
def ocr_batch():
    """
    Process multiple pages.
    Body: { "images": ["base64_page1", "base64_page2", …], "refine_threshold": 0.72 }
    Returns: { "success": true, "pages": [{page, text, lines, …}], "combined_text": "…" }
    """
    data = request.get_json(force=True)
    images = data.get("images", [])
    threshold = float(data.get("refine_threshold", 0.72))

    if not images:
        return jsonify({"success": False, "error": "No images provided"}), 400

    log.info(f"Batch OCR: {len(images)} page(s)")
    pages = []
    try:
        for i, b64 in enumerate(images):
            log.info(f"Processing page {i+1}/{len(images)}…")
            result = ocr_page(b64, threshold)
            pages.append({"page": i + 1, **result})

        combined = "\n\n".join(
            f"=== PAGE {p['page']} ===\n{p['text']}" for p in pages
        )
        return jsonify({"success": True, "pages": pages, "combined_text": combined,
                        "total_pages": len(pages)})

    except Exception as exc:
        log.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("OCR_PORT", 5005))
    log.info(f"Starting OCR server on http://localhost:{port}")
    log.info("Models will load on first request (may take 30-60 s the first time).")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
