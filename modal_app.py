import modal

app = modal.App("learndiag-ocr")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "paddleocr==2.7.3",
        "paddlepaddle==2.6.2",
        "numpy==1.26.4",
        "opencv-python-headless==4.8.1.78",
        "pillow",
        "fastapi[standard]",
    )
)

# ── Shared OCR model (loaded once per container) ──────────────────────────────
@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
)
class OCRService:
    @modal.enter()
    def load_model(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    @modal.method()
    def run_ocr(self, image_b64: str) -> dict:
        import base64, io
        import numpy as np
        from PIL import Image

        img = Image.open(
            io.BytesIO(base64.b64decode(image_b64))
        ).convert("RGB")
        result = self.ocr.ocr(np.array(img))
        lines = []
        if result and result[0]:
            lines = [line[1][0] for line in result[0]]
        return {"text": "\n".join(lines), "success": True}


# ── OCR batch endpoint (called by Vercel proxy) ───────────────────────────────
@app.function(image=image, timeout=600)
@modal.web_endpoint(method="POST", label="ocr-batch")
def ocr_batch(data: dict):
    """
    Expects: { "images": ["base64...", ...], "refine_threshold": 0.72 }
    Returns: { "success": true, "pages": [{"text": "..."}], "combined_text": "..." }
    """
    images = data.get("images", [])
    service = OCRService()
    pages = []
    for b64 in images:
        try:
            result = service.run_ocr.remote(b64)
            pages.append(result)
        except Exception as e:
            pages.append({"text": "", "success": False, "error": str(e)})

    combined = "\n\n".join(p.get("text", "") for p in pages)
    return {
        "success": True,
        "pages": pages,
        "combined_text": combined,
    }


# ── Health check ──────────────────────────────────────────────────────────────
@app.function(image=image)
@modal.web_endpoint(method="GET", label="health")
def health():
    return {"status": "ok"}
