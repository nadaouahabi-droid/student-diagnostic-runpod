import modal

app = modal.App("ocr-app")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "paddleocr==2.7.3",
        "paddlepaddle==2.6.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.8.1.78",
        "transformers==4.41.2",
        "torch==2.2.0",
        "pillow"
    )
)

@app.function(
    image=image,
    gpu="T4",               # or "A10G"
    timeout=300,
    scaledown_window=300    # reduce cold start impact
)
@modal.web_endpoint(method="POST")
def ocr_endpoint(data: dict):
    import base64, io
    import numpy as np
    from PIL import Image
    from paddleocr import PaddleOCR

    # Lazy load
    global ocr
    if "ocr" not in globals():
        ocr = PaddleOCR(use_angle_cls=True, lang="en")

    img_b64 = data["image"]
    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")

    result = ocr.ocr(np.array(img))

    text = "\n".join([line[1][0] for line in result[0]])

    return {"text": text}
