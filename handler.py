"""
RunPod Serverless Handler — Student Diagnostic System
======================================================
Starts Ollama on worker init (once, not per request).
Models are pre-baked into the Docker image on local NVMe.

FIXES:
- Ollama starts at import time (RunPod-safe)
- Added debug logs for deployment verification
- Fixed silent OCR failures
- Enabled logs for debugging
"""

import runpod
import subprocess
import threading
import time
import requests
import os

# ── Configuration ──────────────────────────────────────────────────────────
OLLAMA_URL    = "http://127.0.0.1:11434"
VISION_MODEL  = os.environ.get("VISION_MODEL", "qwen2.5vl:7b-q4_K_M")
TEXT_MODEL    = os.environ.get("TEXT_MODEL",   "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_READY  = threading.Event()

# ── Start Ollama once at worker boot ───────────────────────────────────────
def start_ollama():
    """Launch Ollama server in background and wait until it accepts requests."""
    print("[init] Starting Ollama server...", flush=True)

    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env["OLLAMA_MODELS"] = "/root/.ollama/models"

    proc = subprocess.Popen(
        ["/usr/local/bin/ollama", "serve"],
        env=env,
        stdout=None,   # show logs
        stderr=None,
    )

    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print(f"[init] Ollama ready. Models: {models}", flush=True)
                OLLAMA_READY.set()
                return proc
        except Exception:
            pass
        time.sleep(2)

    proc.kill()
    raise RuntimeError("[init] Ollama failed to start within 3 minutes")

# ── Ollama chat helper ─────────────────────────────────────────────────────
def ollama_chat(model, messages, options=None, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {"temperature": 0.05},
    }

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")

# ── Request handler ────────────────────────────────────────────────────────
def handler(job):

    if not OLLAMA_READY.wait(timeout=180):
        return {"error": "Ollama not ready — timed out after 3 minutes"}

    job_input = job.get("input", {})
    action    = job_input.get("action", "ollama-chat")

    # ── health ────────────────────────────────────────────────────────────
    if action == "health":
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            return {
                "status": "ok",
                "services": {
                    "ollama": "running",
                    "ocr_server": "ollama-vision"
                },
                "models_loaded": models,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── ollama-chat ───────────────────────────────────────────────────────
    if action == "ollama-chat":
        model    = job_input.get("model", TEXT_MODEL)
        messages = job_input.get("messages", [])
        options  = job_input.get("options", {"temperature": 0.05})
        timeout  = int(job_input.get("timeout_seconds", 900))

        try:
            content = ollama_chat(model, messages, options, timeout)
            return {"content": content}
        except requests.exceptions.Timeout:
            return {"error": f"Ollama request timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    # ── ocr-batch ─────────────────────────────────────────────────────────
    if action == "ocr-batch":
        images  = job_input.get("images", [])
        timeout = int(job_input.get("timeout_seconds", 900))

        if not images:
            return {
                "success": False,
                "error": "No images provided",
                "combined_text": "",
                "pages": []
            }

        pages    = []
        combined = []
        success  = True

        for i, img_b64 in enumerate(images):
            prompt = (
                f"You are an expert OCR specialist analysing exam page {i + 1} of {len(images)}.\n"
                "Extract ALL text exactly as written — every question number, mark scheme bracket, "
                "student answer, tick mark (✓), cross mark (✗), circled score, and teacher correction.\n"
                "Include crossed-out working verbatim.\n"
                "Preserve structure. Do not summarise."
            )

            messages = [{
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }]

            try:
                text = ollama_chat(VISION_MODEL, messages, {"temperature": 0.05}, timeout)
            except Exception as e:
                text = f"[OCR error on page {i + 1}: {e}]"
                success = False

            pages.append({"page": i + 1, "text": text})
            combined.append(f"=== PAGE {i + 1} ===\n{text}")

        return {
            "success": success,
            "combined_text": "\n\n".join(combined),
            "pages": pages,
        }

    return {"error": f"Unknown action: '{action}'"}

print("🚀 HANDLER VERSION V3 LOADED", flush=True)

start_ollama()

print("[init] RunPod handler starting...", flush=True)
runpod.serverless.start({"handler": handler})
