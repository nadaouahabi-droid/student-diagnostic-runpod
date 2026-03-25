"""
RunPod Serverless Handler — Student Diagnostic System v10
Manages: PaddleOCR server (port 5005) + Ollama (port 11434)
"""

import runpod
import subprocess
import requests
import time
import os

def wait_for(name, url, retries=60, delay=2):
    print(f"[boot] Waiting for {name} at {url}...")
    for i in range(retries):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"[boot] {name} ready after {i*delay}s")
                return True
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(f"[boot] {name} did not start within {retries*delay}s")


def boot():
    env = os.environ.copy()
    env["OLLAMA_ORIGINS"] = "*"
    env["OLLAMA_MODELS"]  = "/runpod-volume/ollama-models"  
    subprocess.Popen(["ollama", "serve"], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for("Ollama", "http://localhost:11434/api/tags")

    print("[boot] Pre-warming vision model into VRAM...")
    requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5vl:7b-q8_0",
        "stream": False,
        "messages": [{"role": "user", "content": "hi", "images": []}],
        "options": {"num_predict": 1}
    }, timeout=120)

    print("[boot] Pre-warming text model into VRAM...")
    requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
        "options": {"num_predict": 1}
    }, timeout=120)

    subprocess.Popen(["python", "-u", "/app/ocr_server.py"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for("OCR Server", "http://localhost:5005/health")

    print("[boot] All services ready. Handler open.")


boot()   


def handler(job):
    """
    Routes on job["input"]["action"]:

      "health"       → service status check
      "ocr-batch"    → PaddleOCR + TrOCR via /ocr-batch  (Stage A)
      "ollama-chat"  → Ollama /api/chat                   (Stages B & C)
    """
    inp    = job.get("input", {})
    action = inp.get("action", "health")

    if action == "health":
        services = {}
        for name, url in [("ocr_server", "http://localhost:5005/health"),
                           ("ollama",     "http://localhost:11434/api/tags")]:
            try:
                r = requests.get(url, timeout=5)
                services[name] = "ok" if r.status_code == 200 else f"error {r.status_code}"
            except Exception as e:
                services[name] = f"unreachable: {e}"
        return {"status": "ok", "services": services}

    if action == "ocr-batch":
        payload = {
            "images":           inp.get("images", []),
            "refine_threshold": inp.get("refine_threshold", 0.72),
        }
        try:
            r = requests.post(
                "http://localhost:5005/ocr-batch",
                json=payload,
                timeout=inp.get("timeout_seconds", 300),
            )
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    if action == "ollama-chat":
        payload = {
            "model":    inp.get("model", "qwen2.5:7b-instruct-q4_K_M"),
            "stream":   False,   
            "messages": inp.get("messages", []),
            "options":  inp.get("options", {
                "temperature": 0.05,
                "num_predict": 6144,
                "num_ctx":     8192,
            }),
        }
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=inp.get("timeout_seconds", 600),
            )
            data = r.json()
            return {"content": data.get("message", {}).get("content", "")}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown action: '{action}'"}


runpod.serverless.start({"handler": handler})
