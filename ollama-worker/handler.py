import runpod
import subprocess
import time
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

os.environ['OLLAMA_MODELS']  = '/runpod-volume/ollama-models'
os.environ['OLLAMA_ORIGINS'] = '*'

OLLAMA_URL = "http://127.0.0.1:11434"

OLLAMA_READY = False


def start_ollama():
    global OLLAMA_READY

    subprocess.Popen(["ollama", "serve"])

    for _ in range(120):
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                OLLAMA_READY = True
                return
        except:
            pass
        time.sleep(2)

    raise RuntimeError("Ollama failed to start")


start_ollama()


def ollama_chat(model, messages, options=None, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {"temperature": 0.05},
    }

    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "ollama-chat")

    if action == "health":
        return {"status": "ok", "ollama_ready": OLLAMA_READY}

    if not OLLAMA_READY:
        return {"error": "Ollama not ready"}

    if action == "ollama-chat":
        model    = job_input.get("model", "qwen2.5:7b-instruct-q4_K_M")
        messages = job_input.get("messages", [])
        options  = job_input.get("options", {"temperature": 0.05})
        timeout  = int(job_input.get("timeout_seconds", 900))

        try:
            content = ollama_chat(model, messages, options, timeout)
            return {"content": content}
        except Exception as e:
            return {"error": str(e)}

    return {"error": "Unknown action"}


runpod.serverless.start({"handler": handler})
