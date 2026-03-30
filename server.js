"""
Render Web Server — Student Diagnostic System
"""
from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ✅ Point Flask to your public folder
app = Flask(__name__, static_folder="public", static_url_path="")

RUNPOD_API_KEY     = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE_URL    = f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type":  "application/json",
}


# ── Serve index.html at root ──────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("public", "index.html")


# ── Catch-all: serve any static file from public/ ────────────
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("public", path)


# ── RunPod proxy routes ───────────────────────────────────────
@app.route("/api/run", methods=["POST"])
def run_job():
    payload = request.get_json()
    try:
        r = requests.post(
            f"{RUNPOD_BASE_URL}/run",
            json={"input": payload},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        log.error(f"RunPod /run error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<job_id>", methods=["GET"])
def job_status(job_id):
    try:
        r = requests.get(
            f"{RUNPOD_BASE_URL}/status/{job_id}",
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        log.error(f"RunPod /status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/runsync", methods=["POST"])
def run_sync():
    payload = request.get_json()
    try:
        r = requests.post(
            f"{RUNPOD_BASE_URL}/runsync",
            json={"input": payload},
            headers=HEADERS,
            timeout=120,
        )
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        log.error(f"RunPod /runsync error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
