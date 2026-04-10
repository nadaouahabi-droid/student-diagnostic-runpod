from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="public", static_url_path="")

HF_API_URL = os.environ.get("HF_API_URL", "")

# ── Serve frontend ───────────────────────────
@app.route("/")
def index():
    return send_from_directory("public", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("public", path)

# ── /api/ai — called by index.html ───────────
@app.route("/api/ai", methods=["POST"])
def ai():
    payload = request.get_json()

    try:
        # 🔹 Direct call to HF Space (NO polling)
        r = requests.post(
            f"{HF_API_URL}/ocr",   # or /analyze depending on use
            json=payload,
            timeout=120
        )
        r.raise_for_status()

        return jsonify(r.json())

    except requests.exceptions.Timeout:
        return jsonify({"error": "HF request timed out"}), 504
    except Exception as e:
        log.error(f"[HF] Error: {e}")
        return jsonify({"error": str(e)}), 500
