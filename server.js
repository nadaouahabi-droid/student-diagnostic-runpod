from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="public", static_url_path="")

RUNPOD_API_KEY     = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE_URL    = f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type":  "application/json",
}


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("public", path)


# ── /api/ai — called by index.html ───────────────────────────
# Submits job to RunPod and polls until complete
@app.route("/api/ai", methods=["POST"])
def ai():
    payload = request.get_json()
    job_input = payload.get("input", payload)
    timeout_seconds = int(job_input.get("timeout_seconds", 600))

    try:
        # Submit job
        r = requests.post(
            f"{RUNPOD_BASE_URL}/run",
            json={"input": job_input},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        job_id = r.json().get("id")
        if not job_id:
            return jsonify({"error": "No job ID returned from RunPod"}), 500

        log.info(f"[runpod] Job submitted: {job_id}")

        # Poll until complete
        import time
        deadline = time.time() + timeout_seconds + 60
        while time.time() < deadline:
            poll = requests.get(
                f"{RUNPOD_BASE_URL}/status/{job_id}",
                headers=HEADERS,
                timeout=30,
            )
            poll.raise_for_status()
            data   = poll.json()
            status = data.get("status")

            log.info(f"[runpod] Job {job_id} status: {status}")

            if status == "COMPLETED":
                return jsonify(data)
            elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                return jsonify({"error": f"RunPod job {status}", "details": data}), 500

            time.sleep(3)

        return jsonify({"error": "Timed out waiting for RunPod job"}), 504

    except requests.exceptions.Timeout:
        return jsonify({"error": "RunPod request timed out"}), 504
    except Exception as e:
        log.error(f"[runpod] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ── /api/status — optional direct poll from browser ──────────
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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
