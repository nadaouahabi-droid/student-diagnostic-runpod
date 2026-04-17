#!/bin/bash
# populate.sh — Pull qwen3 models onto a RunPod Network Volume
# Compatible with the built Docker image (Ollama already present)
# and with a bare RunPod instance (auto-installs Ollama if missing).
#
# Usage:
#   bash populate.sh
#
# Env overrides:
#   VOLUME_PATH   — mount point of the network volume (default: /runpod-volume)
#   OLLAMA_PORT   — port for the temporary ollama server  (default: 11434)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
VOLUME_PATH="${VOLUME_PATH:-/runpod-volume}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_ADDR="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_MODELS="${VOLUME_PATH}/ollama/models"   # must match Dockerfile ENV
export OLLAMA_HOST="${OLLAMA_ADDR}"

MODELS=(
    "qwen3:8b-q4_K_M"
    "qwen3-vision:8b-q8_0"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] ✅ $*"; }
err()  { echo "[$(date '+%H:%M:%S')] ❌ $*" >&2; exit 1; }
hr()   { echo "────────────────────────────────────────"; }

# ── 1. Volume check ───────────────────────────────────────────────────────────
hr
log "Checking volume at ${VOLUME_PATH}"
mountpoint -q "${VOLUME_PATH}" \
    || df -h | grep -q "${VOLUME_PATH}" \
    || err "Volume not mounted at ${VOLUME_PATH}. Attach your network volume first."

AVAIL=$(df -h "${VOLUME_PATH}" | awk 'NR==2 {print $4}')
log "Volume available space: ${AVAIL}"

mkdir -p "${OLLAMA_MODELS}"
ok "Model dir ready: ${OLLAMA_MODELS}"

# ── 2. Ollama binary ──────────────────────────────────────────────────────────
hr
if command -v ollama &>/dev/null; then
    log "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
    log "Ollama not found — installing …"
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed: $(ollama --version 2>/dev/null)"
fi

# ── 3. Start Ollama server ────────────────────────────────────────────────────
hr
log "Starting Ollama server on ${OLLAMA_ADDR} …"

# Kill any stale Ollama process on this port before starting
if lsof -ti tcp:"${OLLAMA_PORT}" &>/dev/null; then
    log "Port ${OLLAMA_PORT} in use — stopping existing process …"
    lsof -ti tcp:"${OLLAMA_PORT}" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

OLLAMA_HOST="${OLLAMA_ADDR}" \
OLLAMA_MODELS="${OLLAMA_MODELS}" \
OLLAMA_NOPRUNE=1 \
    ollama serve >"/tmp/ollama_populate.log" 2>&1 &
OLLAMA_PID=$!

# ── 4. Wait for API readiness ─────────────────────────────────────────────────
hr
log "Waiting for Ollama API …"
MAX_WAIT=90
WAITED=0
until curl -sf "http://${OLLAMA_ADDR}/api/tags" >/dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
        log "--- Ollama log ---"
        cat /tmp/ollama_populate.log || true
        err "Ollama API did not become ready after ${MAX_WAIT}s."
    fi
    log "  waiting … ${WAITED}s / ${MAX_WAIT}s"
done
ok "Ollama API ready (${WAITED}s)"

# ── 5. Pull models ────────────────────────────────────────────────────────────
hr
PULL_ERRORS=0
for MODEL in "${MODELS[@]}"; do
    log "Pulling ${MODEL} …"
    if OLLAMA_HOST="${OLLAMA_ADDR}" OLLAMA_MODELS="${OLLAMA_MODELS}" \
            ollama pull "${MODEL}"; then
        ok "Pulled ${MODEL}"
    else
        echo "⚠️  Failed to pull ${MODEL}" >&2
        PULL_ERRORS=$((PULL_ERRORS + 1))
    fi
    hr
done

# ── 6. Verify ─────────────────────────────────────────────────────────────────
log "Registered models:"
OLLAMA_HOST="${OLLAMA_ADDR}" ollama list

log "Manifest tree:"
find "${OLLAMA_MODELS}/manifests" -type f 2>/dev/null \
    | sort \
    | sed "s|${OLLAMA_MODELS}/manifests/||" \
    || log "  (no manifests found)"

log "Blobs on disk:"
du -sh "${OLLAMA_MODELS}/blobs" 2>/dev/null || log "  (no blobs dir yet)"

log "Volume usage summary:"
df -h "${VOLUME_PATH}"

# ── 7. Shutdown ───────────────────────────────────────────────────────────────
hr
log "Stopping Ollama server (PID ${OLLAMA_PID}) …"
kill "${OLLAMA_PID}" 2>/dev/null || true
wait "${OLLAMA_PID}" 2>/dev/null || true

# ── 8. Final result ───────────────────────────────────────────────────────────
hr
if [[ ${PULL_ERRORS} -gt 0 ]]; then
    err "${PULL_ERRORS} model(s) failed to pull. Check output above."
fi
ok "All models installed at ${OLLAMA_MODELS}"
ok "You can now deploy your RunPod serverless worker."
