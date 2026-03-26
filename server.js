/**
 * Student Diagnostic System — Secure Backend Proxy
 * =================================================
 * The RunPod API key lives HERE, in process.env, loaded from .env at startup.
 * The browser only ever calls /api/ai and /api/health on this same-origin server.
 * The key is never sent to or stored in the browser.
 *
 * Usage:
 *   npm install express dotenv express-rate-limit
 *   node server.js
 *
 * Environment variables (put in .env — never commit that file):
 *   RUNPOD_ENDPOINT_URL   Full RunPod /runsync URL
 *   RUNPOD_API_KEY        Your RunPod API key
 *   PORT                  (optional) Defaults to 3000
 *   ALLOWED_ORIGIN        (optional) CORS origin, e.g. https://yourapp.example.com
 *                         Defaults to same-origin only (no CORS header = browser blocks cross-origin)
 */

'use strict';

require('dotenv').config();       // loads .env into process.env
const express    = require('express');
const rateLimit  = require('express-rate-limit');

// ── Validate required env vars at boot ──────────────────────────────────────
const RUNPOD_ENDPOINT_URL = process.env.RUNPOD_ENDPOINT_URL;
const RUNPOD_API_KEY      = process.env.RUNPOD_API_KEY;

if (!RUNPOD_ENDPOINT_URL || !RUNPOD_API_KEY) {
  console.error(
    '\n[FATAL] Missing environment variables.\n' +
    'Create a .env file with:\n' +
    '  RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync\n' +
    '  RUNPOD_API_KEY=your_runpod_key_here\n'
  );
  process.exit(1);
}

const PORT           = parseInt(process.env.PORT || '3000', 10);
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || null;   // null = same-origin only

const app = express();

// ── Security headers ─────────────────────────────────────────────────────────
app.use((req, res, next) => {
  // Prevent browsers from inferring content type (XSS mitigation)
  res.setHeader('X-Content-Type-Options', 'nosniff');
  // Prevent this page from being iframed (clickjacking)
  res.setHeader('X-Frame-Options', 'DENY');
  // Basic CSP — tighten this for production
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; img-src 'self' data: blob:; font-src https://fonts.gstatic.com; connect-src 'self'");
  next();
});

// ── CORS (only if ALLOWED_ORIGIN is set) ─────────────────────────────────────
if (ALLOWED_ORIGIN) {
  app.use((req, res, next) => {
    if (req.headers.origin === ALLOWED_ORIGIN) {
      res.setHeader('Access-Control-Allow-Origin', ALLOWED_ORIGIN);
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    }
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
  });
}

// ── Body parsing — cap at 50 MB to handle base64 exam images ─────────────────
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));   // serve student-diagnostic.html from /public

// ── Rate limiting — prevents API-key abuse if the proxy is somehow exposed ───
const apiLimiter = rateLimit({
  windowMs : 60 * 1000,    // 1 minute window
  max      : 30,           // max 30 requests per minute per IP
  message  : { error: 'Too many requests — please slow down.' },
  standardHeaders: true,
  legacyHeaders  : false,
});
app.use('/api/', apiLimiter);

// ── Helper: forward a request body to RunPod with the server-side key ────────
async function forwardToRunPod(bodyObj, timeoutMs = 600_000) {
  // node 18+ ships with fetch() built-in; use node-fetch for older node versions
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(RUNPOD_ENDPOINT_URL, {
      method : 'POST',
      headers: {
        'Content-Type' : 'application/json',
        // ← The API key is attached HERE, server-side only.
        //   The browser never sees this header.
        'Authorization': 'Bearer ' + RUNPOD_API_KEY,
      },
      body  : JSON.stringify(bodyObj),
      signal: controller.signal,
    });
    clearTimeout(timer);
    return response;
  } catch (err) {
    clearTimeout(timer);
    throw err;
  }
}

// ── POST /api/ai — main inference proxy ──────────────────────────────────────
// The browser sends: { input: { action, model, messages, options, ... } }
// This server adds the Authorization header and forwards to RunPod.
app.post('/api/ai', async (req, res) => {
  try {
    if (!req.body || !req.body.input) {
      return res.status(400).json({ error: 'Missing "input" field in request body.' });
    }

    const upstream = await forwardToRunPod(req.body);

    // Stream the RunPod response status + body back to the browser
    res.status(upstream.status);
    const text = await upstream.text();

    // Validate it's JSON before forwarding (avoids leaking RunPod error HTML)
    try {
      const parsed = JSON.parse(text);
      return res.json(parsed);
    } catch {
      // RunPod returned non-JSON (e.g. plain error message)
      return res.status(502).json({ error: 'Upstream returned unexpected format.', detail: text.slice(0, 200) });
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      return res.status(504).json({ error: 'RunPod request timed out.' });
    }
    console.error('[/api/ai error]', err.message);
    return res.status(500).json({ error: 'Proxy error: ' + err.message });
  }
});

// ── POST /api/health — connection test ───────────────────────────────────────
app.post('/api/health', async (req, res) => {
  try {
    const upstream = await forwardToRunPod({ input: { action: 'health' } }, 30_000);
    const text = await upstream.text();
    try {
      return res.status(upstream.status).json(JSON.parse(text));
    } catch {
      return res.status(502).json({ error: 'Upstream non-JSON response', detail: text.slice(0, 200) });
    }
  } catch (err) {
    console.error('[/api/health error]', err.message);
    return res.status(500).json({ error: err.message });
  }
});

// ── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n✓ Student Diagnostic proxy running on http://localhost:${PORT}`);
  console.log(`  RunPod endpoint : ${RUNPOD_ENDPOINT_URL}`);
  console.log(`  API key         : ****${RUNPOD_API_KEY.slice(-4)}  (last 4 chars shown)\n`);
});
