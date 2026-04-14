export const config = {
  api: {
    bodyParser: {
      sizeLimit: '20mb',      
    },
    responseLimit: false,
  },
  maxDuration: 300,        
};

const HF_URL = process.env.HF_API_URL;

export default async function handler(req, res) {
  const { action } = req.body;

  if (action === "process_page" || action === "flan-text") {
    // Per-action timeout: OCR pages need 4 min, analysis needs 15 min
    const timeoutMs = action === "flan-text" ? 270_000 : 240_000;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${HF_URL}/api/ai`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
        signal: controller.signal,   
      });
      clearTimeout(timer);
      if (!response.ok) {
        const text = await response.text();
        return res.status(response.status).json({ error: text });
      }
      return res.status(200).json(await response.json());
    } catch (err) {
      clearTimeout(timer);
      const isTimeout = err.name === "AbortError";
      return res.status(isTimeout ? 504 : 500).json({
        error: isTimeout ? "Backend timed out — try again" : err.message,
      });
    }
  }

  return res.status(400).json({
    error: "Unknown action — provide action: process_page or flan-text",
  });
}
