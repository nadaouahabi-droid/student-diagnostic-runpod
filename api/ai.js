const HF_URL = process.env.HF_API_URL;

export default async function handler(req, res) {
    const { action, image, images, prompt, system } = req.body;

    let endpoint;
    let body;

    if (action === "paddleocr") {
        endpoint = `${HF_URL}/api/ai`;
        body = req.body;                 

    } else if (action === "trocr") {
        endpoint = `${HF_URL}/api/ai`;
        body = req.body;

    } else if (action === "flan-text") {
        endpoint = `${HF_URL}/api/ai`;
        body = req.body;

    } else if (images) {
        endpoint = `${HF_URL}/ocr-batch`;      // batch image upload
        body = { images };

    } else if (image) {
        endpoint = `${HF_URL}/ocr`;            // single image upload
        body = { image, filetype: req.body.filetype || "image" };

    } else if (prompt) {
        endpoint = `${HF_URL}/analyze`;        // direct text prompt
        body = { prompt };

    } else {
        return res.status(400).json({ error: "Unknown request — no action, image, or prompt" });
    }

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const text = await response.text();
            return res.status(response.status).json({ error: text });
        }

        const data = await response.json();
        res.status(200).json(data);

    } catch (err) {
        res.status(500).json({ error: err.message });
    }
}
