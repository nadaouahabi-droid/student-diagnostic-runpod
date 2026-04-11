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
        endpoint = `${HF_URL}/api/ai`;

        // process each page using the GOOD pipeline
        const results = [];

        for (let i = 0; i < images.length; i++) {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    action: "process_page",
                    image: images[i],
                    page_num: i + 1
                }),
            });

            const data = await response.json();
            results.push(data);
        }

        return res.status(200).json({ results });

    } else if (image) {
        endpoint = `${HF_URL}/api/ai`;
        body = {
            action: "process_page",
            image,
            page_num: 1
        };

    } else if (prompt) {
        endpoint = `${HF_URL}/analyze`;
        body = { prompt };

    } else {
        return res.status(400).json({
            error: "Unknown request — no action, image, or prompt"
        });
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
        return res.status(200).json(data);

    } catch (err) {
        return res.status(500).json({ error: err.message });
    }
}
