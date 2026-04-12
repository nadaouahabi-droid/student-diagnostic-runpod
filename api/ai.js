const HF_URL = process.env.HF_API_URL;

export default async function handler(req, res) {
    const { action, image, images, prompt, system } = req.body;

    if (action === "process_page" || action === "flan-text") {
        try {
            const response = await fetch(`${HF_URL}/api/ai`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(req.body),
            });
            if (!response.ok) {
                const text = await response.text();
                return res.status(response.status).json({ error: text });
            }
            return res.status(200).json(await response.json());
        } catch (err) {
            return res.status(500).json({ error: err.message });
        }
    }

    if (images && images.length > 0) {
        try {
            const results = [];
            for (let i = 0; i < images.length; i++) {
                const response = await fetch(`${HF_URL}/api/ai`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        action:   "process_page",
                        image:    images[i],
                        page_num: i + 1,
                    }),
                });
                if (!response.ok) {
                    const text = await response.text();
                    results.push({ error: text, page_num: i + 1 });
                    continue;
                }
                results.push(await response.json());
            }
            return res.status(200).json({ results });
        } catch (err) {
            return res.status(500).json({ error: err.message });
        }
    }

    if (image) {
        try {
            const response = await fetch(`${HF_URL}/api/ai`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    action:   "process_page",
                    image,
                    page_num: 1,
                }),
            });
            if (!response.ok) {
                const text = await response.text();
                return res.status(response.status).json({ error: text });
            }
            return res.status(200).json(await response.json());
        } catch (err) {
            return res.status(500).json({ error: err.message });
        }
    }

    if (prompt) {
        try {
            const response = await fetch(`${HF_URL}/api/ai`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    action: "flan-text",
                    prompt,
                    system: system || "",
                }),
            });
            if (!response.ok) {
                const text = await response.text();
                return res.status(response.status).json({ error: text });
            }
            return res.status(200).json(await response.json());
        } catch (err) {
            return res.status(500).json({ error: err.message });
        }
    }

    return res.status(400).json({
        error: "Unknown request — provide action, image(s), or prompt",
    });
}
