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

    return res.status(400).json({
        error: "Unknown request — provide action, image(s), or prompt",
    });
}
