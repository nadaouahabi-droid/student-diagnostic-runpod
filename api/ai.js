const HF_URL = process.env.HF_API_URL;

export default async function handler(req, res) {
    const response = await fetch(`${HF_URL}/ocr`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(req.body)
    });

    const data = await response.json();
    res.status(200).json(data);
}
