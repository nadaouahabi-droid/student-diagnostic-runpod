export default async function handler(req, res) {
  try {
    let body = req.body;
    if (typeof body === "string") {
      body = JSON.parse(body);
    }

    const OCR_ENDPOINT = process.env.OCR_ENDPOINT;
    const API_KEY = process.env.RUNPOD_API_KEY;

    const response = await fetch(OCR_ENDPOINT, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: {
          action: "ocr-batch",
          images: body.images
        },
      }),
    });

    const data = await response.json();

    return res.status(200).json({
      step: "ocr_started",
      jobId: data.id
    });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
