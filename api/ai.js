export default async function handler(req, res) {
  try {
    let body = req.body;
    if (typeof body === "string") {
      body = JSON.parse(body);
    }

    const response = await fetch(
      `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/run`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.RUNPOD_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: body.input || body,
        }),
      }
    );

    const data = await response.json();

    return res.status(200).json({
      jobId: data.id
    });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
