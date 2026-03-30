export default async function handler(req, res) {
  try {
    const response = await fetch(
      `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/runsync`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.RUNPOD_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: req.body.input,
        }),
      }
    );

    const data = await response.json();
    res.status(200).json(data);

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
