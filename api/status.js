export default async function handler(req, res) {
  const { jobId } = req.query;

  try {
    const response = await fetch(
      `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/status/${jobId}`,
      {
        headers: {
          "Authorization": `Bearer ${process.env.RUNPOD_API_KEY}`,
        },
      }
    );

    const data = await response.json();
    res.status(200).json(data);

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
