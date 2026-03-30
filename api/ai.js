export default async function handler(req, res) {
  try {
    // ✅ Safe body handling
    let body = req.body;
    if (typeof body === "string") {
      try {
        body = JSON.parse(body);
      } catch (e) {
        return res.status(400).json({ error: "Invalid JSON from frontend", raw: body });
      }
    }

    console.log("BODY:", body);

    // ✅ Call RunPod
    const response = await fetch(
      `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/runsync`,
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

    // ✅ SAFE response parsing (THIS FIXES YOUR ERROR)
    const text = await response.text();
    console.log("RUNPOD RAW:", text);

    try {
      const data = JSON.parse(text);
      return res.status(200).json(data);
    } catch (e) {
      return res.status(500).json({
        error: "RunPod returned non-JSON",
        raw: text,
      });
    }

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
