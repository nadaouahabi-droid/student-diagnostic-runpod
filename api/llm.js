export default async function handler(req, res) {
  try {
    let body = req.body;
    if (typeof body === "string") {
      body = JSON.parse(body);
    }

    const LLM_ENDPOINT = process.env.LLM_ENDPOINT;
    const API_KEY = process.env.RUNPOD_API_KEY;

    const response = await fetch(LLM_ENDPOINT, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: {
          action: "ollama-chat",
          messages: [
            { role: "user", content: body.text }
          ]
        },
      }),
    });

    const data = await response.json();

    return res.status(200).json({
      step: "llm_started",
      jobId: data.id
    });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
