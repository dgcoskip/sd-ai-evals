import OpenAI from "openai";
import "dotenv/config";

const openai = new OpenAI();

export async function getCompletion(prompt, options) {
  try {
    const model = options.model || "gpt-4.1-mini";
    const response = await openai.chat.completions.create({
      model: model,
      messages: [{ role: "user", content: prompt }],
      ...options,
    });
    let result = response.choices[0].message.content;
    if (options.seed) {
      result += ` (system_fingerprint: ${response.system_fingerprint})`;
    }
    return result;
  } catch (error) {
    console.error("Error getting completion:", error);
    return null;
  }
}

export async function runExperiment(prompt, options, runs = 10) {
  const results = [];
  for (let i = 0; i < runs; i++) {
    const result = await getCompletion(prompt, options);
    results.push(result);
  }
  return results;
}
