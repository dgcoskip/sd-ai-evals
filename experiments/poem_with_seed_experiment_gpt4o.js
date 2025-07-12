import { runExperiment } from "./openai.js";

async function main() {
  const prompt = "write me any one line poem";
  console.log("Experiment:", prompt, "Model: gpt-4.1-mini-2025-04-14");

  const temp0_results = await runExperiment(prompt, { model: "gpt-4.1-mini-2025-04-14", temperature: 0 }, 10);
  console.log("Temperature 0 results:", temp0_results);

  const temp0_seed_results = await runExperiment(prompt, { model: "gpt-4.1-mini-2025-04-14", temperature: 0, seed: 12345 }, 10);
  console.log("Temperature 0 with seed results:", temp0_seed_results);
}

main();
