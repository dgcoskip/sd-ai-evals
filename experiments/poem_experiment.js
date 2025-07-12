import { runExperiment } from "./openai.js";

async function main() {
  const prompt = "write me any one line poem";
  console.log("Experiment:", prompt);
  const temp1_8_results_2 = await runExperiment(prompt, { temperature: 1.8 });
  console.log("Temperature 1.8 results:", temp1_8_results_2);

  const temp0_results_2 = await runExperiment(prompt, { temperature: 0 });
  console.log("Temperature 0 results:", temp0_results_2);
}

main();
