import { runExperiment } from "./openai.js";

async function main() {
  const prompt = "complete this statement: 'after the rain comes the'";
  console.log("Experiment:", prompt);
  const temp1_8_results = await runExperiment(prompt, { temperature: 1.8 });
  console.log("Temperature 1.8 results:", temp1_8_results);

  const temp0_results = await runExperiment(prompt, { temperature: 0 });
  console.log("Temperature 0 results:", temp0_results);
}

main();
