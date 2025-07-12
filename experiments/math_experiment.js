import { runExperiment } from "./openai.js";

async function main() {
  console.log("Experiment 1: 2+2");
  const temp1_8_results_1 = await runExperiment("2+2", { temperature: 1.8 });
  console.log("Temperature 1.8 results:", temp1_8_results_1);

  const temp0_results_1 = await runExperiment("2+2", { temperature: 0 });
  console.log("Temperature 0 results:", temp0_results_1);
}

main();
