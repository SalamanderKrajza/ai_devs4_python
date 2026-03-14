/**
 * Image Guidance Agent (Interactive)
 */

import { createInterface } from "node:readline/promises";
import { createMcpClient, listMcpTools } from "./src/mcp/client.js";
import { nativeTools } from "./src/native/tools.js";
import { createReadline, runRepl } from "./src/repl.js";
import { onShutdown } from "./src/helpers/shutdown.js";
import { logStats } from "./src/helpers/stats.js";
import log from "./src/helpers/logger.js";

const EXAMPLES = [
  "Create a female magician in a walking pose",
  "Generate a running knight with a red cape",
  "Create a friendly robot explorer using the walking pose reference",
  "Analyze the latest generated image for pose consistency and style quality"
];

const DEMO_FILES = [
  "workspace/demo/walking-pose-input.png",
  "workspace/demo/walking-pose-final.jpg",
  "workspace/demo/running-pose-input.png",
  "workspace/demo/running-pose-final.jpg"
];

const confirmRun = async () => {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.log("\n⚠️  UWAGA: Uruchomienie tego agenta może zużyć zauważalną liczbę tokenów i wygenerować obrazy.");
  console.log("   Jeśli nie chcesz uruchamiać go teraz, zajrzyj najpierw do folderu workspace/demo/:");
  DEMO_FILES.forEach((path) => console.log(`   ${path}`));
  console.log("");

  const answer = await rl.question("Czy chcesz kontynuować? (yes/y): ");
  rl.close();

  const normalized = answer.trim().toLowerCase();
  if (normalized !== "yes" && normalized !== "y") {
    console.log("Przerwano.");
    process.exit(0);
  }
};

const printTools = () => {
  log.heading("TOOLS");

  for (const tool of nativeTools) {
    log.info(`${tool.name.padEnd(14)} — ${tool.description.split(".")[0]}`);
  }
};

const printExamples = () => {
  log.heading("EXAMPLES", "For demo purposes, try these queries:");
  EXAMPLES.forEach((example) => log.example(example));
  log.hint("Type 'exit' to quit, 'clear' to reset conversation");
};

const main = async () => {
  log.box("Image Guidance Agent");
  await confirmRun();
  printTools();

  let mcpClient;
  let rl;

  try {
    log.start("Connecting to MCP server...");
    mcpClient = await createMcpClient();
    const mcpTools = await listMcpTools(mcpClient);
    log.success(`MCP: ${mcpTools.map((tool) => tool.name).join(", ")}`);

    printExamples();

    rl = createReadline();
    const shutdown = onShutdown(async () => {
      logStats();
      rl?.close();
      if (mcpClient) await mcpClient.close();
    });

    await runRepl({ mcpClient, mcpTools, rl });
    await shutdown();
  } catch (error) {
    rl?.close();
    if (mcpClient) {
      await mcpClient.close().catch(() => {});
    }
    throw error;
  }
};

main().catch((err) => {
  log.error("Startup error", err.message);
  process.exit(1);
});
