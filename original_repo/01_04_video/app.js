/**
 * Video Processing Agent (Interactive)
 */

import { createInterface } from "node:readline/promises";
import { createMcpClient, listMcpTools, closeMcpClient } from "./src/mcp/client.js";
import { nativeTools } from "./src/native/tools.js";
import { createReadline, runRepl } from "./src/repl.js";
import { onShutdown } from "./src/helpers/shutdown.js";
import { logStats } from "./src/helpers/stats.js";
import log from "./src/helpers/logger.js";

const EXAMPLE_QUERY = "List 4 big claims breakdown from this video https://www.youtube.com/watch?v=Iar4yweKGoI";

const confirmRun = async () => {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.log("\n⚠️  UWAGA: Uruchomienie tego agenta może zużyć zauważalną liczbę tokenów i przetworzyć wideo.");
  console.log("   Jeśli nie chcesz uruchamiać go teraz, najpierw sprawdź plik demo:");
  console.log("   Demo: workspace/demo/breakdown.md");
  console.log("");

  const answer = await rl.question("Czy chcesz kontynuować? (yes/y): ");
  rl.close();

  const normalized = answer.trim().toLowerCase();
  if (normalized !== "yes" && normalized !== "y") {
    console.log("Przerwano.");
    process.exit(0);
  }
};

const main = async () => {
  log.box("Video Processing Agent\nType 'exit' to quit, 'clear' to reset");
  await confirmRun();

  let mcpClient = null;
  let rl = null;

  try {
    log.start("Connecting to MCP server...");
    mcpClient = await createMcpClient();
    
    const mcpTools = await listMcpTools(mcpClient);
    log.success(`MCP tools: ${mcpTools.map(t => t.name).join(", ")}`);
    log.info(`Native tools: ${nativeTools.map(t => t.name).join(", ")}`);
    console.log(`\n  Example: ${EXAMPLE_QUERY}\n`);

    rl = createReadline();

    const shutdown = onShutdown(async () => {
      logStats();
      rl?.close();
      if (mcpClient) await closeMcpClient(mcpClient);
    });

    await runRepl({ mcpClient, mcpTools, rl });
    await shutdown();

  } catch (error) {
    log.error("Error", error.message);
    rl?.close();
    if (mcpClient) await closeMcpClient(mcpClient);
  }
};

main().catch((err) => {
  log.error("Startup error", err.message);
  process.exit(1);
});
