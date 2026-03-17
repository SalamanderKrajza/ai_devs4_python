/**
 * Agentic RAG (Interactive)
 */

import { createInterface } from "node:readline/promises";
import { createMcpClient, listMcpTools, closeMcpClient } from "./src/mcp/client.js";
import { createReadline, runRepl } from "./src/repl.js";
import { onShutdown } from "./src/helpers/shutdown.js";
import { logStats } from "./src/helpers/stats.js";
import log from "./src/helpers/logger.js";

const DEMO_FILE = "demo/example.md";

const confirmRun = async () => {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.log("\n⚠️  UWAGA: Uruchomienie tego agenta może zużyć zauważalną liczbę tokenów.");
  console.log("   Jeśli nie chcesz uruchamiać go teraz, najpierw sprawdź plik demo:");
  console.log(`   Demo: ${DEMO_FILE}`);
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
  log.box("Agentic RAG\nCommands: 'exit' | 'clear'");
  await confirmRun();

  let mcpClient = null;
  let rl = null;

  try {
    log.start("Connecting to MCP server...");
    mcpClient = await createMcpClient();
    
    const mcpTools = await listMcpTools(mcpClient);
    log.success(`MCP tools: ${mcpTools.map(t => t.name).join(", ")}\n`);

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
