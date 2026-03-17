/**
 * JSON Image Agent (Interactive)
 */

import { createMcpClient, listMcpTools, closeMcpClient } from "./src/mcp/client.js";
import { nativeTools } from "./src/native/tools.js";
import { createReadline, runRepl } from "./src/repl.js";
import { onShutdown } from "./src/helpers/shutdown.js";
import { logStats } from "./src/helpers/stats.js";
import log from "./src/helpers/logger.js";

const main = async () => {
  log.box("JSON Image Agent\nType 'exit' to quit, 'clear' to reset");

  let mcpClient = null;
  let rl = null;

  try {
    log.start("Connecting to MCP server...");
    mcpClient = await createMcpClient();
    
    const mcpTools = await listMcpTools(mcpClient);
    log.success(`MCP tools: ${mcpTools.map(t => t.name).join(", ")}`);
    log.info(`Native tools: ${nativeTools.map(t => t.name).join(", ")}\n`);

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
