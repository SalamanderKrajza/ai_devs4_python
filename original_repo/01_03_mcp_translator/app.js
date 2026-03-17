/**
 * MCP Translator Agent
 *
 * Connects to files-mcp, starts a file-watching translation loop,
 * and exposes HTTP endpoints for on-demand translation.
 */

import { createMcpClient, listMcpTools } from "./src/mcp/client.js";
import { runTranslationLoop } from "./src/translator.js";
import { startHttpServer } from "./src/server.js";
import { server as serverConfig } from "./src/config.js";
import log from "./src/helpers/logger.js";

let mcpClient = null;
let mcpTools = [];

const main = async () => {
  log.box("MCP Translator Agent\nAccurate translations to English with tone, formatting & nuances");

  // Connect to files-mcp (stdio transport, config in mcp.json)
  log.start("Connecting to MCP server...");
  mcpClient = await createMcpClient();
  mcpTools = await listMcpTools(mcpClient);
  log.success(`Connected with ${mcpTools.length} tools: ${mcpTools.map(t => t.name).join(", ")}`);

  // Watch workspace/translate/ for new files
  runTranslationLoop(mcpClient, mcpTools);

  // HTTP API for on-demand translation
  const server = startHttpServer(serverConfig, () => ({ mcpClient, mcpTools }));

  const shutdown = async () => {
    log.warn("Shutting down...");
    if (mcpClient) await mcpClient.close();
    server.close();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
};

main().catch((error) => {
  log.error("Startup error", error.message);
  process.exit(1);
});
