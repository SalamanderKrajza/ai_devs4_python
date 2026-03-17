/**
 * Image Recognition Agent
 */

import { createMcpClient, listMcpTools } from "./src/mcp/client.js";
import { run } from "./src/agent.js";
import { nativeTools } from "./src/native/tools.js";
import log from "./src/helpers/logger.js";
import { logStats } from "./src/helpers/stats.js";

const CLASSIFICATION_QUERY = `Classify all images in the images/ folder based on the character knowledge files.
Read the knowledge files first, then analyze each image and copy it to the appropriate character folder(s).`;

const main = async () => {
  log.box("Image Recognition Agent\nClassify images by character");

  let mcpClient;

  try {
    log.start("Connecting to MCP server...");
    mcpClient = await createMcpClient();
    const mcpTools = await listMcpTools(mcpClient);
    log.success(`MCP: ${mcpTools.map((tool) => tool.name).join(", ")}`);
    log.success(`Native: ${nativeTools.map((tool) => tool.name).join(", ")}`);

    log.start("Starting image classification...");
    const result = await run(CLASSIFICATION_QUERY, { mcpClient, mcpTools });
    log.success("Classification complete");
    log.info(result.response);
    logStats();
  } catch (error) {
    throw error;
  } finally {
    if (mcpClient) {
      await mcpClient.close().catch(() => {});
    }
  }
};

main().catch((error) => {
  log.error("Startup error", error.message);
  process.exit(1);
});
