/**
 * File & Email Agent (Interactive)
 */

import { createMcpClient, listMcpTools, closeMcpClient } from "./src/mcp/client.js";
import { nativeTools } from "./src/native/tools.js";
import { createReadline, runRepl } from "./src/repl.js";
import { onShutdown } from "./src/helpers/shutdown.js";
import { logStats } from "./src/helpers/stats.js";
import log from "./src/helpers/logger.js";

const EXAMPLES = [
  'List all files in the workspace',
  'Read workspace/whitelist.json and show me its contents',
  'Write "Hello from the agent!" to workspace/output/hello.txt',
  'Send an email to alice@aidevs.pl with subject "Hello" and a short greeting',
  'Search for any markdown files in the workspace',
];

const main = async () => {
  log.box("File & Email Agent\nCommands: 'exit' | 'clear' | 'untrust'");

  let mcpClient = null;
  let rl = null;

  try {
    log.start("Connecting to MCP server...");
    mcpClient = await createMcpClient();
    
    const mcpTools = await listMcpTools(mcpClient);
    log.success(`MCP tools: ${mcpTools.map(t => t.name).join(", ")}`);
    log.info(`Native tools: ${nativeTools.map(t => t.name).join(", ")}`);

    console.log("");
    log.info("Example queries:");
    for (const example of EXAMPLES) {
      log.info(`  • ${example}`);
    }
    console.log("");

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
