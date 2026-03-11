/**
 * MCP Upload Agent — scans workspace files and uploads them via MCP servers.
 *
 * Demonstrates connecting to multiple MCP servers simultaneously:
 *  - files (stdio):       local filesystem via files-mcp
 *  - uploadthing (http):  remote upload service via StreamableHTTP
 *
 * Tool names are prefixed with the server name (e.g. files__fs_read,
 * uploadthing__upload_files) so the agent loop can route calls to
 * the correct server.
 */

import {
  createAllMcpClients,
  listAllMcpTools,
  closeAllClients,
  ConfigurationError
} from "./src/mcp/client.js";
import { run } from "./src/agent.js";
import log from "./src/helpers/logger.js";
import { logStats } from "./src/helpers/stats.js";
import { resolveModelForProvider } from "../config.js";

const model = resolveModelForProvider("gpt-5.4");
const maxOutputTokens = 16384;
const instructions = `You are a file upload assistant.

Use the {{file:path}} placeholder for the base64 field when uploading — the system resolves it automatically.

Example: { "files": [{ "base64": "{{file:example.md}}", "name": "example.md", "type": "text/markdown" }] }

Workflow:
1. fs_read with mode:"list" to see workspace files
2. Upload each file not already in uploaded.md using {{file:path}} syntax
3. Update uploaded.md with a table of filename, URL, and timestamp

Rules:
- Never read or encode file content yourself — always use {{file:path}}
- Skip uploaded.md itself and files already listed in it
- Handle errors gracefully

When done, say "Upload complete: X files uploaded, Y skipped."`;

const main = async () => {
  log.box("MCP Upload Agent\nUpload workspace files via uploadthing");

  let mcpClients = null;

  try {
    // Connect to all servers defined in mcp.json (files via stdio, uploadthing via HTTP)
    log.start("Connecting to MCP servers...");
    mcpClients = await createAllMcpClients();
    const mcpTools = await listAllMcpTools(mcpClients);
    log.success(`Connected with ${mcpTools.length} tools from ${Object.keys(mcpClients).length} servers`);

    // Run the upload task — the agent lists files, uploads, and updates uploaded.md
    log.start("Starting upload task...");
    const conversation = [
      { role: "user", content: "Check the workspace for files, upload any that haven't been uploaded yet, and update uploaded.md with the results." }
    ];
    await run(conversation, { mcpClients, mcpTools, model, instructions, maxOutputTokens });
    logStats();

  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log.error(message);

    if (!(error instanceof ConfigurationError)) {
      console.error(error);
    }

    process.exitCode = 1;
  } finally {
    if (mcpClients) {
      log.start("Closing connections...");
      await closeAllClients(mcpClients);
    }
  }
};

main().catch((error) => {
  log.error("Startup error", error.message);
  process.exit(1);
});
