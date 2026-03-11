/**
 * Multi-server MCP client — connects to all servers in mcp.json.
 *
 * Supports two transport types:
 *  - stdio: spawns a local process (e.g. files-mcp)
 *  - http:  connects to a remote StreamableHTTP endpoint (e.g. uploadthing)
 *
 * Tool names are prefixed with the server name (e.g. "files__fs_read")
 * so the agent can route calls to the correct server via callMcpTool.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import {
  loadMcpConfig,
  validateMcpConfig,
  validateHttpServerConfig,
  PROJECT_ROOT,
  ConfigurationError
} from "./config.js";
import log from "../helpers/logger.js";

export { ConfigurationError };

const createStdioTransport = (serverConfig) => {
  log.info(`Spawning stdio server: ${serverConfig.command} ${serverConfig.args.join(" ")}`);

  return new StdioClientTransport({
    command: serverConfig.command,
    args: serverConfig.args,
    env: {
      PATH: process.env.PATH,
      HOME: process.env.HOME,
      NODE_ENV: process.env.NODE_ENV,
      ...serverConfig.env
    },
    cwd: PROJECT_ROOT,
    stderr: "inherit"
  });
};

const createHttpTransport = (serverName, serverConfig) => {
  validateHttpServerConfig(serverName, serverConfig);
  log.info(`Connecting to HTTP server: ${serverConfig.url}`);

  return new StreamableHTTPClientTransport(new URL(serverConfig.url));
};

const createMcpClient = async (serverName, config) => {
  const serverConfig = config.mcpServers[serverName];

  if (!serverConfig) {
    throw new Error(`MCP server "${serverName}" not found in mcp.json`);
  }

  const client = new Client(
    { name: "mcp-upload-client", version: "1.0.0" },
    { capabilities: {} }
  );

  const transport = serverConfig.transport === "http"
    ? createHttpTransport(serverName, serverConfig)
    : createStdioTransport(serverConfig);

  await client.connect(transport);
  log.success(`Connected to ${serverName} via ${serverConfig.transport || "stdio"}`);

  return client;
};

export const createAllMcpClients = async () => {
  const config = await loadMcpConfig();
  const clients = {};

  validateMcpConfig(config);

  try {
    for (const serverName of Object.keys(config.mcpServers)) {
      clients[serverName] = await createMcpClient(serverName, config);
    }
  } catch (error) {
    await closeAllClients(clients);
    throw error;
  }

  return clients;
};

// Some servers return non-standard outputSchema that fails SDK validation.
// Falls back to a raw request without strict Zod parsing.
const listMcpTools = async (client) => {
  try {
    const result = await client.listTools();
    return result.tools;
  } catch (error) {
    if (error.name === "$ZodError" || error.message?.includes("outputSchema")) {
      log.warn(`Tool listing validation failed, using raw request`);
      const result = await client.request(
        { method: "tools/list", params: {} },
        { parse: (data) => data }
      );
      return result.tools || [];
    }
    throw error;
  }
};

// Collects tools from all servers, prefixing names with the server name
// so the agent can route calls back to the right server.
// Logs the discovered tools grouped by server.
export const listAllMcpTools = async (clients) => {
  const allTools = [];

  for (const [serverName, client] of Object.entries(clients)) {
    const tools = await listMcpTools(client);
    log.info(`  ${serverName}: ${tools.map((t) => t.name).join(", ")}`);

    for (const tool of tools) {
      allTools.push({
        ...tool,
        _server: serverName,
        _originalName: tool.name,
        name: `${serverName}__${tool.name}`
      });
    }
  }

  return allTools;
};

// Routes a prefixed tool call (e.g. "files__fs_read") to the right server
export const callMcpTool = async (clients, name, args) => {
  const [serverName, ...toolParts] = name.split("__");
  const toolName = toolParts.join("__");

  const client = clients[serverName];
  if (!client) {
    throw new Error(`Unknown MCP server: ${serverName}`);
  }

  const result = await client.callTool({ name: toolName, arguments: args });

  const textContent = result.content.find((c) => c.type === "text");
  if (textContent) {
    try { return JSON.parse(textContent.text); }
    catch { return textContent.text; }
  }
  return result;
};

export const mcpToolsToOpenAI = (mcpTools) =>
  mcpTools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
    strict: false
  }));

export const closeAllClients = async (clients) => {
  for (const [name, client] of Object.entries(clients)) {
    try {
      await client.close();
      log.info(`Closed ${name} client`);
    } catch (e) {
      log.warn(`Error closing ${name}: ${e.message}`);
    }
  }
};
