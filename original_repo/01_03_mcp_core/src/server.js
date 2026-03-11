#!/usr/bin/env node
/**
 * MCP Server — registers tools, resources, and prompts, then listens over stdio.
 *
 * In MCP, the server is a capability provider. It exposes:
 *  - tools:      actions the LLM can invoke (e.g. calculate, summarize)
 *  - resources:  read-only data (e.g. config, runtime stats)
 *  - prompts:    reusable message templates with parameters
 *
 * The server runs as a subprocess, communicating with the client via stdin/stdout.
 * Definitions live in tools.js, resources.js, and prompts.js — this file
 * is just the registry and transport wiring.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { tools } from "./tools.js";
import { resources } from "./resources.js";
import { prompts } from "./prompts.js";

const createServer = () => {
  const server = new McpServer(
    { name: "mcp-core-demo", version: "1.0.0" },
    { capabilities: { tools: {}, resources: {}, prompts: {} } }
  );

  tools.forEach(({ name, config, handler }) =>
    server.registerTool(name, config, handler)
  );

  resources.forEach(({ id, uri, config, handler }) =>
    server.registerResource(id, uri, config, handler)
  );

  prompts.forEach(({ name, config, handler }) =>
    server.registerPrompt(name, config, handler)
  );

  return server;
};

const main = async () => {
  const server = createServer();
  await server.connect(new StdioServerTransport());

  const exit = async () => { await server.close(); process.exit(0); };
  process.on("SIGINT", exit);
  process.on("SIGTERM", exit);
};

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
