/**
 * MCP client — connects to a server via in-memory transport.
 *
 * In-memory transport is used here because the server runs in the same
 * process (unlike mcp_core which uses stdio for a subprocess).
 * The wrapper functions bridge MCP tool format to OpenAI function format
 * so the agent can treat MCP tools like any other tool.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

export const createMcpClient = async (server) => {
  const client = new Client(
    { name: "demo-mcp-client", version: "1.0.0" },
    { capabilities: {} }
  );

  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  return client;
};

export const listMcpTools = async (client) => {
  const { tools } = await client.listTools();
  return tools;
};

// Calls an MCP tool and parses the text result
export const callMcpTool = async (client, name, args) => {
  const result = await client.callTool({ name, arguments: args });
  const textContent = result.content.find((c) => c.type === "text");
  return textContent ? JSON.parse(textContent.text) : result;
};

// Converts MCP tool schemas → OpenAI function-calling format
export const mcpToolsToOpenAI = (mcpTools) =>
  mcpTools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
    strict: true
  }));
