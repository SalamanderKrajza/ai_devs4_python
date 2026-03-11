/**
 * MCP Native Demo — one agent using both MCP tools and native JS tools.
 *
 * Shows how MCP tools (from a server) and plain function tools can be
 * unified behind a single handler map and driven by the same agent loop.
 * The model doesn't know which tools are MCP and which are native.
 */

import { createMcpServer } from "./src/mcp/server.js";
import { createMcpClient, listMcpTools, mcpToolsToOpenAI, callMcpTool } from "./src/mcp/client.js";
import { nativeTools, nativeHandlers } from "./src/native/tools.js";
import { createAgent } from "./src/agent.js";
import { MCP_LABEL, NATIVE_LABEL } from "./src/log.js";
import { resolveModelForProvider } from "../config.js";

const model = resolveModelForProvider("gpt-5.2");
const instructions = `You are a helpful assistant with access to various tools.
You can check weather, get time, perform calculations, and transform text.
Use the appropriate tool for each task. Be concise.`;

const main = async () => {
  // Start in-memory MCP server and connect a client
  const mcpServer = createMcpServer();
  const mcpClient = await createMcpClient(mcpServer);
  const mcpTools = await listMcpTools(mcpClient);

  // Unified handler map — MCP and native tools behind the same { execute, label } interface
  const handlers = Object.fromEntries([
    ...mcpTools.map((t) => [t.name, {
      execute: (args) => callMcpTool(mcpClient, t.name, args),
      label: MCP_LABEL
    }]),
    ...Object.entries(nativeHandlers).map(([name, fn]) => [name, {
      execute: fn,
      label: NATIVE_LABEL
    }])
  ]);

  const tools = [...mcpToolsToOpenAI(mcpTools), ...nativeTools];
  const agent = createAgent({ model, tools, instructions, handlers });

  console.log(`MCP tools: ${mcpTools.map((t) => t.name).join(", ")}`);
  console.log(`Native tools: ${Object.keys(nativeHandlers).join(", ")}`);

  const queries = [
    "What's the weather in Tokyo?",
    "What time is it in Europe/London?",
    "Calculate 42 multiplied by 17",
    "Convert 'hello world' to uppercase",
    "What's 25 + 17, and what's the weather in Paris?"
  ];

  for (const query of queries) {
    await agent.processQuery(query);
  }

  await mcpClient.close();
  await mcpServer.close();
};

main().catch(console.error);
