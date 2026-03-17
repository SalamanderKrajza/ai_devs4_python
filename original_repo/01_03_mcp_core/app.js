/**
 * MCP Core Demo — exercises all MCP capabilities via stdio transport.
 *
 * The client spawns the server as a subprocess and communicates via stdin/stdout.
 * This is how real MCP integrations work (e.g. Claude Desktop, Cursor).
 */

import { createMcpClient } from "./src/client.js";
import { heading, log, parseToolResult } from "./src/log.js";
import { resolveModelForProvider } from "../config.js";

const model = resolveModelForProvider("gpt-5.1");

const main = async () => {
  const client = await createMcpClient({ model });

  try {
    heading("TOOLS", "Actions the server exposes for the LLM to invoke");

    const { tools } = await client.listTools();
    log("listTools", tools.map((t) => `${t.name} — ${t.description}`));

    const calcResult = await client.callTool({
      name: "calculate",
      arguments: { operation: "multiply", a: 42, b: 17 }
    });
    log("callTool(calculate)", parseToolResult(calcResult));

    const summaryResult = await client.callTool({
      name: "summarize_with_confirmation",
      arguments: {
        text: "The Model Context Protocol (MCP) is a standardized protocol that allows applications to provide context for LLMs. It separates the concerns of providing context from the actual LLM interaction. MCP servers expose tools, resources, and prompts that clients can discover and use.",
        maxLength: 30
      }
    });
    log("callTool(summarize_with_confirmation)", parseToolResult(summaryResult));

    heading("RESOURCES", "Read-only data the server makes available to clients");

    const { resources } = await client.listResources();
    log("listResources", resources.map((r) => `${r.uri} — ${r.name ?? r.description}`));

    const configResource = await client.readResource({ uri: "config://project" });
    log("readResource(config://project)", JSON.parse(configResource.contents[0].text));

    const statsResource = await client.readResource({ uri: "data://stats" });
    log("readResource(data://stats)", JSON.parse(statsResource.contents[0].text));

    heading("PROMPTS", "Reusable message templates with parameters");

    const { prompts } = await client.listPrompts();
    log("listPrompts", prompts.map((p) => `${p.name} — ${p.description}`));

    const { messages } = await client.getPrompt({
      name: "code-review",
      arguments: {
        code: "function add(a, b) { return a + b; }",
        language: "javascript",
        focus: "readability"
      }
    });
    log("getPrompt(code-review)", messages.map((m) =>
      `[${m.role}] ${m.content?.text ?? JSON.stringify(m.content)}`
    ));
  } finally {
    await client.close();
  }
};

main().catch(console.error);
