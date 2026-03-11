/**
 * In-memory MCP server with mock tools (weather, time).
 *
 * Unlike mcp_core which uses stdio transport, this server runs
 * in the same process and connects via InMemoryTransport.
 * The tools are intentionally simple — the point of this example
 * is the unified agent loop, not the tool implementations.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export const createMcpServer = () => {
  const server = new McpServer(
    { name: "demo-mcp-server", version: "1.0.0" },
    { capabilities: { tools: {} } }
  );

  server.registerTool(
    "get_weather",
    {
      description: "Get current weather for a city",
      inputSchema: { city: z.string().describe("City name") }
    },
    async ({ city }) => {
      const conditions = ["sunny", "cloudy", "rainy", "snowy"];
      const condition = conditions[Math.floor(Math.random() * conditions.length)];
      const temp = Math.floor(Math.random() * 35) - 5;

      return {
        content: [{ type: "text", text: JSON.stringify({ city, condition, temperature: `${temp}°C` }) }]
      };
    }
  );

  server.registerTool(
    "get_time",
    {
      description: "Get current time in a specified timezone",
      inputSchema: { timezone: z.string().describe("Timezone (e.g., 'UTC', 'America/New_York')") }
    },
    async ({ timezone }) => {
      try {
        const time = new Date().toLocaleString("en-US", { timeZone: timezone });
        return {
          content: [{ type: "text", text: JSON.stringify({ timezone, time }) }]
        };
      } catch {
        return {
          content: [{ type: "text", text: JSON.stringify({ error: `Invalid timezone: ${timezone}` }) }],
          isError: true
        };
      }
    }
  );

  return server;
};
