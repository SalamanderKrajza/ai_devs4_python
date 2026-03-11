/**
 * MCP resource definitions for the demo server.
 *
 * Resources are read-only data the server exposes to clients.
 * They can be static (fixed content) or dynamic (generated per request).
 * Clients discover them via listResources and read via readResource.
 */

const startTime = Date.now();
let requestCount = 0;

export const resources = [
  // Static resource — always returns the same project metadata
  {
    id: "project-config",
    uri: "config://project",
    config: { title: "Project Configuration", description: "Current project settings", mimeType: "application/json" },
    handler: async () => ({
      contents: [{
        uri: "config://project",
        mimeType: "application/json",
        text: JSON.stringify({
          name: "mcp-core-demo",
          version: "1.0.0",
          features: ["tools", "resources", "prompts", "elicitation", "sampling"]
        }, null, 2)
      }]
    })
  },
  // Dynamic resource — content changes on every read
  {
    id: "runtime-stats",
    uri: "data://stats",
    config: { title: "Runtime Statistics", description: "Dynamic server stats", mimeType: "application/json" },
    handler: async () => {
      requestCount++;
      return {
        contents: [{
          uri: "data://stats",
          mimeType: "application/json",
          text: JSON.stringify({
            uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
            request_count: requestCount,
            timestamp: new Date().toISOString()
          }, null, 2)
        }]
      };
    }
  }
];
