/**
 * MCP server configuration — loads and validates mcp.json.
 *
 * This example connects to two servers defined in mcp.json:
 *  - files:       stdio transport (spawns files-mcp as a subprocess)
 *  - uploadthing: HTTP transport (connects to a remote deployment)
 *
 * The HTTP server URL must be configured before running — see README.md.
 */

import { readFile } from "fs/promises";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, "../..");
const MCP_CONFIG_PATH = "01_03_upload_mcp/mcp.json";
const UPLOADTHING_PLACEHOLDER = "https://URL_TO_YOUR_MCP_SERVER/mcp";

export class ConfigurationError extends Error {
  constructor(message) {
    super(message);
    this.name = "ConfigurationError";
  }
}

export const validateHttpServerConfig = (serverName, serverConfig) => {
  const url = serverConfig?.url?.trim?.() ?? "";

  if (!url) {
    throw new ConfigurationError(
      `Invalid ${MCP_CONFIG_PATH}: set mcpServers.${serverName}.url to the deployed MCP endpoint from the AI_devs lesson, e.g. https://your-domain.example/mcp`
    );
  }

  if (url === UPLOADTHING_PLACEHOLDER || url.includes("URL_TO_YOUR_MCP_SERVER")) {
    throw new ConfigurationError(
      `Invalid ${MCP_CONFIG_PATH}: replace the mcpServers.${serverName}.url placeholder with the deployed MCP endpoint from the AI_devs lesson, e.g. https://your-domain.example/mcp`
    );
  }

  let parsedUrl;

  try {
    parsedUrl = new URL(url);
  } catch {
    throw new ConfigurationError(
      `Invalid ${MCP_CONFIG_PATH}: mcpServers.${serverName}.url must be a full http(s) URL, e.g. https://your-domain.example/mcp`
    );
  }

  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    throw new ConfigurationError(
      `Invalid ${MCP_CONFIG_PATH}: mcpServers.${serverName}.url must use http or https`
    );
  }
};

export const validateMcpConfig = (config) => {
  if (!config?.mcpServers || typeof config.mcpServers !== "object") {
    throw new ConfigurationError(
      `Invalid ${MCP_CONFIG_PATH}: expected a top-level "mcpServers" object`
    );
  }

  for (const [serverName, serverConfig] of Object.entries(config.mcpServers)) {
    if ((serverConfig.transport ?? "stdio") === "http") {
      validateHttpServerConfig(serverName, serverConfig);
    }
  }
};

export const loadMcpConfig = async () => {
  const configPath = join(PROJECT_ROOT, "mcp.json");
  const content = await readFile(configPath, "utf-8");
  return JSON.parse(content);
};

export { PROJECT_ROOT };
