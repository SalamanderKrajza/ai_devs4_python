/**
 * Logging and result helpers for the MCP core demo.
 */

const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  cyan: "\x1b[36m",
  magenta: "\x1b[35m",
};

const truncate = (value, maxLength = 50) => {
  const text = String(value);
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
};

const getErrorMessage = (cause) =>
  cause instanceof Error ? cause.message : String(cause);

export const heading = (title, description) => {
  console.log(`\n${c.bold}═══ ${title} ═══${c.reset}`);
  if (description) console.log(`${c.dim}${description}${c.reset}`);
};

export const log = (label, data) => {
  console.log(`\n${c.bold}${c.cyan}▶ ${label}${c.reset}`);

  if (data === undefined) return;

  const lines = Array.isArray(data) ? data
    : typeof data === "string" ? [data]
    : JSON.stringify(data, null, 2).split("\n");

  lines.forEach((line) => console.log(`${c.dim}  ${line}${c.reset}`));
};

export const parseToolResult = (result) => {
  const text = result.content?.find((c) => c.type === "text")?.text ?? "";

  if (result.isError) {
    throw new Error(text || "Tool call failed");
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
};

export const clientLog = {
  spawningServer: (serverPath) => {
    console.log(`\n${c.green}🚀 Spawning MCP server: ${serverPath}${c.reset}`);
  },
  connected: () => {
    console.log(`${c.green}✓ Connected to MCP server via stdio${c.reset}`);
  },
  samplingRequest: ({ messages, maxTokens }) => {
    console.log(`\n${c.magenta}  📡 Sampling — server asked the client to call an LLM${c.reset}`);
    console.log(`${c.dim}     Messages: ${messages.length}, max tokens: ${maxTokens ?? "default"}${c.reset}`);
  },
  samplingResponse: (text) => {
    console.log(`${c.dim}     LLM responded: "${truncate(text)}"${c.reset}`);
  },
  samplingError: (cause) => {
    console.error(`${c.red}     Sampling error: ${getErrorMessage(cause)}${c.reset}`);
  },
  elicitationRequest: ({ mode }) => {
    console.log(`\n${c.yellow}  🔔 Elicitation — server asked the client for user confirmation${c.reset}`);
    console.log(`${c.dim}     Mode: ${mode}${c.reset}`);
  },
  autoAcceptedElicitation: (content) => {
    console.log(`${c.dim}     Auto-accepted with: ${JSON.stringify(content)}${c.reset}`);
  }
};
