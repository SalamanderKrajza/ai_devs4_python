/**
 * Logging helpers for the MCP native demo.
 *
 * Separates presentation from logic so the agent loop stays clean.
 * Color-coded labels distinguish MCP tools from native tools in output.
 */

const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  green: "\x1b[32m",
  red: "\x1b[31m",
  yellow: "\x1b[33m",
  cyan: "\x1b[36m",
};

export const MCP_LABEL = `${c.cyan}🔌 MCP${c.reset}`;
export const NATIVE_LABEL = `${c.yellow}⚡ Native${c.reset}`;

export const logQuery = (query) => {
  console.log(`\n${c.bold}${"═".repeat(60)}${c.reset}`);
  console.log(`${c.bold}Query: ${query}${c.reset}`);
  console.log(`${c.bold}${"═".repeat(60)}${c.reset}`);
};

export const logToolCall = (label, name, args) => {
  console.log(`  ${label} ${c.bold}${name}${c.reset}(${c.dim}${JSON.stringify(args)}${c.reset})`);
};

export const logToolResult = (result) => {
  console.log(`       ${c.green}✓${c.reset} ${c.dim}${JSON.stringify(result)}${c.reset}`);
};

export const logToolError = (message) => {
  console.log(`       ${c.red}✗ Error: ${message}${c.reset}`);
};

export const logToolCount = (count) => {
  console.log(`\n${c.dim}Tool calls: ${count}${c.reset}`);
};

export const logResponse = (text) => {
  console.log(`\n${c.green}Assistant:${c.reset} ${text}`);
};
