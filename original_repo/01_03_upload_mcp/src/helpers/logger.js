/**
 * Logger — colored terminal output with smart summarization.
 *
 * Inspired by 04_04_system's logger: [LABEL] tags, summarized tool
 * args/results instead of raw JSON dumps, dim secondary info.
 */

const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
};

const tag = (name, color) => `${c.bold}${color}[${name}]${c.reset}`;

const truncate = (text, max = 120) =>
  text.length > max ? text.slice(0, max - 1) + "…" : text;

const summarizeArgs = (name, args) => {
  if (name.includes("fs_read") && args.path) return `${args.path} (${args.mode ?? "content"})`;
  if (name.includes("fs_write") && args.path) return `${args.path} (${args.operation})`;
  if (name.includes("fs_search") && args.query) return `"${args.query}" in ${args.path ?? "."}`;
  if (name.includes("fs_manage") && args.operation) return `${args.operation} ${args.path}`;
  if (name.includes("upload_files") && args.files) return args.files.map((f) => f.name).join(", ");
  if (name.includes("list_files")) return "";
  return truncate(JSON.stringify(args), 80);
};

const summarizeResult = (name, result) => {
  if (typeof result === "string") return truncate(result, 100);
  if (result?.success && result?.type === "file") return `✓ ${result.path} (${result.content?.totalLines ?? "?"} lines)`;
  if (result?.success && result?.type === "directory") return `✓ ${result.summary}`;
  if (result?.status === "applied") return `✓ ${result.result?.action ?? "written"}`;
  if (result?.status === "error") return `✗ ${result.error?.message ?? "failed"}`;
  if (result?.status) return result.status;
  return `${JSON.stringify(result).length} chars`;
};

const tryParseJson = (str) => {
  try { return JSON.parse(str); }
  catch { return null; }
};

export const log = {
  info: (msg) => console.log(`${c.dim}  ${msg}${c.reset}`),
  success: (msg) => console.log(`  ${c.green}✓${c.reset} ${msg}`),
  warn: (msg) => console.log(`  ${c.yellow}⚠${c.reset} ${msg}`),
  error: (msg, detail) => console.error(`  ${c.red}✗ ${msg}${detail ? `: ${detail}` : ""}${c.reset}`),
  debug: () => {},

  box: (msg) => {
    const lines = msg.split("\n");
    const width = Math.max(...lines.map((l) => l.length)) + 4;
    console.log(`\n${c.bold}╭${"─".repeat(width)}╮${c.reset}`);
    lines.forEach((l) => console.log(`${c.bold}│  ${l.padEnd(width - 2)}│${c.reset}`));
    console.log(`${c.bold}╰${"─".repeat(width)}╯${c.reset}\n`);
  },

  start: (msg) => console.log(`  ${c.cyan}◐${c.reset} ${msg}`),
  ready: (msg) => console.log(`  ${c.green}✓${c.reset} ${c.bold}${msg}${c.reset}`),

  tool: (name, args) => {
    const summary = summarizeArgs(name, args);
    console.log(`  ${tag("TOOL", c.magenta)} ${c.cyan}${name}${c.reset} ${c.dim}${summary}${c.reset}`);
  },

  toolResult: (name, success, detail = "") => {
    if (success) {
      const parsed = tryParseJson(detail);
      const summary = parsed ? summarizeResult(name, parsed) : truncate(detail, 100);
      console.log(`  ${c.dim}  → ${summary}${c.reset}`);
    } else {
      console.log(`  ${c.red}  → ✗ ${truncate(detail, 100)}${c.reset}`);
    }
  },

  api: (action, historyLength) => {
    const info = historyLength !== undefined ? ` (${historyLength} msgs)` : "";
    console.log(`\n  ${tag("LLM", c.blue)} ${action}${c.dim}${info}${c.reset}`);
  },

  apiDone: (usage) => {
    if (!usage) return;
    const { input_tokens: i = 0, output_tokens: o = 0 } = usage;
    const cached = usage.input_tokens_details?.cached_tokens ?? 0;
    const rate = i > 0 ? Math.round((cached / i) * 100) : 0;
    console.log(`  ${c.dim}  → ${i} in, ${o} out, ${rate}% cached${c.reset}`);
  },

  query: (text) => {
    console.log(`\n${tag("QUERY", c.blue)} ${truncate(text, 120)}`);
  },

  response: (text) => {
    console.log(`\n${tag("DONE", c.green)} ${truncate(text, 120)}\n`);
  },
};

export default log;
