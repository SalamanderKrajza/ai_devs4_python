/**
 * Simple colored logger for terminal output.
 */

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
  white: "\x1b[37m",
  bgBlue: "\x1b[44m",
  bgMagenta: "\x1b[45m"
};

const timestamp = () => new Date().toLocaleTimeString("en-US", { hour12: false });

const log = {
  info: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.green}✓${colors.reset} ${msg}`),
  error: (title, msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.red}✗ ${title}${colors.reset} ${msg || ""}`),
  warn: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.yellow}⚠${colors.reset} ${msg}`),
  start: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.cyan}→${colors.reset} ${msg}`),
  
  box: (text) => {
    const lines = text.split("\n");
    const width = Math.max(...lines.map(l => l.length)) + 4;
    console.log(`\n${colors.cyan}${"─".repeat(width)}${colors.reset}`);
    for (const line of lines) {
      console.log(`${colors.cyan}│${colors.reset} ${colors.bright}${line.padEnd(width - 3)}${colors.reset}${colors.cyan}│${colors.reset}`);
    }
    console.log(`${colors.cyan}${"─".repeat(width)}${colors.reset}\n`);
  },

  heading: (title, description) => {
    console.log(`\n${colors.bright}═══ ${title} ═══${colors.reset}`);
    if (description) console.log(`${colors.dim}${description}${colors.reset}`);
  },

  detail: (label, data) => {
    console.log(`\n${colors.bright}${colors.cyan}▶ ${label}${colors.reset}`);
    if (data === undefined) return;
    const lines = Array.isArray(data) ? data
      : typeof data === "string" ? [data]
      : JSON.stringify(data, null, 2).split("\n");
    lines.forEach((line) => console.log(`${colors.dim}  ${line}${colors.reset}`));
  },

  example: (text) => console.log(`  ${colors.green}→${colors.reset} ${colors.bright}${text}${colors.reset}`),
  hint: (text) => console.log(`\n${colors.dim}${text}${colors.reset}\n`),

  query: (q) => console.log(`\n${colors.bgBlue}${colors.white} QUERY ${colors.reset} ${q}\n`),
  response: (r) => console.log(`\n${colors.green}Response:${colors.reset} ${r.substring(0, 500)}${r.length > 500 ? "..." : ""}\n`),
  
  api: (step, msgCount) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.magenta}◆${colors.reset} ${step} (${msgCount} messages)`),
  apiDone: (usage) => {
    if (usage) {
      console.log(`${colors.dim}         tokens: ${usage.input_tokens} in / ${usage.output_tokens} out${colors.reset}`);
    }
  },

  tool: (name, args) => {
    const argStr = JSON.stringify(args);
    const truncated = argStr.length > 100 ? argStr.substring(0, 100) + "..." : argStr;
    console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.yellow}⚡${colors.reset} ${name} ${colors.dim}${truncated}${colors.reset}`);
  },
  
  toolResult: (name, success, output) => {
    const icon = success ? `${colors.green}✓${colors.reset}` : `${colors.red}✗${colors.reset}`;
    const truncated = output.length > 150 ? output.substring(0, 150) + "..." : output;
    console.log(`${colors.dim}         ${icon} ${truncated}${colors.reset}`);
  },

  gemini: (action, detail) => {
    console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.bgMagenta}${colors.white} GEMINI ${colors.reset} ${action}`);
    if (detail) console.log(`${colors.dim}         ${detail}${colors.reset}`);
  },

  geminiResult: (success, msg) => {
    const icon = success ? `${colors.green}✓${colors.reset}` : `${colors.red}✗${colors.reset}`;
    console.log(`${colors.dim}         ${icon} ${msg}${colors.reset}`);
  }
};

export default log;
