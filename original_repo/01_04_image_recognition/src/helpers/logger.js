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
  bgBlue: "\x1b[44m"
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

  query: (q) => console.log(`\n${colors.bgBlue}${colors.white} QUERY ${colors.reset} ${q}\n`),
  response: (r) => console.log(`\n${colors.green}Response:${colors.reset} ${r.substring(0, 200)}${r.length > 200 ? "..." : ""}\n`),
  
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

  vision: (path, question) => {
    console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.blue}👁${colors.reset} Vision: ${path}`);
    console.log(`${colors.dim}         Q: ${question}${colors.reset}`);
  },

  visionResult: (answer) => {
    const truncated = answer.length > 200 ? answer.substring(0, 200) + "..." : answer;
    console.log(`${colors.dim}         A: ${truncated}${colors.reset}`);
  }
};

export default log;
