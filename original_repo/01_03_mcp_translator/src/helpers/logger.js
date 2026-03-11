/**
 * Logging with consola - beautiful CLI output out of the box.
 */

import { consola, createConsola } from "consola";

// Create custom consola instance with our preferences
// Level: 0=silent, 1=error, 2=warn, 3=info, 4=debug, 5=trace
const logger = createConsola({
  level: 4, // Enable debug by default for now
  fancy: true,
  formatOptions: {
    date: false, // Disable - position is inconsistent based on line length
    colors: true,
    compact: false
  }
});

// Re-export consola methods with some custom wrappers
export const log = {
  // Standard levels
  info: (msg, ...args) => logger.info(msg, ...args),
  success: (msg, ...args) => logger.success(msg, ...args),
  warn: (msg, ...args) => logger.warn(msg, ...args),
  error: (msg, ...args) => logger.error(msg, ...args),
  debug: (msg, ...args) => logger.debug(msg, ...args),
  
  // Box for banners
  box: (msg) => logger.box(msg),
  
  // Start/ready for server
  start: (msg, ...args) => logger.start(msg, ...args),
  ready: (msg, ...args) => logger.ready(msg, ...args),
  
  // Tool calls
  tool: (name, args) => {
    const argsStr = typeof args === "object" ? JSON.stringify(args) : args;
    const truncated = argsStr.length > 200 ? argsStr.slice(0, 197) + "..." : argsStr;
    logger.info(`🔧 ${name} ${truncated}`);
  },
  
  toolResult: (name, success, detail = "") => {
    const truncated = detail.length > 200 ? detail.slice(0, 197) + "..." : detail;
    if (success) {
      logger.success(`   ↳ ${truncated || "OK"}`);
    } else {
      logger.fail(`   ↳ ${truncated || "Failed"}`);
    }
  },
  
  // API calls
  api: (action, historyLength) => {
    const historyInfo = historyLength !== undefined ? ` (${historyLength} messages)` : "";
    logger.info(`🤖 ${action}${historyInfo}`);
  },
  apiDone: (usage) => {
    if (!usage) {
      logger.success(`   ↳ done`);
      return;
    }
    const input = usage.input_tokens || 0;
    const output = usage.output_tokens || 0;
    const cached = usage.input_tokens_details?.cached_tokens || 0;
    const cacheRate = input > 0 ? Math.round((cached / input) * 100) : 0;
    logger.success(`   ↳ in:${input} out:${output} | cached:${cached} (${cacheRate}%)`);
  },
  
  // Query/response
  query: (text) => {
    const truncated = text.length > 60 ? text.slice(0, 57) + "..." : text;
    logger.info(`▶ Query: ${truncated}`);
  },
  
  response: (text) => {
    const truncated = text.length > 80 ? text.slice(0, 77) + "..." : text;
    logger.success(`◀ Response: ${truncated}`);
  },
  
  // Server endpoints
  endpoint: (method, path, desc) => {
    logger.info(`  ${method.padEnd(5)} ${path.padEnd(18)} ${desc}`);
  }
};

export default log;
