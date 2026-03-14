/**
 * Interactive REPL for the image editing agent.
 */

import * as readline from "readline/promises";
import { run } from "./agent.js";
import { resetStats } from "./helpers/stats.js";
import log from "./helpers/logger.js";

export const createReadline = () =>
  readline.createInterface({ input: process.stdin, output: process.stdout });

export const runRepl = async ({ mcpClient, mcpTools, rl }) => {
  let history = [];

  while (true) {
    const input = await rl.question("You: ").catch(() => "exit");

    if (input.toLowerCase() === "exit") break;

    if (input.toLowerCase() === "clear") {
      history = [];
      resetStats();
      log.success("Conversation cleared\n");
      continue;
    }

    if (!input.trim()) continue;

    try {
      const result = await run(input, { mcpClient, mcpTools, conversationHistory: history });
      history = result.conversationHistory;
      console.log(`\nAssistant: ${result.response}\n`);
    } catch (err) {
      log.error("Error", err.message);
      console.log("");
    }
  }
};
