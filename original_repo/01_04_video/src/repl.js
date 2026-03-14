/**
 * Interactive REPL for the video processing agent.
 */

import * as readline from "readline/promises";
import { run, createConversation } from "./agent.js";
import { resetStats } from "./helpers/stats.js";
import log from "./helpers/logger.js";

/**
 * Create readline interface.
 */
export const createReadline = () => 
  readline.createInterface({ 
    input: process.stdin, 
    output: process.stdout 
  });

/**
 * Run the interactive REPL loop.
 * 
 * @param {object} options
 * @param {object} options.mcpClient - MCP client instance
 * @param {array} options.mcpTools - Available MCP tools
 * @param {object} options.rl - Readline interface
 * @returns {Promise<void>}
 */
export const runRepl = async ({ mcpClient, mcpTools, rl }) => {
  let conversation = createConversation();

  while (true) {
    const input = await rl.question("You: ").catch(() => "exit");
    
    if (input.toLowerCase() === "exit") {
      break;
    }

    if (input.toLowerCase() === "clear") {
      conversation = createConversation();
      resetStats();
      log.success("Conversation cleared\n");
      continue;
    }

    if (!input.trim()) {
      continue;
    }

    try {
      const result = await run(input, { 
        mcpClient, 
        mcpTools,
        conversationHistory: conversation.history
      });
      
      conversation.history = result.conversationHistory;
      console.log(`\nAssistant: ${result.response}\n`);
    } catch (err) {
      log.error("Error", err.message);
      console.log("");
    }
  }
};
