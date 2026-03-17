/**
 * Interactive REPL for the file & email agent.
 * Includes confirmation UI for sensitive operations with trust option.
 */

import * as readline from "readline/promises";
import { run, createConversation } from "./agent.js";
import { resetStats } from "./helpers/stats.js";
import log from "./helpers/logger.js";

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  cyan: "\x1b[36m",
  bgGreen: "\x1b[42m",
  bgRed: "\x1b[41m",
  bgBlue: "\x1b[44m",
  white: "\x1b[37m"
};

/**
 * Create readline interface.
 */
export const createReadline = () => 
  readline.createInterface({ 
    input: process.stdin, 
    output: process.stdout 
  });

/**
 * Format email details for confirmation UI.
 */
const formatEmailConfirmation = (args) => {
  const recipients = Array.isArray(args.to) ? args.to.join(", ") : args.to;
  const bodyLines = (args.body || "(empty)").split("\n");
  const formattedBody = bodyLines.map(line => `│  ${line}`).join("\n");

  return `
${colors.cyan}┌──────────────────────────────────────────────────────────────────┐${colors.reset}
${colors.cyan}│${colors.reset}  ${colors.bright}📧 EMAIL CONFIRMATION REQUIRED${colors.reset}                                 ${colors.cyan}│${colors.reset}
${colors.cyan}├──────────────────────────────────────────────────────────────────┤${colors.reset}
${colors.cyan}│${colors.reset}                                                                  ${colors.cyan}│${colors.reset}
${colors.cyan}│${colors.reset}  ${colors.bright}To:${colors.reset}      ${recipients.padEnd(52)}${colors.cyan}│${colors.reset}
${colors.cyan}│${colors.reset}  ${colors.bright}Subject:${colors.reset} ${(args.subject || "(no subject)").padEnd(52)}${colors.cyan}│${colors.reset}
${colors.cyan}│${colors.reset}  ${colors.bright}Format:${colors.reset}  ${(args.format || "text").padEnd(52)}${colors.cyan}│${colors.reset}
${args.reply_to ? `${colors.cyan}│${colors.reset}  ${colors.bright}Reply-To:${colors.reset} ${args.reply_to.padEnd(51)}${colors.cyan}│${colors.reset}\n` : ""}${colors.cyan}│${colors.reset}                                                                  ${colors.cyan}│${colors.reset}
${colors.cyan}├──────────────────────────────────────────────────────────────────┤${colors.reset}
${colors.cyan}│${colors.reset}  ${colors.bright}Body:${colors.reset}                                                         ${colors.cyan}│${colors.reset}
${colors.cyan}│${colors.reset}                                                                  ${colors.cyan}│${colors.reset}
${formattedBody}
${colors.cyan}│${colors.reset}                                                                  ${colors.cyan}│${colors.reset}
${colors.cyan}└──────────────────────────────────────────────────────────────────┘${colors.reset}

  ${colors.bgGreen}${colors.white} [Y] Send ${colors.reset}    ${colors.bgBlue}${colors.white} [T] Trust & Send ${colors.reset}    ${colors.bgRed}${colors.white} [N] Cancel ${colors.reset}
`;
};

/**
 * Create a confirmation handler for sensitive tools.
 * Supports trusting tools to skip future confirmations.
 */
const createConfirmationHandler = (rl, trustedTools) => async (toolName, args) => {
  // Skip confirmation for trusted tools
  if (trustedTools.has(toolName)) {
    console.log(`\n  ${colors.blue}⚡ Auto-approved (trusted):${colors.reset} ${toolName}\n`);
    return true;
  }

  if (toolName === "send_email") {
    console.log(formatEmailConfirmation(args));
    
    const answer = await rl.question(`  ${colors.bright}Your choice:${colors.reset} `);
    const choice = answer.toLowerCase().trim();
    
    console.log("");
    
    if (choice === "t" || choice === "trust") {
      trustedTools.add(toolName);
      console.log(`  ${colors.blue}✓ Trusted "${toolName}" for this session${colors.reset}`);
      console.log(`  ${colors.green}✓ Sending email...${colors.reset}\n`);
      return true;
    }
    
    if (choice === "y" || choice === "yes") {
      console.log(`  ${colors.green}✓ Sending email...${colors.reset}\n`);
      return true;
    }
    
    console.log(`  ${colors.red}✗ Email cancelled${colors.reset}\n`);
    return false;
  }
  
  // Unknown tool requiring confirmation - ask generically
  console.log(`\n${colors.yellow}⚠  Action requires confirmation${colors.reset}`);
  console.log(`   Tool: ${toolName}`);
  console.log(`   Args: ${JSON.stringify(args, null, 2)}`);
  console.log(`\n  ${colors.bgGreen}${colors.white} [Y] Proceed ${colors.reset}    ${colors.bgBlue}${colors.white} [T] Trust & Proceed ${colors.reset}    ${colors.bgRed}${colors.white} [N] Cancel ${colors.reset}\n`);
  
  const answer = await rl.question(`  ${colors.bright}Your choice:${colors.reset} `);
  const choice = answer.toLowerCase().trim();
  
  if (choice === "t" || choice === "trust") {
    trustedTools.add(toolName);
    console.log(`\n  ${colors.blue}✓ Trusted "${toolName}" for this session${colors.reset}\n`);
    return true;
  }
  
  return choice === "y" || choice === "yes";
};

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
  const trustedTools = new Set();
  const confirmTool = createConfirmationHandler(rl, trustedTools);

  while (true) {
    const input = await rl.question(`${colors.bright}You:${colors.reset} `).catch(() => "exit");
    
    if (input.toLowerCase() === "exit") {
      break;
    }

    if (input.toLowerCase() === "clear") {
      conversation = createConversation();
      trustedTools.clear();
      resetStats();
      log.success("Conversation and trust cleared\n");
      continue;
    }

    if (input.toLowerCase() === "untrust") {
      trustedTools.clear();
      log.success("All tools untrusted\n");
      continue;
    }

    if (!input.trim()) {
      continue;
    }

    try {
      const result = await run(input, { 
        mcpClient, 
        mcpTools,
        conversationHistory: conversation.history,
        confirmTool
      });
      
      conversation.history = result.conversationHistory;
      console.log(`\n${colors.green}Assistant:${colors.reset} ${result.response}\n`);
    } catch (err) {
      log.error("Error", err.message);
      console.log("");
    }
  }
};
