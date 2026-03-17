/**
 * Agent Loop
 * 
 * Executes chat → tool calls → results cycle until completion.
 * Supports both MCP tools and native tools.
 * Maintains conversation state for follow-up questions.
 * Supports confirmation hooks for sensitive operations.
 */

import { chat, extractToolCalls, extractText } from "./helpers/api.js";
import { callMcpTool, mcpToolsToOpenAI } from "./mcp/client.js";
import { nativeTools, isNativeTool, executeNativeTool } from "./native/tools.js";
import log from "./helpers/logger.js";

const MAX_STEPS = 50;

/**
 * Tools that require user confirmation before execution.
 */
const TOOLS_REQUIRING_CONFIRMATION = new Set(["send_email"]);

// ─────────────────────────────────────────────────────────────
// Tool Execution
// ─────────────────────────────────────────────────────────────

const runTool = async (mcpClient, toolCall, confirmTool) => {
  const args = JSON.parse(toolCall.arguments);
  const toolName = toolCall.name;

  log.tool(toolName, args);

  // Check if tool requires confirmation
  if (TOOLS_REQUIRING_CONFIRMATION.has(toolName) && confirmTool) {
    const confirmed = await confirmTool(toolName, args);
    
    if (!confirmed) {
      const output = JSON.stringify({ 
        success: false, 
        error: "User rejected the action",
        rejected: true 
      });
      log.toolResult(toolName, false, "Rejected by user");
      return { type: "function_call_output", call_id: toolCall.call_id, output };
    }
  }

  try {
    let result;
    
    if (isNativeTool(toolName)) {
      result = await executeNativeTool(toolName, args);
    } else {
      result = await callMcpTool(mcpClient, toolName, args);
    }

    const output = JSON.stringify(result);
    log.toolResult(toolName, true, output);
    return { type: "function_call_output", call_id: toolCall.call_id, output };
  } catch (error) {
    const output = JSON.stringify({ error: error.message });
    log.toolResult(toolName, false, error.message);
    return { type: "function_call_output", call_id: toolCall.call_id, output };
  }
};

const runTools = async (mcpClient, toolCalls, confirmTool) => {
  // Run sequentially to allow for confirmations
  const results = [];
  for (const tc of toolCalls) {
    const result = await runTool(mcpClient, tc, confirmTool);
    results.push(result);
  }
  return results;
};

// ─────────────────────────────────────────────────────────────
// Agent Loop (with conversation state)
// ─────────────────────────────────────────────────────────────

/**
 * Run the agent with a query.
 * 
 * @param {string} query - User query
 * @param {object} options - Configuration
 * @param {object} options.mcpClient - MCP client instance
 * @param {array} options.mcpTools - Available MCP tools
 * @param {array} options.conversationHistory - Previous messages (for follow-ups)
 * @param {function} options.confirmTool - Callback for tool confirmation (toolName, args) => Promise<boolean>
 * @returns {object} { response, conversationHistory }
 */
export const run = async (query, { mcpClient, mcpTools, conversationHistory = [], confirmTool }) => {
  const tools = [...mcpToolsToOpenAI(mcpTools), ...nativeTools];
  
  // Start with existing history or empty array
  const messages = [...conversationHistory, { role: "user", content: query }];

  log.query(query);

  for (let step = 1; step <= MAX_STEPS; step++) {
    log.api(`Step ${step}`, messages.length);
    const response = await chat({ input: messages, tools });
    log.apiDone(response.usage);

    const toolCalls = extractToolCalls(response);

    if (toolCalls.length === 0) {
      const text = extractText(response) ?? "No response";
      log.response(text);
      
      // Add assistant response to history
      messages.push(...response.output);
      
      return { 
        response: text, 
        conversationHistory: messages
      };
    }

    messages.push(...response.output);

    const results = await runTools(mcpClient, toolCalls, confirmTool);
    messages.push(...results);
  }

  throw new Error(`Max steps (${MAX_STEPS}) reached`);
};

/**
 * Create a new conversation context.
 */
export const createConversation = () => ({
  history: []
});
