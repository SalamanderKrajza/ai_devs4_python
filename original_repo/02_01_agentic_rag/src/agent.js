/**
 * Agent Loop
 * 
 * Executes chat → tool calls → results cycle until completion.
 * Uses MCP tools for file operations.
 * Maintains conversation state for follow-up questions.
 */

import { chat, extractToolCalls, extractText, extractReasoning } from "./helpers/api.js";
import { callMcpTool, mcpToolsToOpenAI } from "./mcp/client.js";
import log from "./helpers/logger.js";

const MAX_STEPS = 50;

// ─────────────────────────────────────────────────────────────
// Tool Execution
// ─────────────────────────────────────────────────────────────

const runTool = async (mcpClient, toolCall) => {
  const args = JSON.parse(toolCall.arguments);
  const toolName = toolCall.name;

  log.tool(toolName, args);

  try {
    const result = await callMcpTool(mcpClient, toolName, args);
    const output = JSON.stringify(result);
    log.toolResult(toolName, true, output);
    return { type: "function_call_output", call_id: toolCall.call_id, output };
  } catch (error) {
    const output = JSON.stringify({ error: error.message });
    log.toolResult(toolName, false, error.message);
    return { type: "function_call_output", call_id: toolCall.call_id, output };
  }
};

const runTools = (mcpClient, toolCalls) =>
  Promise.all(toolCalls.map(tc => runTool(mcpClient, tc)));

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
 * @returns {object} { response, conversationHistory }
 */
export const run = async (query, { mcpClient, mcpTools, conversationHistory = [] }) => {
  const tools = mcpToolsToOpenAI(mcpTools);
  
  // Start with existing history or empty array
  const messages = [...conversationHistory, { role: "user", content: query }];

  log.query(query);

  for (let step = 1; step <= MAX_STEPS; step++) {
    log.api(`Step ${step}`, messages.length);
    const response = await chat({ input: messages, tools });
    log.apiDone(response.usage);
    log.reasoning(extractReasoning(response));

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

    const results = await runTools(mcpClient, toolCalls);
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
