/**
 * Agent loop — processes queries using a unified set of tool handlers.
 *
 * The agent doesn't know whether a tool is served by MCP or native JS.
 * It just dispatches to the handler map built by app.js. Each handler
 * has { execute, label } so the output shows which backend ran the tool.
 */

import { chat, extractToolCalls, extractText } from "./ai.js";
import { logQuery, logToolCall, logToolResult, logToolError, logToolCount, logResponse } from "./log.js";

const MAX_TOOL_ROUNDS = 10;

const executeToolCall = async (call, handlers) => {
  const args = JSON.parse(call.arguments);
  const handler = handlers[call.name];

  if (!handler) {
    throw new Error(`Unknown tool: ${call.name}`);
  }

  logToolCall(handler.label, call.name, args);

  try {
    const result = await handler.execute(args);
    logToolResult(result);
    return { type: "function_call_output", call_id: call.call_id, output: JSON.stringify(result) };
  } catch (error) {
    logToolError(error.message);
    return { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ error: error.message }) };
  }
};

/**
 * @param {object} config
 * @param {string} config.model — model identifier
 * @param {Array} config.tools — OpenAI-format tool definitions
 * @param {string} config.instructions — system prompt
 * @param {object} config.handlers — { toolName: { execute, label } }
 */
export const createAgent = ({ model, tools, instructions, handlers }) => ({
  async processQuery(query) {
    logQuery(query);

    const chatConfig = { model, tools, instructions };
    let conversation = [{ role: "user", content: query }];

    for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
      const response = await chat({ ...chatConfig, input: conversation });
      const toolCalls = extractToolCalls(response);

      if (toolCalls.length === 0) {
        const text = extractText(response) ?? "No response";
        logResponse(text);
        return text;
      }

      logToolCount(toolCalls.length);
      const toolResults = await Promise.all(
        toolCalls.map((call) => executeToolCall(call, handlers))
      );

      conversation = [...conversation, ...response.output, ...toolResults];
    }

    logResponse("Max tool rounds reached");
    return "Max tool rounds reached";
  }
});
