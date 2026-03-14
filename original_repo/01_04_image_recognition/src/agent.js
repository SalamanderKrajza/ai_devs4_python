/**
 * Agent loop — chat → tool calls → results cycle until completion.
 * Supports both MCP and native tools.
 */

import { chat, extractToolCalls, extractText } from "./api.js";
import { callMcpTool, mcpToolsToOpenAI } from "./mcp/client.js";
import { nativeTools, isNativeTool, executeNativeTool } from "./native/tools.js";
import log from "./helpers/logger.js";

const MAX_STEPS = 100;

const runTool = async (mcpClient, toolCall) => {
  const args = JSON.parse(toolCall.arguments);
  log.tool(toolCall.name, args);

  try {
    const result = isNativeTool(toolCall.name)
      ? await executeNativeTool(toolCall.name, args)
      : await callMcpTool(mcpClient, toolCall.name, args);

    const output = JSON.stringify(result);
    log.toolResult(toolCall.name, true, output);
    return { type: "function_call_output", call_id: toolCall.call_id, output };
  } catch (error) {
    const output = JSON.stringify({ error: error.message });
    log.toolResult(toolCall.name, false, error.message);
    return { type: "function_call_output", call_id: toolCall.call_id, output };
  }
};

const runTools = (mcpClient, toolCalls) =>
  Promise.all(toolCalls.map(tc => runTool(mcpClient, tc)));

export const run = async (query, { mcpClient, mcpTools }) => {
  const tools = [...mcpToolsToOpenAI(mcpTools), ...nativeTools];
  const messages = [{ role: "user", content: query }];

  log.query(query);

  for (let step = 1; step <= MAX_STEPS; step++) {
    log.api(`Step ${step}`, messages.length);
    const response = await chat({ input: messages, tools });
    log.apiDone(response.usage);

    const toolCalls = extractToolCalls(response);

    if (toolCalls.length === 0) {
      const text = extractText(response) ?? "No response";
      return { response: text };
    }

    messages.push(...response.output);

    const results = await runTools(mcpClient, toolCalls);
    messages.push(...results);
  }

  throw new Error(`Max steps (${MAX_STEPS}) reached`);
};
