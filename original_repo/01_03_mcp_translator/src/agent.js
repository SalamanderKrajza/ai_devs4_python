import { chat, extractToolCalls, extractText } from "./helpers/api.js";
import { callMcpTool, mcpToolsToOpenAI } from "./mcp/client.js";
import log from "./helpers/logger.js";

const MAX_STEPS = 80;

const runTool = async (mcpClient, toolCall) => {
  const args = JSON.parse(toolCall.arguments);
  log.tool(toolCall.name, args);

  try {
    const result = await callMcpTool(mcpClient, toolCall.name, args);
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
  Promise.all(toolCalls.map(toolCall => runTool(mcpClient, toolCall)));

/**
 * Agentic loop: chat → tools → results → repeat until done.
 */
export const run = async (query, { mcpClient, mcpTools }) => {
  const tools = mcpToolsToOpenAI(mcpTools);
  const messages = [{ role: "user", content: query }];
  const history = [];

  log.query(query);

  for (let step = 1; step <= MAX_STEPS; step++) {
    log.api(`Step ${step}`, messages.length);
    const response = await chat({ input: messages, tools });
    log.apiDone(response.usage);

    const toolCalls = extractToolCalls(response);

    if (toolCalls.length === 0) {
      const text = extractText(response) ?? "No response";
      log.response(text);
      return { response: text, toolCalls: history };
    }

    messages.push(...response.output);

    for (const tc of toolCalls) {
      history.push({ name: tc.name, arguments: JSON.parse(tc.arguments) });
    }

    const results = await runTools(mcpClient, toolCalls);
    messages.push(...results);
  }

  throw new Error(`Max steps (${MAX_STEPS}) reached`);
};
