/*
  Agent loop — orchestrates the tool-calling workflow for file uploads.

  Flow:  query → model → (tool calls → resolve file refs → MCP → results → model) → final answer

  The model can request tool calls in its response. When it does, we resolve
  any {{file:path}} placeholders in the arguments, execute via the MCP client,
  feed the results back, and let the model continue. This repeats until the
  model produces a plain text answer or we hit MAX_STEPS.
*/

import { chat as complete } from "./ai.js";
import { callMcpTool, mcpToolsToOpenAI } from "./mcp/client.js";
import { resolveFileRefs } from "./files/resolver.js";
import log from "./helpers/logger.js";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const MAX_STEPS = 50;
const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKSPACE_ROOT = join(__dirname, "..", "workspace");

export const run = async (conversation, { mcpClients, mcpTools, model, instructions, maxOutputTokens }) => {
  const tools = mcpToolsToOpenAI(mcpTools);

  const executeTool = async (call) => {
    const rawArgs = JSON.parse(call.arguments);
    const args = await resolveFileRefs(rawArgs, WORKSPACE_ROOT);

    log.tool(call.name, rawArgs);

    try {
      const result = await callMcpTool(mcpClients, call.name, args);
      const output = JSON.stringify(result);
      log.toolResult(call.name, true, output);
      return { type: "function_call_output", call_id: call.call_id, output };
    } catch (error) {
      const output = JSON.stringify({ error: error.message });
      log.toolResult(call.name, false, error.message);
      return { type: "function_call_output", call_id: call.call_id, output };
    }
  };

  let current = conversation;

  for (let step = 0; step < MAX_STEPS; step++) {
    log.api(`Step ${step + 1}`, current.length);
    const response = await complete({ model, instructions, maxOutputTokens, input: current, tools });
    log.apiDone(response.usage);

    const calls = response.output.filter((item) => item.type === "function_call");

    if (calls.length === 0) {
      const text = response.output_text
        ?? response.output.find((item) => item.type === "message")?.content?.[0]?.text
        ?? "No response";
      log.response(text);
      return { text, conversation: current };
    }

    const results = await Promise.all(calls.map(executeTool));
    current = [...current, ...response.output, ...results];
  }

  throw new Error(`Tool loop did not finish within ${MAX_STEPS} steps.`);
};
