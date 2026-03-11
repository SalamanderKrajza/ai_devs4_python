/**
 * Sampling handler — bridges MCP sampling requests to the AI provider.
 *
 * Sampling is a protocol feature where the server asks the client to
 * generate an LLM completion on its behalf. The server sends messages
 * and constraints, and the client returns the model's response.
 *
 * This lets the server leverage AI without needing its own API keys
 * or provider configuration — the client owns that relationship.
 */

import { completion } from "./ai.js";
import { clientLog } from "./log.js";

/**
 * Creates a sampling request handler for the MCP client.
 *
 * @param {string} model — model identifier passed to the AI provider
 */
export const createSamplingHandler = (model) => async (request) => {
  clientLog.samplingRequest(request.params);

  try {
    const { messages, maxTokens } = request.params;

    // Convert MCP message format → Responses API input format
    const input = messages.map(({ role, content }) => ({
      role,
      content: content?.type === "text" ? content.text : JSON.stringify(content)
    }));

    const text = await completion({ model, input, max_output_tokens: maxTokens ?? 500 });
    clientLog.samplingResponse(text);

    // Return in the MCP sampling response format
    return { role: "assistant", content: { type: "text", text }, model };
  } catch (error) {
    clientLog.samplingError(error);
    throw error;
  }
};
