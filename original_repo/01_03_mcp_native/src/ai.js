/**
 * AI provider client — calls the Responses API (OpenAI / OpenRouter).
 *
 * Handles the request/response cycle and text extraction.
 * Provider/key selection is handled by the root config.js.
 */

import {
  AI_API_KEY,
  EXTRA_API_HEADERS,
  RESPONSES_API_ENDPOINT
} from "../../config.js";

// The Responses API returns text in two possible locations
const extractResponseText = (data) => {
  if (typeof data?.output_text === "string") {
    return data.output_text.trim();
  }

  const message = data?.output?.find((o) => o?.type === "message");
  const part = message?.content?.find((c) => c?.type === "output_text");
  return part?.text?.trim() ?? "";
};

/**
 * Sends a chat request to the Responses API.
 * Returns the raw response (caller extracts tool calls or text).
 */
export const chat = async ({ model, input, tools, toolChoice = "auto", instructions }) => {
  const body = { model, input };
  if (tools?.length) { body.tools = tools; body.tool_choice = toolChoice; }
  if (instructions) body.instructions = instructions;

  const response = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify(body)
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(data?.error?.message || `API request failed (${response.status})`);
  }

  return data;
};

export const extractToolCalls = (response) =>
  (response.output ?? []).filter((item) => item.type === "function_call");

export const extractText = (response) =>
  extractResponseText(response) || null;
