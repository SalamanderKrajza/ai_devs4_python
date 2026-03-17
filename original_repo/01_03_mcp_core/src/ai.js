/**
 * AI provider client — calls the Responses API (OpenAI / OpenRouter).
 *
 * Thin wrapper around the Responses API. Sends a completion request
 * and returns the text output. Provider/key selection is handled by
 * the root config.js based on environment variables.
 */

import {
  AI_API_KEY,
  EXTRA_API_HEADERS,
  RESPONSES_API_ENDPOINT
} from "../../config.js";

// The Responses API can return text in two places:
//   1. data.output_text (direct shorthand)
//   2. data.output[].content[].text (nested message format)
const extractText = (data) => {
  if (typeof data?.output_text === "string") {
    return data.output_text.trim();
  }

  const message = data?.output?.find((o) => o?.type === "message");
  const part = message?.content?.find((c) => c?.type === "output_text");
  return part?.text?.trim() ?? "";
};

export const completion = async ({ model, input, max_output_tokens }) => {
  const response = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify({ model, input, max_output_tokens })
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(data?.error?.message || `API request failed (${response.status})`);
  }

  const text = extractText(data);
  if (!text) throw new Error("Empty response");

  return text;
};
