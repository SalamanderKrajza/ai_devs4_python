import { api } from "../config.js";
import {
  AI_API_KEY,
  EXTRA_API_HEADERS,
  RESPONSES_API_ENDPOINT
} from "../../../config.js";
import { recordUsage } from "./stats.js";

const extractResponseText = (data) => {
  if (typeof data?.output_text === "string" && data.output_text.trim()) {
    return data.output_text;
  }

  const messages = Array.isArray(data?.output)
    ? data.output.filter((item) => item?.type === "message")
    : [];

  const textPart = messages
    .flatMap((message) => (Array.isArray(message?.content) ? message.content : []))
    .find((part) => part?.type === "output_text" && typeof part?.text === "string");

  return textPart?.text ?? "";
};

/**
 * Calls the configured Responses API provider.
 */
export const chat = async ({
  model = api.model,
  input,
  tools,
  toolChoice = "auto",
  instructions = api.instructions,
  maxOutputTokens = api.maxOutputTokens
}) => {
  const body = { model, input };

  if (tools?.length) body.tools = tools;
  if (tools?.length) body.tool_choice = toolChoice;
  if (instructions) body.instructions = instructions;
  if (maxOutputTokens) body.max_output_tokens = maxOutputTokens;

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
    throw new Error(data?.error?.message || `Responses API request failed (${response.status})`);
  }

  recordUsage(data.usage);
  return data;
};

export const extractToolCalls = (response) =>
  (response.output ?? []).filter((item) => item.type === "function_call");

export const extractText = (response) => extractResponseText(response) || null;
