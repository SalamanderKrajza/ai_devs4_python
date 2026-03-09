import { AI_API_KEY, EXTRA_API_HEADERS } from "../../config.js";
import { api } from "./config.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const isRetryable = (status) =>
  status === 429 || status === 503 || status === 502 || status === 500;

const buildRequestBody = ({ model, input, textFormat, tools, include, reasoning, previousResponseId }) => {
  const body = { model, input };

  const optionalFields = {
    text: textFormat ? { format: textFormat } : undefined,
    tools,
    include,
    reasoning,
    previous_response_id: previousResponseId
  };

  for (const [key, value] of Object.entries(optionalFields)) {
    if (value !== undefined) {
      body[key] = value;
    }
  }

  return body;
};

const fetchWithRetry = async (url, options) => {
  let lastError;

  for (let attempt = 0; attempt < api.retries; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), api.timeoutMs);

    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);

      if (response.ok) {
        return response;
      }

      const errorText = await response.text();
      lastError = new Error(`Responses API error (${response.status}): ${errorText}`);

      if (!isRetryable(response.status)) {
        throw lastError;
      }

      const delay = api.retryDelayMs * 2 ** attempt;
      console.warn(`  Retry ${attempt + 1}/${api.retries} after ${delay}ms (status ${response.status})`);
      await sleep(delay);
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === "AbortError") {
        lastError = new Error(`Request timed out after ${api.timeoutMs}ms`);
      } else if (error.message.includes("Responses API error")) {
        throw error;
      } else {
        lastError = error;
      }

      if (attempt < api.retries - 1) {
        const delay = api.retryDelayMs * 2 ** attempt;
        console.warn(`  Retry ${attempt + 1}/${api.retries} after ${delay}ms (${lastError.message})`);
        await sleep(delay);
      }
    }
  }

  throw lastError;
};

/**
 * Calls the configured responses API with automatic retry and timeout.
 */
export const chat = async ({ model, input, textFormat, tools, include, reasoning, previousResponseId }) => {
  if (!model || typeof model !== "string") {
    throw new Error("chat: model is required and must be a string");
  }

  if (input === undefined || input === null) {
    throw new Error("chat: input is required");
  }

  const body = buildRequestBody({ model, input, textFormat, tools, include, reasoning, previousResponseId });

  const response = await fetchWithRetry(api.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify(body)
  });

  const data = await response.json();

  if (data.error) {
    throw new Error(data.error.message);
  }

  return data;
};

/**
 * Extracts the output text from an API response.
 */
export const extractText = (response) => {
  if (typeof response?.output_text === "string" && response.output_text.trim()) {
    return response.output_text;
  }

  const messages = response?.output?.filter((item) => item.type === "message") ?? [];

  const text = messages
    .flatMap((msg) => msg.content ?? [])
    .find((part) => part.type === "output_text")?.text;

  if (!text) {
    const types = response?.output?.map((item) => item.type).join(", ") || "none";
    throw new Error(`No output_text in response. Found types: ${types}`);
  }

  return text;
};

/**
 * Extracts and parses JSON from an API response.
 */
export const extractJson = (response, label = "response") => {
  const text = extractText(response);

  try {
    return JSON.parse(text);
  } catch (error) {
    const preview = text.length > 200 ? `${text.slice(0, 200)}...` : text;
    throw new Error(`Failed to parse JSON for ${label}: ${error.message}\nOutput: ${preview}`);
  }
};

/**
 * Extracts unique web search sources from an API response.
 */
export const extractSources = (response) => {
  const calls = response?.output?.filter((item) => item.type === "web_search_call") ?? [];

  const callSources = calls
    .flatMap((call) => call?.action?.sources ?? [])
    .filter((source) => source?.url);

  const citationSources = [];

  const collectCitations = (node) => {
    if (!node || typeof node !== "object") return;

    if (Array.isArray(node)) {
      for (const item of node) {
        collectCitations(item);
      }
      return;
    }

    const citation = node.url_citation;
    if (citation?.url) {
      citationSources.push({
        title: citation.title ?? null,
        url: citation.url
      });
    }

    for (const value of Object.values(node)) {
      collectCitations(value);
    }
  };

  collectCitations(response);

  const sources = [...callSources, ...citationSources];
  const seen = new Set();

  return sources.filter((source) => {
    if (seen.has(source.url)) return false;
    seen.add(source.url);
    return true;
  });
};

// Legacy aliases for backward compatibility
export const callResponses = chat;
export const parseJsonOutput = extractJson;
export const getWebSources = extractSources;
