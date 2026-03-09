import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MIN_NODE_VERSION = 24;
const ROOT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT_ENV_FILE = path.join(ROOT_DIR, ".env");
const RESPONSES_ENDPOINTS = {
  openai: "https://api.openai.com/v1/responses",
  openrouter: "https://openrouter.ai/api/v1/responses"
};
const VALID_PROVIDERS = new Set(["openai", "openrouter"]);

const [major] = process.versions.node.split(".").map(Number);
if (major < MIN_NODE_VERSION) {
  console.error(`\x1b[31mError: Node.js ${MIN_NODE_VERSION}+ is required\x1b[0m`);
  console.error(`       Current version: ${process.versions.node}`);
  console.error("       Please upgrade: https://nodejs.org/");
  process.exit(1);
}

if (existsSync(ROOT_ENV_FILE) && typeof process.loadEnvFile === "function") {
  try {
    process.loadEnvFile(ROOT_ENV_FILE);
  } catch (error) {
    console.error("\x1b[31mError: Failed to load .env file\x1b[0m");
    console.error(`       File: ${ROOT_ENV_FILE}`);
    console.error(`       Reason: ${error.message}`);
    process.exit(1);
  }
}

const OPENAI_API_KEY = process.env.OPENAI_API_KEY?.trim() ?? "";
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY?.trim() ?? "";
const requestedProvider = process.env.AI_PROVIDER?.trim().toLowerCase() ?? "";
const hasOpenAIKey = Boolean(OPENAI_API_KEY);
const hasOpenRouterKey = Boolean(OPENROUTER_API_KEY);

if (!hasOpenAIKey && !hasOpenRouterKey) {
  console.error("\x1b[31mError: API key is not set\x1b[0m");
  console.error(`       Create: ${ROOT_ENV_FILE}`);
  console.error("       Add one of:");
  console.error("       OPENAI_API_KEY=sk-...");
  console.error("       OPENROUTER_API_KEY=sk-or-v1-...");
  process.exit(1);
}

if (requestedProvider && !VALID_PROVIDERS.has(requestedProvider)) {
  console.error("\x1b[31mError: AI_PROVIDER must be one of: openai, openrouter\x1b[0m");
  process.exit(1);
}

const resolveProvider = () => {
  if (requestedProvider) {
    if (requestedProvider === "openai" && !hasOpenAIKey) {
      console.error("\x1b[31mError: AI_PROVIDER=openai requires OPENAI_API_KEY\x1b[0m");
      process.exit(1);
    }

    if (requestedProvider === "openrouter" && !hasOpenRouterKey) {
      console.error("\x1b[31mError: AI_PROVIDER=openrouter requires OPENROUTER_API_KEY\x1b[0m");
      process.exit(1);
    }

    return requestedProvider;
  }

  return hasOpenAIKey ? "openai" : "openrouter";
};

export const AI_PROVIDER = resolveProvider();
export const AI_API_KEY = AI_PROVIDER === "openai" ? OPENAI_API_KEY : OPENROUTER_API_KEY;
export const RESPONSES_API_ENDPOINT = RESPONSES_ENDPOINTS[AI_PROVIDER];
export const EXTRA_API_HEADERS = AI_PROVIDER === "openrouter"
  ? {
      ...(process.env.OPENROUTER_HTTP_REFERER
        ? { "HTTP-Referer": process.env.OPENROUTER_HTTP_REFERER }
        : {}),
      ...(process.env.OPENROUTER_APP_NAME
        ? { "X-Title": process.env.OPENROUTER_APP_NAME }
        : {})
    }
  : {};

export const resolveModelForProvider = (model) => {
  if (typeof model !== "string" || !model.trim()) {
    throw new Error("Model must be a non-empty string");
  }

  if (AI_PROVIDER !== "openrouter" || model.includes("/")) {
    return model;
  }

  return model.startsWith("gpt-") ? `openai/${model}` : model;
};

// Backward-compatible alias used in existing examples.
export { OPENAI_API_KEY, OPENROUTER_API_KEY };
