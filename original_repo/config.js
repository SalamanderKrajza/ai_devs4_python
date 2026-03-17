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
const EMBEDDINGS_ENDPOINTS = {
  openai: "https://api.openai.com/v1/embeddings",
  openrouter: "https://openrouter.ai/api/v1/embeddings"
};
const CHAT_API_BASE_URLS = {
  openai: "https://api.openai.com/v1",
  openrouter: "https://openrouter.ai/api/v1"
};
const OPENROUTER_ONLINE_SUFFIX = ":online";
const VALID_OPENAI_SEARCH_CONTEXT_SIZES = new Set(["low", "medium", "high"]);
const VALID_OPENROUTER_WEB_ENGINES = new Set(["native", "exa"]);
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
export const EMBEDDINGS_API_ENDPOINT = EMBEDDINGS_ENDPOINTS[AI_PROVIDER];
export const CHAT_API_BASE_URL = CHAT_API_BASE_URLS[AI_PROVIDER];
export const OPENROUTER_EXTRA_HEADERS = {
  ...(process.env.OPENROUTER_HTTP_REFERER
    ? { "HTTP-Referer": process.env.OPENROUTER_HTTP_REFERER }
    : {}),
  ...(process.env.OPENROUTER_APP_NAME
    ? { "X-Title": process.env.OPENROUTER_APP_NAME }
    : {})
};
export const EXTRA_API_HEADERS = AI_PROVIDER === "openrouter"
  ? OPENROUTER_EXTRA_HEADERS
  : {};

const isPlainObject = (value) =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const ensureTrimmedString = (value, fieldName) => {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${fieldName} must be a non-empty string`);
  }

  return value.trim();
};

const normalizeOpenRouterOnlineModel = (model) =>
  model.endsWith(OPENROUTER_ONLINE_SUFFIX)
    ? model
    : `${model}${OPENROUTER_ONLINE_SUFFIX}`;

const stripOpenRouterOnlineSuffix = (model) =>
  model.endsWith(OPENROUTER_ONLINE_SUFFIX)
    ? model.slice(0, -OPENROUTER_ONLINE_SUFFIX.length)
    : model;

export const resolveModelForProvider = (model) => {
  if (typeof model !== "string" || !model.trim()) {
    throw new Error("Model must be a non-empty string");
  }

  if (AI_PROVIDER !== "openrouter" || model.includes("/")) {
    return model;
  }

  return `openai/${model}`;
};

const normalizeWebSearchConfig = (webSearch) => {
  if (!webSearch) {
    return null;
  }

  if (webSearch === true) {
    return {};
  }

  if (!isPlainObject(webSearch)) {
    throw new Error("webSearch must be either boolean or an object");
  }

  if (webSearch.enabled === false) {
    return null;
  }

  const config = {};

  if (webSearch.searchContextSize !== undefined) {
    const searchContextSize = ensureTrimmedString(
      webSearch.searchContextSize,
      "webSearch.searchContextSize"
    );

    if (!VALID_OPENAI_SEARCH_CONTEXT_SIZES.has(searchContextSize)) {
      throw new Error('webSearch.searchContextSize must be one of: "low", "medium", "high"');
    }

    config.searchContextSize = searchContextSize;
  }

  if (webSearch.engine !== undefined) {
    const engine = ensureTrimmedString(webSearch.engine, "webSearch.engine");

    if (!VALID_OPENROUTER_WEB_ENGINES.has(engine)) {
      throw new Error('webSearch.engine must be one of: "native", "exa"');
    }

    config.engine = engine;
  }

  if (webSearch.maxResults !== undefined) {
    if (!Number.isInteger(webSearch.maxResults) || webSearch.maxResults <= 0) {
      throw new Error("webSearch.maxResults must be a positive integer");
    }

    config.maxResults = webSearch.maxResults;
  }

  if (webSearch.searchPrompt !== undefined) {
    config.searchPrompt = ensureTrimmedString(
      webSearch.searchPrompt,
      "webSearch.searchPrompt"
    );
  }

  return config;
};

const addUniqueTool = (tools, tool) => {
  if (!Array.isArray(tools) || tools.length === 0) {
    return [tool];
  }

  return tools.some((candidate) => candidate?.type === tool.type)
    ? tools
    : [...tools, tool];
};

const mergeOpenRouterPlugins = (plugins, plugin) => {
  if (!Array.isArray(plugins) || plugins.length === 0) {
    return [plugin];
  }

  const existingIndex = plugins.findIndex((candidate) => candidate?.id === plugin.id);

  if (existingIndex === -1) {
    return [...plugins, plugin];
  }

  const mergedPlugin = { ...plugins[existingIndex], ...plugin };
  return plugins.map((candidate, index) => (
    index === existingIndex ? mergedPlugin : candidate
  ));
};

export const buildResponsesRequest = ({ model, tools, plugins, webSearch = false, ...rest }) => {
  const request = {
    model: resolveModelForProvider(model),
    ...rest
  };

  if (tools) {
    request.tools = tools;
  }

  if (plugins) {
    request.plugins = plugins;
  }

  const webSearchConfig = normalizeWebSearchConfig(webSearch);

  if (!webSearchConfig) {
    return request;
  }

  if (AI_PROVIDER === "openrouter") {
    const hasPluginOverrides = (
      webSearchConfig.engine !== undefined
      || webSearchConfig.maxResults !== undefined
      || webSearchConfig.searchPrompt !== undefined
    );

    if (!hasPluginOverrides) {
      request.model = normalizeOpenRouterOnlineModel(request.model);
      return request;
    }

    request.model = stripOpenRouterOnlineSuffix(request.model);
    request.plugins = mergeOpenRouterPlugins(request.plugins, {
      id: "web",
      ...(webSearchConfig.engine ? { engine: webSearchConfig.engine } : {}),
      ...(webSearchConfig.maxResults ? { max_results: webSearchConfig.maxResults } : {}),
      ...(webSearchConfig.searchPrompt ? { search_prompt: webSearchConfig.searchPrompt } : {})
    });

    return request;
  }

  request.tools = addUniqueTool(request.tools, { type: "web_search_preview" });

  if (webSearchConfig.searchContextSize) {
    request.web_search_options = {
      search_context_size: webSearchConfig.searchContextSize
    };
  }

  return request;
};

// Backward-compatible alias used in existing examples.
export { OPENAI_API_KEY, OPENROUTER_API_KEY };
