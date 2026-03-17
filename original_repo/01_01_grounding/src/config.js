import path from "node:path";
import { fileURLToPath } from "node:url";
import { RESPONSES_API_ENDPOINT, resolveModelForProvider } from "../../config.js";

const DIRNAME = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(DIRNAME, "..");
const ROOT_DIR = path.resolve(PROJECT_DIR, "..");

export const paths = {
  root: ROOT_DIR,
  project: PROJECT_DIR,
  notes: path.join(PROJECT_DIR, "notes"),
  output: path.join(PROJECT_DIR, "output"),
  template: path.join(PROJECT_DIR, "template.html"),
  concepts: path.join(PROJECT_DIR, "output", "concepts.json"),
  dedupe: path.join(PROJECT_DIR, "output", "dedupe.json"),
  search: path.join(PROJECT_DIR, "output", "search_results.json"),
  grounded: path.join(PROJECT_DIR, "output", "grounded.html")
};

export const models = {
  extract: resolveModelForProvider("gpt-5.4"),
  search: resolveModelForProvider("gpt-5.4"),
  ground: resolveModelForProvider("gpt-5.4")
};

export const api = {
  endpoint: RESPONSES_API_ENDPOINT,
  timeoutMs: 180_000,
  retries: 3,
  retryDelayMs: 1000
};

const isFlag = (arg) => arg.startsWith("--");
const args = process.argv.slice(2);

const parseBatchSize = () => {
  if (args.includes("--no-batch")) {
    return 1;
  }

  const batchArg = args.find((arg) => arg.startsWith("--batch="));

  if (batchArg) {
    const value = parseInt(batchArg.split("=")[1], 10);
    return Number.isNaN(value) || value < 1 ? 3 : Math.min(value, 10);
  }

  return 3; // default batch size
};

export const cli = {
  force: args.includes("--force"),
  inputFile: args.find((arg) => !isFlag(arg)) ?? null,
  batchSize: parseBatchSize()
};
