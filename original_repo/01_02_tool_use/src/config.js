import { resolve } from "path";
import { mkdir } from "fs/promises";
import { resolveModelForProvider } from "../../config.js";

export const sandbox = {
  root: resolve(import.meta.dirname, "..", "sandbox")
};

await mkdir(sandbox.root, { recursive: true });

export const api = {
  model: resolveModelForProvider("gpt-4.1"),
  instructions: `You are a helpful assistant with access to a sandboxed filesystem. 
You can list, read, write, and delete files within the sandbox.
Always use the available tools to interact with files.
Be concise in your responses.`
};
