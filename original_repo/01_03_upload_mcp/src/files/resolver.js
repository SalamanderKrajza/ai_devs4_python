/**
 * File reference resolver — replaces {{file:path}} placeholders with base64.
 *
 * The model references workspace files using {{file:path}} syntax in tool
 * arguments. Before calling the MCP server, this resolver walks the argument
 * tree and replaces each placeholder with the actual base64-encoded content.
 * This avoids the model having to read and encode files itself.
 */

import { readFile } from "fs/promises";
import { join } from "path";
import log from "../helpers/logger.js";

const FILE_REF_PATTERN = /\{\{file:([^}]+)\}\}/g;

const readAsBase64 = async (workspaceRoot, relativePath) => {
  const fullPath = join(workspaceRoot, relativePath);
  const buffer = await readFile(fullPath);
  return buffer.toString("base64");
};

const resolveInString = async (str, workspaceRoot) => {
  const matches = [...str.matchAll(FILE_REF_PATTERN)];
  if (matches.length === 0) return str;

  let result = str;
  for (const match of matches) {
    const [placeholder, filePath] = match;
    try {
      const base64 = await readAsBase64(workspaceRoot, filePath);
      log.info(`   📎 Resolved: ${filePath} → ${base64.length} chars`);
      result = result.replace(placeholder, base64);
    } catch (error) {
      log.warn(`   ⚠️ Failed: ${filePath} - ${error.message}`);
    }
  }
  return result;
};

/**
 * Recursively resolves {{file:path}} in strings, arrays, and objects.
 */
export const resolveFileRefs = async (value, workspaceRoot) => {
  if (typeof value === "string") {
    return resolveInString(value, workspaceRoot);
  }
  if (Array.isArray(value)) {
    return Promise.all(value.map((v) => resolveFileRefs(v, workspaceRoot)));
  }
  if (typeof value === "object" && value !== null) {
    const entries = await Promise.all(
      Object.entries(value).map(async ([k, v]) => [k, await resolveFileRefs(v, workspaceRoot)])
    );
    return Object.fromEntries(entries);
  }
  return value;
};
