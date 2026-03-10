import { mkdir, rm } from "fs/promises";
import { resolve, relative } from "path";
import { sandbox } from "../config.js";

export const initializeSandbox = async () => {
  await rm(sandbox.root, { recursive: true, force: true });
  await mkdir(sandbox.root, { recursive: true });
};

export const resolveSandboxPath = (relativePath) => {
  const resolved = resolve(sandbox.root, relativePath);
  const rel = relative(sandbox.root, resolved);

  if (rel.startsWith("..") || resolve(rel) === resolved) {
    throw new Error(`Access denied: path "${relativePath}" is outside sandbox`);
  }

  return resolved;
};
