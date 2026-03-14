import { access } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = dirname(fileURLToPath(import.meta.url));
const filesMcpDir = join(rootDir, "mcp", "files-mcp");
const nodeModulesDir = join(filesMcpDir, "node_modules");

const hasInstalledDependencies = async () => {
  try {
    await access(nodeModulesDir);
    return true;
  } catch {
    return false;
  }
};

const getInstallerCommand = () => {
  if (process.env.npm_execpath) {
    return {
      command: process.execPath,
      args: [process.env.npm_execpath, "install", "--prefix", filesMcpDir]
    };
  }

  return {
    command: process.platform === "win32" ? "npm.cmd" : "npm",
    args: ["install", "--prefix", filesMcpDir]
  };
};

const installFilesMcp = () =>
  new Promise((resolve, reject) => {
    const { command, args } = getInstallerCommand();
    const child = spawn(command, args, { stdio: "inherit" });

    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`Failed to install files-mcp dependencies (exit code ${code ?? "unknown"}).`));
    });
  });

if (!(await hasInstalledDependencies())) {
  console.log("[ensure-files-mcp] Installing mcp/files-mcp dependencies...");
  await installFilesMcp();
}
