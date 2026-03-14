import { translator as config } from "./config.js";
import { run } from "./agent.js";
import { callMcpTool } from "./mcp/client.js";
import log from "./helpers/logger.js";
import { logStats } from "./helpers/stats.js";

const MAX_TRANSLATIONS = 3;

const inProgress = new Set();
const loggedSkipped = new Set();
let completedCount = 0;

/**
 * Lists files in a directory.
 */
const listFiles = async (mcpClient, dir, filterByExtension = false) => {
  try {
    const result = await callMcpTool(mcpClient, "fs_read", { path: dir, mode: "list" });
    if (!result.entries) return [];
    
    const getName = (e) => e.name || e.path?.split(/[/\\]/).pop();
    
    return result.entries
      .filter(e => e.kind === "file" || e.type === "file")
      .filter(e => !filterByExtension || config.supportedExtensions.some(ext => getName(e)?.endsWith(ext)))
      .map(getName);
  } catch {
    return [];
  }
};

/**
 * Translates a single file using the agent.
 */
const translateFile = async (filename, mcpClient, mcpTools) => {
  // Skip if already in progress (only log once per file)
  if (inProgress.has(filename)) {
    if (!loggedSkipped.has(filename)) {
      log.debug(`${filename} - translation in progress, waiting...`);
      loggedSkipped.add(filename);
    }
    return null;
  }
  
  // Clear the "logged skipped" flag when starting fresh
  loggedSkipped.delete(filename);
  inProgress.add(filename);
  
  const sourcePath = `${config.sourceDir}/${filename}`;
  const targetPath = `${config.targetDir}/${filename}`;
  
  log.info(`📄 Translating: ${filename}`);
  
  const prompt = `Translate "${sourcePath}" to English and save to "${targetPath}".`;

  try {
    const result = await run(prompt, { mcpClient, mcpTools });
    completedCount++;
    log.success(`✅ Translated: ${filename} (${completedCount}/${MAX_TRANSLATIONS})`);
    logStats();
    return result;
  } catch (error) {
    log.error(`Translation failed: ${filename}`, error.message);
    return null;
  } finally {
    inProgress.delete(filename);
  }
};

/**
 * Ensures required directories exist.
 */
const ensureDirectories = async (mcpClient) => {
  try {
    await callMcpTool(mcpClient, "fs_manage", {
      operation: "mkdir",
      path: config.sourceDir,
      recursive: true
    });
  } catch { /* ignore */ }
  
  try {
    await callMcpTool(mcpClient, "fs_manage", {
      operation: "mkdir", 
      path: config.targetDir,
      recursive: true
    });
  } catch { /* ignore */ }
};

/**
 * Main watch loop - detects files and asks agent to translate.
 */
export const runTranslationLoop = async (mcpClient, mcpTools) => {
  log.start(`Watching ${config.sourceDir} (every ${config.pollInterval}ms)`);
  log.info(`Output: ${config.targetDir}`);
  
  await ensureDirectories(mcpClient);
  
  const tick = async () => {
    try {
      const sourceFiles = await listFiles(mcpClient, config.sourceDir, true);
      const translatedFiles = await listFiles(mcpClient, config.targetDir);
      const pending = sourceFiles.filter(f => !translatedFiles.includes(f));
      
      for (const filename of pending) {
        if (completedCount >= MAX_TRANSLATIONS) {
          log.warn(`Reached translation limit (${MAX_TRANSLATIONS}). Restart the script to continue.`);
          return;
        }
        await translateFile(filename, mcpClient, mcpTools);
      }
    } catch (error) {
      log.error("Watch loop error", error.message);
    }
  };
  
  // Initial run
  await tick();
  
  // Start polling
  setInterval(tick, config.pollInterval);
};
