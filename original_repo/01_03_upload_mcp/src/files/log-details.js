/**
 * Logging helpers for file upload and read operations.
 *
 * Shows previews of uploaded file content and first lines of read files
 * so the user can verify what the agent is doing.
 */

import log from "../helpers/logger.js";

const decodePreview = (base64, maxLen = 80) => {
  try {
    const decoded = Buffer.from(base64, "base64").toString("utf-8");
    return decoded.length > maxLen ? decoded.slice(0, maxLen) + "…" : decoded;
  } catch {
    return "[binary]";
  }
};

export const logUploadDetails = (args) => {
  if (!args.files?.length) return;

  for (const file of args.files) {
    const preview = decodePreview(file.base64);
    log.info(`  📤 ${file.name} (${file.type}) — ${preview}`);
  }
};

export const logReadDetails = (result) => {
  if (!result?.content?.text) return;

  const firstLine = result.content.text.split("\n")[0]?.replace(/^\d+\|/, "").trim();
  if (firstLine) {
    log.info(`  📥 ${firstLine}`);
  }
};
