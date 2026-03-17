import { readFile, writeFile, mkdir, rename, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

export const ensureDir = async (dirPath) => {
  await mkdir(dirPath, { recursive: true });
};

export const resolveMarkdownPath = async (notesDir, inputFile) => {
  await ensureDir(notesDir);

  if (inputFile) {
    const candidate = path.isAbsolute(inputFile)
      ? inputFile
      : path.join(notesDir, inputFile);

    if (!candidate.endsWith(".md")) {
      throw new Error("Please provide a .md file name.");
    }

    if (!existsSync(candidate)) {
      throw new Error(`File not found: ${candidate}`);
    }

    return candidate;
  }

  const entries = await readdir(notesDir, { withFileTypes: true });
  const mdFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  if (!mdFiles.length) {
    throw new Error(`No .md files found in ${notesDir}. Add a markdown file to process.`);
  }

  return path.join(notesDir, mdFiles[0]);
};

export const readJsonIfExists = async (filePath) => {
  try {
    const text = await readFile(filePath, "utf8");
    return JSON.parse(text);
  } catch (error) {
    if (error?.code === "ENOENT") {
      return null;
    }
    throw error;
  }
};

export const safeWriteJson = async (filePath, data) => {
  await ensureDir(path.dirname(filePath));
  const tempPath = `${filePath}.tmp`;
  await writeFile(tempPath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  await rename(tempPath, filePath);
};
