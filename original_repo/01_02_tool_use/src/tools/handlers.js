import { readdir, readFile, writeFile, unlink, mkdir, stat } from "fs/promises";
import { resolveSandboxPath } from "../utils/sandbox.js";

export const handlers = {
  async list_files({ path }) {
    const fullPath = resolveSandboxPath(path);
    const entries = await readdir(fullPath, { withFileTypes: true });

    return entries.map((entry) => ({
      name: entry.name,
      type: entry.isDirectory() ? "directory" : "file"
    }));
  },

  async read_file({ path }) {
    const fullPath = resolveSandboxPath(path);
    const content = await readFile(fullPath, "utf-8");
    return { content };
  },

  async write_file({ path, content }) {
    const fullPath = resolveSandboxPath(path);
    await writeFile(fullPath, content, "utf-8");
    return { success: true, message: `File written: ${path}` };
  },

  async delete_file({ path }) {
    const fullPath = resolveSandboxPath(path);
    await unlink(fullPath);
    return { success: true, message: `File deleted: ${path}` };
  },

  async create_directory({ path }) {
    const fullPath = resolveSandboxPath(path);
    await mkdir(fullPath, { recursive: true });
    return { success: true, message: `Directory created: ${path}` };
  },

  async file_info({ path }) {
    const fullPath = resolveSandboxPath(path);
    const stats = await stat(fullPath);

    return {
      size: stats.size,
      isDirectory: stats.isDirectory(),
      created: stats.birthtime.toISOString(),
      modified: stats.mtime.toISOString()
    };
  }
};
