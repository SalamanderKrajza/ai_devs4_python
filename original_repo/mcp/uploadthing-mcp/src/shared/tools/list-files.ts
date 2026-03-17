/**
 * List files from UploadThing with optional detailed info.
 */

import { z } from 'zod';
import {
  getFileInfo,
  getUsageInfo,
  listFiles,
} from '../services/uploadthing-client.js';
import { defineTool } from './types.js';

export const listFilesInputSchema = z.object({
  limit: z
    .number()
    .int()
    .min(1)
    .max(500)
    .optional()
    .describe(
      'Maximum files to return. Default: 50, Max: 500. Use with offset for pagination.',
    ),
  offset: z
    .number()
    .int()
    .min(0)
    .optional()
    .describe(
      'Number of files to skip. Default: 0. Use with limit for pagination. Example: offset=50 skips first 50 files.',
    ),
  fileKey: z
    .string()
    .optional()
    .describe(
      'Get detailed info for a specific file. When provided, limit/offset are ignored. Use keys from previous list_files or upload_files results.',
    ),
  includeUsage: z
    .boolean()
    .optional()
    .describe(
      'Include storage usage stats (total bytes, file count, limit). Default: false. Useful for monitoring quotas.',
    ),
});

const FileInfoSchema = z.object({
  key: z.string().describe('Unique file key for use in other operations'),
  name: z.string().describe('File name'),
  size: z.number().describe('File size in bytes'),
  status: z
    .string()
    .describe('Upload status: "Uploaded" (ready), "Uploading" (in progress), "Failed", "Deletion Pending"'),
  customId: z
    .string()
    .nullable()
    .optional()
    .describe('Custom identifier if set during upload'),
  uploadedAt: z.number().describe('Upload timestamp in milliseconds since epoch'),
  url: z.string().optional().describe('File URL (included when querying specific file)'),
});

const UsageSchema = z.object({
  totalBytes: z.number().describe('Total storage used in bytes'),
  filesUploaded: z.number().describe('Total number of files'),
  limitBytes: z.number().describe('Storage limit in bytes for your plan'),
});

export const listFilesTool = defineTool({
  name: 'list_files',
  title: 'List Files',
  description: `List uploaded files from UploadThing with pagination, or get info about a specific file.

MODES:
1. List all: No fileKey → returns paginated file list
2. Single file: With fileKey → returns detailed info for that file

PAGINATION:
- Use limit (default 50, max 500) and offset (default 0)
- Check hasMore to know if more files exist
- Example: First page limit=50, second page limit=50&offset=50

TIPS:
- Use includeUsage=true to check storage quotas before uploading
- Save file keys from upload_files results for later reference
- File status "Uploaded" means ready to use`,
  inputSchema: listFilesInputSchema,
  outputSchema: {
    files: z.array(FileInfoSchema).describe('List of files matching query'),
    hasMore: z.boolean().describe('True if more files available beyond current page'),
    usage: UsageSchema.optional().describe('Storage usage (when includeUsage=true)'),
  },
  annotations: {
    title: 'List UploadThing Files',
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: true,
  },
  handler: async (args) => {
    try {
      // Single file lookup mode
      if (args.fileKey) {
        const fileInfo = await getFileInfo(args.fileKey);

        if (!fileInfo) {
          return {
            content: [
              {
                type: 'text',
                text: `NOT_FOUND: fileKey="${args.fileKey}" does not exist.\nRecovery: Call list_files without fileKey to get valid keys, then retry with correct key.`,
              },
            ],
            isError: true,
          };
        }

        const usage = args.includeUsage ? await getUsageInfo() : undefined;

        return {
          content: [
            {
              type: 'text',
              text: `File: ${fileInfo.name}\nKey: ${fileInfo.key}\nSize: ${formatBytes(fileInfo.size)}\nStatus: ${fileInfo.status}\nURL: ${fileInfo.url}`,
            },
          ],
          structuredContent: {
            files: [fileInfo],
            hasMore: false,
            usage,
          },
        };
      }

      // List mode with pagination
      const filesResult = await listFiles({
        limit: args.limit,
        offset: args.offset,
      });

      const usage = args.includeUsage ? await getUsageInfo() : undefined;

      const result = {
        files: filesResult.files.map((f) => ({
          key: f.key,
          name: f.name,
          size: f.size,
          status: f.status,
          customId: f.customId ?? null,
          uploadedAt: f.uploadedAt,
        })),
        hasMore: filesResult.hasMore,
        usage,
      };

      // Build concise text summary
      let text = `Found ${filesResult.files.length} file(s)`;
      if (filesResult.hasMore) {
        const nextOffset = (args.offset || 0) + filesResult.files.length;
        text += ` (more available, use offset=${nextOffset})`;
      }
      text += ':\n';
      text += filesResult.files
        .slice(0, 10) // Show max 10 in text to avoid noise
        .map((f) => `- ${f.name} (${formatBytes(f.size)}) [${f.key}]`)
        .join('\n');

      if (filesResult.files.length > 10) {
        text += `\n... and ${filesResult.files.length - 10} more (see structuredContent)`;
      }

      if (usage) {
        const usedPercent = Math.round((usage.totalBytes / usage.limitBytes) * 100);
        text += `\n\nStorage: ${formatBytes(usage.totalBytes)} / ${formatBytes(usage.limitBytes)} (${usedPercent}% used, ${usage.filesUploaded} files)`;
      }

      return {
        content: [{ type: 'text', text }],
        structuredContent: result,
      };
    } catch (error) {
      const err = error as Error & { code?: string };
      return formatListError(err, args);
    }
  },
});

function formatListError(
  error: Error & { code?: string },
  args: { fileKey?: string; limit?: number; offset?: number },
): { content: Array<{ type: 'text'; text: string }>; isError: true } {
  const code = error.code || 'UNKNOWN';

  const recovery: Record<string, string> = {
    FORBIDDEN: 'UPLOADTHING_TOKEN invalid. Server admin must fix token configuration.',
    NOT_FOUND: `fileKey="${args.fileKey}" not found. Call list_files without fileKey to get valid keys.`,
    BAD_REQUEST: `Invalid params. Ensure: limit is 1-500, offset >= 0. Current: limit=${args.limit}, offset=${args.offset}`,
    MISSING_ENV: 'Server misconfigured. UPLOADTHING_TOKEN env var missing.',
  };

  const rec = recovery[code] || 'Retry list_files. If persistent, server admin should check UploadThing dashboard.';

  return {
    content: [
      {
        type: 'text',
        text: `LIST_ERROR: ${code}\nMessage: ${error.message}\nRecovery: ${rec}`,
      },
    ],
    isError: true,
  };
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Number.parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`;
}
