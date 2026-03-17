/**
 * Upload files to UploadThing.
 * Accepts base64-encoded files and uploads them to UploadThing.
 */

import { z } from 'zod';
import { uploadFiles } from '../services/uploadthing-client.js';
import { defineTool } from './types.js';

const fileInputSchema = z.object({
  base64: z
    .string()
    .describe(
      'Base64-encoded file content (raw base64, without data: URI prefix). Example: "iVBORw0KGgo..."',
    ),
  name: z
    .string()
    .describe(
      'File name with extension. Must include valid extension matching the type. Example: "photo.jpg", "document.pdf"',
    ),
  type: z
    .string()
    .describe(
      'MIME type of the file. Supported types: image/* (jpeg, png, gif, webp), video/*, audio/*, application/pdf, text/*. Example: "image/png"',
    ),
  customId: z
    .string()
    .max(128)
    .optional()
    .describe(
      'Optional custom identifier for later retrieval (max 128 chars). Useful for linking files to your own records.',
    ),
});

export const uploadFilesInputSchema = z.object({
  files: z
    .array(fileInputSchema)
    .min(1)
    .max(10)
    .describe(
      'Array of files to upload. Minimum 1, maximum 10 files per request. Total size should not exceed your plan limits.',
    ),
  acl: z
    .enum(['public-read', 'private'])
    .optional()
    .describe(
      'Access control. "public-read" (default): Anyone with URL can access. "private": Requires signed URL to access.',
    ),
});

const UploadedFileSchema = z.object({
  key: z.string().describe('Unique file key for referencing this file in other operations'),
  url: z.string().describe('Direct URL to access the file (public files) or base URL (private files)'),
  name: z.string().describe('Original file name as uploaded'),
  size: z.number().describe('File size in bytes'),
});

export const uploadFilesTool = defineTool({
  name: 'upload_files',
  title: 'Upload Files',
  description: `Upload files to UploadThing cloud storage.

INPUTS:
- files: Array of {base64, name, type} objects (1-10 files)
- acl: Optional "public-read" or "private"

RETURNS: Array of uploaded file info with keys and URLs.

TIPS:
- Base64 should be raw encoded content, not a data: URI
- File name must have correct extension matching MIME type
- Use customId to link files to your records for easy retrieval
- For private files, use manage_files with action "get_url" to get signed access URLs`,
  inputSchema: uploadFilesInputSchema,
  outputSchema: {
    files: z.array(UploadedFileSchema).describe('Successfully uploaded files'),
    count: z.number().describe('Number of files uploaded'),
  },
  annotations: {
    title: 'Upload Files to UploadThing',
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: true,
  },
  handler: async (args) => {
    try {
      const uploaded = await uploadFiles(args.files, { acl: args.acl });

      const result = {
        files: uploaded.map((f) => ({
          key: f.key,
          url: f.url,
          name: f.name,
          size: f.size,
        })),
        count: uploaded.length,
      };

      const summary = uploaded.map((f) => `- ${f.name}: ${f.url}`).join('\n');

      return {
        content: [
          {
            type: 'text',
            text: `Successfully uploaded ${uploaded.length} file(s):\n${summary}`,
          },
        ],
        structuredContent: result,
      };
    } catch (error) {
      const err = error as Error & { code?: string; message: string };
      return formatUploadError(err, args.files);
    }
  },
});

function formatUploadError(
  error: Error & { code?: string },
  files: Array<{ name: string; type: string }>,
): { content: Array<{ type: 'text'; text: string }>; isError: true } {
  const code = error.code || 'UNKNOWN';
  const fileNames = files.map((f) => f.name).join(', ');

  const recovery: Record<string, { action: string; params?: Record<string, unknown> }> = {
    TOO_LARGE: {
      action: 'Reduce file size below plan limit, then retry upload_files with smaller file',
    },
    TOO_SMALL: {
      action: 'Verify base64 is not empty. Re-encode the source file and retry upload_files',
    },
    TOO_MANY_FILES: {
      action: 'Split into multiple upload_files calls with max 10 files each',
    },
    FORBIDDEN: {
      action: 'UPLOADTHING_TOKEN invalid or expired. Server admin must update token',
    },
    BAD_REQUEST: {
      action: 'Check: 1) MIME type matches extension, 2) base64 has no "data:" prefix, 3) no line breaks in base64',
    },
    UPLOAD_FAILED: {
      action: 'Retry upload_files. If persistent, verify file is not corrupted',
    },
    FILE_LIMIT_EXCEEDED: {
      action: 'Free space first: call manage_files with action="delete" and fileKeys of unused files, then retry',
      params: { suggestedTool: 'manage_files', suggestedAction: 'delete' },
    },
  };

  const rec = recovery[code] || { action: 'Retry upload_files. Check file content is valid' };

  return {
    content: [
      {
        type: 'text',
        text: `UPLOAD_ERROR: ${code}\nFiles: ${fileNames}\nMessage: ${error.message}\nRecovery: ${rec.action}`,
      },
    ],
    isError: true,
  };
}
