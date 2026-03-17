/**
 * Manage files on UploadThing: delete, rename, update ACL, get URL.
 */

import { z } from 'zod';
import {
  deleteFiles,
  getFileUrl,
  renameFile,
  updateFileAcl,
} from '../services/uploadthing-client.js';
import { defineTool } from './types.js';

export const manageFilesInputSchema = z.object({
  action: z
    .enum(['delete', 'rename', 'update_acl', 'get_url'])
    .describe(
      'Action to perform: "delete" removes files, "rename" changes file name, "update_acl" changes visibility, "get_url" gets signed access URL for private files',
    ),
  fileKeys: z
    .array(z.string())
    .optional()
    .describe(
      'File keys to delete (only for action="delete"). Get keys from upload_files or list_files results. Can delete multiple files at once.',
    ),
  fileKey: z
    .string()
    .optional()
    .describe(
      'Single file key (required for rename, update_acl, get_url). Get key from upload_files or list_files results.',
    ),
  newName: z
    .string()
    .optional()
    .describe(
      'New file name with extension (required for action="rename"). Example: "new-photo.jpg"',
    ),
  acl: z
    .enum(['public-read', 'private'])
    .optional()
    .describe(
      'Access control (required for action="update_acl"). "public-read": anyone can access via URL. "private": requires signed URL from get_url action.',
    ),
});

export const manageFilesTool = defineTool({
  name: 'manage_files',
  title: 'Manage Files',
  description: `Manage uploaded files on UploadThing.

ACTIONS:
1. delete: Remove files permanently (use fileKeys array)
2. rename: Change file name (use fileKey + newName)
3. update_acl: Change visibility (use fileKey + acl)
4. get_url: Get signed URL for private file access (use fileKey)

REQUIRED PARAMS PER ACTION:
- delete: fileKeys (array of keys)
- rename: fileKey + newName
- update_acl: fileKey + acl ("public-read" or "private")
- get_url: fileKey

TIPS:
- Get file keys from upload_files or list_files results
- delete is permanent and cannot be undone
- Private files need get_url to generate temporary access URLs
- Renamed files keep the same key`,
  inputSchema: manageFilesInputSchema,
  outputSchema: {
    success: z.boolean().describe('Whether the operation succeeded'),
    action: z.string().describe('The action that was performed'),
    message: z.string().describe('Human-readable result message'),
    data: z.record(z.unknown()).optional().describe('Action-specific result data'),
  },
  annotations: {
    title: 'Manage UploadThing Files',
    readOnlyHint: false,
    destructiveHint: true,
    idempotentHint: false,
    openWorldHint: true,
  },
  handler: async (args) => {
    const { action } = args;

    try {
      switch (action) {
        case 'delete':
          return await handleDelete(args);
        case 'rename':
          return await handleRename(args);
        case 'update_acl':
          return await handleUpdateAcl(args);
        case 'get_url':
          return await handleGetUrl(args);
        default:
          return {
            content: [
              {
                type: 'text',
                text: `Unknown action: ${action}\n\nValid actions: delete, rename, update_acl, get_url`,
              },
            ],
            isError: true,
          };
      }
    } catch (error) {
      const err = error as Error & { code?: string };
      return formatManageError(err, action, args);
    }
  },
});

type ManageArgs = {
  action: 'delete' | 'rename' | 'update_acl' | 'get_url';
  fileKeys?: string[];
  fileKey?: string;
  newName?: string;
  acl?: 'public-read' | 'private';
};

type ToolResponse = {
  content: Array<{ type: 'text'; text: string }>;
  structuredContent?: Record<string, unknown>;
  isError?: boolean;
};

async function handleDelete(args: ManageArgs): Promise<ToolResponse> {
  if (!args.fileKeys || args.fileKeys.length === 0) {
    return {
      content: [
        {
          type: 'text',
          text: `MISSING_PARAM: fileKeys required for action="delete".\nRecovery: Call list_files to get file keys, then call manage_files with action="delete" and fileKeys=["key1","key2"]`,
        },
      ],
      isError: true,
    };
  }

  const result = await deleteFiles(args.fileKeys);

  const message =
    result.deletedCount === args.fileKeys.length
      ? `Successfully deleted ${result.deletedCount} file(s).`
      : `Deleted ${result.deletedCount} of ${args.fileKeys.length} files. Some files may not have existed.`;

  return {
    content: [{ type: 'text', text: message }],
    structuredContent: {
      success: true,
      action: 'delete',
      message,
      data: { deletedCount: result.deletedCount, requestedCount: args.fileKeys.length },
    },
  };
}

async function handleRename(args: ManageArgs): Promise<ToolResponse> {
  if (!args.fileKey) {
    return {
      content: [
        {
          type: 'text',
          text: `MISSING_PARAM: fileKey required for action="rename".\nRecovery: Call list_files to get file keys, then call manage_files with action="rename", fileKey="<key>", newName="<name>"`,
        },
      ],
      isError: true,
    };
  }

  if (!args.newName) {
    return {
      content: [
        {
          type: 'text',
          text: `MISSING_PARAM: newName required for action="rename".\nRecovery: Call manage_files with action="rename", fileKey="${args.fileKey}", newName="<new-filename.ext>"`,
        },
      ],
      isError: true,
    };
  }

  await renameFile(args.fileKey, args.newName);

  const message = `Renamed file to "${args.newName}". The file key remains the same: ${args.fileKey}`;

  return {
    content: [{ type: 'text', text: message }],
    structuredContent: {
      success: true,
      action: 'rename',
      message,
      data: { fileKey: args.fileKey, newName: args.newName },
    },
  };
}

async function handleUpdateAcl(args: ManageArgs): Promise<ToolResponse> {
  if (!args.fileKey) {
    return {
      content: [
        {
          type: 'text',
          text: `MISSING_PARAM: fileKey required for action="update_acl".\nRecovery: Call list_files to get file keys, then call manage_files with action="update_acl", fileKey="<key>", acl="public-read"|"private"`,
        },
      ],
      isError: true,
    };
  }

  if (!args.acl) {
    return {
      content: [
        {
          type: 'text',
          text: `MISSING_PARAM: acl required for action="update_acl".\nRecovery: Call manage_files with action="update_acl", fileKey="${args.fileKey}", acl="public-read" or acl="private"`,
        },
      ],
      isError: true,
    };
  }

  await updateFileAcl(args.fileKey, args.acl);

  const accessNote =
    args.acl === 'private'
      ? 'The file now requires a signed URL for access. Use action "get_url" to generate one.'
      : 'The file is now publicly accessible via its URL.';

  const message = `Updated file ACL to "${args.acl}". ${accessNote}`;

  return {
    content: [{ type: 'text', text: message }],
    structuredContent: {
      success: true,
      action: 'update_acl',
      message,
      data: { fileKey: args.fileKey, acl: args.acl },
    },
  };
}

async function handleGetUrl(args: ManageArgs): Promise<ToolResponse> {
  if (!args.fileKey) {
    return {
      content: [
        {
          type: 'text',
          text: `MISSING_PARAM: fileKey required for action="get_url".\nRecovery: Call list_files to get file keys, then call manage_files with action="get_url", fileKey="<key>"`,
        },
      ],
      isError: true,
    };
  }

  const result = await getFileUrl(args.fileKey);

  const message = `Signed URL generated for file ${args.fileKey}:\n${result.url}\n\nNote: This URL provides temporary access. For public files, consider using update_acl with "public-read".`;

  return {
    content: [{ type: 'text', text: message }],
    structuredContent: {
      success: true,
      action: 'get_url',
      message: 'Signed URL generated',
      data: { fileKey: args.fileKey, url: result.url },
    },
  };
}

function formatManageError(
  error: Error & { code?: string },
  action: string,
  args: ManageArgs,
): ToolResponse {
  const code = error.code || 'UNKNOWN';
  const fileKey = args.fileKey || args.fileKeys?.join(',') || 'none';

  const recovery: Record<string, string> = {
    NOT_FOUND: `fileKey="${fileKey}" not found. Call list_files to get valid keys, then retry.`,
    FORBIDDEN: 'UPLOADTHING_TOKEN lacks permission. Server admin must update token.',
    BAD_REQUEST: `Invalid params for action="${action}". Check required params: delete→fileKeys, rename→fileKey+newName, update_acl→fileKey+acl, get_url→fileKey`,
    MISSING_ENV: 'Server misconfigured. UPLOADTHING_TOKEN env var missing.',
  };

  const rec = recovery[code] || `Retry manage_files. Verify fileKey with list_files first.`;

  return {
    content: [
      {
        type: 'text',
        text: `MANAGE_ERROR: ${code}\nAction: ${action}\nFileKey: ${fileKey}\nMessage: ${error.message}\nRecovery: ${rec}`,
      },
    ],
    isError: true,
  };
}
