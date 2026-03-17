/**
 * Centralized tool metadata for the UploadThing MCP server.
 */

export interface ToolMetadata {
  name: string;
  title: string;
  description: string;
}

export const toolsMetadata = {
  upload_files: {
    name: 'upload_files',
    title: 'Upload Files',
    description: `Upload one or more files to UploadThing.

Accepts base64-encoded file content with name and MIME type.
Returns URLs and file keys for accessing the uploaded files.

Features:
- Upload up to 10 files at once
- Set ACL (public-read or private)
- Assign custom IDs for easy retrieval`,
  },
  list_files: {
    name: 'list_files',
    title: 'List Files',
    description: `List uploaded files from UploadThing with pagination.

Options:
- Paginate with limit/offset
- Get detailed info for a specific file by key
- Include storage usage statistics

Returns file metadata including key, name, size, status, and upload time.`,
  },
  manage_files: {
    name: 'manage_files',
    title: 'Manage Files',
    description: `Manage files on UploadThing.

Actions:
- delete: Remove files by keys
- rename: Change file name
- update_acl: Set public-read or private
- get_url: Get access URL for a file`,
  },
} as const satisfies Record<string, ToolMetadata>;

/**
 * Type-safe helper to get metadata for a tool.
 */
export function getToolMetadata(toolName: keyof typeof toolsMetadata): ToolMetadata {
  return toolsMetadata[toolName];
}

/**
 * Get all registered tool names.
 */
export function getToolNames(): string[] {
  return Object.keys(toolsMetadata);
}

/**
 * Server-level metadata
 */
export const serverMetadata = {
  title: 'UploadThing MCP Server',
  instructions:
    'Use these tools to upload, list, and manage files on UploadThing. Files can be uploaded as base64-encoded content.',
} as const;
