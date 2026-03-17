/**
 * UploadThing API client using the official SDK.
 * Handles all interactions with UploadThing's REST API.
 */

import { UTApi, UTFile } from 'uploadthing/server';
import { logger } from '../utils/logger.js';

// Singleton UTApi instance
let utapiInstance: UTApi | null = null;

/**
 * Get or create the UTApi instance.
 * Uses UPLOADTHING_TOKEN from environment.
 */
export function getUTApi(): UTApi {
  if (!utapiInstance) {
    const token = process.env.UPLOADTHING_TOKEN;
    if (!token) {
      throw new Error('UPLOADTHING_TOKEN environment variable is required');
    }
    utapiInstance = new UTApi({ token });
    logger.info('uploadthing', { message: 'UTApi client initialized' });
  }
  return utapiInstance;
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface UploadFileInput {
  /** Base64-encoded file content */
  base64: string;
  /** File name with extension */
  name: string;
  /** MIME type (e.g., "image/png") */
  type: string;
  /** Optional custom ID for the file */
  customId?: string;
}

export interface UploadedFile {
  key: string;
  url: string;
  name: string;
  size: number;
  type: string;
  customId?: string | null;
}

export interface ListFilesOptions {
  limit?: number;
  offset?: number;
}

export interface ListFilesResult {
  files: Array<{
    id: string;
    key: string;
    name: string;
    size: number;
    status: string;
    customId?: string | null;
    uploadedAt: number;
  }>;
  hasMore: boolean;
}

export interface FileInfo {
  key: string;
  name: string;
  size: number;
  status: string;
  customId?: string | null;
  uploadedAt: number;
  url: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Upload Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Upload files to UploadThing.
 * Converts base64 input to UTFile and uploads via UTApi.
 */
export async function uploadFiles(
  files: UploadFileInput[],
  options?: { acl?: 'public-read' | 'private' },
): Promise<UploadedFile[]> {
  const utapi = getUTApi();

  // Convert base64 to UTFile objects (supports customId)
  const fileObjects = files.map((f) => {
    const buffer = Buffer.from(f.base64, 'base64');
    return new UTFile([buffer], f.name, {
      type: f.type,
      customId: f.customId,
    });
  });

  logger.debug('uploadthing', {
    message: 'Uploading files',
    count: fileObjects.length,
    names: fileObjects.map((f) => f.name),
  });

  const response = await utapi.uploadFiles(fileObjects, {
    acl: options?.acl,
  });

  const results: UploadedFile[] = [];
  const responseArray = Array.isArray(response) ? response : [response];

  for (let i = 0; i < responseArray.length; i++) {
    const res = responseArray[i];
    if (res.error) {
      logger.error('uploadthing', {
        message: 'Upload failed',
        fileName: files[i].name,
        error: res.error.message,
      });
      throw new Error(`Upload failed for ${files[i].name}: ${res.error.message}`);
    }
    results.push({
      key: res.data.key,
      url: res.data.ufsUrl,
      name: res.data.name,
      size: res.data.size,
      type: res.data.type,
      customId: res.data.customId,
    });
  }

  logger.info('uploadthing', {
    message: 'Files uploaded successfully',
    count: results.length,
  });

  return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// List Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * List files from UploadThing with pagination.
 */
export async function listFiles(options?: ListFilesOptions): Promise<ListFilesResult> {
  const utapi = getUTApi();

  const response = await utapi.listFiles({
    limit: options?.limit ?? 50,
    offset: options?.offset ?? 0,
  });

  return {
    files: response.files.map((f) => ({
      id: f.id,
      key: f.key,
      name: f.name,
      size: f.size,
      status: f.status,
      customId: f.customId,
      uploadedAt: f.uploadedAt,
    })),
    hasMore: response.hasMore,
  };
}

/**
 * Get info about a specific file by key.
 */
export async function getFileInfo(fileKey: string): Promise<FileInfo | null> {
  const utapi = getUTApi();

  // List files and find by key (UTApi doesn't have a direct get by key)
  const response = await utapi.listFiles({ limit: 500 });
  const file = response.files.find((f) => f.key === fileKey);

  if (!file) {
    return null;
  }

  // Generate signed URL for the file (works for both public and private)
  const signedUrl = await utapi.generateSignedURL(fileKey);

  return {
    key: file.key,
    name: file.name,
    size: file.size,
    status: file.status,
    customId: file.customId,
    uploadedAt: file.uploadedAt,
    url: signedUrl.ufsUrl,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Management Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Delete files by their keys.
 */
export async function deleteFiles(fileKeys: string[]): Promise<{ deletedCount: number }> {
  const utapi = getUTApi();

  logger.debug('uploadthing', {
    message: 'Deleting files',
    keys: fileKeys,
  });

  const response = await utapi.deleteFiles(fileKeys);

  logger.info('uploadthing', {
    message: 'Files deleted',
    deletedCount: response.deletedCount,
  });

  return { deletedCount: response.deletedCount };
}

/**
 * Rename a file.
 */
export async function renameFile(
  fileKey: string,
  newName: string,
): Promise<{ success: boolean }> {
  const utapi = getUTApi();

  logger.debug('uploadthing', {
    message: 'Renaming file',
    key: fileKey,
    newName,
  });

  // RenameFileUpdate = { fileKey: string; newName: string }
  await utapi.renameFiles({ fileKey, newName });

  logger.info('uploadthing', {
    message: 'File renamed',
    key: fileKey,
    newName,
  });

  return { success: true };
}

/**
 * Update file ACL (access control).
 */
export async function updateFileAcl(
  fileKey: string,
  acl: 'public-read' | 'private',
): Promise<{ success: boolean }> {
  const utapi = getUTApi();

  logger.debug('uploadthing', {
    message: 'Updating file ACL',
    key: fileKey,
    acl,
  });

  await utapi.updateACL(fileKey, acl);

  logger.info('uploadthing', {
    message: 'File ACL updated',
    key: fileKey,
    acl,
  });

  return { success: true };
}

/**
 * Get a signed URL for accessing a file (works for private files).
 */
export async function getFileUrl(fileKey: string): Promise<{ url: string }> {
  const utapi = getUTApi();

  const response = await utapi.generateSignedURL(fileKey);

  return { url: response.ufsUrl };
}

/**
 * Get usage information for the app.
 */
export async function getUsageInfo(): Promise<{
  totalBytes: number;
  filesUploaded: number;
  limitBytes: number;
}> {
  const utapi = getUTApi();
  const response = await utapi.getUsageInfo();

  return {
    totalBytes: response.totalBytes,
    filesUploaded: response.filesUploaded,
    limitBytes: response.limitBytes,
  };
}
