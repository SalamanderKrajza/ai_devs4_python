/**
 * Type-safe access to internal MCP SDK Server properties.
 *
 * The McpServer class wraps a lower-level Server instance that provides
 * direct access to request/notification methods and client capabilities.
 * This module provides typed helpers to access these internals safely.
 */

import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

/**
 * Shape of the internal server with commonly used methods.
 * These are not part of the public McpServer API but are needed
 * for advanced features like client requests and capability checks.
 */
interface LowLevelServer {
  request?: (
    params: { method: string; params?: unknown },
    schema?: { parse: (r: unknown) => unknown },
  ) => Promise<unknown>;
  notification?: (params: { method: string; params?: unknown }) => Promise<void>;
  setRequestHandler?: (
    method: string,
    handler: (request: unknown) => Promise<unknown>,
  ) => void;
  getClientCapabilities?: () => ClientCapabilities;
  getClientVersion?: () => string;
  oninitialized?: () => void;
}

/**
 * Client capabilities shape for type-safe access.
 */
interface ClientCapabilities {
  roots?: { listChanged?: boolean };
  sampling?: { tools?: boolean };
  elicitation?: { form?: unknown; url?: boolean };
  [key: string]: unknown;
}

/**
 * McpServer with internal methods typed (use with type assertion).
 */
interface McpServerWithInternals {
  server?: LowLevelServer;
  sendResourceUpdated?: (params: { uri: string }) => void;
}

/**
 * Get the low-level server instance from McpServer.
 * Returns the internal Server or falls back to the McpServer itself.
 */
export function getLowLevelServer(server: McpServer): LowLevelServer {
  const extended = server as unknown as McpServerWithInternals;
  return (extended.server ?? server) as unknown as LowLevelServer;
}

/**
 * Get McpServer with internal methods typed.
 */
export function getServerWithInternals(server: McpServer): McpServerWithInternals {
  return server as unknown as McpServerWithInternals;
}

/**
 * JSON-RPC error with code property.
 */
export interface JsonRpcError extends Error {
  code?: number;
  data?: unknown;
}

/**
 * Check if an error is a JSON-RPC error with a specific code.
 */
export function isJsonRpcError(error: unknown, code?: number): error is JsonRpcError {
  if (typeof error !== 'object' || error === null) return false;
  const err = error as JsonRpcError;
  if (typeof err.code !== 'number') return false;
  if (code !== undefined && err.code !== code) return false;
  return true;
}

/**
 * JSON-RPC error code for method not found.
 */
export const JSON_RPC_METHOD_NOT_FOUND = -32601;
