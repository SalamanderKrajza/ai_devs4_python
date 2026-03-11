/**
 * Provider token types and conversion utilities.
 *
 * ProviderTokens (snake_case) - storage/OAuth API format, defined in storage/interface.ts
 * ProviderInfo (camelCase) - tool handler format, defined here
 */

// Re-export ProviderTokens from storage (canonical source)
export type { ProviderTokens } from '../storage/interface.js';

import type { ProviderTokens } from '../storage/interface.js';

/**
 * Provider info in camelCase for tool handlers.
 * Converted from ProviderTokens when passing to tools.
 */
export interface ProviderInfo {
  accessToken: string;
  refreshToken?: string;
  expiresAt?: number;
  scopes?: string[];
}

/**
 * Convert snake_case ProviderTokens to camelCase ProviderInfo.
 * Use when bridging storage layer to tool context.
 */
export function toProviderInfo(tokens: ProviderTokens): ProviderInfo {
  return {
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,
    expiresAt: tokens.expires_at,
    scopes: tokens.scopes,
  };
}

/**
 * Convert camelCase ProviderInfo to snake_case ProviderTokens.
 * Use when storing tool-provided data.
 */
export function toProviderTokens(info: ProviderInfo): ProviderTokens {
  return {
    access_token: info.accessToken,
    refresh_token: info.refreshToken,
    expires_at: info.expiresAt,
    scopes: info.scopes,
  };
}
