/**
 * Canonical authentication types.
 * Single source of truth for auth strategy definitions.
 */

/**
 * Supported authentication strategies.
 *
 * - 'oauth': Full OAuth 2.1 PKCE flow with RS token → provider token mapping
 * - 'bearer': Simple static Bearer token (from BEARER_TOKEN env)
 * - 'api_key': API key in custom header (from API_KEY env)
 * - 'custom': Arbitrary headers from CUSTOM_HEADERS config
 * - 'none': No authentication required
 */
export type AuthStrategy = 'oauth' | 'bearer' | 'api_key' | 'custom' | 'none';

/**
 * Auth headers extracted from incoming requests.
 */
export interface AuthHeaders {
  authorization?: string;
  'x-api-key'?: string;
  'x-auth-token'?: string;
  [key: string]: string | undefined;
}

/**
 * Resolved authentication result.
 * Contains headers ready for API calls and token information.
 */
export interface ResolvedAuth {
  /** Auth strategy used */
  strategy: AuthStrategy;
  /** Headers to pass to API calls */
  headers: Record<string, string>;
  /** Raw access token (if bearer/oauth) */
  accessToken?: string;
}
