// Unified storage interfaces for both Node.js and Cloudflare Workers
// Provider-agnostic version from Spotify MCP

export type ProviderTokens = {
  access_token: string;
  refresh_token?: string;
  expires_at?: number;
  scopes?: string[];
};

export type RsRecord = {
  rs_access_token: string;
  rs_refresh_token: string;
  provider: ProviderTokens;
  created_at: number;
};

export type Transaction = {
  codeChallenge: string;
  state?: string;
  scope?: string;
  createdAt: number;
  sid?: string;
  provider?: ProviderTokens;
};

export type SessionRecord = {
  /** API key that owns this session (for multi-tenant support) */
  apiKey?: string;
  rs_access_token?: string;
  rs_refresh_token?: string;
  provider?: ProviderTokens | null;
  created_at: number;
  last_accessed: number;
  /** Whether MCP initialize handshake completed */
  initialized?: boolean;
  /** Negotiated MCP protocol version */
  protocolVersion?: string;
};

/**
 * Token storage interface - all operations are async to support both
 * sync (Node Map + File) and async (Cloudflare KV) backends
 */
export interface TokenStore {
  // RS token mapping
  storeRsMapping(
    rsAccess: string,
    provider: ProviderTokens,
    rsRefresh?: string,
  ): Promise<RsRecord>;

  getByRsAccess(rsAccess: string): Promise<RsRecord | null>;

  getByRsRefresh(rsRefresh: string): Promise<RsRecord | null>;

  updateByRsRefresh(
    rsRefresh: string,
    provider: ProviderTokens,
    maybeNewRsAccess?: string,
  ): Promise<RsRecord | null>;

  // Transaction storage (PKCE flow)
  saveTransaction(txnId: string, txn: Transaction, ttlSeconds?: number): Promise<void>;

  getTransaction(txnId: string): Promise<Transaction | null>;

  deleteTransaction(txnId: string): Promise<void>;

  // Code storage (authorization codes)
  saveCode(code: string, txnId: string, ttlSeconds?: number): Promise<void>;

  getTxnIdByCode(code: string): Promise<string | null>;

  deleteCode(code: string): Promise<void>;
}

/** Maximum sessions allowed per API key */
export const MAX_SESSIONS_PER_API_KEY = 5;

/**
 * Session storage interface for multi-tenant MCP servers.
 * Supports tracking sessions per API key with limits and LRU eviction.
 */
export interface SessionStore {
  /**
   * Create a new session for an API key.
   * Automatically enforces MAX_SESSIONS_PER_API_KEY limit with LRU eviction.
   */
  create(sessionId: string, apiKey: string): Promise<SessionRecord>;

  /**
   * Get a session by ID. Returns null if not found or expired.
   * Updates last_accessed timestamp on access.
   */
  get(sessionId: string): Promise<SessionRecord | null>;

  /**
   * Update session data (e.g., mark as initialized, store protocol version).
   */
  update(sessionId: string, data: Partial<SessionRecord>): Promise<void>;

  /**
   * Delete a session.
   */
  delete(sessionId: string): Promise<void>;

  /**
   * Get all sessions for an API key.
   */
  getByApiKey(apiKey: string): Promise<SessionRecord[]>;

  /**
   * Count active sessions for an API key.
   */
  countByApiKey(apiKey: string): Promise<number>;

  /**
   * Delete the oldest session for an API key (LRU eviction).
   */
  deleteOldestByApiKey(apiKey: string): Promise<void>;

  /**
   * Ensure a session exists (legacy compat). Creates minimal session if missing.
   * @deprecated Use create() for new code
   */
  ensure(sessionId: string): Promise<void>;

  /**
   * Put a session record directly (legacy compat).
   * @deprecated Use create() or update() for new code
   */
  put(sessionId: string, value: SessionRecord): Promise<void>;
}
