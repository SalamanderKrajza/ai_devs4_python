// Cloudflare KV storage with encryption support
// Provider-agnostic version from Spotify MCP

import type {
  ProviderTokens,
  RsRecord,
  SessionRecord,
  SessionStore,
  TokenStore,
  Transaction,
} from './interface.js';
import { MAX_SESSIONS_PER_API_KEY } from './interface.js';
import { MemorySessionStore, MemoryTokenStore } from './memory.js';

// Cloudflare KV namespace type
type KVNamespace = {
  get(key: string): Promise<string | null>;
  put(
    key: string,
    value: string,
    options?: { expiration?: number; expirationTtl?: number },
  ): Promise<void>;
  delete(key: string): Promise<void>;
};

type EncryptFn = (plaintext: string) => Promise<string> | string;
type DecryptFn = (ciphertext: string) => Promise<string> | string;

function ttl(seconds: number): number {
  return Math.floor(Date.now() / 1000) + seconds;
}

function toJson(value: unknown): string {
  return JSON.stringify(value);
}

function fromJson<T>(value: string | null): T | null {
  if (!value) {
    return null;
  }
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

export class KvTokenStore implements TokenStore {
  private kv: KVNamespace;
  private encrypt: EncryptFn;
  private decrypt: DecryptFn;
  private fallback: MemoryTokenStore;

  constructor(
    kv: KVNamespace,
    options?: {
      encrypt?: EncryptFn;
      decrypt?: DecryptFn;
      fallback?: MemoryTokenStore;
    },
  ) {
    this.kv = kv;
    this.encrypt = options?.encrypt ?? ((s) => s);
    this.decrypt = options?.decrypt ?? ((s) => s);
    this.fallback = options?.fallback ?? new MemoryTokenStore();
  }

  private async putJson(
    key: string,
    value: unknown,
    options?: { expiration?: number; expirationTtl?: number },
  ): Promise<void> {
    try {
      const raw = await this.encrypt(toJson(value));
      await this.kv.put(key, raw, options);
    } catch (error) {
      // KV write failed (likely quota exceeded) - log but don't crash
      // Fallback memory store will still have the data
      console.error('[KV] Write failed:', (error as Error).message);
      throw error; // Re-throw so caller knows KV failed
    }
  }

  private async getJson<T>(key: string): Promise<T | null> {
    const raw = await this.kv.get(key);
    if (!raw) {
      return null;
    }
    const plain = await this.decrypt(raw);
    return fromJson<T>(plain);
  }

  async storeRsMapping(
    rsAccess: string,
    provider: ProviderTokens,
    rsRefresh?: string,
  ): Promise<RsRecord> {
    const rec: RsRecord = {
      rs_access_token: rsAccess,
      rs_refresh_token: rsRefresh ?? crypto.randomUUID(),
      provider: { ...provider },
      created_at: Date.now(),
    };

    // CRITICAL: Store in memory fallback FIRST
    // If KV fails (quota/network), memory still has it
    await this.fallback.storeRsMapping(rsAccess, provider, rsRefresh);

    // Then try KV (may fail due to quota)
    try {
      await Promise.all([
        this.putJson(`rs:access:${rec.rs_access_token}`, rec),
        this.putJson(`rs:refresh:${rec.rs_refresh_token}`, rec),
      ]);
    } catch (error) {
      console.warn(
        '[KV] Failed to persist RS mapping (using memory fallback):',
        (error as Error).message,
      );
      // Don't throw - memory fallback has the data
    }

    return rec;
  }

  async getByRsAccess(rsAccess: string): Promise<RsRecord | null> {
    const rec = await this.getJson<RsRecord>(`rs:access:${rsAccess}`);
    return rec ?? (await this.fallback.getByRsAccess(rsAccess));
  }

  async getByRsRefresh(rsRefresh: string): Promise<RsRecord | null> {
    const rec = await this.getJson<RsRecord>(`rs:refresh:${rsRefresh}`);
    return rec ?? (await this.fallback.getByRsRefresh(rsRefresh));
  }

  async updateByRsRefresh(
    rsRefresh: string,
    provider: ProviderTokens,
    maybeNewRsAccess?: string,
  ): Promise<RsRecord | null> {
    const existing = await this.getJson<RsRecord>(`rs:refresh:${rsRefresh}`);
    if (!existing) {
      return this.fallback.updateByRsRefresh(rsRefresh, provider, maybeNewRsAccess);
    }

    const rsAccessChanged =
      maybeNewRsAccess && maybeNewRsAccess !== existing.rs_access_token;
    const next: RsRecord = {
      rs_access_token: maybeNewRsAccess || existing.rs_access_token,
      rs_refresh_token: rsRefresh,
      provider: { ...provider },
      created_at: Date.now(),
    };

    // Update memory fallback first
    await this.fallback.updateByRsRefresh(rsRefresh, provider, maybeNewRsAccess);

    // Then try KV (may fail due to quota)
    // Optimize: only delete old access key if RS access token actually changed
    try {
      if (rsAccessChanged) {
        // RS access token changed: delete old + write new access + write refresh (3 ops)
        await Promise.all([
          this.kv.delete(`rs:access:${existing.rs_access_token}`),
          this.putJson(`rs:access:${next.rs_access_token}`, next),
          this.putJson(`rs:refresh:${rsRefresh}`, next),
        ]);
      } else {
        // RS access token unchanged: update both keys in place (2 ops, no delete)
        await Promise.all([
          this.putJson(`rs:access:${existing.rs_access_token}`, next),
          this.putJson(`rs:refresh:${rsRefresh}`, next),
        ]);
      }
    } catch (error) {
      console.warn(
        '[KV] Failed to update RS mapping (using memory fallback):',
        (error as Error).message,
      );
      // Don't throw - memory fallback has the data
    }

    return next;
  }

  async saveTransaction(
    txnId: string,
    txn: Transaction,
    ttlSeconds = 600,
  ): Promise<void> {
    // Memory fallback first (critical for OAuth flow)
    await this.fallback.saveTransaction(txnId, txn);

    // KV is optional (nice to have for persistence across instances)
    try {
      await this.putJson(`txn:${txnId}`, txn, { expiration: ttl(ttlSeconds) });
    } catch (error) {
      console.warn(
        '[KV] Failed to save transaction (using memory):',
        (error as Error).message,
      );
      // Don't throw - memory has it
    }
  }

  async getTransaction(txnId: string): Promise<Transaction | null> {
    const txn = await this.getJson<Transaction>(`txn:${txnId}`);
    return txn ?? (await this.fallback.getTransaction(txnId));
  }

  async deleteTransaction(txnId: string): Promise<void> {
    // Skip KV delete - transactions have TTL and will auto-expire
    // This saves 1 write operation per OAuth flow
    await this.fallback.deleteTransaction(txnId);
  }

  async saveCode(code: string, txnId: string, ttlSeconds = 600): Promise<void> {
    // Memory fallback first (critical for OAuth flow)
    await this.fallback.saveCode(code, txnId);

    // KV is optional
    try {
      await this.putJson(`code:${code}`, { v: txnId }, { expiration: ttl(ttlSeconds) });
    } catch (error) {
      console.warn(
        '[KV] Failed to save code (using memory):',
        (error as Error).message,
      );
      // Don't throw - memory has it
    }
  }

  async getTxnIdByCode(code: string): Promise<string | null> {
    const obj = await this.getJson<{ v: string }>(`code:${code}`);
    return obj?.v ?? (await this.fallback.getTxnIdByCode(code));
  }

  async deleteCode(code: string): Promise<void> {
    // Skip KV delete - codes have TTL and will auto-expire
    // This saves 1 write operation per OAuth flow
    await this.fallback.deleteCode(code);
  }
}

const SESSION_KEY_PREFIX = 'session:';
const SESSION_APIKEY_PREFIX = 'session:apikey:';
const SESSION_TTL_SECONDS = 24 * 60 * 60;

/**
 * KV-based session store for Cloudflare Workers.
 * Provides persistent session storage with multi-tenant support.
 *
 * Storage structure:
 * - session:{sessionId} → SessionRecord (main session data)
 * - session:apikey:{apiKey} → string[] (list of session IDs for this API key)
 */
export class KvSessionStore implements SessionStore {
  private kv: KVNamespace;
  private encrypt: EncryptFn;
  private decrypt: DecryptFn;
  private fallback: MemorySessionStore;

  constructor(
    kv: KVNamespace,
    options?: {
      encrypt?: EncryptFn;
      decrypt?: DecryptFn;
      fallback?: MemorySessionStore;
    },
  ) {
    this.kv = kv;
    this.encrypt = options?.encrypt ?? ((s) => s);
    this.decrypt = options?.decrypt ?? ((s) => s);
    this.fallback = options?.fallback ?? new MemorySessionStore();
  }

  private async putSession(sessionId: string, value: SessionRecord): Promise<void> {
    const raw = await this.encrypt(toJson(value));
    await this.kv.put(`${SESSION_KEY_PREFIX}${sessionId}`, raw, {
      expiration: ttl(SESSION_TTL_SECONDS),
    });
  }

  private async getSession(sessionId: string): Promise<SessionRecord | null> {
    const raw = await this.kv.get(`${SESSION_KEY_PREFIX}${sessionId}`);
    if (!raw) {
      return this.fallback.get(sessionId);
    }
    const plain = await this.decrypt(raw);
    return fromJson<SessionRecord>(plain);
  }

  private async getApiKeySessionIds(apiKey: string): Promise<string[]> {
    const raw = await this.kv.get(`${SESSION_APIKEY_PREFIX}${apiKey}`);
    if (!raw) return [];
    return fromJson<string[]>(raw) ?? [];
  }

  private async setApiKeySessionIds(
    apiKey: string,
    sessionIds: string[],
  ): Promise<void> {
    if (sessionIds.length === 0) {
      await this.kv.delete(`${SESSION_APIKEY_PREFIX}${apiKey}`);
    } else {
      await this.kv.put(`${SESSION_APIKEY_PREFIX}${apiKey}`, toJson(sessionIds), {
        expiration: ttl(SESSION_TTL_SECONDS),
      });
    }
  }

  async create(sessionId: string, apiKey: string): Promise<SessionRecord> {
    // Enforce session limit per API key
    const currentCount = await this.countByApiKey(apiKey);
    if (currentCount >= MAX_SESSIONS_PER_API_KEY) {
      await this.deleteOldestByApiKey(apiKey);
    }

    const now = Date.now();
    const record: SessionRecord = {
      apiKey,
      created_at: now,
      last_accessed: now,
      initialized: false,
    };

    // Store session
    await this.putSession(sessionId, record);
    await this.fallback.create(sessionId, apiKey);

    // Update API key → session IDs index
    const sessionIds = await this.getApiKeySessionIds(apiKey);
    if (!sessionIds.includes(sessionId)) {
      sessionIds.push(sessionId);
      await this.setApiKeySessionIds(apiKey, sessionIds);
    }

    return record;
  }

  async get(sessionId: string): Promise<SessionRecord | null> {
    const session = await this.getSession(sessionId);
    if (!session) return null;

    // Update last_accessed
    const now = Date.now();
    session.last_accessed = now;

    // Update in KV (fire and forget for performance)
    this.putSession(sessionId, session).catch(() => {});

    return session;
  }

  async update(sessionId: string, data: Partial<SessionRecord>): Promise<void> {
    const session = await this.getSession(sessionId);
    if (!session) return;

    const updated: SessionRecord = {
      ...session,
      ...data,
      last_accessed: Date.now(),
    };

    await this.putSession(sessionId, updated);
    await this.fallback.update(sessionId, data);
  }

  async delete(sessionId: string): Promise<void> {
    // Get session to find API key
    const session = await this.getSession(sessionId);

    // Remove from KV
    await this.kv.delete(`${SESSION_KEY_PREFIX}${sessionId}`);
    await this.fallback.delete(sessionId);

    // Update API key index
    if (session?.apiKey) {
      const sessionIds = await this.getApiKeySessionIds(session.apiKey);
      const filtered = sessionIds.filter((id) => id !== sessionId);
      await this.setApiKeySessionIds(session.apiKey, filtered);
    }
  }

  async getByApiKey(apiKey: string): Promise<SessionRecord[]> {
    const sessionIds = await this.getApiKeySessionIds(apiKey);
    const sessions: SessionRecord[] = [];

    for (const sessionId of sessionIds) {
      const session = await this.getSession(sessionId);
      if (session) {
        sessions.push(session);
      }
    }

    // Sort by last_accessed descending
    return sessions.sort((a, b) => b.last_accessed - a.last_accessed);
  }

  async countByApiKey(apiKey: string): Promise<number> {
    const sessionIds = await this.getApiKeySessionIds(apiKey);
    return sessionIds.length;
  }

  async deleteOldestByApiKey(apiKey: string): Promise<void> {
    const sessions = await this.getByApiKey(apiKey);
    if (sessions.length === 0) return;

    // Find oldest (last in the sorted array since it's sorted by last_accessed DESC)
    const oldest = sessions[sessions.length - 1];

    // Find session ID for oldest session
    const sessionIds = await this.getApiKeySessionIds(apiKey);
    for (const sessionId of sessionIds) {
      const session = await this.getSession(sessionId);
      if (session && session.created_at === oldest.created_at) {
        await this.delete(sessionId);
        return;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Legacy compatibility methods
  // ─────────────────────────────────────────────────────────────────────────

  async ensure(sessionId: string): Promise<void> {
    const existing = await this.fallback.get(sessionId);
    if (!existing) {
      const now = Date.now();
      await this.fallback.put(sessionId, { created_at: now, last_accessed: now });
    }
  }

  async put(sessionId: string, value: SessionRecord): Promise<void> {
    await this.putSession(sessionId, value);
    await this.fallback.put(sessionId, value);

    // Update API key index if present
    if (value.apiKey) {
      const sessionIds = await this.getApiKeySessionIds(value.apiKey);
      if (!sessionIds.includes(sessionId)) {
        sessionIds.push(sessionId);
        await this.setApiKeySessionIds(value.apiKey, sessionIds);
      }
    }
  }
}
