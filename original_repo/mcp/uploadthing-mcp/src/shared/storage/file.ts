// File-backed storage for Node.js with encryption and strict permissions
// Provider-agnostic version from Spotify MCP

import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { createEncryptor, type Encryptor } from '../crypto/aes-gcm.js';
import { sharedLogger as logger } from '../utils/logger.js';
import type { ProviderTokens, RsRecord, TokenStore, Transaction } from './interface.js';
import { MemoryTokenStore } from './memory.js';

/** File permission: owner read/write only (600) */
const SECURE_FILE_MODE = 0o600;

/** Directory permission: owner only (700) */
const SECURE_DIR_MODE = 0o700;

type PersistShape = {
  version: number;
  encrypted: boolean;
  records: Array<RsRecord>;
};

export class FileTokenStore implements TokenStore {
  private memory: MemoryTokenStore;
  private persistPath: string | null;
  private encryptor: Encryptor | null = null;
  private saveDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingSave: Promise<void> | null = null;

  /**
   * Create a file-backed token store.
   *
   * @param persistPath - Path to the JSON file for persistence
   * @param encryptionKey - Base64url-encoded 32-byte key for AES-256-GCM encryption
   */
  constructor(persistPath?: string, encryptionKey?: string) {
    this.memory = new MemoryTokenStore();
    this.persistPath = persistPath ?? null;

    if (encryptionKey) {
      try {
        this.encryptor = createEncryptor(encryptionKey);
        logger.debug('file_token_store', { message: 'Encryption enabled' });
      } catch (error) {
        logger.error('file_token_store', {
          message: 'Failed to initialize encryption',
          error: (error as Error).message,
        });
        throw error;
      }
    } else if (process.env.NODE_ENV === 'production') {
      logger.warning('file_token_store', {
        message: 'No encryption key provided! Tokens stored in plaintext.',
      });
    }

    // Load is async now, but constructor is sync - we'll load lazily
    this.loadAsync().catch((err) => {
      logger.error('file_token_store', {
        message: 'Initial load failed',
        error: err.message,
      });
    });
  }

  private async loadAsync(): Promise<void> {
    if (!this.persistPath) {
      logger.debug('file_token_store', { message: 'No persistPath, skipping load' });
      return;
    }

    try {
      if (!existsSync(this.persistPath)) {
        logger.debug('file_token_store', {
          message: 'File does not exist',
          path: this.persistPath,
        });
        return;
      }

      let raw = readFileSync(this.persistPath, 'utf8');

      // Try to parse as JSON first (unencrypted or old format)
      let data: PersistShape;
      try {
        data = JSON.parse(raw) as PersistShape;
      } catch {
        // If parse fails and we have encryptor, try decrypting
        if (this.encryptor) {
          try {
            raw = await this.encryptor.decrypt(raw);
            data = JSON.parse(raw) as PersistShape;
          } catch (decryptError) {
            logger.error('file_token_store', {
              message: 'Failed to decrypt file',
              error: (decryptError as Error).message,
            });
            return;
          }
        } else {
          logger.error('file_token_store', {
            message: 'File appears encrypted but no key provided',
          });
          return;
        }
      }

      if (!data || !Array.isArray(data.records)) {
        logger.warning('file_token_store', { message: 'Invalid file format' });
        return;
      }

      // Check if file was encrypted but we don't have a key
      if (data.encrypted && !this.encryptor) {
        logger.warning('file_token_store', {
          message: 'File was saved encrypted but no encryption key provided',
        });
      }

      logger.info('file_token_store', {
        message: 'Loading records',
        count: data.records.length,
        path: this.persistPath,
        encrypted: data.encrypted ?? false,
      });

      // Filter out expired records during load
      const now = Date.now();
      const validRecords = data.records.filter((rec) => {
        // Skip records with expired provider tokens
        if (rec.provider.expires_at && now >= rec.provider.expires_at) {
          return false;
        }
        return true;
      });

      // Populate internal memory maps
      for (const rec of validRecords) {
        const memoryMap = this.memory as unknown as {
          rsAccessMap: Map<string, RsRecord & { expiresAt: number }>;
          rsRefreshMap: Map<string, RsRecord & { expiresAt: number }>;
        };

        const recordWithExpiry = {
          ...rec,
          expiresAt: rec.provider.expires_at ?? now + 7 * 24 * 60 * 60 * 1000,
        };

        memoryMap.rsAccessMap.set(rec.rs_access_token, recordWithExpiry);
        memoryMap.rsRefreshMap.set(rec.rs_refresh_token, recordWithExpiry);
      }

      logger.debug('file_token_store', {
        message: 'Records loaded successfully',
        total: data.records.length,
        valid: validRecords.length,
        expired: data.records.length - validRecords.length,
      });
    } catch (error) {
      logger.error('file_token_store', {
        message: 'Load failed',
        error: (error as Error).message,
      });
    }
  }

  private scheduleSave(): void {
    if (!this.persistPath) {
      return;
    }

    // Debounce saves to avoid excessive disk writes
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
    }

    this.saveDebounceTimer = setTimeout(() => {
      this.pendingSave = this.saveAsync();
      this.pendingSave.catch((err) => {
        logger.error('file_token_store', {
          message: 'Save failed',
          error: err.message,
        });
      });
    }, 100);
  }

  private async saveAsync(): Promise<void> {
    if (!this.persistPath) {
      return;
    }

    try {
      const dir = dirname(this.persistPath);
      if (!existsSync(dir)) {
        logger.debug('file_token_store', { message: 'Creating directory', dir });
        mkdirSync(dir, { recursive: true, mode: SECURE_DIR_MODE });
      }

      // Get all records from internal memory map
      const memoryMap = this.memory as unknown as {
        rsAccessMap: Map<string, RsRecord>;
      };
      const records = Array.from(memoryMap.rsAccessMap.values());

      const data: PersistShape = {
        version: 1,
        encrypted: Boolean(this.encryptor),
        records,
      };

      let content = JSON.stringify(data, null, 2);

      // Encrypt if key is available (async)
      if (this.encryptor) {
        content = await this.encryptor.encrypt(content);
      }

      // Write with secure permissions
      writeFileSync(this.persistPath, content, {
        encoding: 'utf8',
        mode: SECURE_FILE_MODE,
      });

      // Ensure permissions are set (in case file already existed)
      try {
        chmodSync(this.persistPath, SECURE_FILE_MODE);
      } catch {
        // Ignore chmod errors (might fail on some systems)
      }

      logger.debug('file_token_store', {
        message: 'File saved',
        records: records.length,
        encrypted: Boolean(this.encryptor),
      });
    } catch (error) {
      logger.error('file_token_store', {
        message: 'Save failed',
        error: (error as Error).message,
      });
    }
  }

  async storeRsMapping(
    rsAccess: string,
    provider: ProviderTokens,
    rsRefresh?: string,
  ): Promise<RsRecord> {
    logger.debug('file_token_store', {
      message: 'Storing RS mapping',
      hasRefresh: Boolean(rsRefresh),
      persistPath: this.persistPath,
    });

    const result = await this.memory.storeRsMapping(rsAccess, provider, rsRefresh);
    this.scheduleSave();
    return result;
  }

  async getByRsAccess(rsAccess: string): Promise<RsRecord | null> {
    return this.memory.getByRsAccess(rsAccess);
  }

  async getByRsRefresh(rsRefresh: string): Promise<RsRecord | null> {
    return this.memory.getByRsRefresh(rsRefresh);
  }

  async updateByRsRefresh(
    rsRefresh: string,
    provider: ProviderTokens,
    maybeNewRsAccess?: string,
  ): Promise<RsRecord | null> {
    const result = await this.memory.updateByRsRefresh(
      rsRefresh,
      provider,
      maybeNewRsAccess,
    );
    this.scheduleSave();
    return result;
  }

  async saveTransaction(
    txnId: string,
    txn: Transaction,
    ttlSeconds?: number,
  ): Promise<void> {
    // Transactions are memory-only (don't persist OAuth flow state)
    return this.memory.saveTransaction(txnId, txn, ttlSeconds);
  }

  async getTransaction(txnId: string): Promise<Transaction | null> {
    return this.memory.getTransaction(txnId);
  }

  async deleteTransaction(txnId: string): Promise<void> {
    return this.memory.deleteTransaction(txnId);
  }

  async saveCode(code: string, txnId: string, ttlSeconds?: number): Promise<void> {
    // Codes are memory-only (don't persist OAuth flow state)
    return this.memory.saveCode(code, txnId, ttlSeconds);
  }

  async getTxnIdByCode(code: string): Promise<string | null> {
    return this.memory.getTxnIdByCode(code);
  }

  async deleteCode(code: string): Promise<void> {
    return this.memory.deleteCode(code);
  }

  /**
   * Force immediate save (useful before shutdown).
   */
  async flush(): Promise<void> {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
      this.saveDebounceTimer = null;
    }
    await this.saveAsync();
  }

  /**
   * Stop cleanup intervals.
   */
  stopCleanup(): void {
    this.memory.stopCleanup();
  }

  /**
   * Get store statistics.
   */
  getStats(): { rsTokens: number; transactions: number; codes: number } {
    return this.memory.getStats();
  }
}
