/**
 * MCP OAuth provider — file-backed token storage with PKCE support.
 *
 * Implements OAuthClientProvider from the MCP SDK.
 * Tokens are persisted per-server in .mcp.oauth.json.
 * Pending auth URLs are stored in-memory for the API to surface.
 */
import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { OAuthClientProvider } from '@modelcontextprotocol/sdk/client/auth.js'
import type { OAuthClientMetadata, OAuthTokens } from '@modelcontextprotocol/sdk/shared/auth.js'
import { logger } from '../lib/logger.js'

const log = logger.child({ name: 'mcp:oauth' })

const OAUTH_FILE = '.mcp.oauth.json'

interface ServerOAuthData {
  tokens?: OAuthTokens
  clientId?: string
  clientSecret?: string
  codeVerifier?: string
}

type OAuthStore = Record<string, ServerOAuthData>

// ─────────────────────────────────────────────────────────────────────────────
// File storage (serialized via async lock to prevent concurrent write races)
// ─────────────────────────────────────────────────────────────────────────────

let lockChain = Promise.resolve()

async function withFileLock<T>(fn: () => Promise<T>): Promise<T> {
  let release!: () => void
  const gate = new Promise<void>(r => { release = r })
  const prev = lockChain
  lockChain = gate
  await prev
  try {
    return await fn()
  } finally {
    release()
  }
}

async function loadStore(rootDir: string): Promise<OAuthStore> {
  const path = resolve(rootDir, OAUTH_FILE)
  const raw = await readFile(path, 'utf-8').catch(() => '{}')
  try {
    return JSON.parse(raw) as OAuthStore
  } catch {
    return {}
  }
}

async function saveStore(rootDir: string, store: OAuthStore): Promise<void> {
  const path = resolve(rootDir, OAUTH_FILE)
  await writeFile(path, JSON.stringify(store, null, 2), 'utf-8')
}

async function readServerData(rootDir: string, serverName: string): Promise<ServerOAuthData> {
  const store = await loadStore(rootDir)
  return store[serverName] ?? {}
}

async function writeServerData(rootDir: string, serverName: string, data: Partial<ServerOAuthData>): Promise<void> {
  await withFileLock(async () => {
    const store = await loadStore(rootDir)
    store[serverName] = { ...store[serverName], ...data }
    await saveStore(rootDir, store)
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Pending auth URLs — in-memory, consumed by API routes
// ─────────────────────────────────────────────────────────────────────────────

const pendingAuths = new Map<string, URL>()

export function getPendingAuthUrl(serverName: string): URL | undefined {
  return pendingAuths.get(serverName)
}

export function clearPendingAuth(serverName: string): void {
  pendingAuths.delete(serverName)
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider factory
// ─────────────────────────────────────────────────────────────────────────────

export function createOAuthProvider(
  serverName: string,
  rootDir: string,
  callbackUrl: string,
): OAuthClientProvider {
  const _redirectUrl = new URL(callbackUrl)

  return {
    get redirectUrl() {
      return _redirectUrl
    },

    get clientMetadata(): OAuthClientMetadata {
      return {
        redirect_uris: [_redirectUrl.toString()],
        token_endpoint_auth_method: 'none',
        grant_types: ['authorization_code'],
        response_types: ['code'],
        client_name: `agent-mcp-${serverName}`,
        scope: 'read write',
      }
    },

    async clientInformation() {
      const data = await readServerData(rootDir, serverName)
      if (!data.clientId) return undefined
      return {
        client_id: data.clientId,
        ...(data.clientSecret && { client_secret: data.clientSecret }),
      }
    },

    async saveClientInformation(info) {
      await writeServerData(rootDir, serverName, {
        clientId: info.client_id,
        clientSecret: 'client_secret' in info ? (info.client_secret as string) : undefined,
      })
    },

    async tokens() {
      const data = await readServerData(rootDir, serverName)
      return data.tokens
    },

    async saveTokens(tokens: OAuthTokens) {
      await writeServerData(rootDir, serverName, { tokens })
    },

    async redirectToAuthorization(authorizationUrl: URL) {
      // Store the URL for the API to surface — don't open a browser
      pendingAuths.set(serverName, authorizationUrl)
      log.info({ server: serverName, url: authorizationUrl.toString() }, 'OAuth authorization required')
    },

    async saveCodeVerifier(codeVerifier: string) {
      await writeServerData(rootDir, serverName, { codeVerifier })
    },

    async codeVerifier() {
      const data = await readServerData(rootDir, serverName)
      if (!data.codeVerifier) throw new Error(`No code verifier for ${serverName}`)
      return data.codeVerifier
    },
  }
}
