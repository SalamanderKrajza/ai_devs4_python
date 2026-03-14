/**
 * MCP client manager — creates, connects, and manages MCP server connections.
 */
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import type { McpConfig, McpServerConfig, McpToolInfo } from './types.js'
import { createOAuthProvider } from './oauth.js'
import { logger } from '../lib/logger.js'

const log = logger.child({ name: 'mcp' })

const SEPARATOR = '__'
const CALL_TIMEOUT_MS = 30_000

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

export async function loadMcpConfig(rootDir: string): Promise<McpConfig> {
  const raw = await readFile(resolve(rootDir, '.mcp.json'), 'utf-8').catch(() => null)
  if (!raw) return { mcpServers: {} }

  try {
    return JSON.parse(raw) as McpConfig
  } catch (err) {
    throw new Error(`.mcp.json parse error: ${(err as Error).message}`)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transport
// ─────────────────────────────────────────────────────────────────────────────

function createStdioTransport(serverConfig: McpServerConfig & { transport?: 'stdio' }, rootDir: string) {
  return new StdioClientTransport({
    command: serverConfig.command,
    args: serverConfig.args,
    env: {
      PATH: process.env.PATH ?? '',
      HOME: process.env.HOME ?? '',
      NODE_ENV: process.env.NODE_ENV ?? '',
      ...serverConfig.env,
    },
    cwd: serverConfig.cwd ?? rootDir,
    stderr: 'inherit',
  })
}

function createHttpTransport(
  serverName: string,
  serverConfig: McpServerConfig & { transport: 'http' },
  rootDir: string,
  baseUrl: string,
) {
  const callbackUrl = `${baseUrl}/mcp/${serverName}/callback`
  const authProvider = createOAuthProvider(serverName, rootDir, callbackUrl)

  return new StreamableHTTPClientTransport(
    new URL(serverConfig.url),
    {
      authProvider,
      ...(serverConfig.headers && {
        requestInit: { headers: serverConfig.headers },
      }),
    },
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Content helpers
// ─────────────────────────────────────────────────────────────────────────────

function extractText(content: unknown): string {
  if (!Array.isArray(content)) return ''
  return content
    .filter(c => c != null && typeof c === 'object' && c.type === 'text' && typeof c.text === 'string')
    .map(c => c.text as string)
    .join('\n')
}

function parsePrefixedName(prefixedName: string) {
  const idx = prefixedName.indexOf(SEPARATOR)
  if (idx === -1) return undefined
  return {
    server: prefixedName.slice(0, idx),
    tool: prefixedName.slice(idx + SEPARATOR.length),
  }
}

function mapTools(serverName: string, tools: Array<{ name: string; description?: string; inputSchema: unknown }>): McpToolInfo[] {
  return tools.map(t => ({
    server: serverName,
    originalName: t.name,
    prefixedName: `${serverName}${SEPARATOR}${t.name}`,
    description: t.description,
    inputSchema: (t.inputSchema ?? {}) as Record<string, unknown>,
  }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Manager
// ─────────────────────────────────────────────────────────────────────────────

export interface McpManager {
  servers(): string[]
  serverStatus(name: string): 'connected' | 'auth_required' | 'disconnected'
  listTools(): Promise<McpToolInfo[]>
  listServerTools(serverName: string): Promise<McpToolInfo[]>
  callTool(prefixedName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<string>
  parseName(prefixedName: string): { server: string; tool: string } | undefined
  finishAuth(serverName: string, authorizationCode: string): Promise<void>
  close(): Promise<void>
}

export async function createMcpManager(rootDir: string, baseUrl: string): Promise<McpManager> {
  const config = await loadMcpConfig(rootDir)
  const clients = new Map<string, Client>()
  const transports = new Map<string, StreamableHTTPClientTransport | StdioClientTransport>()
  const authRequired = new Set<string>()

  // Connect to all configured servers
  const entries = Object.entries(config.mcpServers)
  const results = await Promise.allSettled(
    entries.map(async ([name, serverConfig]) => {
      const client = new Client(
        { name: 'agent-mcp-client', version: '1.0.0' },
        { capabilities: {} },
      )

      const transport = serverConfig.transport === 'http'
        ? createHttpTransport(name, serverConfig, rootDir, baseUrl)
        : createStdioTransport(serverConfig, rootDir)

      transports.set(name, transport)

      try {
        await client.connect(transport)
        return { name, client, needsAuth: false }
      } catch (err) {
        // If auth is required, the transport throws UnauthorizedError
        // but the OAuth provider has stored the auth URL
        const isAuthError = (err as Error).message?.includes('Unauthorized')
          || (err as Error).constructor?.name === 'UnauthorizedError'
        if (isAuthError && serverConfig.transport === 'http') {
          return { name, client, needsAuth: true }
        }
        throw err
      }
    }),
  )

  for (const result of results) {
    if (result.status === 'fulfilled') {
      const { name, client, needsAuth } = result.value
      if (needsAuth) {
        authRequired.add(name)
        log.info({ server: name }, 'awaiting OAuth authorization')
      } else {
        clients.set(name, client)
        log.info({ server: name }, 'connected')
      }
    } else {
      log.error({ err: result.reason }, 'connection failed')
    }
  }

  function serverStatus(name: string): 'connected' | 'auth_required' | 'disconnected' {
    if (clients.has(name)) return 'connected'
    if (authRequired.has(name)) return 'auth_required'
    return 'disconnected'
  }

  async function finishAuth(serverName: string, authorizationCode: string): Promise<void> {
    const transport = transports.get(serverName)
    if (!transport || !(transport instanceof StreamableHTTPClientTransport)) {
      throw new Error(`No HTTP transport for server: ${serverName}`)
    }

    await transport.finishAuth(authorizationCode)

    // Reconnect — create a fresh client on the existing transport
    const client = new Client(
      { name: 'agent-mcp-client', version: '1.0.0' },
      { capabilities: {} },
    )
    await client.connect(transport)
    clients.set(serverName, client)
    authRequired.delete(serverName)
    log.info({ server: serverName }, 'OAuth complete, connected')
  }

  async function listServerTools(serverName: string): Promise<McpToolInfo[]> {
    const client = clients.get(serverName)
    if (!client) return []
    const result = await client.listTools()
    return mapTools(serverName, result.tools)
  }

  async function listTools(): Promise<McpToolInfo[]> {
    const all = await Promise.all([...clients.keys()].map(listServerTools))
    return all.flat()
  }

  async function callTool(prefixedName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<string> {
    const parsed = parsePrefixedName(prefixedName)
    if (!parsed) throw new Error(`Invalid MCP tool name: ${prefixedName}`)

    const client = clients.get(parsed.server)
    if (!client) {
      if (authRequired.has(parsed.server)) {
        throw new Error(`MCP server "${parsed.server}" requires OAuth authorization`)
      }
      throw new Error(`MCP server not connected: ${parsed.server}`)
    }

    const timeout = AbortSignal.timeout(CALL_TIMEOUT_MS)
    const combined = signal ? AbortSignal.any([signal, timeout]) : timeout

    const result = await client.callTool(
      { name: parsed.tool, arguments: args },
      undefined,
      { signal: combined },
    )

    if (result.isError) {
      const msg = extractText(result.content)
      throw new Error(msg || 'MCP tool returned an error')
    }

    if (result.structuredContent) {
      return JSON.stringify(result.structuredContent)
    }

    const text = extractText(result.content)
    return text || JSON.stringify(result.content)
  }

  async function close() {
    const closeResults = await Promise.allSettled(
      [...clients.entries()].map(async ([name, client]) => {
        await client.close()
        log.info({ server: name }, 'closed')
      }),
    )
    for (const r of closeResults) {
      if (r.status === 'rejected') {
        log.error({ err: r.reason }, 'close failed')
      }
    }
    clients.clear()
    transports.clear()
    authRequired.clear()
  }

  return {
    servers: () => [...new Set([...clients.keys(), ...authRequired])],
    serverStatus,
    listTools,
    listServerTools,
    callTool,
    parseName: parsePrefixedName,
    finishAuth,
    close,
  }
}
