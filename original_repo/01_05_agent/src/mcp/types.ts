/**
 * MCP configuration and client types
 */

export interface McpStdioServer {
  transport?: 'stdio'
  command: string
  args?: string[]
  env?: Record<string, string>
  cwd?: string
}

export interface McpHttpServer {
  transport: 'http'
  url: string
  headers?: Record<string, string>
}

export type McpServerConfig = McpStdioServer | McpHttpServer

export interface McpConfig {
  mcpServers: Record<string, McpServerConfig>
}

export interface McpOAuthConfig {
  [serverName: string]: {
    accessToken?: string
    refreshToken?: string
    [key: string]: unknown
  }
}

/** Resolved MCP tool with server prefix */
export interface McpToolInfo {
  server: string
  originalName: string
  prefixedName: string
  description?: string
  inputSchema: Record<string, unknown>
}
