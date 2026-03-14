/**
 * MCP OAuth routes — authorization flow and server status.
 */
import { Hono } from 'hono'
import type { ApiResponse } from '../errors/index.js'
import { getPendingAuthUrl, clearPendingAuth } from '../mcp/index.js'
import type { RuntimeContext } from '../runtime/index.js'

type Env = { Variables: { runtime: RuntimeContext } }

const mcp = new Hono<Env>()

// ─────────────────────────────────────────────────────────────────────────────
// GET /mcp/servers — list all MCP servers and their status
// ─────────────────────────────────────────────────────────────────────────────

mcp.get('/servers', (c) => {
  const runtime = c.get('runtime')
  if (!runtime) {
    return c.json<ApiResponse>({ data: null, error: { message: 'Runtime not initialized' } }, 500)
  }

  const servers = runtime.mcp.servers().map(name => ({
    name,
    status: runtime.mcp.serverStatus(name),
  }))

  return c.json<ApiResponse>({ data: { servers }, error: null })
})

// ─────────────────────────────────────────────────────────────────────────────
// GET /mcp/:server/auth — start OAuth flow, returns authorization URL
// ─────────────────────────────────────────────────────────────────────────────

mcp.get('/:server/auth', (c) => {
  const serverName = c.req.param('server')
  const runtime = c.get('runtime')
  if (!runtime) {
    return c.json<ApiResponse>({ data: null, error: { message: 'Runtime not initialized' } }, 500)
  }

  const status = runtime.mcp.serverStatus(serverName)
  if (status === 'connected') {
    return c.json<ApiResponse>({ data: { status: 'already_connected' }, error: null })
  }
  if (status === 'disconnected') {
    return c.json<ApiResponse>({ data: null, error: { message: `Server "${serverName}" not configured` } }, 404)
  }

  const authUrl = getPendingAuthUrl(serverName)
  if (!authUrl) {
    return c.json<ApiResponse>({
      data: null,
      error: { message: `No pending authorization for "${serverName}". Restart the server.` },
    }, 400)
  }

  return c.json<ApiResponse>({
    data: {
      server: serverName,
      status: 'auth_required',
      authorizationUrl: authUrl.toString(),
    },
    error: null,
  })
})

// ─────────────────────────────────────────────────────────────────────────────
// GET /mcp/:server/callback — OAuth callback, renders styled status page
// ─────────────────────────────────────────────────────────────────────────────

mcp.get('/:server/callback', async (c) => {
  const serverName = c.req.param('server')
  const code = c.req.query('code')
  const error = c.req.query('error')
  const errorDescription = c.req.query('error_description')
  const runtime = c.get('runtime')

  // Error from authorization server
  if (error) {
    return c.html(renderCallbackPage({
      server: serverName,
      success: false,
      message: errorDescription || error,
    }))
  }

  if (!code) {
    return c.html(renderCallbackPage({
      server: serverName,
      success: false,
      message: 'Missing authorization code',
    }), 400)
  }

  if (!runtime) {
    return c.html(renderCallbackPage({
      server: serverName,
      success: false,
      message: 'Server runtime not available',
    }), 500)
  }

  try {
    await runtime.mcp.finishAuth(serverName, code)
    clearPendingAuth(serverName)
    return c.html(renderCallbackPage({
      server: serverName,
      success: true,
      message: 'Connected successfully',
    }))
  } catch (err) {
    return c.html(renderCallbackPage({
      server: serverName,
      success: false,
      message: (err as Error).message,
    }), 500)
  }
})

// ─────────────────────────────────────────────────────────────────────────────
// Callback page renderer
// ─────────────────────────────────────────────────────────────────────────────

function renderCallbackPage(opts: { server: string; success: boolean; message: string }): string {
  const { server, success, message } = opts
  const icon = success ? '&#10003;' : '&#10007;'
  const title = success ? 'Connection Established' : 'Connection Failed'
  const accent = success ? '#10b981' : '#ef4444'
  const bgAccent = success ? '#ecfdf5' : '#fef2f2'

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MCP — ${title}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .card {
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
      max-width: 440px;
      width: 100%;
      padding: 2.5rem;
      text-align: center;
    }
    .icon {
      width: 64px;
      height: 64px;
      border-radius: 50%;
      background: ${bgAccent};
      color: ${accent};
      font-size: 28px;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 1.5rem;
    }
    h1 {
      font-size: 1.375rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    .server {
      display: inline-block;
      background: #f1f5f9;
      color: #475569;
      padding: 0.25rem 0.75rem;
      border-radius: 6px;
      font-size: 0.875rem;
      font-family: 'SF Mono', SFMono-Regular, Menlo, monospace;
      margin-bottom: 1rem;
    }
    .message {
      color: #64748b;
      font-size: 0.9375rem;
      line-height: 1.5;
    }
    .close-hint {
      margin-top: 1.5rem;
      font-size: 0.8125rem;
      color: #94a3b8;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">${icon}</div>
    <h1>${title}</h1>
    <div class="server">${server}</div>
    <p class="message">${message}</p>
    <p class="close-hint">You can close this tab.</p>
  </div>
</body>
</html>`
}

export { mcp }
