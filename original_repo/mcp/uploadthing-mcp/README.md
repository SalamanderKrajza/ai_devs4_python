# UploadThing MCP Server

> Warning: You connect this server to your MCP client at your own responsibility. Language models can make mistakes, misinterpret instructions, or perform unintended actions. Review tool outputs and verify changes before relying on uploaded files.

A streamable HTTP MCP server for [UploadThing](https://uploadthing.com/) that lets you upload, list, and manage files — locally or remotely.

### Notice

This repo works in two ways:

- As a Node/Hono server for local workflows
- As a Cloudflare Worker for remote interactions

### Motivation

UploadThing is a file hosting service optimized for developers. This MCP server enables AI agents to:

- Upload files from URLs or base64 content directly to UploadThing
- List and search uploaded files with filtering options
- Rename or delete files programmatically
- Handle batch operations efficiently

The server is designed with clear schema descriptions and human-readable responses, making it easy for LLMs to understand what happened and suggest next steps.

## Tools

| Tool | Description |
|------|-------------|
| `upload_files` | Upload files from URLs or base64 content to UploadThing |
| `list_files` | List uploaded files with optional filtering |
| `manage_files` | Rename or delete files |

## Installation & development

Prerequisites: [Bun](https://bun.sh/), [Node.js 24+](https://nodejs.org), [UploadThing](https://uploadthing.com) account.

You also need an MCP client such as:

- [HeyAlice](https://heyalice.app)
- [Claude](https://claude.ai)
- [Cursor](https://cursor.sh)

### Quick start (local workflow with API key)

Get your UploadThing token from [uploadthing.com/dashboard](https://uploadthing.com/dashboard) → API Keys.

```bash
cd uploadthing-mcp
bun install
cp .env.example .env
# Update UPLOADTHING_TOKEN in .env
bun dev
```

Connect to Alice (Settings → MCP) or use Claude Desktop:

```json
{
  "mcpServers": {
    "uploadthing": {
      "command": "bunx",
      "args": [
        "mcp-remote",
        "http://localhost:3000/mcp",
        "--header",
        "Authorization: Bearer ${UPLOADTHING_TOKEN}"
      ]
    }
  }
}
```

### Cloudflare Worker

```bash
bun x wrangler secret put UPLOADTHING_TOKEN
bun x wrangler deploy
```

Endpoint: `https://<worker-name>.<account>.workers.dev/mcp`

---

## Environment

Required:

```env
UPLOADTHING_TOKEN=your-uploadthing-token
```

Optional:

```env
PORT=3000
HOST=0.0.0.0
NODE_ENV=development
```

---

## Client configuration

MCP Inspector (quick test):

```bash
bunx @modelcontextprotocol/inspector
# Connect to: http://localhost:3000/mcp
```

Claude Desktop / Cursor via mcp-remote:

```json
{
  "mcpServers": {
    "uploadthing": {
      "command": "bunx",
      "args": [
        "mcp-remote",
        "http://localhost:3000/mcp",
        "--transport",
        "http-only"
      ],
      "env": { "NO_PROXY": "127.0.0.1,localhost" }
    }
  }
}
```

---

## Examples

### 1) Upload a file from URL

```json
{
  "name": "upload_files",
  "arguments": {
    "files": [
      {
        "url": "https://example.com/image.png",
        "name": "my-image.png"
      }
    ]
  }
}
```

### 2) List all uploaded files

```json
{
  "name": "list_files",
  "arguments": {
    "limit": 20
  }
}
```

### 3) Rename a file

```json
{
  "name": "manage_files",
  "arguments": {
    "action": "rename",
    "fileKey": "abc123",
    "newName": "renamed-file.png"
  }
}
```

### 4) Delete files

```json
{
  "name": "manage_files",
  "arguments": {
    "action": "delete",
    "fileKeys": ["abc123", "def456"]
  }
}
```

## License

MIT
