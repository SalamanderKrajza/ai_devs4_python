# 01_03_mcp_translator

Translation agent built on `files-mcp` that watches a folder and writes English versions of incoming files.

## Run

```bash
npm run lesson3:mcp_translator
```

## Example curl

```bash
curl -X POST "http://localhost:3000/api/translate" -H "Content-Type: application/json" -d '{"text":"To jest przykladowy tekst po polsku."}'
```

## What it does

1. Connects to the `files` MCP server defined in `mcp.json`
2. Watches `workspace/translate/` for supported files
3. Saves translated output to `workspace/translated/`
4. Exposes `POST /api/chat` and `POST /api/translate`

## Tools

| Tool | Description |
|------|-------------|
| `fs_read` | Read files and explore directories |
| `fs_search` | Find files and search content |
| `fs_write` | Create and update files |
| `fs_manage` | Structural file operations |

## Notes

Put source files in `workspace/translate/`. The example creates missing directories on startup.
