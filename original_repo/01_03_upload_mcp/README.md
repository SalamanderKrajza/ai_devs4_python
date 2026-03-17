# 01_03_upload_mcp

Upload assistant that combines a local file MCP server with a remote upload MCP server.

## Run

```bash
npm run lesson3:upload_mcp
```

## Required setup

Before running, edit `mcp.json` and replace:

```json
"url": "https://URL_TO_YOUR_MCP_SERVER/mcp"
```

with the real URL of the MCP deployment you created in the AI_devs lesson, for example:

```json
"url": "https://your-domain.example/mcp"
```

If you leave the placeholder value in place, the example will stop with a validation error.

## What it does

1. Connects to the MCP servers listed in `mcp.json`
2. Lists files in `workspace/`
3. Uploads files that are not already recorded
4. Saves results to `uploaded.md`

## Tools

| Server | Tool | Description |
|--------|------|-------------|
| `files` (stdio) | `fs_read`, `fs_search`, `fs_write`, `fs_manage` | Local file operations in `workspace/` |
| `uploadthing` (http) | *(from your deployment)* | Upload files to the configured remote server |

## Notes

Place files to upload in `workspace/`. The agent skips `uploaded.md` and previously uploaded entries.
