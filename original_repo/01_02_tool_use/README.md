# 01_02_tool_use

Function calling with sandboxed filesystem tools — the model lists, reads, writes, and deletes files through tool definitions.

## Run

```bash
npm run lesson2:tool_use
```

## What it does

1. Defines 6 filesystem tools (`list_files`, `read_file`, `write_file`, `delete_file`, `create_directory`, `file_info`)
2. Resets `sandbox/` to an empty state before running the demo
3. Runs each example query as a separate conversation
4. Executes tool calls, appends results within that query, and prints the final answer
5. All operations are sandboxed — path traversal is blocked programmatically

## Tools

| Tool | Description |
|------|-------------|
| `list_files` | List files and directories at a path |
| `read_file` | Read file contents |
| `write_file` | Create or overwrite a file |
| `delete_file` | Delete a file |
| `create_directory` | Create a directory (recursive) |
| `file_info` | Get file/directory metadata |

See `TOOLS.md` for full schemas and examples.
