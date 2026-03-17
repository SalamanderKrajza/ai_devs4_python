# 01_03_mcp_core

Core MCP capabilities over stdio: tools, resources, prompts, elicitation, and sampling.

## Run

```bash
npm run lesson3:mcp_core
```

## What it does

1. Spawns a local MCP server as a subprocess over stdio
2. Lists the available tools, resources, and prompts
3. Calls `calculate` directly through MCP
4. Runs `summarize_with_confirmation` to demonstrate elicitation and sampling

## MCP capabilities

| Type | Name | Description |
|------|------|-------------|
| Tool | `calculate` | Basic arithmetic (add, subtract, multiply, divide) |
| Tool | `summarize_with_confirmation` | Summarizes text after elicitation (user confirmation) and sampling (LLM completion) |
| Resource | `config://project` | Static project configuration |
| Resource | `data://stats` | Dynamic runtime statistics |
| Prompt | `code-review` | Code review template with args (code, language, focus) |

## Notes

The client handles sampling with the shared workspace AI config, so setup lives in the root `README.md` and `.env`.
