# 01_03_mcp_native

One agent using both MCP tools and native function tools in the same loop.

## Run

```bash
npm run lesson3:mcp_native
```

## What it does

1. Starts an in-memory MCP server with weather and time tools
2. Adds native tools for calculation and text transformation
3. Exposes all tools to one model as a single toolset
4. Runs a few demo queries, including a mixed-tool example

## Tools

| Tool | Description |
|------|-------------|
| `get_weather` (MCP) | Mock weather data for a city |
| `get_time` (MCP) | Current time in a timezone |
| `calculate` (native) | Basic math (add, subtract, multiply, divide) |
| `uppercase` (native) | Convert text to uppercase |
