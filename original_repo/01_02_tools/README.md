# 01_02_tools

Minimal tool use with the Responses API — the model can use provider-native web search, then send the result through a mocked email tool.

## Run

```bash
bun run lesson2:tools
```

Backward-compatible alias:

```bash
bun run lesson2:minimal
```

## What it does

1. Defines two custom tools: `get_weather` and `send_email`
2. Enables provider-native web search through a shared helper in `config.js`
3. Sends the user message and available tools to the Responses API
4. Executes each tool in regular JavaScript
5. Sends the tool results back to the model
6. Prints the final natural-language answer

## Tools

| Tool | Description |
|------|-------------|
| `get_weather` | Return mock weather data for a city |
| `send_email` | Return a mocked confirmation that an email was sent |
| built-in web search | OpenAI uses `web_search_preview`; OpenRouter uses `:online` or the `web` plugin |
