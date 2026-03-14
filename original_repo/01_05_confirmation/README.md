# 01_05_confirmation

Interactive file-and-email agent with whitelist enforcement and a terminal confirmation step before sending email.

## Run

```bash
npm run lesson5:confirmation
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. Set `RESEND_API_KEY` and `RESEND_FROM`.
4. Edit `workspace/whitelist.json` with allowed recipients or domains.

## What it does

1. Connects to the local file MCP server from `mcp.json`
2. Lets the agent read, search, and write workspace files
3. Drafts emails with the model using `gpt-5.4`
4. Requires explicit terminal confirmation before `send_email` is executed

## Notes

The whitelist supports exact emails like `user@example.com` and whole domains like `@example.com`. Use `clear` to reset the conversation and `exit` to quit the REPL.
