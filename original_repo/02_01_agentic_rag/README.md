# 02_01_agentic_rag

Agentic RAG with multi-step retrieval over MCP file tools.

## Run

```bash
npm run lesson6:agentic_rag
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.

## What it does

1. Connects to an MCP file server (`list`, `search`, `read`)
2. Runs an agentic loop — the model decides which files to scan, search, and read
3. Iterates through multiple search angles (synonyms, related terms) before answering
4. Maintains conversation history for follow-up questions

## Notes

On startup the agent asks for confirmation because it can consume a noticeable number of tokens. A pre-recorded example session is available in `demo/example.md`. Use `clear` to reset conversation and `exit` to quit the REPL.
