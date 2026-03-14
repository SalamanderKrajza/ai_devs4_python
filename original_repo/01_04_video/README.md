# 01_04_video

Video analysis, transcription, extraction, and question answering with Gemini plus MCP file tools.

## Run

```bash
npm run lesson4:video
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. Set `GEMINI_API_KEY` for video processing.
4. Put local videos in `workspace/input/` when not using YouTube URLs.

## What it does

1. Reads video files through `files-mcp`
2. Analyzes, transcribes, or extracts structured information from video
3. Saves JSON outputs to `workspace/output/` when requested
4. Supports both local video files and public YouTube URLs

## Notes

Use `clear` to reset the conversation and `exit` to quit the REPL.
