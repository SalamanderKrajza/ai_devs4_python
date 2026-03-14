# 01_04_json_image

Token-efficient image generation from JSON prompt templates.

## Run

```bash
npm run lesson4:json_image
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. For image generation, set `OPENROUTER_API_KEY` or `GEMINI_API_KEY`.

## What it does

1. Copies `workspace/template.json` into `workspace/prompts/`
2. Edits only the subject section of the copied JSON
3. Generates an image from the full JSON prompt
4. Saves outputs to `workspace/output/`

## Notes

This example is built around reproducible prompt files rather than one-off raw prompts. If both keys are present, image generation prefers OpenRouter with `google/gemini-3.1-flash-image-preview`. Use `clear` to reset the conversation and `exit` to quit the REPL.
