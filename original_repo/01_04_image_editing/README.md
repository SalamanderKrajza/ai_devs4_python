# 01_04_image_editing

Interactive image generation and editing with quality checks.

## Run

```bash
npm run lesson4:image_editing
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. For image generation and editing, set `OPENROUTER_API_KEY` or `GEMINI_API_KEY`.
4. Add optional source images to `workspace/input/`.

## What it does

1. Reads `workspace/style-guide.md` before generation
2. Generates new images or edits reference images
3. Analyzes the result for prompt adherence and quality issues
4. Saves final files to `workspace/output/`

## Notes

If both keys are present, image generation prefers OpenRouter with `google/gemini-3.1-flash-image-preview`. Use `clear` to reset the conversation and `exit` to quit the REPL.
