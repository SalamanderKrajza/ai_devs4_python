# 01_04_image_recognition

Vision-based image classification using MCP file operations and a native image-understanding tool.

## Run

```bash
npm run lesson4:image_recognition
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. Put source images in `images/`.
4. Keep character profiles in `knowledge/`.

## What it does

1. Reads the character descriptions from `knowledge/`
2. Analyzes each file in `images/`
3. Matches visible traits against the profiles
4. Sorts files into `images/organized/<category>/`

## Notes

Use `knowledge/*.md` to define the classification rules before running the example.
