# 01_04_reports

PDF report generation from HTML, local assets, and AI-generated images.

## Run

```bash
npm run lesson4:reports
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. For AI image generation inside reports, set `OPENROUTER_API_KEY` or `GEMINI_API_KEY`.
4. Put local assets in `workspace/input/` when needed.

## What it does

1. Reads `workspace/template.html` and `workspace/style-guide.md`
2. Writes working HTML files to `workspace/html/`
3. Optionally generates images with OpenRouter or Gemini
4. Converts the final HTML to PDF in `workspace/output/`

## Notes

If both keys are present, report image generation prefers OpenRouter with `google/gemini-3.1-flash-image-preview`. Use `clear` to reset the conversation and `exit` to quit the REPL.
