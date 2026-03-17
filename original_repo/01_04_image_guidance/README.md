# 01_04_image_guidance

Pose-guided image generation from JSON templates and reference images.

## Run

```bash
npm run lesson4:image_guidance
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. For image generation, set `OPENROUTER_API_KEY` or `GEMINI_API_KEY`.
4. Add pose reference files to `workspace/reference/`.
5. Keep `workspace/template.json` as the base style template.

## What it does

1. Copies `workspace/template.json` into `workspace/prompts/`
2. Edits only the subject section of the copied JSON
3. Uses a pose reference image from `workspace/reference/`
4. Generates final images to `workspace/output/`

## Notes

The default workflow expects a walking pose reference such as `workspace/reference/walking-pose.png`. If both keys are present, image generation prefers OpenRouter with `google/gemini-3.1-flash-image-preview`. Use `clear` to reset the conversation and `exit` to quit the REPL.
