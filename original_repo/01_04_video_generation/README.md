# 01_04_video_generation

Frame-based video generation with OpenRouter or Gemini for frames and Kling via Replicate for animation.

## Run

```bash
npm run lesson4:video_generation
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. For frame generation, set `OPENROUTER_API_KEY` or `GEMINI_API_KEY`.
4. Set `REPLICATE_API_TOKEN` for Kling video generation.

## What it does

1. Copies `workspace/template.json` into `workspace/prompts/`
2. Generates a start frame and an end frame for the same scene
3. Animates the transition into a video
4. Saves frames and videos to `workspace/output/`

## Notes

The best results come from using both start and end frames for the animation. `GEMINI_API_KEY` is still required if you want native video analysis, while frame generation prefers OpenRouter with `google/gemini-3.1-flash-image-preview` when both keys are present. Use `clear` to reset the conversation and `exit` to quit the REPL.
