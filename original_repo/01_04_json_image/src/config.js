import { OPENROUTER_API_KEY, resolveModelForProvider } from "../../config.js";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY?.trim() ?? "";
const hasGeminiImageBackend = Boolean(GEMINI_API_KEY);
const hasOpenRouterImageBackend = Boolean(OPENROUTER_API_KEY);

if (!hasGeminiImageBackend && !hasOpenRouterImageBackend) {
  console.error("\x1b[31mError: image generation backend is not configured\x1b[0m");
  console.error("       Add one of these to the repo root .env file:");
  console.error("       OPENROUTER_API_KEY=sk-or-v1-...   # uses google/gemini-3.1-flash-image-preview");
  console.error("       GEMINI_API_KEY=...                # uses native Gemini image generation");
  process.exit(1);
}

export { GEMINI_API_KEY };
export const IMAGE_BACKEND = hasOpenRouterImageBackend ? "openrouter" : "gemini";

export const api = {
  model: resolveModelForProvider("gpt-5.2"),
  visionModel: resolveModelForProvider("gpt-5.2"),
  maxOutputTokens: 16384,
  instructions: `You are an image generation agent using JSON-based prompting with minimal token usage.

## WORKFLOW (Token-Efficient)

1. **COPY template**: Copy workspace/template.json → workspace/prompts/{subject_name}_{timestamp}.json
2. **EDIT subject only**: Use MCP file tools to edit ONLY the "subject" section in the copied file
3. **READ prompt file**: Read the complete JSON from the prompt file
4. **GENERATE**: Pass the JSON content to create_image with format settings from the template
5. **REPORT**: Return the generated image path and prompt file path

## PROCESS STEPS

1. Copy template.json to workspace/prompts/ with descriptive filename
   Example: workspace/prompts/phoenix_1769959315686.json

2. Edit the copied file - ONLY modify the "subject" object:
   {
     "subject": {
       "main": "phoenix",
       "details": "rising from flames, wings fully spread, feathers transforming to fire",
       "orientation": "three-quarter view, facing slightly left",
       "position": "centered horizontally and vertically",
       "scale": "occupies 60% of frame height"
     }
   }
   Keep orientation, position, scale from template unless user specifies otherwise.

3. Read the complete JSON from the prompt file

4. Pass JSON content to create_image. Extract technical settings from the JSON:
   - aspect_ratio: use technical.aspect_ratio from JSON (e.g., "1:1", "16:9")
   - image_size: use technical.resolution from JSON (e.g., "1k", "2k", "4k")

## RULES
- **COPY FIRST**: Always create a new prompt file, never edit template.json directly
- **MINIMAL EDITS**: Only edit the "subject" section, preserve everything else
- **VERSION FILES**: Each generation gets its own prompt file for history
- **READ BEFORE GENERATE**: Always read the complete JSON before passing to create_image
- **USE TEMPLATE SETTINGS**: Always use aspect_ratio and resolution from the template's technical section

## FILE NAMING
- Format: {subject_slug}_{timestamp}.json
- Example: dragon_breathing_fire_1769959315686.json
- Keep names short but descriptive`
};

export const gemini = {
  apiKey: GEMINI_API_KEY,
  imageBackend: IMAGE_BACKEND,
  imageModel: IMAGE_BACKEND === "openrouter"
    ? "google/gemini-3.1-flash-image-preview"
    : "gemini-3-pro-image-preview",
  endpoint: "https://generativelanguage.googleapis.com/v1beta/interactions",
  openRouterEndpoint: "https://openrouter.ai/api/v1/chat/completions"
};

export const outputFolder = "workspace/output";
