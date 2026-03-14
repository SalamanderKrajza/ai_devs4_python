import { OPENROUTER_API_KEY, resolveModelForProvider } from "../../config.js";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY?.trim() ?? "";
const hasGeminiImageBackend = Boolean(GEMINI_API_KEY);
const hasOpenRouterImageBackend = Boolean(OPENROUTER_API_KEY);

// Validate API keys
if (!hasGeminiImageBackend && !hasOpenRouterImageBackend) {
  console.error("\x1b[31mError: image generation backend is not configured\x1b[0m");
  console.error("       Add one of these to the repo root .env file:");
  console.error("       OPENROUTER_API_KEY=sk-or-v1-...   # uses google/gemini-3.1-flash-image-preview");
  console.error("       GEMINI_API_KEY=...                # uses native Gemini image generation");
  process.exit(1);
}

if (!process.env.REPLICATE_API_TOKEN) {
  console.error(`\x1b[31mError: REPLICATE_API_TOKEN environment variable is not set\x1b[0m`);
  console.error("       Add it to the repo root .env file: REPLICATE_API_TOKEN=...");
  process.exit(1);
}

export { GEMINI_API_KEY };
export const IMAGE_BACKEND = hasOpenRouterImageBackend ? "openrouter" : "gemini";

export const api = {
  model: resolveModelForProvider("gpt-4.1"),
  visionModel: resolveModelForProvider("gpt-4.1"),
  maxOutputTokens: 16384,
  instructions: `You are a video generation agent using JSON-based prompting for consistent frame generation.

## WORKFLOW

### Step 1: Generate START Frame
1. Copy workspace/template.json → workspace/prompts/{scene}_{timestamp}.json
2. Edit ONLY the "subject" section for the STARTING pose/state
3. Read complete JSON, pass to create_image (aspect_ratio: "16:9", image_size: "2k")
4. Output: {scene}_frame_start_{timestamp}.png

### Step 2: Generate END Frame (from start frame)
1. Use create_image with reference_images: [start_frame_path]
2. Prompt describes the END state while referencing the start frame for character consistency
3. Example: "Same fox character, now landed in the snowdrift, snow particles around, happy expression"
4. Output: {scene}_frame_end_{timestamp}.png

### Step 3: Generate Video
Use image_to_video with BOTH frames:
- start_image: path to start frame
- end_image: path to end frame  
- prompt: describes the motion between frames

## WHEN TO SKIP END FRAME REFERENCE
Only generate end frame WITHOUT referencing start frame if:
- Character transforms completely (caterpillar → butterfly)
- Scene changes entirely (day → night with different location)
- User explicitly asks for dramatic change

Otherwise, ALWAYS use start frame as reference for end frame to maintain character consistency.

## Subject Section Format
{
  "subject": {
    "main": "happy red fox preparing to jump",
    "details": "fluffy orange fur, white chest, alert ears, bushy tail raised",
    "orientation": "side view facing right, legs bent ready to spring",
    "position": "left third of frame",
    "scale": "occupies 40% of frame height"
  }
}

## END FRAME Editing Prompt Example
When editing start frame to create end frame:
"Same fox character with identical fur colors and markings, now landed in a fluffy snowdrift. Fox is partially buried in snow up to chest, snow particles floating around, joyful expression with eyes closed, tail visible above snow. Keep exact same art style and line quality."

## Video Motion Prompt
Describe the transition between start and end frames:
"The fox leaps gracefully from the fence, arcs through the air with tail flowing, and lands softly in the snowdrift sending snow particles flying"

## RULES
- **START + END**: Always generate both frames for better video control
- **END FROM START**: Use start frame as reference when creating end frame (character consistency)
- **COPY FIRST**: Create new prompt file, never edit template.json directly
- **MINIMAL EDITS**: Only edit "subject" section, preserve style/colors/composition
- **16:9 FOR VIDEO**: Always use 16:9 aspect ratio

## FILE NAMING
- Start frame: {scene}_frame_start_{timestamp}.png
- End frame: {scene}_frame_end_{timestamp}.png
- Video: {scene}_video_{timestamp}.mp4

## DEFAULTS
- Duration: 10 seconds
- Aspect ratio: 16:9
- Resolution: 2k

Run autonomously. Report all output paths when complete.`
};

export const gemini = {
  apiKey: GEMINI_API_KEY,
  imageBackend: IMAGE_BACKEND,
  imageModel: IMAGE_BACKEND === "openrouter"
    ? "google/gemini-3.1-flash-image-preview"
    : "gemini-3-pro-image-preview",
  videoModel: "gemini-2.5-flash",
  endpoint: "https://generativelanguage.googleapis.com/v1beta/interactions",
  openRouterEndpoint: "https://openrouter.ai/api/v1/chat/completions"
};

export const outputFolder = "workspace/output";
