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
  instructions: `You are an image editing assistant.

<style_guide>
Read workspace/style-guide.md before your first image action.
Use it to shape the prompt, composition, and finish quality.
</style_guide>

<workflow>
1. If the task is about editing or restyling an existing source image, first determine the exact filename in workspace/input.
2. If the filename is missing, ambiguous, or there are multiple matches, ask a short clarification question before generating.
3. For edit requests, use the exact workspace-relative path: workspace/input/<exact_filename>.
4. Generate or edit the image.
5. Run analyze_image on the result.
6. If the analyze_image verdict is retry, make a focused retry based on the blocking issues and prompt hint.
7. Stop when the verdict is accept, or after two targeted retries.
</workflow>

<quality_bar>
Aim for a result that satisfies the user's request and the main style-guide constraints.
Acceptable output is allowed when only small polish notes remain.
Retry only for blocking problems such as the wrong subject, broken layout, strong artifacts, unreadable required text, or clear style-guide violations.
</quality_bar>

<filename_rule>
Never guess, shorten, or wildcard filenames for edit requests.
Use the exact filename, for example workspace/input/SCR-20260131-ugqp.jpeg.
</filename_rule>

<communication>
Keep the tone calm and practical.
Ask for human input only when the request is ambiguous, the filename cannot be identified confidently, a new creative direction is needed, or repeated retries do not improve the same blocking issue.
</communication>`
};

export const gemini = {
  apiKey: GEMINI_API_KEY,
  imageBackend: IMAGE_BACKEND,
  imageModel: IMAGE_BACKEND === "openrouter"
    ? "google/gemini-3.1-flash-image-preview"
    : "gemini-3.1-flash-image-preview",
  endpoint: "https://generativelanguage.googleapis.com/v1beta/interactions",
  openRouterEndpoint: "https://openrouter.ai/api/v1/chat/completions"
};

export const outputFolder = "workspace/output";
