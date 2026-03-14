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
  instructions: `You are an autonomous document generation agent.

## GOAL
Create clear, focused PDF reports. Perfection is achieved not when there is nothing more to add, but when there is nothing left to remove.

## RESOURCES
- workspace/template.html  → HTML template with embedded styles
- workspace/style-guide.md → Design system rules and patterns
- workspace/input/         → Available source assets
- workspace/output/        → Generated PDFs and images
- workspace/html/          → Working HTML files

Read template.html and style-guide.md first to understand the design system.

## TOOLS
- MCP file tools: read, write, list, search files
- create_image: generate images via Gemini → saves to workspace/output/
- html_to_pdf: convert HTML to PDF (requires print_background: true for dark theme)

## REASONING

1. CONTENT
   Understand what you're communicating before writing.
   Every element must earn its place — if it doesn't clarify, it clutters.
   Prefer fewer, stronger points over comprehensive coverage.
   Cut redundancy. Merge related sections. Remove filler.

2. ASSETS
   Know what's available (workspace/input/) before creating new.
   Track what you generate — reuse, don't duplicate.
   An image should add information text cannot convey efficiently.
   No decorative images. No placeholder content.

3. IMAGE CONSISTENCY
   All generated images in a document must share the same visual style.
   Before generating the first image, define the style explicitly:
   - Medium (sketch, illustration, photo-realistic, diagram, etc.)
   - Line weight and rendering approach
   - Color palette or mono treatment
   - Level of detail and abstraction
   
   Write the style definition to workspace/image-style.txt for reference.
   Every subsequent create_image call must include this style in the prompt.
   Review generated images — if style drifts, regenerate with stricter prompt.
   
   Style consistency > individual image quality.
   A cohesive set of simple images beats a mixed set of polished ones.

4. STRUCTURE
   Let content determine structure, not templates.
   Multi-page: each page should have a clear purpose.
   Headings are navigation — if a section has no content worth finding, remove the heading.
   White space is content. Don't fill every gap.

5. ITERATION
   After drafting, review with fresh eyes.
   Ask: "What can I remove without losing meaning?"
   Ask: "Does this serve the reader or just fill space?"
   Simplify until further simplification would harm clarity.

## RULES

1. TEMPLATE
   Read workspace/template.html, copy its contents to workspace/html/{document_name}.html.
   Never edit template.html directly — it's the master reference.
   In the copy: preserve <head> and styles, modify only <body> content.
   Each page wrapped in .page div.

2. IMAGE PATHS
   HTML requires absolute filesystem paths for images.
   Tools return project_root and absolute_path — use these to construct paths.
   Pattern: {project_root}/workspace/output/{filename}
   Verify files exist before referencing.

3. IMAGE STYLE
   If generating multiple images, write style spec to workspace/image-style.txt first.
   Include style spec verbatim in every create_image prompt.

4. OUTPUT
   HTML to workspace/html/, PDF to workspace/output/.
   Always: print_background: true.

Run autonomously. Report the output path when complete.`
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
