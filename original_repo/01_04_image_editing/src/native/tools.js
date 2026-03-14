/**
 * Native tools for image editing agent.
 * 
 * Tools:
 * - create_image: Generate or edit images (reference_images optional)
 * - analyze_image: Evaluate image quality and prompt adherence
 */

import { readFile, writeFile, mkdir } from "fs/promises";
import { join, extname, dirname } from "path";
import { fileURLToPath } from "url";
import { generateImage, editImage, editImageWithReferences } from "./gemini.js";
import { vision } from "./vision.js";
import log from "../helpers/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, "../..");

/**
 * MIME type mapping for common image formats.
 */
const getMimeType = (filepath) => {
  const ext = extname(filepath).toLowerCase();
  const mimeTypes = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp"
  };
  return mimeTypes[ext] || "image/png";
};

/**
 * Get file extension from MIME type.
 */
const getExtension = (mimeType) => {
  const extensions = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp"
  };
  return extensions[mimeType] || ".png";
};

/**
 * Generate a unique filename with timestamp.
 */
const generateFilename = (prefix, mimeType) => {
  const timestamp = Date.now();
  const ext = getExtension(mimeType);
  return `${prefix}_${timestamp}${ext}`;
};

/**
 * Ensure directory exists.
 */
const ensureDir = async (dir) => {
  try {
    await mkdir(dir, { recursive: true });
  } catch (e) {
    if (e.code !== "EEXIST") throw e;
  }
};

const extractTaggedValue = (text, tag) => {
  const match = text.match(new RegExp(`^${tag}:\\s*(.+)$`, "im"));
  return match?.[1]?.trim() ?? "";
};

const extractBulletSection = (text, section) => {
  const lines = text.split("\n");
  const header = `${section}:`;
  const startIndex = lines.findIndex((line) => line.trim().toUpperCase() === header);

  if (startIndex === -1) {
    return [];
  }

  const items = [];

  for (let index = startIndex + 1; index < lines.length; index++) {
    const trimmed = lines[index].trim();

    if (!trimmed) {
      continue;
    }

    if (/^[A-Z_ ]+:$/.test(trimmed)) {
      break;
    }

    if (trimmed.startsWith("- ")) {
      items.push(trimmed.slice(2).trim());
    }
  }

  return items;
};

const parseAnalysisReport = (analysis) => {
  const rawVerdict = extractTaggedValue(analysis, "VERDICT").toUpperCase();
  const scoreText = extractTaggedValue(analysis, "SCORE");
  const score = Number.parseInt(scoreText, 10);

  return {
    verdict: rawVerdict === "RETRY" ? "retry" : "accept",
    score: Number.isFinite(score) ? score : null,
    blockingIssues: extractBulletSection(analysis, "BLOCKING_ISSUES"),
    minorIssues: extractBulletSection(analysis, "MINOR_ISSUES"),
    nextPromptHints: extractBulletSection(analysis, "NEXT_PROMPT_HINT")
  };
};

/**
 * Native tool definitions in OpenAI function format.
 */
export const nativeTools = [
  {
    type: "function",
    name: "create_image",
    description: "Generate or edit images. For edits, reference_images must use exact workspace-relative filenames such as workspace/input/SCR-20260131-ugqp.jpeg. Never use wildcards or guessed paths.",
    parameters: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "Description of image to generate, or instructions for editing reference images. Be specific about style, composition, colors, changes."
        },
        output_name: {
          type: "string",
          description: "Base name for the output file (without extension). Will be saved to workspace/output/"
        },
        reference_images: {
          type: "array",
          items: { type: "string" },
          description: "Optional exact workspace-relative paths to reference image(s) for editing, for example workspace/input/SCR-20260131-ugqp.jpeg. Empty array = generate from scratch."
        },
        aspect_ratio: {
          type: "string",
          enum: ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
          description: "Optional aspect ratio for the output image. If omitted, follow the style guide or user request."
        },
        image_size: {
          type: "string",
          enum: ["1k", "2k", "4k"],
          description: "Optional image size. If omitted, follow the style guide or user request."
        }
      },
      required: ["prompt", "output_name", "reference_images"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "analyze_image",
    description: "Analyze a generated or edited image and return an ACCEPT or RETRY verdict. RETRY should be used only for blocking issues, while minor polish notes should still allow ACCEPT.",
    parameters: {
      type: "object",
      properties: {
        image_path: {
          type: "string",
          description: "Path to the image file relative to the project root"
        },
        original_prompt: {
          type: "string",
          description: "The original prompt or instructions used to generate/edit the image"
        },
        check_aspects: {
          type: "array",
          items: { 
            type: "string",
            enum: ["prompt_adherence", "visual_artifacts", "anatomy", "text_rendering", "style_consistency", "composition"]
          },
          description: "Specific aspects to check. If not provided, checks all aspects."
        }
      },
      required: ["image_path", "original_prompt"],
      additionalProperties: false
    },
    strict: false
  }
];

/**
 * Native tool handlers.
 */
export const nativeHandlers = {
  /**
   * Create an image - generate from scratch or edit with references.
   */
  async create_image({ prompt, output_name, reference_images, aspect_ratio, image_size }) {
    const isEditing = reference_images && reference_images.length > 0;
    const mode = isEditing ? "edit" : "generate";

    try {
      const options = {};
      if (aspect_ratio) options.aspectRatio = aspect_ratio;
      if (image_size) options.imageSize = image_size;

      let result;

      if (isEditing) {
        // Load reference images
        const loadedImages = [];
        for (const imagePath of reference_images) {
          const fullPath = join(PROJECT_ROOT, imagePath);
          const imageBuffer = await readFile(fullPath);
          const imageBase64 = imageBuffer.toString("base64");
          const mimeType = getMimeType(imagePath);
          loadedImages.push({ data: imageBase64, mimeType });
        }

        if (loadedImages.length === 1) {
          result = await editImage(
            prompt,
            loadedImages[0].data,
            loadedImages[0].mimeType,
            options
          );
        } else {
          result = await editImageWithReferences(prompt, loadedImages, options);
        }
      } else {
        // Generate from scratch
        result = await generateImage(prompt, options);
      }
      
      // Save to output folder
      const outputDir = join(PROJECT_ROOT, "workspace/output");
      await ensureDir(outputDir);
      
      const filename = generateFilename(output_name, result.mimeType);
      const outputPath = join(outputDir, filename);
      
      const imageBuffer = Buffer.from(result.data, "base64");
      await writeFile(outputPath, imageBuffer);
      
      const relativePath = `workspace/output/${filename}`;
      log.success(`Image saved: ${relativePath}`);
      
      return { 
        success: true,
        mode,
        output_path: relativePath,
        mime_type: result.mimeType,
        prompt_used: prompt,
        reference_images: reference_images || []
      };
    } catch (error) {
      log.error("create_image", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Analyze an image for quality issues.
   */
  async analyze_image({ image_path, original_prompt, check_aspects }) {
    try {
      const fullPath = join(PROJECT_ROOT, image_path);
      const imageBuffer = await readFile(fullPath);
      const imageBase64 = imageBuffer.toString("base64");
      const mimeType = getMimeType(image_path);

      // Build analysis question based on aspects to check
      const aspects = check_aspects || [
        "prompt_adherence",
        "visual_artifacts", 
        "anatomy",
        "text_rendering",
        "style_consistency",
        "composition"
      ];

      const analysisPrompt = `Analyze this AI-generated image for quality issues. The original prompt was:
"${original_prompt}"

Please evaluate the following aspects:

${aspects.includes("prompt_adherence") ? `1. PROMPT ADHERENCE: Does the image accurately represent what was requested? What elements match or are missing?` : ""}
${aspects.includes("visual_artifacts") ? `2. VISUAL ARTIFACTS: Are there any glitches, distortions, blur, noise, or unnatural patterns?` : ""}
${aspects.includes("anatomy") ? `3. ANATOMY: If there are people/animals, check for correct proportions, especially hands, fingers, faces, and limbs.` : ""}
${aspects.includes("text_rendering") ? `4. TEXT RENDERING: If text was requested, is it readable and correctly spelled?` : ""}
${aspects.includes("style_consistency") ? `5. STYLE CONSISTENCY: Is the visual style coherent throughout the image?` : ""}
${aspects.includes("composition") ? `6. COMPOSITION: Is the framing and layout balanced and appropriate?` : ""}

Use this exact output format:

VERDICT: ACCEPT or RETRY
SCORE: <1-10>
BLOCKING_ISSUES:
- <only issues that materially break the brief; use "none" if there are none>
MINOR_ISSUES:
- <optional polish notes that do not require another retry; use "none" if there are none>
NEXT_PROMPT_HINT:
- <targeted retry hint only if VERDICT is RETRY; otherwise use "none">

Decision rules:
- Use ACCEPT when the main subject, layout intent, and style-guide essentials are satisfied, even if minor polish notes remain.
- Use RETRY only when there are blocking issues such as wrong subject, broken composition, unreadable required text, severe artifacts, or clear style-guide violations.
- Do NOT use RETRY for small polish improvements alone.`;

      log.vision(image_path, "Quality analysis");

      const analysis = await vision({
        imageBase64,
        mimeType,
        question: analysisPrompt
      });

      log.visionResult(analysis.substring(0, 150) + "...");

      const report = parseAnalysisReport(analysis);

      return {
        success: true,
        image_path,
        original_prompt,
        aspects_checked: aspects,
        verdict: report.verdict,
        score: report.score,
        blocking_issues: report.blockingIssues,
        minor_issues: report.minorIssues,
        next_prompt_hints: report.nextPromptHints,
        analysis
      };
    } catch (error) {
      log.error("analyze_image", error.message);
      return { success: false, error: error.message };
    }
  }
};

/**
 * Check if a tool is native (not MCP).
 */
export const isNativeTool = (name) => name in nativeHandlers;

/**
 * Execute a native tool.
 */
export const executeNativeTool = async (name, args) => {
  const handler = nativeHandlers[name];
  if (!handler) throw new Error(`Unknown native tool: ${name}`);
  return handler(args);
};
