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
import { vision } from "../helpers/api.js";
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

/**
 * Native tool definitions in OpenAI function format.
 */
export const nativeTools = [
  {
    type: "function",
    name: "create_image",
    description: "Generate or edit images using Gemini. If reference_images is empty, generates from prompt. If reference_images provided, edits/combines them based on the prompt.",
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
          description: "Optional paths to reference image(s) for editing. Empty array = generate from scratch."
        },
        aspect_ratio: {
          type: "string",
          enum: ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
          description: "Aspect ratio of the output image. Default is 1:1"
        },
        image_size: {
          type: "string",
          enum: ["1k", "2k", "4k"],
          description: "Resolution of the output image. Default is 1k"
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
    description: "Analyze a generated or edited image for quality issues. Checks for prompt adherence, visual artifacts, style consistency, and common AI generation mistakes.",
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
    
    log.tool("create_image", { 
      mode,
      prompt: prompt.substring(0, 50) + "...", 
      output_name,
      references: reference_images?.length || 0
    });

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
    log.tool("analyze_image", { image_path });

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

Please evaluate the following aspects and provide a detailed assessment:

${aspects.includes("prompt_adherence") ? `1. PROMPT ADHERENCE: Does the image accurately represent what was requested? What elements match or are missing?` : ""}
${aspects.includes("visual_artifacts") ? `2. VISUAL ARTIFACTS: Are there any glitches, distortions, blur, noise, or unnatural patterns?` : ""}
${aspects.includes("anatomy") ? `3. ANATOMY: If there are people/animals, check for correct proportions, especially hands, fingers, faces, and limbs.` : ""}
${aspects.includes("text_rendering") ? `4. TEXT RENDERING: If text was requested, is it readable and correctly spelled?` : ""}
${aspects.includes("style_consistency") ? `5. STYLE CONSISTENCY: Is the visual style coherent throughout the image?` : ""}
${aspects.includes("composition") ? `6. COMPOSITION: Is the framing and layout balanced and appropriate?` : ""}

Provide:
- An overall quality score (1-10)
- List of specific issues found
- Whether the image is acceptable or needs regeneration
- Suggestions for improving the prompt if regeneration is needed`;

      log.vision(image_path, "Quality analysis");

      const analysis = await vision({
        imagePath: image_path,
        imageBase64,
        mimeType,
        question: analysisPrompt
      });

      log.visionResult(analysis.substring(0, 150) + "...");

      return {
        success: true,
        image_path,
        original_prompt,
        aspects_checked: aspects,
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
