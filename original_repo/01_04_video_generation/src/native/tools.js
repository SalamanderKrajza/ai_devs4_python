/**
 * Native tools for video generation agent.
 * 
 * Tools:
 * - create_image: Generate frames using JSON templates (Gemini)
 * - analyze_image: Quality check images (Gemini)
 * - generate_video: Text-to-video (Kling)
 * - image_to_video: Start frame → video (Kling)
 * - analyze_video: Understand generated video (Gemini)
 */

import { readFile, writeFile, mkdir } from "fs/promises";
import { join, extname, dirname } from "path";
import { fileURLToPath } from "url";
import { generateImage, editImage, editImageWithReferences, processVideo } from "./gemini.js";
import { generateVideo as klingGenerateVideo, imageToVideo as klingImageToVideo, downloadVideo } from "./replicate.js";
import { vision } from "../helpers/api.js";
import { recordGemini } from "../helpers/stats.js";
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
  // ─────────────────────────────────────────────────────────────
  // Image Generation Tools
  // ─────────────────────────────────────────────────────────────
  {
    type: "function",
    name: "create_image",
    description: "Generate or edit images using Gemini. Use for creating video frames. If reference_images is empty, generates from prompt. If reference_images provided, edits/combines them.",
    parameters: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "Description of image to generate. For best results, pass the complete JSON template content as the prompt."
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
          description: "Aspect ratio of the output image. Default is 16:9 for video frames."
        },
        image_size: {
          type: "string",
          enum: ["1k", "2k", "4k"],
          description: "Resolution of the output image. Default is 2k"
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
    description: "Analyze a generated image for quality issues before using it as a video frame. Checks for prompt adherence, visual artifacts, style consistency.",
    parameters: {
      type: "object",
      properties: {
        image_path: {
          type: "string",
          description: "Path to the image file relative to the project root"
        },
        original_prompt: {
          type: "string",
          description: "The original prompt or description used to generate the image"
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
  },

  // ─────────────────────────────────────────────────────────────
  // Video Generation Tools
  // ─────────────────────────────────────────────────────────────
  {
    type: "function",
    name: "generate_video",
    description: "Generate a video from a text prompt using Kling AI. Creates a 10-second video by default. Use this when you don't have a start frame.",
    parameters: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "Detailed description of the video scene, motion, and action. Be specific about what should happen."
        },
        output_name: {
          type: "string",
          description: "Base name for the output video file (saved to workspace/output/)"
        },
        duration: {
          type: "number",
          enum: [5, 10],
          description: "Video duration in seconds. Default: 10"
        },
        aspect_ratio: {
          type: "string",
          enum: ["16:9", "9:16", "1:1"],
          description: "Video aspect ratio. Default: 16:9"
        },
        negative_prompt: {
          type: "string",
          description: "Things to avoid in the video (e.g., 'blurry, distorted, text')"
        }
      },
      required: ["prompt", "output_name"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "image_to_video",
    description: "Generate a video from a start frame image using Kling AI. Use this when you have already created a start frame with create_image. Optionally provide an end frame for more control.",
    parameters: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "Description of motion and action to animate from the start frame. Describe what should move and how."
        },
        start_image: {
          type: "string",
          description: "Path to the start frame image (e.g., workspace/output/frame_start.png)"
        },
        end_image: {
          type: "string",
          description: "Optional path to the end frame image for controlled transitions"
        },
        output_name: {
          type: "string",
          description: "Base name for the output video file (saved to workspace/output/)"
        },
        duration: {
          type: "number",
          enum: [5, 10],
          description: "Video duration in seconds. Default: 10"
        },
        negative_prompt: {
          type: "string",
          description: "Things to avoid in the video"
        }
      },
      required: ["prompt", "start_image", "output_name"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "analyze_video",
    description: "Analyze a generated video for quality, motion, and content. Use to review videos before delivering to user.",
    parameters: {
      type: "object",
      properties: {
        video_path: {
          type: "string",
          description: "Path to the video file relative to project root"
        },
        analysis_focus: {
          type: "string",
          enum: ["general", "motion", "quality", "prompt_adherence"],
          description: "What to focus on: general overview, motion quality, visual quality, or how well it matches the prompt"
        },
        original_prompt: {
          type: "string",
          description: "The original prompt used to generate the video (for prompt_adherence check)"
        }
      },
      required: ["video_path"],
      additionalProperties: false
    },
    strict: false
  }
];

/**
 * Native tool handlers.
 */
export const nativeHandlers = {
  // ─────────────────────────────────────────────────────────────
  // Image Generation Handlers
  // ─────────────────────────────────────────────────────────────
  
  /**
   * Create an image - generate from scratch or edit with references.
   */
  async create_image({ prompt, output_name, reference_images, aspect_ratio = "16:9", image_size = "2k" }) {
    const isEditing = reference_images && reference_images.length > 0;
    const mode = isEditing ? "edit" : "generate";
    
    log.tool("create_image", { 
      mode,
      prompt: prompt.substring(0, 50) + "...", 
      output_name,
      references: reference_images?.length || 0
    });

    try {
      const options = { aspectRatio: aspect_ratio, imageSize: image_size };

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
        absolute_path: outputPath,
        project_root: PROJECT_ROOT,
        mime_type: result.mimeType,
        prompt_used: prompt.substring(0, 200) + "..."
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

      const aspects = check_aspects || [
        "prompt_adherence",
        "visual_artifacts", 
        "style_consistency",
        "composition"
      ];

      const analysisPrompt = `Analyze this AI-generated image for use as a video frame. The original prompt was:
"${original_prompt}"

Please evaluate:
${aspects.includes("prompt_adherence") ? `1. PROMPT ADHERENCE: Does the image match what was requested?` : ""}
${aspects.includes("visual_artifacts") ? `2. VISUAL ARTIFACTS: Any glitches, distortions, or unnatural patterns?` : ""}
${aspects.includes("style_consistency") ? `3. STYLE: Is the visual style coherent?` : ""}
${aspects.includes("composition") ? `4. COMPOSITION: Is framing suitable for video animation?` : ""}

Provide:
- Quality score (1-10)
- Issues found
- Whether it's suitable as a video frame
- Suggestions if regeneration needed`;

      log.vision(image_path, "Frame quality analysis");

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
        aspects_checked: aspects,
        analysis
      };
    } catch (error) {
      log.error("analyze_image", error.message);
      return { success: false, error: error.message };
    }
  },

  // ─────────────────────────────────────────────────────────────
  // Video Generation Handlers
  // ─────────────────────────────────────────────────────────────

  /**
   * Generate video from text prompt.
   */
  async generate_video({ 
    prompt, 
    output_name, 
    duration = 10, 
    aspect_ratio = "16:9",
    negative_prompt = ""
  }) {
    log.tool("generate_video", { 
      prompt: prompt.substring(0, 50) + "...", 
      duration,
      aspect_ratio
    });

    try {
      const result = await klingGenerateVideo({
        prompt,
        duration,
        aspectRatio: aspect_ratio,
        negativePrompt: negative_prompt
      });

      // Download the video to workspace
      const localPath = await downloadVideo(result.url, output_name);

      return {
        success: true,
        output_path: localPath,
        video_url: result.url,
        prompt,
        duration,
        aspect_ratio
      };
    } catch (error) {
      log.error("generate_video", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Generate video from start frame image.
   */
  async image_to_video({ 
    prompt, 
    start_image, 
    end_image,
    output_name, 
    duration = 10,
    negative_prompt = ""
  }) {
    log.tool("image_to_video", { 
      start_image, 
      end_image: end_image || "none",
      duration
    });

    try {
      const result = await klingImageToVideo({
        prompt,
        startImagePath: start_image,
        endImagePath: end_image,
        duration,
        negativePrompt: negative_prompt
      });

      // Download the video to workspace
      const localPath = await downloadVideo(result.url, output_name);

      return {
        success: true,
        output_path: localPath,
        video_url: result.url,
        prompt,
        start_image,
        end_image: end_image || null,
        duration
      };
    } catch (error) {
      log.error("image_to_video", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Analyze a generated video.
   */
  async analyze_video({ video_path, analysis_focus = "general", original_prompt }) {
    log.tool("analyze_video", { video_path, analysis_focus });

    try {
      // Read video file
      const fullPath = join(PROJECT_ROOT, video_path);
      const videoBuffer = await readFile(fullPath);
      const videoBase64 = videoBuffer.toString("base64");

      const prompts = {
        general: `Analyze this video comprehensively. Describe:
- Overall content and what's happening
- Visual quality (resolution, artifacts, consistency)
- Motion quality (smoothness, naturalness)
- Key moments with timestamps
- Overall assessment`,

        motion: `Analyze the motion quality in this video:
- Smoothness of movement
- Natural vs artificial motion
- Any jitter, stuttering, or unnatural transitions
- How well objects/subjects maintain consistency
- Motion artifacts or distortions`,

        quality: `Analyze the visual quality of this video:
- Resolution and clarity
- Color consistency
- Artifacts or glitches
- Frame-to-frame consistency
- Overall production quality`,

        prompt_adherence: `The video was generated with this prompt: "${original_prompt}"

Analyze how well the video matches the prompt:
- Which elements from the prompt are present?
- What's missing or different?
- How accurate is the motion/action?
- Overall adherence score (1-10)`
      };

      const prompt = prompts[analysis_focus] || prompts.general;

      const analysis = await processVideo({
        videoBase64,
        mimeType: "video/mp4",
        prompt
      });

      log.success(`Video analyzed (${analysis_focus})`);

      return {
        success: true,
        video_path,
        analysis_focus,
        analysis
      };
    } catch (error) {
      log.error("analyze_video", error.message);
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
