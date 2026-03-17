/**
 * Native tools for video processing agent.
 * 
 * Tools:
 * - analyze_video: Analyze video content (visual, audio, action)
 * - transcribe_video: Transcribe speech with timestamps and speakers
 * - extract_video: Extract scenes, keyframes, objects, or text
 * - query_video: Ask any question about a video
 */

import { readFile, writeFile, mkdir } from "fs/promises";
import { join, extname, dirname } from "path";
import { fileURLToPath } from "url";
import { 
  uploadVideoFile, 
  analyzeVideo, 
  transcribeVideo,
  extractFromVideo,
  processVideo
} from "./gemini.js";
import log from "../helpers/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, "../..");

// File size threshold for upload vs inline (20MB)
const INLINE_SIZE_LIMIT = 20 * 1024 * 1024;

/**
 * MIME type mapping for video formats.
 */
const getVideoMimeType = (filepath) => {
  const ext = extname(filepath).toLowerCase();
  const mimeTypes = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp"
  };
  return mimeTypes[ext] || "video/mp4";
};

/**
 * Check if input is a YouTube URL.
 */
const isYouTubeUrl = (input) => {
  return input.includes("youtube.com/watch") || input.includes("youtu.be/");
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
 * Load video file and prepare for Gemini.
 * Uses upload API for large files, inline for small ones.
 * Returns fileUri directly for YouTube URLs.
 */
const loadVideo = async (videoPath) => {
  // Handle YouTube URLs
  if (isYouTubeUrl(videoPath)) {
    log.info(`YouTube URL detected`);
    return { fileUri: videoPath, mimeType: "video/mp4" };
  }

  const fullPath = join(PROJECT_ROOT, videoPath);
  const buffer = await readFile(fullPath);
  const mimeType = getVideoMimeType(videoPath);
  const displayName = videoPath.split("/").pop();

  if (buffer.length > INLINE_SIZE_LIMIT) {
    // Upload large files
    log.info(`Video file > 20MB, using upload API...`);
    const uploaded = await uploadVideoFile(buffer, mimeType, displayName);
    return { fileUri: uploaded.fileUri, mimeType };
  } else {
    // Use inline for small files
    return { videoBase64: buffer.toString("base64"), mimeType };
  }
};

/**
 * Native tool definitions in OpenAI function format.
 */
export const nativeTools = [
  {
    type: "function",
    name: "analyze_video",
    description: "Analyze video content - visual elements, audio, actions, and overall composition. Supports local files and YouTube URLs. Returns structured analysis with timestamps.",
    parameters: {
      type: "object",
      properties: {
        video_path: {
          type: "string",
          description: "Path to video file relative to project root (e.g., workspace/input/video.mp4) OR a YouTube URL (e.g., https://www.youtube.com/watch?v=...)"
        },
        analysis_type: {
          type: "string",
          enum: ["general", "visual", "audio", "action"],
          description: "Type of analysis: 'general' (comprehensive), 'visual' (cinematography, scenes), 'audio' (speech, music, sounds), 'action' (events, movements). Default: general"
        },
        custom_prompt: {
          type: "string",
          description: "Optional custom analysis prompt to override the default"
        },
        start_time: {
          type: "string",
          description: "Optional start time for clipping (e.g., '30s' or '1m30s')"
        },
        end_time: {
          type: "string",
          description: "Optional end time for clipping"
        },
        fps: {
          type: "number",
          description: "Frames per second to sample (default: 1). Lower for long videos, higher for fast action."
        },
        output_name: {
          type: "string",
          description: "Optional base name for saving analysis JSON to workspace/output/"
        }
      },
      required: ["video_path"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "transcribe_video",
    description: "Transcribe speech from video with timestamps and speaker detection. Also captures non-speech audio. Supports local files and YouTube URLs.",
    parameters: {
      type: "object",
      properties: {
        video_path: {
          type: "string",
          description: "Path to video file relative to project root OR a YouTube URL"
        },
        include_timestamps: {
          type: "boolean",
          description: "Include timestamps for each segment. Default: true"
        },
        detect_speakers: {
          type: "boolean",
          description: "Identify and label different speakers. Default: true"
        },
        translate_to: {
          type: "string",
          description: "Target language for translation (e.g., 'English', 'Spanish'). If not provided, keeps original language."
        },
        start_time: {
          type: "string",
          description: "Optional start time for clipping (e.g., '30s' or '1m30s')"
        },
        end_time: {
          type: "string",
          description: "Optional end time for clipping"
        },
        output_name: {
          type: "string",
          description: "Optional base name for saving transcription JSON to workspace/output/"
        }
      },
      required: ["video_path"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "extract_video",
    description: "Extract specific elements from video: scenes, keyframes, objects, or text. Returns structured data with timestamps.",
    parameters: {
      type: "object",
      properties: {
        video_path: {
          type: "string",
          description: "Path to video file relative to project root OR a YouTube URL"
        },
        extraction_type: {
          type: "string",
          enum: ["scenes", "keyframes", "objects", "text"],
          description: "What to extract: 'scenes' (distinct scenes with timestamps), 'keyframes' (representative moments), 'objects' (people/things), 'text' (on-screen text). Default: scenes"
        },
        start_time: {
          type: "string",
          description: "Optional start time for clipping"
        },
        end_time: {
          type: "string",
          description: "Optional end time for clipping"
        },
        fps: {
          type: "number",
          description: "Frames per second to sample. Higher = more detail but more tokens."
        },
        output_name: {
          type: "string",
          description: "Optional base name for saving extraction JSON to workspace/output/"
        }
      },
      required: ["video_path"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "query_video",
    description: "Ask any question about a video. Use for custom queries that don't fit analyze/transcribe/extract patterns. Can reference specific timestamps.",
    parameters: {
      type: "object",
      properties: {
        video_path: {
          type: "string",
          description: "Path to video file relative to project root OR a YouTube URL"
        },
        question: {
          type: "string",
          description: "Question or prompt about the video content. Can reference timestamps like 'What happens at 01:30?'"
        },
        start_time: {
          type: "string",
          description: "Optional start time to focus on specific segment"
        },
        end_time: {
          type: "string",
          description: "Optional end time to focus on specific segment"
        }
      },
      required: ["video_path", "question"],
      additionalProperties: false
    },
    strict: false
  }
];

/**
 * Build videoMetadata object from options.
 */
const buildVideoMetadata = ({ start_time, end_time, fps }) => {
  const metadata = {};
  if (start_time) metadata.start_offset = start_time;
  if (end_time) metadata.end_offset = end_time;
  if (fps) metadata.fps = fps;
  return Object.keys(metadata).length > 0 ? metadata : undefined;
};

/**
 * Native tool handlers.
 */
export const nativeHandlers = {
  /**
   * Analyze video content.
   */
  async analyze_video({ 
    video_path, 
    analysis_type = "general", 
    custom_prompt,
    start_time,
    end_time,
    fps,
    output_name 
  }) {
    log.tool("analyze_video", { 
      video_path: video_path.substring(0, 50), 
      analysis_type,
      clip: start_time || end_time ? `${start_time || "0"}-${end_time || "end"}` : "full"
    });

    try {
      const video = await loadVideo(video_path);
      const videoMetadata = buildVideoMetadata({ start_time, end_time, fps });

      const result = await analyzeVideo({
        ...video,
        analysisType: analysis_type,
        customPrompt: custom_prompt,
        videoMetadata
      });

      // Save to file if output_name provided
      if (output_name) {
        const outputDir = join(PROJECT_ROOT, "workspace/output");
        await ensureDir(outputDir);
        const timestamp = Date.now();
        const outputPath = join(outputDir, `${output_name}_${timestamp}.json`);
        await writeFile(outputPath, JSON.stringify(result, null, 2));
        log.success(`Analysis saved: workspace/output/${output_name}_${timestamp}.json`);
        
        return {
          success: true,
          video_path,
          analysis_type,
          output_path: `workspace/output/${output_name}_${timestamp}.json`,
          analysis: result
        };
      }

      log.success(`Analyzed: ${result.video_type}`);
      
      return {
        success: true,
        video_path,
        analysis_type,
        analysis: result
      };
    } catch (error) {
      log.error("analyze_video", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Transcribe speech from video.
   */
  async transcribe_video({ 
    video_path, 
    include_timestamps = true, 
    detect_speakers = true,
    translate_to,
    start_time,
    end_time,
    output_name 
  }) {
    log.tool("transcribe_video", { 
      video_path: video_path.substring(0, 50),
      timestamps: include_timestamps,
      speakers: detect_speakers,
      translate: translate_to || "no"
    });

    try {
      const video = await loadVideo(video_path);
      const videoMetadata = buildVideoMetadata({ start_time, end_time });

      const result = await transcribeVideo({
        ...video,
        includeTimestamps: include_timestamps,
        detectSpeakers: detect_speakers,
        targetLanguage: translate_to,
        videoMetadata
      });

      // Save to file if output_name provided
      if (output_name) {
        const outputDir = join(PROJECT_ROOT, "workspace/output");
        await ensureDir(outputDir);
        const timestamp = Date.now();
        const outputPath = join(outputDir, `${output_name}_${timestamp}.json`);
        await writeFile(outputPath, JSON.stringify(result, null, 2));
        log.success(`Transcription saved: workspace/output/${output_name}_${timestamp}.json`);
        
        return {
          success: true,
          video_path,
          output_path: `workspace/output/${output_name}_${timestamp}.json`,
          transcription: result
        };
      }

      log.success(`Transcribed: ${result.segments?.length || 0} segments`);
      
      return {
        success: true,
        video_path,
        transcription: result
      };
    } catch (error) {
      log.error("transcribe_video", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Extract elements from video.
   */
  async extract_video({ 
    video_path, 
    extraction_type = "scenes",
    start_time,
    end_time,
    fps,
    output_name 
  }) {
    log.tool("extract_video", { 
      video_path: video_path.substring(0, 50), 
      extraction_type 
    });

    try {
      const video = await loadVideo(video_path);
      const videoMetadata = buildVideoMetadata({ start_time, end_time, fps });

      const result = await extractFromVideo({
        ...video,
        extractionType: extraction_type,
        videoMetadata
      });

      // Save to file if output_name provided
      if (output_name) {
        const outputDir = join(PROJECT_ROOT, "workspace/output");
        await ensureDir(outputDir);
        const timestamp = Date.now();
        const outputPath = join(outputDir, `${output_name}_${timestamp}.json`);
        await writeFile(outputPath, JSON.stringify(result, null, 2));
        log.success(`Extraction saved: workspace/output/${output_name}_${timestamp}.json`);
        
        return {
          success: true,
          video_path,
          extraction_type,
          output_path: `workspace/output/${output_name}_${timestamp}.json`,
          extraction: result
        };
      }

      log.success(`Extracted ${extraction_type}`);
      
      return {
        success: true,
        video_path,
        extraction_type,
        extraction: result
      };
    } catch (error) {
      log.error("extract_video", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Ask a custom question about video.
   */
  async query_video({ video_path, question, start_time, end_time }) {
    log.tool("query_video", { 
      video_path: video_path.substring(0, 50), 
      question: question.substring(0, 50) + "..." 
    });

    try {
      const video = await loadVideo(video_path);
      const videoMetadata = buildVideoMetadata({ start_time, end_time });

      const result = await processVideo({
        ...video,
        prompt: question,
        videoMetadata
      });

      log.success(`Query answered (${result.length} chars)`);
      
      return {
        success: true,
        video_path,
        question,
        answer: result
      };
    } catch (error) {
      log.error("query_video", error.message);
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
