/**
 * Google Gemini API wrapper for video processing.
 * Uses gemini-2.5-flash for video understanding.
 */

import { gemini } from "../config.js";
import { recordGemini } from "../helpers/stats.js";
import log from "../helpers/logger.js";

const UPLOAD_ENDPOINT = "https://generativelanguage.googleapis.com/upload/v1beta/files";
const GENERATE_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${gemini.videoModel}:generateContent`;

/**
 * Upload a video file to Gemini Files API.
 * Required for files > 20MB or for reuse across multiple requests.
 * 
 * @param {Buffer} videoBuffer - Video file buffer
 * @param {string} mimeType - MIME type (video/mp4, video/webm, etc.)
 * @param {string} displayName - Display name for the file
 * @returns {Promise<{fileUri: string, name: string}>} Uploaded file info
 */
export const uploadVideoFile = async (videoBuffer, mimeType, displayName) => {
  log.gemini("Uploading video file", displayName);

  // Step 1: Initialize resumable upload
  const initResponse = await fetch(UPLOAD_ENDPOINT, {
    method: "POST",
    headers: {
      "x-goog-api-key": gemini.apiKey,
      "X-Goog-Upload-Protocol": "resumable",
      "X-Goog-Upload-Command": "start",
      "X-Goog-Upload-Header-Content-Length": videoBuffer.length.toString(),
      "X-Goog-Upload-Header-Content-Type": mimeType,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      file: { display_name: displayName }
    })
  });

  if (!initResponse.ok) {
    const error = await initResponse.text();
    throw new Error(`Upload init failed: ${error}`);
  }

  const uploadUrl = initResponse.headers.get("x-goog-upload-url");
  if (!uploadUrl) {
    throw new Error("No upload URL received from Gemini");
  }

  // Step 2: Upload the actual bytes
  const uploadResponse = await fetch(uploadUrl, {
    method: "POST",
    headers: {
      "Content-Length": videoBuffer.length.toString(),
      "X-Goog-Upload-Offset": "0",
      "X-Goog-Upload-Command": "upload, finalize"
    },
    body: videoBuffer
  });

  if (!uploadResponse.ok) {
    const error = await uploadResponse.text();
    throw new Error(`Upload failed: ${error}`);
  }

  const fileInfo = await uploadResponse.json();
  
  if (!fileInfo.file?.uri) {
    throw new Error("No file URI in upload response");
  }

  log.geminiResult(true, `Uploaded: ${fileInfo.file.name}`);
  recordGemini("upload");

  return {
    fileUri: fileInfo.file.uri,
    name: fileInfo.file.name,
    mimeType: fileInfo.file.mimeType
  };
};

/**
 * Process video with Gemini (analysis, transcription, etc.)
 * 
 * @param {object} options - Processing options
 * @param {string} options.fileUri - Uploaded file URI or YouTube URL
 * @param {string} options.videoBase64 - Base64 video data (use for inline, <20MB)
 * @param {string} options.mimeType - Video MIME type
 * @param {string} options.prompt - Instructions for processing
 * @param {object} options.responseSchema - Optional JSON schema for structured output
 * @param {object} options.videoMetadata - Optional video metadata (fps, start_offset, end_offset)
 * @returns {Promise<string|object>} Processing result
 */
export const processVideo = async ({ 
  fileUri, 
  videoBase64, 
  mimeType, 
  prompt, 
  responseSchema,
  videoMetadata 
}) => {
  log.gemini("Processing video", prompt.substring(0, 80));

  const parts = [];

  // Add video as file_data (uploaded/YouTube) or inline_data
  if (fileUri) {
    const filePart = {
      file_data: {
        file_uri: fileUri
      }
    };
    // Only add mime_type if not a YouTube URL
    if (!fileUri.includes("youtube.com") && !fileUri.includes("youtu.be")) {
      filePart.file_data.mime_type = mimeType;
    }
    // Add video metadata if provided
    if (videoMetadata) {
      filePart.video_metadata = videoMetadata;
    }
    parts.push(filePart);
  } else if (videoBase64) {
    const inlinePart = {
      inline_data: {
        mime_type: mimeType,
        data: videoBase64
      }
    };
    if (videoMetadata) {
      inlinePart.video_metadata = videoMetadata;
    }
    parts.push(inlinePart);
  } else {
    throw new Error("Either fileUri or videoBase64 must be provided");
  }

  // Add text prompt after video (best practice per docs)
  parts.push({ text: prompt });

  const body = {
    contents: [{ parts }]
  };

  // Add structured output schema if provided
  if (responseSchema) {
    body.generation_config = {
      response_mime_type: "application/json",
      response_schema: responseSchema
    };
  }

  const response = await fetch(GENERATE_ENDPOINT, {
    method: "POST",
    headers: {
      "x-goog-api-key": gemini.apiKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  const data = await response.json();

  if (data.error) {
    throw new Error(data.error.message || JSON.stringify(data.error));
  }

  recordGemini("process");

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  
  if (!text) {
    throw new Error("No text response from Gemini");
  }

  log.geminiResult(true, `Processed video (${text.length} chars)`);

  // Parse JSON if schema was provided
  if (responseSchema) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  return text;
};

/**
 * Analyze video content (visual and audio).
 * 
 * @param {object} options - Analysis options
 * @param {string} options.fileUri - Uploaded file URI or YouTube URL
 * @param {string} options.videoBase64 - Base64 video data (alternative)
 * @param {string} options.mimeType - Video MIME type
 * @param {string} options.analysisType - Type: "general", "visual", "audio", "action"
 * @param {string} options.customPrompt - Custom analysis prompt (optional)
 * @param {object} options.videoMetadata - Optional clipping/fps settings
 * @returns {Promise<object>} Analysis result
 */
export const analyzeVideo = async (options) => {
  const {
    fileUri,
    videoBase64,
    mimeType,
    analysisType = "general",
    customPrompt,
    videoMetadata
  } = options;

  const prompts = {
    general: `Analyze this video comprehensively. Describe:
- Type of video content (tutorial, vlog, presentation, movie clip, etc.)
- Main subject and topics covered
- Key visual elements and scenes
- Audio content (speech, music, sound effects)
- Overall quality and production value
- Notable moments with timestamps (MM:SS format)`,
    
    visual: `Analyze the visual elements of this video. Describe:
- Scene composition and cinematography
- Color palette and lighting
- Text overlays, graphics, or animations
- Objects and people visible
- Visual transitions and effects
- Key visual moments with timestamps`,
    
    audio: `Analyze the audio content of this video. Describe:
- Speech content and speakers
- Background music (genre, mood)
- Sound effects
- Audio quality
- Key audio moments with timestamps`,
    
    action: `Analyze the actions and events in this video. Describe:
- Sequence of events with timestamps
- Key actions performed
- Interactions between subjects
- Important transitions or changes
- Climactic or significant moments`
  };

  const prompt = customPrompt || prompts[analysisType] || prompts.general;

  const schema = {
    type: "OBJECT",
    properties: {
      video_type: {
        type: "STRING",
        description: "Type of video content"
      },
      summary: {
        type: "STRING",
        description: "Brief summary of the video"
      },
      duration_estimate: {
        type: "STRING",
        description: "Estimated duration of the video"
      },
      key_moments: {
        type: "ARRAY",
        items: {
          type: "OBJECT",
          properties: {
            timestamp: { type: "STRING", description: "Timestamp in MM:SS format" },
            description: { type: "STRING", description: "What happens at this moment" }
          }
        },
        description: "Key moments with timestamps"
      },
      visual_elements: {
        type: "ARRAY",
        items: { type: "STRING" },
        description: "Notable visual elements"
      },
      audio_elements: {
        type: "ARRAY",
        items: { type: "STRING" },
        description: "Notable audio elements"
      },
      quality_assessment: {
        type: "STRING",
        description: "Video quality assessment"
      }
    },
    required: ["video_type", "summary"]
  };

  return processVideo({
    fileUri,
    videoBase64,
    mimeType,
    prompt,
    responseSchema: schema,
    videoMetadata
  });
};

/**
 * Transcribe speech from video with timestamps.
 * 
 * @param {object} options - Transcription options
 * @param {string} options.fileUri - Uploaded file URI or YouTube URL
 * @param {string} options.videoBase64 - Base64 video data (alternative)
 * @param {string} options.mimeType - Video MIME type
 * @param {boolean} options.includeTimestamps - Include timestamps (default: true)
 * @param {boolean} options.detectSpeakers - Detect speakers (default: true)
 * @param {string} options.targetLanguage - Translate to this language (optional)
 * @param {object} options.videoMetadata - Optional clipping settings
 * @returns {Promise<object>} Structured transcription
 */
export const transcribeVideo = async (options) => {
  const {
    fileUri,
    videoBase64,
    mimeType,
    includeTimestamps = true,
    detectSpeakers = true,
    targetLanguage,
    videoMetadata
  } = options;

  let prompt = "Transcribe all speech from this video.\n\nRequirements:\n";
  
  if (detectSpeakers) {
    prompt += "- Identify distinct speakers (e.g., Speaker 1, Speaker 2, or names if visible/mentioned).\n";
  }
  if (includeTimestamps) {
    prompt += "- Provide accurate timestamps for each segment (Format: MM:SS).\n";
  }
  prompt += "- Detect the primary language.\n";
  if (targetLanguage) {
    prompt += `- Translate all segments to ${targetLanguage}.\n`;
  }
  prompt += "- Note any significant non-speech audio (music, sound effects) with timestamps.\n";
  prompt += "- Provide a brief summary at the beginning.";

  const schema = {
    type: "OBJECT",
    properties: {
      summary: {
        type: "STRING",
        description: "A concise summary of the spoken content."
      },
      duration_estimate: {
        type: "STRING",
        description: "Estimated duration of the video."
      },
      primary_language: {
        type: "STRING",
        description: "Primary language detected."
      },
      segments: {
        type: "ARRAY",
        description: "List of transcribed segments.",
        items: {
          type: "OBJECT",
          properties: {
            speaker: { type: "STRING" },
            timestamp: { type: "STRING" },
            content: { type: "STRING" },
            ...(targetLanguage && { translation: { type: "STRING" } })
          },
          required: ["content", ...(includeTimestamps ? ["timestamp"] : []), ...(detectSpeakers ? ["speaker"] : [])]
        }
      },
      non_speech_audio: {
        type: "ARRAY",
        items: {
          type: "OBJECT",
          properties: {
            timestamp: { type: "STRING" },
            description: { type: "STRING" }
          }
        },
        description: "Notable non-speech audio moments"
      }
    },
    required: ["summary", "segments"]
  };

  return processVideo({
    fileUri,
    videoBase64,
    mimeType,
    prompt,
    responseSchema: schema,
    videoMetadata
  });
};

/**
 * Extract frames/scenes from video.
 * 
 * @param {object} options - Extraction options
 * @param {string} options.fileUri - Uploaded file URI
 * @param {string} options.videoBase64 - Base64 video data
 * @param {string} options.mimeType - Video MIME type
 * @param {string} options.extractionType - "scenes", "keyframes", "objects", "text"
 * @param {object} options.videoMetadata - Optional clipping/fps settings
 * @returns {Promise<object>} Extraction result
 */
export const extractFromVideo = async (options) => {
  const {
    fileUri,
    videoBase64,
    mimeType,
    extractionType = "scenes",
    videoMetadata
  } = options;

  const prompts = {
    scenes: `Identify and describe all distinct scenes in this video.
For each scene provide:
- Start timestamp (MM:SS)
- End timestamp (MM:SS) 
- Description of the scene
- Key visual elements
- Mood/tone`,

    keyframes: `Identify the key frames in this video - moments that best represent the content.
For each keyframe provide:
- Timestamp (MM:SS)
- Description of what's shown
- Why this frame is significant`,

    objects: `Identify all notable objects, people, and elements visible in this video.
For each item provide:
- What it is
- Timestamps when visible (MM:SS)
- Context/relevance to the video`,

    text: `Extract all text visible in this video (on-screen text, titles, captions, signs, etc.)
For each text element provide:
- The text content
- Timestamp when visible (MM:SS)
- Location on screen
- Purpose (title, caption, sign, etc.)`
  };

  const prompt = prompts[extractionType] || prompts.scenes;

  const schemas = {
    scenes: {
      type: "OBJECT",
      properties: {
        total_scenes: { type: "INTEGER" },
        scenes: {
          type: "ARRAY",
          items: {
            type: "OBJECT",
            properties: {
              scene_number: { type: "INTEGER" },
              start_time: { type: "STRING" },
              end_time: { type: "STRING" },
              description: { type: "STRING" },
              visual_elements: { type: "ARRAY", items: { type: "STRING" } },
              mood: { type: "STRING" }
            }
          }
        }
      }
    },
    keyframes: {
      type: "OBJECT",
      properties: {
        total_keyframes: { type: "INTEGER" },
        keyframes: {
          type: "ARRAY",
          items: {
            type: "OBJECT",
            properties: {
              timestamp: { type: "STRING" },
              description: { type: "STRING" },
              significance: { type: "STRING" }
            }
          }
        }
      }
    },
    objects: {
      type: "OBJECT",
      properties: {
        objects: {
          type: "ARRAY",
          items: {
            type: "OBJECT",
            properties: {
              name: { type: "STRING" },
              timestamps: { type: "ARRAY", items: { type: "STRING" } },
              context: { type: "STRING" }
            }
          }
        }
      }
    },
    text: {
      type: "OBJECT",
      properties: {
        text_elements: {
          type: "ARRAY",
          items: {
            type: "OBJECT",
            properties: {
              content: { type: "STRING" },
              timestamp: { type: "STRING" },
              location: { type: "STRING" },
              purpose: { type: "STRING" }
            }
          }
        }
      }
    }
  };

  return processVideo({
    fileUri,
    videoBase64,
    mimeType,
    prompt,
    responseSchema: schemas[extractionType] || schemas.scenes,
    videoMetadata
  });
};
