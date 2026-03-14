/**
 * Native tools for audio processing agent.
 * 
 * Tools:
 * - transcribe_audio: Transcribe speech with timestamps and speaker detection
 * - analyze_audio: Analyze audio content (speech, music, sounds)
 * - generate_audio: Text-to-speech generation (single or multi-speaker)
 */

import { readFile, writeFile, mkdir } from "fs/promises";
import { join, extname, dirname } from "path";
import { fileURLToPath } from "url";
import { 
  uploadAudioFile, 
  transcribeAudio, 
  analyzeAudio, 
  processAudio,
  generateSpeech,
  generateMultiSpeakerSpeech,
  TTS_VOICES
} from "./gemini.js";
import log from "../helpers/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, "../..");

// File size threshold for upload vs inline (20MB)
const INLINE_SIZE_LIMIT = 20 * 1024 * 1024;

/**
 * MIME type mapping for audio formats.
 */
const getAudioMimeType = (filepath) => {
  const ext = extname(filepath).toLowerCase();
  const mimeTypes = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm"
  };
  return mimeTypes[ext] || "audio/mpeg";
};

/**
 * Check if input is a YouTube URL.
 */
const isYouTubeUrl = (input) => {
  return input.includes("youtube.com/watch") || input.includes("youtu.be/")
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
 * Write PCM data as WAV file.
 * Gemini TTS outputs 24kHz, 16-bit, mono PCM.
 */
const writeWavFile = async (filepath, pcmBuffer) => {
  const sampleRate = 24000;
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = pcmBuffer.length;
  const headerSize = 44;
  
  const wavBuffer = Buffer.alloc(headerSize + dataSize);
  
  // RIFF header
  wavBuffer.write("RIFF", 0);
  wavBuffer.writeUInt32LE(36 + dataSize, 4);
  wavBuffer.write("WAVE", 8);
  
  // fmt chunk
  wavBuffer.write("fmt ", 12);
  wavBuffer.writeUInt32LE(16, 16); // chunk size
  wavBuffer.writeUInt16LE(1, 20);  // PCM format
  wavBuffer.writeUInt16LE(numChannels, 22);
  wavBuffer.writeUInt32LE(sampleRate, 24);
  wavBuffer.writeUInt32LE(byteRate, 28);
  wavBuffer.writeUInt16LE(blockAlign, 32);
  wavBuffer.writeUInt16LE(bitsPerSample, 34);
  
  // data chunk
  wavBuffer.write("data", 36);
  wavBuffer.writeUInt32LE(dataSize, 40);
  pcmBuffer.copy(wavBuffer, headerSize);
  
  await writeFile(filepath, wavBuffer);
};

/**
 * Load audio file and prepare for Gemini.
 * Uses upload API for large files, inline for small ones.
 * Supports YouTube URLs directly.
 */
const loadAudio = async (audioPath) => {
  // Handle YouTube URLs
  if (isYouTubeUrl(audioPath)) {
    log.info(`YouTube URL detected`);
    return { fileUri: audioPath, mimeType: "video/mp4" };
  }

  const fullPath = join(PROJECT_ROOT, audioPath);
  const buffer = await readFile(fullPath);
  const mimeType = getAudioMimeType(audioPath);
  const displayName = audioPath.split("/").pop();

  if (buffer.length > INLINE_SIZE_LIMIT) {
    // Upload large files
    log.info(`Audio file > 20MB, using upload API...`);
    const uploaded = await uploadAudioFile(buffer, mimeType, displayName);
    return { fileUri: uploaded.fileUri, mimeType };
  } else {
    // Use inline for small files
    return { audioBase64: buffer.toString("base64"), mimeType };
  }
};

/**
 * Native tool definitions in OpenAI function format.
 */
export const nativeTools = [
  {
    type: "function",
    name: "transcribe_audio",
    description: "Transcribe audio to text with timestamps and speaker detection. Supports local files (MP3, WAV, AIFF, AAC, OGG, FLAC) and YouTube URLs. Can detect speakers, emotions, and translate to other languages.",
    parameters: {
      type: "object",
      properties: {
        audio_path: {
          type: "string",
          description: "Path to audio file relative to project root (e.g., workspace/input/recording.mp3) OR a YouTube URL (e.g., https://www.youtube.com/watch?v=...)"
        },
        include_timestamps: {
          type: "boolean",
          description: "Include timestamps for each segment. Default: true"
        },
        detect_speakers: {
          type: "boolean",
          description: "Identify and label different speakers. Default: true"
        },
        detect_emotions: {
          type: "boolean",
          description: "Detect speaker emotions (happy, sad, angry, neutral). Default: false"
        },
        translate_to: {
          type: "string",
          description: "Target language for translation (e.g., 'English', 'Spanish'). If not provided, keeps original language."
        },
        output_name: {
          type: "string",
          description: "Optional base name for saving transcription JSON to workspace/output/"
        }
      },
      required: ["audio_path"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "analyze_audio",
    description: "Analyze audio content - identify sounds, music characteristics, speech patterns, or general audio analysis. Supports local files and YouTube URLs. Does NOT transcribe - use transcribe_audio for that.",
    parameters: {
      type: "object",
      properties: {
        audio_path: {
          type: "string",
          description: "Path to audio file relative to project root OR a YouTube URL"
        },
        analysis_type: {
          type: "string",
          enum: ["general", "music", "speech", "sounds"],
          description: "Type of analysis: 'general' (comprehensive), 'music' (genre, tempo, instruments), 'speech' (speakers, style, clarity), 'sounds' (identify sound sources). Default: general"
        },
        custom_prompt: {
          type: "string",
          description: "Optional custom analysis prompt to override the default for the analysis type"
        },
        output_name: {
          type: "string",
          description: "Optional base name for saving analysis JSON to workspace/output/"
        }
      },
      required: ["audio_path"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "query_audio",
    description: "Ask any question about audio content. Supports local files and YouTube URLs. Use for custom queries that don't fit transcribe or analyze patterns.",
    parameters: {
      type: "object",
      properties: {
        audio_path: {
          type: "string",
          description: "Path to audio file relative to project root OR a YouTube URL"
        },
        question: {
          type: "string",
          description: "Question or prompt about the audio content"
        }
      },
      required: ["audio_path", "question"],
      additionalProperties: false
    },
    strict: false
  },
  {
    type: "function",
    name: "generate_audio",
    description: `Generate speech audio from text using Gemini TTS. Supports single-speaker and multi-speaker (up to 2) generation. Style, tone, pace, and accent are controllable via natural language in the text prompt.

Available voices: ${Object.entries(TTS_VOICES).map(([name, style]) => `${name} (${style})`).join(", ")}

For style control, include directions in the text like: "Say cheerfully: Hello!" or "In a whisper: The secret is..."
For multi-speaker, format as dialogue: "Speaker1: Hello! Speaker2: Hi there!"`,
    parameters: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Text to convert to speech. Include style directions for tone/pace control. For multi-speaker, use 'SpeakerName: dialogue' format."
        },
        voice: {
          type: "string",
          description: "Voice name for single-speaker. Options: Kore (Firm), Puck (Upbeat), Charon (Informative), Aoede (Breezy), Fenrir (Excitable), etc. Default: Kore"
        },
        speakers: {
          type: "array",
          items: {
            type: "object",
            properties: {
              speaker: { type: "string", description: "Speaker name as used in the text" },
              voice: { type: "string", description: "Voice name for this speaker" }
            },
            required: ["speaker", "voice"]
          },
          description: "For multi-speaker: array of {speaker, voice} mappings. Max 2 speakers."
        },
        output_name: {
          type: "string",
          description: "Base name for output WAV file (saved to workspace/output/)"
        }
      },
      required: ["text", "output_name"],
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
   * Transcribe audio with optional features.
   */
  async transcribe_audio({ 
    audio_path, 
    include_timestamps = true, 
    detect_speakers = true, 
    detect_emotions = false,
    translate_to,
    output_name 
  }) {
    log.tool("transcribe_audio", { 
      audio_path, 
      timestamps: include_timestamps,
      speakers: detect_speakers,
      emotions: detect_emotions,
      translate: translate_to || "no"
    });

    try {
      const audio = await loadAudio(audio_path);

      const result = await transcribeAudio({
        ...audio,
        includeTimestamps: include_timestamps,
        detectSpeakers: detect_speakers,
        detectEmotions: detect_emotions,
        targetLanguage: translate_to
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
          audio_path,
          output_path: `workspace/output/${output_name}_${timestamp}.json`,
          transcription: result
        };
      }

      log.success(`Transcribed: ${result.segments?.length || 0} segments`);
      
      return {
        success: true,
        audio_path,
        transcription: result
      };
    } catch (error) {
      log.error("transcribe_audio", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Analyze audio content.
   */
  async analyze_audio({ audio_path, analysis_type = "general", custom_prompt, output_name }) {
    log.tool("analyze_audio", { audio_path, analysis_type });

    try {
      const audio = await loadAudio(audio_path);

      const result = await analyzeAudio({
        ...audio,
        analysisType: analysis_type,
        customPrompt: custom_prompt
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
          audio_path,
          analysis_type,
          output_path: `workspace/output/${output_name}_${timestamp}.json`,
          analysis: result
        };
      }

      log.success(`Analyzed: ${result.audio_type}`);
      
      return {
        success: true,
        audio_path,
        analysis_type,
        analysis: result
      };
    } catch (error) {
      log.error("analyze_audio", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Ask a custom question about audio.
   */
  async query_audio({ audio_path, question }) {
    log.tool("query_audio", { audio_path, question: question.substring(0, 50) + "..." });

    try {
      const audio = await loadAudio(audio_path);

      const result = await processAudio({
        ...audio,
        prompt: question
      });

      log.success(`Query answered (${result.length} chars)`);
      
      return {
        success: true,
        audio_path,
        question,
        answer: result
      };
    } catch (error) {
      log.error("query_audio", error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Generate speech audio from text.
   */
  async generate_audio({ text, voice = "Kore", speakers, output_name }) {
    const isMultiSpeaker = speakers && speakers.length > 0;
    
    log.tool("generate_audio", { 
      mode: isMultiSpeaker ? "multi-speaker" : "single-speaker",
      voice: isMultiSpeaker ? speakers.map(s => s.voice).join(", ") : voice,
      text_length: text.length
    });

    try {
      let result;

      if (isMultiSpeaker) {
        if (speakers.length > 2) {
          return { success: false, error: "Maximum 2 speakers supported for multi-speaker TTS" };
        }
        result = await generateMultiSpeakerSpeech({ text, speakers });
      } else {
        // Validate voice name
        if (!TTS_VOICES[voice]) {
          const validVoices = Object.keys(TTS_VOICES).join(", ");
          return { success: false, error: `Invalid voice "${voice}". Valid options: ${validVoices}` };
        }
        result = await generateSpeech({ text, voice });
      }

      // Save to output folder
      const outputDir = join(PROJECT_ROOT, "workspace/output");
      await ensureDir(outputDir);
      
      const timestamp = Date.now();
      const filename = `${output_name}_${timestamp}.wav`;
      const outputPath = join(outputDir, filename);
      
      await writeWavFile(outputPath, result.data);
      
      const relativePath = `workspace/output/${filename}`;
      log.success(`Audio saved: ${relativePath}`);

      return {
        success: true,
        mode: isMultiSpeaker ? "multi-speaker" : "single-speaker",
        output_path: relativePath,
        absolute_path: outputPath,
        project_root: PROJECT_ROOT,
        voice: isMultiSpeaker ? speakers : voice,
        text_length: text.length,
        format: "WAV (24kHz, 16-bit, mono)"
      };
    } catch (error) {
      log.error("generate_audio", error.message);
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
