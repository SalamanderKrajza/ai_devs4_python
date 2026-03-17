/**
 * Google Gemini API wrapper for audio processing.
 * - Audio understanding: gemini-2.5-flash-preview
 * - Text-to-speech: gemini-2.5-flash-preview-tts
 */

import { gemini } from "../config.js";
import { recordGemini } from "../helpers/stats.js";
import log from "../helpers/logger.js";

const UPLOAD_ENDPOINT = "https://generativelanguage.googleapis.com/upload/v1beta/files";
const GENERATE_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${gemini.audioModel}:generateContent`;
const TTS_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${gemini.ttsModel}:generateContent`;

/**
 * Available TTS voices with their characteristics.
 */
export const TTS_VOICES = {
  Zephyr: "Bright",
  Puck: "Upbeat",
  Charon: "Informative",
  Kore: "Firm",
  Fenrir: "Excitable",
  Leda: "Youthful",
  Orus: "Firm",
  Aoede: "Breezy",
  Callirrhoe: "Easy-going",
  Autonoe: "Bright",
  Enceladus: "Breathy",
  Iapetus: "Clear",
  Umbriel: "Easy-going",
  Algieba: "Smooth",
  Despina: "Smooth",
  Erinome: "Clear",
  Algenib: "Gravelly",
  Rasalgethi: "Informative",
  Laomedeia: "Upbeat",
  Achernar: "Soft",
  Alnilam: "Firm",
  Schedar: "Even",
  Gacrux: "Mature",
  Pulcherrima: "Forward",
  Achird: "Friendly",
  Zubenelgenubi: "Casual",
  Vindemiatrix: "Gentle",
  Sadachbia: "Lively",
  Sadaltager: "Knowledgeable",
  Sulafat: "Warm"
};

/**
 * Upload an audio file to Gemini Files API.
 * Required for files > 20MB or for reuse across multiple requests.
 * 
 * @param {Buffer} audioBuffer - Audio file buffer
 * @param {string} mimeType - MIME type (audio/mp3, audio/wav, etc.)
 * @param {string} displayName - Display name for the file
 * @returns {Promise<{fileUri: string, name: string}>} Uploaded file info
 */
export const uploadAudioFile = async (audioBuffer, mimeType, displayName) => {
  log.gemini("Uploading audio file", displayName);

  // Step 1: Initialize resumable upload
  const initResponse = await fetch(UPLOAD_ENDPOINT, {
    method: "POST",
    headers: {
      "x-goog-api-key": gemini.apiKey,
      "X-Goog-Upload-Protocol": "resumable",
      "X-Goog-Upload-Command": "start",
      "X-Goog-Upload-Header-Content-Length": audioBuffer.length.toString(),
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
      "Content-Length": audioBuffer.length.toString(),
      "X-Goog-Upload-Offset": "0",
      "X-Goog-Upload-Command": "upload, finalize"
    },
    body: audioBuffer
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
 * Process audio with Gemini (transcription, analysis, etc.)
 * 
 * @param {object} options - Processing options
 * @param {string} options.fileUri - Uploaded file URI (use for large files)
 * @param {string} options.audioBase64 - Base64 audio data (use for inline, <20MB)
 * @param {string} options.mimeType - Audio MIME type
 * @param {string} options.prompt - Instructions for processing
 * @param {object} options.responseSchema - Optional JSON schema for structured output
 * @returns {Promise<string|object>} Processing result
 */
export const processAudio = async ({ fileUri, audioBase64, mimeType, prompt, responseSchema }) => {
  log.gemini("Processing audio", prompt.substring(0, 80));

  const parts = [{ text: prompt }];

  // Add audio as file_data (uploaded) or inline_data
  if (fileUri) {
    parts.push({
      file_data: {
        mime_type: mimeType,
        file_uri: fileUri
      }
    });
  } else if (audioBase64) {
    parts.push({
      inline_data: {
        mime_type: mimeType,
        data: audioBase64
      }
    });
  } else {
    throw new Error("Either fileUri or audioBase64 must be provided");
  }

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

  log.geminiResult(true, `Processed audio (${text.length} chars)`);

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
 * Transcribe audio with speaker diarization and timestamps.
 * 
 * @param {object} options - Transcription options
 * @param {string} options.fileUri - Uploaded file URI
 * @param {string} options.audioBase64 - Base64 audio data (alternative)
 * @param {string} options.mimeType - Audio MIME type
 * @param {boolean} options.includeTimestamps - Include timestamps (default: true)
 * @param {boolean} options.detectSpeakers - Detect speakers (default: true)
 * @param {boolean} options.detectEmotions - Detect emotions (default: false)
 * @param {string} options.targetLanguage - Translate to this language (optional)
 * @returns {Promise<object>} Structured transcription
 */
export const transcribeAudio = async (options) => {
  const {
    fileUri,
    audioBase64,
    mimeType,
    includeTimestamps = true,
    detectSpeakers = true,
    detectEmotions = false,
    targetLanguage
  } = options;

  let prompt = "Process this audio file and generate a detailed transcription.\n\nRequirements:\n";
  
  if (detectSpeakers) {
    prompt += "- Identify distinct speakers (e.g., Speaker 1, Speaker 2, or names if context allows).\n";
  }
  if (includeTimestamps) {
    prompt += "- Provide accurate timestamps for each segment (Format: MM:SS).\n";
  }
  prompt += "- Detect the primary language of each segment.\n";
  if (targetLanguage) {
    prompt += `- Translate all segments to ${targetLanguage}.\n`;
  }
  if (detectEmotions) {
    prompt += "- Identify the primary emotion of the speaker. Choose exactly one: happy, sad, angry, neutral.\n";
  }
  prompt += "- Provide a brief summary of the entire audio at the beginning.";

  const schema = {
    type: "OBJECT",
    properties: {
      summary: {
        type: "STRING",
        description: "A concise summary of the audio content."
      },
      duration_estimate: {
        type: "STRING",
        description: "Estimated duration of the audio."
      },
      primary_language: {
        type: "STRING",
        description: "Primary language detected in the audio."
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
            language: { type: "STRING" },
            ...(targetLanguage && { translation: { type: "STRING" } }),
            ...(detectEmotions && { 
              emotion: { 
                type: "STRING",
                enum: ["happy", "sad", "angry", "neutral"]
              }
            })
          },
          required: ["content", ...(includeTimestamps ? ["timestamp"] : []), ...(detectSpeakers ? ["speaker"] : [])]
        }
      }
    },
    required: ["summary", "segments"]
  };

  return processAudio({
    fileUri,
    audioBase64,
    mimeType,
    prompt,
    responseSchema: schema
  });
};

/**
 * Analyze audio content (music, sounds, speech characteristics).
 * 
 * @param {object} options - Analysis options
 * @param {string} options.fileUri - Uploaded file URI
 * @param {string} options.audioBase64 - Base64 audio data (alternative)
 * @param {string} options.mimeType - Audio MIME type
 * @param {string} options.analysisType - Type: "general", "music", "speech", "sounds"
 * @param {string} options.customPrompt - Custom analysis prompt (optional)
 * @returns {Promise<object>} Analysis result
 */
export const analyzeAudio = async (options) => {
  const {
    fileUri,
    audioBase64,
    mimeType,
    analysisType = "general",
    customPrompt
  } = options;

  const prompts = {
    general: `Analyze this audio file comprehensively. Describe:
- Type of audio (speech, music, ambient sounds, mixed)
- Main content and topics discussed (if speech)
- Notable sounds or instruments (if music/sounds)
- Audio quality assessment
- Any notable characteristics or anomalies`,
    
    music: `Analyze this music audio. Describe:
- Genre and style
- Tempo (BPM estimate) and time signature
- Key and mood
- Instruments identified
- Song structure (verse, chorus, bridge, etc.)
- Vocals (if any): gender, style, language
- Production quality assessment`,
    
    speech: `Analyze the speech in this audio. Describe:
- Number of speakers and their characteristics
- Speaking style (formal, casual, emotional)
- Speech clarity and pace
- Background noise assessment
- Language and accent identification
- Key topics and themes discussed`,
    
    sounds: `Analyze the sounds in this audio. Identify:
- All distinct sound sources
- Environmental context (indoor, outdoor, etc.)
- Temporal patterns (continuous, intermittent)
- Sound quality and recording conditions
- Any notable or unusual sounds`
  };

  const prompt = customPrompt || prompts[analysisType] || prompts.general;

  const schema = {
    type: "OBJECT",
    properties: {
      audio_type: {
        type: "STRING",
        description: "Primary type of audio content"
      },
      summary: {
        type: "STRING",
        description: "Brief summary of the audio"
      },
      details: {
        type: "STRING",
        description: "Detailed analysis results as text"
      },
      quality_assessment: {
        type: "STRING",
        description: "Audio quality assessment"
      },
      notable_elements: {
        type: "ARRAY",
        items: { type: "STRING" },
        description: "Notable elements or characteristics"
      }
    },
    required: ["audio_type", "summary"]
  };

  return processAudio({
    fileUri,
    audioBase64,
    mimeType,
    prompt,
    responseSchema: schema
  });
};

// ─────────────────────────────────────────────────────────────
// Text-to-Speech Generation
// ─────────────────────────────────────────────────────────────

/**
 * Generate single-speaker audio from text.
 * 
 * @param {object} options - TTS options
 * @param {string} options.text - Text to convert to speech
 * @param {string} options.voice - Voice name (e.g., "Kore", "Puck")
 * @returns {Promise<{data: Buffer, mimeType: string}>} PCM audio data (24kHz, 16-bit, mono)
 */
export const generateSpeech = async ({ text, voice = "Kore" }) => {
  log.gemini("Generating speech", `${voice}: ${text.substring(0, 60)}...`);

  const body = {
    contents: [{
      parts: [{ text }]
    }],
    generationConfig: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: {
            voiceName: voice
          }
        }
      }
    }
  };

  const response = await fetch(TTS_ENDPOINT, {
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

  const audioData = data.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  
  if (!audioData) {
    throw new Error("No audio data in TTS response");
  }

  recordGemini("tts");
  log.geminiResult(true, `Generated speech with voice: ${voice}`);

  return {
    data: Buffer.from(audioData, "base64"),
    mimeType: "audio/wav"
  };
};

/**
 * Generate multi-speaker audio from text.
 * 
 * @param {object} options - TTS options
 * @param {string} options.text - Text/script with speaker labels
 * @param {Array<{speaker: string, voice: string}>} options.speakers - Speaker-voice mappings
 * @returns {Promise<{data: Buffer, mimeType: string}>} PCM audio data
 */
export const generateMultiSpeakerSpeech = async ({ text, speakers }) => {
  const speakerNames = speakers.map(s => s.speaker).join(", ");
  log.gemini("Generating multi-speaker speech", speakerNames);

  const speakerVoiceConfigs = speakers.map(({ speaker, voice }) => ({
    speaker,
    voiceConfig: {
      prebuiltVoiceConfig: {
        voiceName: voice
      }
    }
  }));

  const body = {
    contents: [{
      parts: [{ text }]
    }],
    generationConfig: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        multiSpeakerVoiceConfig: {
          speakerVoiceConfigs
        }
      }
    }
  };

  const response = await fetch(TTS_ENDPOINT, {
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

  const audioData = data.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  
  if (!audioData) {
    throw new Error("No audio data in TTS response");
  }

  recordGemini("tts");
  log.geminiResult(true, `Generated multi-speaker speech: ${speakerNames}`);

  return {
    data: Buffer.from(audioData, "base64"),
    mimeType: "audio/wav"
  };
};
