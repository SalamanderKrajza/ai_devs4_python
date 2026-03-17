/**
 * Image generation wrapper with OpenRouter and native Gemini backends.
 * Video analysis still uses native Gemini.
 */

import {
  OPENROUTER_API_KEY,
  OPENROUTER_EXTRA_HEADERS
} from "../../../config.js";
import { gemini } from "../config.js";
import { recordGemini } from "../helpers/stats.js";
import log from "../helpers/logger.js";

const VIDEO_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${gemini.videoModel}:generateContent`;

const normalizeImageSize = (imageSize) => {
  if (typeof imageSize !== "string") return imageSize;
  return imageSize.endsWith("k") ? `${imageSize.slice(0, -1)}K` : imageSize;
};

const buildImageConfig = (options = {}) => {
  const imageConfig = {};

  if (options.aspectRatio) {
    imageConfig.aspect_ratio = options.aspectRatio;
  }

  if (options.imageSize) {
    imageConfig.image_size = normalizeImageSize(options.imageSize);
  }

  return Object.keys(imageConfig).length > 0 ? imageConfig : null;
};

const extractNativeText = (interaction) => {
  const textOutput = interaction.outputs?.find((output) => output.type === "text");
  return textOutput?.text?.trim() ?? "";
};

const extractNativeImage = (interaction, actionLabel) => {
  const imageOutput = interaction.outputs?.find((output) => output.type === "image");

  if (!imageOutput) {
    const message = extractNativeText(interaction);
    if (message) {
      throw new Error(`${actionLabel} failed: ${message}`);
    }

    throw new Error("No image output received from image backend");
  }

  return {
    data: imageOutput.data,
    mimeType: imageOutput.mime_type || "image/png"
  };
};

const extractOpenRouterText = (data) => {
  const content = data?.choices?.[0]?.message?.content;

  if (typeof content === "string") {
    return content.trim();
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => (typeof part === "string" ? part : part?.text ?? ""))
    .join("\n")
    .trim();
};

const parseDataUrl = (dataUrl) => {
  const match = /^data:([^;]+);base64,(.+)$/s.exec(dataUrl ?? "");

  if (!match) {
    throw new Error("Expected OpenRouter image output as a base64 data URL");
  }

  return {
    mimeType: match[1],
    data: match[2]
  };
};

const extractOpenRouterImage = (data, actionLabel) => {
  const images = data?.choices?.[0]?.message?.images ?? [];
  const imageUrl = images[0]?.image_url?.url ?? images[0]?.imageUrl?.url;

  if (!imageUrl) {
    const message = extractOpenRouterText(data);
    if (message) {
      throw new Error(`${actionLabel} failed: ${message}`);
    }

    throw new Error("No image output received from OpenRouter");
  }

  return parseDataUrl(imageUrl);
};

const createNativeInteraction = async ({ prompt, referenceImages, imageConfig }) => {
  const payload = {
    model: gemini.imageModel,
    input: referenceImages.length === 0
      ? prompt
      : [
          { type: "text", text: prompt },
          ...referenceImages.map((image) => ({
            type: "image",
            data: image.data,
            mime_type: image.mimeType
          }))
        ],
    response_modalities: ["IMAGE"]
  };

  if (imageConfig) {
    payload.generation_config = { image_config: imageConfig };
  }

  const response = await fetch(gemini.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": gemini.apiKey
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(data?.error?.message || `Gemini image request failed (${response.status})`);
  }

  return data;
};

const createOpenRouterInteraction = async ({ prompt, referenceImages, imageConfig }) => {
  const content = referenceImages.length === 0
    ? prompt
    : [
        { type: "text", text: prompt },
        ...referenceImages.map((image) => ({
          type: "image_url",
          image_url: {
            url: `data:${image.mimeType};base64,${image.data}`
          }
        }))
      ];

  const response = await fetch(gemini.openRouterEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      ...OPENROUTER_EXTRA_HEADERS
    },
    body: JSON.stringify({
      model: gemini.imageModel,
      messages: [
        {
          role: "user",
          content
        }
      ],
      modalities: ["image", "text"],
      ...(imageConfig ? { image_config: imageConfig } : {})
    })
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(data?.error?.message || `OpenRouter image request failed (${response.status})`);
  }

  return data;
};

const requestImage = async ({
  prompt,
  referenceImages = [],
  options = {},
  logAction,
  successLabel,
  statsType,
  failureLabel
}) => {
  const preview = referenceImages.length === 0
    ? prompt.substring(0, 100)
    : `${referenceImages.length} images`;

  log.gemini(logAction, preview);

  const imageConfig = buildImageConfig(options);
  const response = gemini.imageBackend === "openrouter"
    ? await createOpenRouterInteraction({ prompt, referenceImages, imageConfig })
    : await createNativeInteraction({ prompt, referenceImages, imageConfig });

  recordGemini(statsType);

  const image = gemini.imageBackend === "openrouter"
    ? extractOpenRouterImage(response, failureLabel)
    : extractNativeImage(response, failureLabel);

  log.geminiResult(true, `${successLabel} (${image.mimeType})`);

  return image;
};

export const generateImage = async (prompt, options = {}) =>
  requestImage({
    prompt,
    options,
    logAction: "Generating image",
    successLabel: "Generated image",
    statsType: "generate",
    failureLabel: "Image generation"
  });

export const editImage = async (instructions, imageBase64, mimeType, options = {}) =>
  requestImage({
    prompt: instructions,
    referenceImages: [{ data: imageBase64, mimeType }],
    options,
    logAction: "Editing image",
    successLabel: "Edited image",
    statsType: "edit",
    failureLabel: "Image editing"
  });

export const editImageWithReferences = async (instructions, referenceImages, options = {}) =>
  requestImage({
    prompt: instructions,
    referenceImages,
    options,
    logAction: "Editing with references",
    successLabel: "Generated image with references",
    statsType: "edit",
    failureLabel: "Image editing"
  });

// ─────────────────────────────────────────────────────────────
// Video Analysis (GenerateContent API)
// ─────────────────────────────────────────────────────────────

/**
 * Process/analyze video with Gemini.
 * 
 * @param {object} options - Processing options
 * @param {string} options.videoBase64 - Base64 video data
 * @param {string} options.mimeType - Video MIME type
 * @param {string} options.prompt - Analysis instructions
 * @returns {Promise<string>} Analysis result
 */
export const processVideo = async ({ videoBase64, mimeType, prompt }) => {
  if (!gemini.apiKey) {
    throw new Error("GEMINI_API_KEY is required for native video analysis in this example");
  }

  log.gemini("Analyzing video", prompt.substring(0, 80));

  const body = {
    contents: [{
      parts: [
        {
          inline_data: {
            mime_type: mimeType,
            data: videoBase64
          }
        },
        { text: prompt }
      ]
    }]
  };

  const response = await fetch(VIDEO_ENDPOINT, {
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

  recordGemini("analyze_video");

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  
  if (!text) {
    throw new Error("No text response from Gemini");
  }

  log.geminiResult(true, `Video analyzed (${text.length} chars)`);

  return text;
};
