/**
 * Replicate API wrapper for Kling video generation.
 * Uses kling-v2.5-turbo-pro model.
 */

import Replicate from "replicate";
import { writeFile, readFile } from "fs/promises";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import log from "../helpers/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, "../..");

// Initialize Replicate client
const replicate = new Replicate();

const KLING_MODEL = "kwaivgi/kling-v2.5-turbo-pro";

/**
 * Generate video from text prompt.
 * 
 * @param {object} options - Generation options
 * @param {string} options.prompt - Text description of the video
 * @param {number} options.duration - Duration in seconds (5 or 10)
 * @param {string} options.aspectRatio - Aspect ratio (16:9, 9:16, 1:1)
 * @param {string} options.negativePrompt - Things to avoid
 * @returns {Promise<{url: string, localPath: string}>} Video result
 */
export const generateVideo = async ({
  prompt,
  duration = 10,
  aspectRatio = "16:9",
  negativePrompt = ""
}) => {
  log.gemini("Generating video (Kling)", `${duration}s - ${prompt.substring(0, 50)}...`);

  const input = {
    prompt,
    duration,
    aspect_ratio: aspectRatio,
    negative_prompt: negativePrompt
  };

  try {
    const output = await replicate.run(KLING_MODEL, { input });
    
    // Get the URL from the output
    const videoUrl = output.url ? output.url() : output;
    
    log.geminiResult(true, `Video generated: ${videoUrl}`);

    return {
      url: videoUrl,
      prompt,
      duration,
      aspectRatio
    };
  } catch (error) {
    log.error("generateVideo", error.message);
    throw error;
  }
};

/**
 * Generate video from an image (image-to-video).
 * 
 * @param {object} options - Generation options
 * @param {string} options.prompt - Text description/direction for the video
 * @param {string} options.startImagePath - Path to the start frame image
 * @param {string} options.endImagePath - Optional path to end frame image
 * @param {number} options.duration - Duration in seconds (5 or 10)
 * @param {string} options.aspectRatio - Aspect ratio (ignored if start_image provided)
 * @param {string} options.negativePrompt - Things to avoid
 * @returns {Promise<{url: string, localPath: string}>} Video result
 */
export const imageToVideo = async ({
  prompt,
  startImagePath,
  endImagePath,
  duration = 10,
  aspectRatio = "16:9",
  negativePrompt = ""
}) => {
  log.gemini("Image-to-video (Kling)", `${duration}s from ${startImagePath}`);

  // Read start image
  const startImageFullPath = join(PROJECT_ROOT, startImagePath);
  const startImageBuffer = await readFile(startImageFullPath);

  const input = {
    prompt,
    duration,
    aspect_ratio: aspectRatio,
    negative_prompt: negativePrompt,
    start_image: startImageBuffer
  };

  // Add end image if provided
  if (endImagePath) {
    const endImageFullPath = join(PROJECT_ROOT, endImagePath);
    const endImageBuffer = await readFile(endImageFullPath);
    input.end_image = endImageBuffer;
  }

  try {
    const output = await replicate.run(KLING_MODEL, { input });
    
    // Get the URL from the output
    const videoUrl = output.url ? output.url() : output;
    
    log.geminiResult(true, `Video generated from image: ${videoUrl}`);

    return {
      url: videoUrl,
      prompt,
      duration,
      startImage: startImagePath,
      endImage: endImagePath || null
    };
  } catch (error) {
    log.error("imageToVideo", error.message);
    throw error;
  }
};

/**
 * Download video from URL and save to workspace.
 * 
 * @param {string} url - Video URL
 * @param {string} outputName - Base name for output file
 * @returns {Promise<string>} Local file path
 */
export const downloadVideo = async (url, outputName) => {
  log.info(`Downloading video: ${outputName}`);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download video: ${response.statusText}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  
  const timestamp = Date.now();
  const filename = `${outputName}_${timestamp}.mp4`;
  const outputPath = join(PROJECT_ROOT, "workspace/output", filename);
  
  await writeFile(outputPath, buffer);
  
  const relativePath = `workspace/output/${filename}`;
  log.success(`Video saved: ${relativePath}`);

  return relativePath;
};
