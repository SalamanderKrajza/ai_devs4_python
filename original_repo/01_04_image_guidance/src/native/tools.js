/**
 * Native tools registry for image guidance.
 */

import { analyzeImageDefinition } from "./analyze-image/definition.js";
import { analyzeImage } from "./analyze-image/handler.js";
import { createImage } from "./create-image/handler.js";
import { createImageDefinition } from "./create-image/definition.js";

export const nativeTools = [createImageDefinition, analyzeImageDefinition];

export const nativeHandlers = {
  create_image: createImage,
  analyze_image: analyzeImage
};

export const isNativeTool = (name) => name in nativeHandlers;

export const executeNativeTool = async (name, args) => {
  const handler = nativeHandlers[name];
  if (!handler) throw new Error(`Unknown native tool: ${name}`);
  return handler(args);
};
