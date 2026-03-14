import { editImage, editImageWithReferences, generateImage } from "./gemini.js";
import { loadReferenceImages, saveGeneratedImage } from "../shared/image-files.js";
import log from "../../helpers/logger.js";

const buildImageOptions = ({ aspect_ratio, image_size }) => {
  const options = {};

  if (aspect_ratio) options.aspectRatio = aspect_ratio;
  if (image_size) options.imageSize = image_size;

  return options;
};

const createOrEditImage = async ({ prompt, reference_images, options }) => {
  if (!reference_images?.length) {
    return {
      mode: "generate",
      result: await generateImage(prompt, options)
    };
  }

  const loadedImages = await loadReferenceImages(reference_images);

  const result = loadedImages.length === 1
    ? await editImage(prompt, loadedImages[0].data, loadedImages[0].mimeType, options)
    : await editImageWithReferences(prompt, loadedImages, options);

  return {
    mode: "edit",
    result
  };
};

export const createImage = async ({ prompt, output_name, reference_images, aspect_ratio, image_size }) => {
  try {
    const options = buildImageOptions({ aspect_ratio, image_size });
    const { mode, result } = await createOrEditImage({ prompt, reference_images, options });
    const output_path = await saveGeneratedImage(output_name, result);

    log.success(`Image saved: ${output_path}`);

    return {
      success: true,
      mode,
      output_path,
      mime_type: result.mimeType,
      prompt_used: prompt,
      reference_images: reference_images || []
    };
  } catch (error) {
    log.error("create_image", error.message);
    return { success: false, error: error.message };
  }
};
