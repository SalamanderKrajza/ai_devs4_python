import { readFile, writeFile, mkdir } from "fs/promises";
import { join, extname, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, "../../..");

export const getMimeType = (filepath) => {
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

const getExtension = (mimeType) => {
  const extensions = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp"
  };

  return extensions[mimeType] || ".png";
};

const ensureDir = async (dir) => {
  try {
    await mkdir(dir, { recursive: true });
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
  }
};

export const loadReferenceImages = async (referenceImages) =>
  Promise.all(referenceImages.map(async (imagePath) => {
    const imageBuffer = await readFile(join(PROJECT_ROOT, imagePath));

    return {
      data: imageBuffer.toString("base64"),
      mimeType: getMimeType(imagePath)
    };
  }));

export const readProjectImage = async (imagePath) => {
  const imageBuffer = await readFile(join(PROJECT_ROOT, imagePath));

  return {
    imageBase64: imageBuffer.toString("base64"),
    mimeType: getMimeType(imagePath)
  };
};

export const saveGeneratedImage = async (outputName, result) => {
  const outputDir = join(PROJECT_ROOT, "workspace/output");
  await ensureDir(outputDir);

  const filename = `${outputName}_${Date.now()}${getExtension(result.mimeType)}`;
  const outputPath = join(outputDir, filename);
  const imageBuffer = Buffer.from(result.data, "base64");

  await writeFile(outputPath, imageBuffer);

  return `workspace/output/${filename}`;
};
