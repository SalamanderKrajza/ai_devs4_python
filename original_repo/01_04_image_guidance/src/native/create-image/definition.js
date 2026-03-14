export const createImageDefinition = {
  type: "function",
  name: "create_image",
  description: "Generate or edit images. Reference images should use exact workspace-relative paths such as workspace/reference/walking-pose.png or workspace/reference/running-pose.png.",
  parameters: {
    type: "object",
    properties: {
      prompt: {
        type: "string",
        description: "Description of image to generate, or instructions for editing reference images. Be specific about style, composition, colors, changes."
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
        description: "Optional aspect ratio for the output image. If omitted, follow the template or user request."
      },
      image_size: {
        type: "string",
        enum: ["1k", "2k", "4k"],
        description: "Optional image size. If omitted, follow the template or user request."
      }
    },
    required: ["prompt", "output_name", "reference_images"],
    additionalProperties: false
  },
  strict: false
};
