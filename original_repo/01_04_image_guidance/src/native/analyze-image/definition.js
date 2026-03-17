export const analyzeImageDefinition = {
  type: "function",
  name: "analyze_image",
  description: "Analyze a generated or edited image and return an ACCEPT or RETRY verdict. RETRY should be used only for blocking issues, while minor polish notes should still allow ACCEPT.",
  parameters: {
    type: "object",
    properties: {
      image_path: {
        type: "string",
        description: "Path to the image file relative to the project root"
      },
      original_prompt: {
        type: "string",
        description: "The original prompt or instructions used to generate/edit the image"
      },
      check_aspects: {
        type: "array",
        items: {
          type: "string",
          enum: ["prompt_adherence", "visual_artifacts", "anatomy", "text_rendering", "style_consistency", "composition"]
        },
        description: "Specific aspects to check. If not provided, checks all aspects."
      }
    },
    required: ["image_path", "original_prompt"],
    additionalProperties: false
  },
  strict: false
};
