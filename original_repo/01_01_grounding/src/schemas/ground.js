export const groundSchema = {
  type: "json_schema",
  name: "grounded_paragraph",
  strict: true,
  schema: {
    type: "object",
    description: "HTML output for a single grounded paragraph.",
    properties: {
      html: {
        type: "string",
        description: "HTML fragment for this paragraph with grounded spans."
      }
    },
    required: ["html"],
    additionalProperties: false
  }
};
