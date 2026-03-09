export const searchSchema = {
  type: "json_schema",
  name: "web_search_result",
  strict: true,
  schema: {
    type: "object",
    description: "Web search summary and sources for a single concept.",
    properties: {
      summary: {
        type: "string",
        description: "Concise factual summary grounded in sources."
      },
      keyPoints: {
        type: "array",
        items: { type: "string" },
        minItems: 0,
        description: "2-4 concise bullet-like key points."
      },
      sources: {
        type: "array",
        minItems: 0,
        description: "Cited sources from web search.",
        items: {
          type: "object",
          properties: {
            title: {
              type: ["string", "null"],
              description: "Optional page title."
            },
            url: {
              type: "string",
              description: "Source URL."
            }
          },
          required: ["title", "url"],
          additionalProperties: false
        }
      }
    },
    required: ["summary", "keyPoints", "sources"],
    additionalProperties: false
  }
};
