/**
 * Native tool definitions — plain JS functions in OpenAI function format.
 *
 * These are "native" tools that run directly in the same process,
 * as opposed to MCP tools which are called through the protocol.
 * The agent treats both identically via the unified handler map.
 */

export const nativeTools = [
  {
    type: "function",
    name: "calculate",
    description: "Perform a basic math calculation",
    parameters: {
      type: "object",
      properties: {
        operation: {
          type: "string",
          enum: ["add", "subtract", "multiply", "divide"],
          description: "The math operation to perform"
        },
        a: { type: "number", description: "First operand" },
        b: { type: "number", description: "Second operand" }
      },
      required: ["operation", "a", "b"],
      additionalProperties: false
    },
    strict: true
  },
  {
    type: "function",
    name: "uppercase",
    description: "Convert text to uppercase",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "Text to convert" }
      },
      required: ["text"],
      additionalProperties: false
    },
    strict: true
  }
];

export const nativeHandlers = {
  calculate({ operation, a, b }) {
    const ops = {
      add: () => a + b,
      subtract: () => a - b,
      multiply: () => a * b,
      divide: () => (b === 0 ? { error: "Division by zero" } : a / b)
    };

    const result = ops[operation]?.();
    return typeof result === "object" ? result : { result, expression: `${a} ${operation} ${b}` };
  },

  uppercase({ text }) {
    return { result: text.toUpperCase() };
  }
};
