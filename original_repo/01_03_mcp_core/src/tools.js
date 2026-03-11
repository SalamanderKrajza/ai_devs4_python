/**
 * MCP tool definitions — schemas and handlers for the demo server.
 *
 * Tools are actions the LLM can invoke. Each tool has:
 *  - inputSchema:  Zod schema describing the expected arguments
 *  - handler:      async function that executes the logic and returns a result
 *
 * The summarize tool demonstrates two advanced MCP features:
 *  - elicitation:  server asks the client for user confirmation before proceeding
 *  - sampling:     server asks the client to generate an LLM completion
 * Both use extra.sendRequest() to call back into the client.
 */

import { z } from "zod";

// Helpers for building MCP tool results
const textResult = (text) => ({ content: [{ type: "text", text }] });
const errorResult = (msg) => ({ content: [{ type: "text", text: msg }], isError: true });

// Zod schemas for validating client responses to server-initiated requests
const elicitationResponseSchema = z.object({
  action: z.enum(["accept", "decline", "cancel"]),
  content: z.record(z.unknown()).optional()
});

const samplingResponseSchema = z.object({
  role: z.string(),
  content: z.object({ type: z.string(), text: z.string() }),
  model: z.string().optional()
});

// Builds an elicitation/create request — asks the client to show a confirmation form
const confirmationForm = (text) => ({
  method: "elicitation/create",
  params: {
    mode: "form",
    message: `Do you want to summarize this text?\n\n"${text.slice(0, 100)}${text.length > 100 ? "..." : ""}"`,
    requestedSchema: {
      type: "object",
      properties: {
        confirm: { type: "boolean", title: "Confirm", description: "Proceed with summarization?", default: true },
        style: { type: "string", title: "Summary Style", enum: ["concise", "detailed", "bullet-points"], default: "concise" }
      },
      required: ["confirm"]
    }
  }
});

// Builds a sampling/createMessage request — asks the client to call an LLM
const samplingMessage = (text, style, maxLength) => ({
  method: "sampling/createMessage",
  params: {
    messages: [{ role: "user", content: { type: "text", text: `Summarize in a ${style} style. Max ${maxLength} words.\n\nText: ${text}` } }],
    maxTokens: 200
  }
});

export const tools = [
  {
    name: "summarize_with_confirmation",
    config: {
      title: "Summarize with Confirmation",
      description: "Summarizes text after getting user confirmation. Demonstrates elicitation and sampling.",
      inputSchema: {
        text: z.string().describe("The text to summarize"),
        maxLength: z.number().optional().describe("Maximum summary length in words")
      }
    },
    handler: async ({ text, maxLength = 50 }, extra) => {
      try {
        // Step 1: Ask the client for user confirmation (elicitation)
        const confirmation = await extra.sendRequest(confirmationForm(text), elicitationResponseSchema);

        if (confirmation.action !== "accept" || !confirmation.content?.confirm) {
          return textResult("Summarization cancelled by user.");
        }

        // Step 2: Ask the client to summarize via its LLM (sampling)
        const style = confirmation.content?.style || "concise";
        const result = await extra.sendRequest(samplingMessage(text, style, maxLength), samplingResponseSchema);

        return textResult(`Summary (${style} style):\n\n${result.content.text}`);
      } catch (error) {
        return errorResult(`Error: ${error.message}. Elicitation/sampling may not be supported by the client.`);
      }
    }
  },
  {
    name: "calculate",
    config: {
      title: "Calculator",
      description: "Performs basic arithmetic operations",
      inputSchema: {
        operation: z.enum(["add", "subtract", "multiply", "divide"]).describe("The operation"),
        a: z.number().describe("First operand"),
        b: z.number().describe("Second operand")
      }
    },
    handler: async ({ operation, a, b }) => {
      const ops = {
        add: () => a + b,
        subtract: () => a - b,
        multiply: () => a * b,
        divide: () => (b !== 0 ? a / b : "Error: Division by zero")
      };

      return textResult(JSON.stringify({ operation, a, b, result: ops[operation]() }));
    }
  }
];
