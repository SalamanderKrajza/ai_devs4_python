/**
 * MCP prompt definitions for the demo server.
 *
 * Prompts are reusable message templates with parameters. They let
 * servers publish structured instructions that clients can discover
 * via listPrompts and instantiate via getPrompt with runtime arguments.
 */

import { z } from "zod";

export const prompts = [
  {
    name: "code-review",
    config: {
      title: "Code Review",
      description: "Template for code review requests",
      argsSchema: {
        code: z.string().describe("The code to review"),
        language: z.string().optional().describe("Programming language"),
        focus: z.enum(["security", "performance", "readability", "all"]).optional()
      }
    },
    handler: async ({ code, language = "unknown", focus = "all" }) => {
      const focusMap = {
        security: "Focus on security vulnerabilities and input validation.",
        performance: "Focus on performance and optimization.",
        readability: "Focus on code clarity and maintainability.",
        all: "Review for security, performance, and readability."
      };

      return {
        messages: [{
          role: "user",
          content: { type: "text", text: `Review this ${language} code.\n\n${focusMap[focus]}\n\n\`\`\`${language}\n${code}\n\`\`\`` }
        }]
      };
    }
  }
];
