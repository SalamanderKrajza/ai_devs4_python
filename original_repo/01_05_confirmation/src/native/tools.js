/**
 * Native tools for email sending with whitelist validation.
 */

import { readFile } from "fs/promises";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { sendEmail } from "./resend.js";
import log from "../helpers/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, "../..");
const WHITELIST_PATH = join(PROJECT_ROOT, "workspace/whitelist.json");

/**
 * Load whitelist from JSON file.
 */
const loadWhitelist = async () => {
  try {
    const content = await readFile(WHITELIST_PATH, "utf-8");
    const data = JSON.parse(content);
    return data.allowed_recipients || [];
  } catch (error) {
    log.error("whitelist", `Failed to load whitelist: ${error.message}`);
    return [];
  }
};

/**
 * Check if email is allowed by whitelist.
 * Supports exact email matches and domain patterns (@example.com).
 */
const isEmailAllowed = (email, whitelist) => {
  const normalized = email.toLowerCase();
  const domain = normalized.split("@")[1];

  return whitelist.some(pattern => {
    const p = pattern.toLowerCase();
    if (p.startsWith("@")) {
      return domain === p.slice(1);
    }
    return normalized === p;
  });
};

/**
 * Validate all recipients against whitelist.
 */
const validateRecipients = (recipients, whitelist) => {
  const blocked = recipients.filter(email => !isEmailAllowed(email, whitelist));
  return blocked.length === 0 
    ? { valid: true } 
    : { valid: false, blocked };
};

/**
 * Convert plain text to minimal HTML.
 */
const textToHtml = (text) => {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>\n");
  return `<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333;">${escaped}</div>`;
};

/**
 * Native tool definitions in OpenAI function format.
 */
export const nativeTools = [
  {
    type: "function",
    name: "send_email",
    description: "Send an email to one or more recipients. Recipients must be in the whitelist (workspace/whitelist.json). Supports plain text or HTML content.",
    parameters: {
      type: "object",
      properties: {
        to: {
          type: "array",
          items: { type: "string" },
          description: "Recipient email address(es). Must be in the whitelist."
        },
        subject: {
          type: "string",
          description: "Email subject line."
        },
        body: {
          type: "string",
          description: "Email content. Plain text (newlines preserved) or HTML."
        },
        format: {
          type: "string",
          enum: ["text", "html"],
          description: "Content format: 'text' for plain text, 'html' for HTML. Default: text"
        },
        reply_to: {
          type: "string",
          description: "Optional reply-to email address."
        }
      },
      required: ["to", "subject", "body"],
      additionalProperties: false
    },
    strict: false
  }
];

/**
 * Native tool handlers.
 */
export const nativeHandlers = {
  async send_email({ to, subject, body, format = "text", reply_to }) {
    const recipients = Array.isArray(to) ? to : [to];

    log.tool("send_email", { 
      to: recipients.join(", "), 
      subject: subject.substring(0, 50) 
    });

    try {
      // Load and validate whitelist
      const whitelist = await loadWhitelist();
      
      if (whitelist.length === 0) {
        return { 
          success: false, 
          error: "Whitelist is empty or not configured. Add allowed recipients to workspace/whitelist.json" 
        };
      }

      const validation = validateRecipients(recipients, whitelist);
      
      if (!validation.valid) {
        const blockedList = validation.blocked.join(", ");
        return { 
          success: false, 
          error: `Recipients not in whitelist: ${blockedList}. Update workspace/whitelist.json to allow them.` 
        };
      }

      // Prepare content
      const isHtml = format === "html";
      const html = isHtml ? body : textToHtml(body);
      const text = isHtml ? body.replace(/<[^>]*>/g, "") : body;

      // Send email
      const result = await sendEmail({
        to: recipients,
        subject,
        html,
        text,
        replyTo: reply_to
      });

      log.success(`Email sent: ${result.id}`);

      return {
        success: true,
        id: result.id,
        to: recipients,
        subject
      };
    } catch (error) {
      log.error("send_email", error.message);
      return { success: false, error: error.message };
    }
  }
};

/**
 * Check if a tool is native (not MCP).
 */
export const isNativeTool = (name) => name in nativeHandlers;

/**
 * Execute a native tool.
 */
export const executeNativeTool = async (name, args) => {
  const handler = nativeHandlers[name];
  if (!handler) throw new Error(`Unknown native tool: ${name}`);
  return handler(args);
};
