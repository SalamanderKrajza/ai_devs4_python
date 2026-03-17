/**
 * Elicitation handler — auto-accepts MCP forms with inferred defaults.
 *
 * Elicitation is a protocol feature where the server asks the client
 * for structured user input (e.g. "confirm this action?"). The server
 * sends a JSON Schema describing the form, and the client responds
 * with filled-in values.
 *
 * In this demo we auto-accept every form by inferring defaults from
 * the schema — no real user interaction needed.
 */

import { clientLog } from "./log.js";

// Picks a reasonable default for a single schema property:
//   explicit default → boolean true → first enum value
const inferDefault = (prop) => {
  if (prop?.default !== undefined) return prop.default;
  if (prop?.type === "boolean") return true;
  if (prop?.enum?.length) return prop.enum[0];
};

// Walks the schema properties and builds a { key: value } map
// using inferred defaults, skipping any field without one.
const autoFillDefaults = (schema) =>
  Object.fromEntries(
    Object.entries(schema?.properties ?? {})
      .map(([key, prop]) => [key, inferDefault(prop)])
      .filter(([, v]) => v !== undefined)
  );

/**
 * Creates an elicitation request handler for the MCP client.
 *
 * @param {object} options
 * @param {function} options.onElicitation — custom handler; if omitted, auto-accepts
 */
export const createElicitationHandler = (options = {}) => async (request) => {
  clientLog.elicitationRequest(request.params);

  // Only "form" mode is supported by the spec right now
  if (request.params.mode !== "form") {
    return { action: "decline" };
  }

  // Let the caller override with real UI if needed
  if (typeof options.onElicitation === "function") {
    return options.onElicitation(request.params);
  }

  // Demo mode: auto-fill the form from schema defaults
  const content = autoFillDefaults(request.params.requestedSchema);
  clientLog.autoAcceptedElicitation(content);

  return { action: "accept", content };
};
