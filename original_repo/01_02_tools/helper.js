export const getToolCalls = (response) =>
  response.output.filter((item) => item.type === "function_call");

export const getFinalText = (response) =>
  response.output_text
  ?? response.output.find((item) => item.type === "message")?.content?.[0]?.text
  ?? "No response";

const supportsColor = Boolean(process.stdout.isTTY && !process.env.NO_COLOR);

const ansi = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  blue: "\x1b[34m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  magenta: "\x1b[35m",
  yellow: "\x1b[33m",
};

const colorize = (text, ...styles) => {
  if (!supportsColor) {
    return text;
  }

  const sequence = styles.map((style) => ansi[style]).join("");
  return `${sequence}${text}${ansi.reset}`;
};

const label = (text, color) => colorize(`[${text}]`, "bold", color);
const formatJson = (value) => JSON.stringify(value, null, 2);

export const logQuestion = (text) => {
  console.log(`${label("USER", "blue")} ${text}\n`);
};

export const logToolCall = (name, args) => {
  console.log(`${label("TOOL", "magenta")} ${colorize(name, "bold")}`);
  console.log(colorize("Arguments:", "cyan"));
  console.log(colorize(formatJson(args), "dim"));
};

export const logToolResult = (result) => {
  console.log(colorize("Result:", "yellow"));
  console.log(colorize(formatJson(result), "dim"));
  console.log("");
};

export const logAnswer = (text) => {
  console.log(`${label("ASSISTANT", "green")} ${text}`);
};

export const executeToolCall = async (call, handlers) => {
  const args = JSON.parse(call.arguments);
  const handler = handlers[call.name];

  if (!handler) {
    throw new Error(`Unknown tool: ${call.name}`);
  }

  logToolCall(call.name, args);
  const result = await handler(args);
  logToolResult(result);

  return {
    type: "function_call_output",
    call_id: call.call_id,
    output: JSON.stringify(result),
  };
};

export const buildNextConversation = async (conversation, toolCalls, handlers) => {
  const toolResults = await Promise.all(
    toolCalls.map((call) => executeToolCall(call, handlers)),
  );

  return [...conversation, ...toolCalls, ...toolResults];
};
