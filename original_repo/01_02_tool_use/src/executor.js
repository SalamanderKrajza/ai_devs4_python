import { chat, extractToolCalls, extractText } from "./api.js";

const MAX_TOOL_ROUNDS = 10;

const logQuery = (query) => {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`Query: ${query}`);
  console.log("=".repeat(60));
};

const logResult = (text) => console.log(`\nA: ${text}`);

const executeToolCalls = async (toolCalls, handlers) => {
  console.log(`\nTool calls: ${toolCalls.length}`);

  return Promise.all(
    toolCalls.map(async (call) => {
      const args = JSON.parse(call.arguments);
      console.log(`  → ${call.name}(${JSON.stringify(args)})`);

      try {
        const handler = handlers[call.name];
        if (!handler) throw new Error(`Unknown tool: ${call.name}`);

        const result = await handler(args);
        console.log(`    ✓ Success`);
        return { type: "function_call_output", call_id: call.call_id, output: JSON.stringify(result) };
      } catch (error) {
        console.log(`    ✗ Error: ${error.message}`);
        return { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ error: error.message }) };
      }
    })
  );
};

export const processQuery = async (query, { model, tools, handlers, instructions }) => {
  const chatConfig = { model, tools, instructions };
  logQuery(query);
  // Each example query is isolated. We keep conversation state
  // only within the current query while the model is calling tools.
  let conversation = [{ role: "user", content: query }];

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const response = await chat({ ...chatConfig, input: conversation });
    const toolCalls = extractToolCalls(response);

    if (toolCalls.length === 0) {
      const text = extractText(response) ?? "No response";
      logResult(text);
      return text;
    }

    const toolResults = await executeToolCalls(toolCalls, handlers);

    conversation = [
      ...conversation,
      ...toolCalls,
      ...toolResults
    ];
  }

  logResult("Max tool rounds reached");
  return "Max tool rounds reached";
};
