import {
  AI_API_KEY,
  buildResponsesRequest,
  EXTRA_API_HEADERS,
  RESPONSES_API_ENDPOINT,
  resolveModelForProvider,
} from "../config.js";
import {
  buildNextConversation,
  getFinalText,
  getToolCalls,
  logAnswer,
  logQuestion,
} from "./helper.js";

const model = resolveModelForProvider("gpt-4.1-mini");

// `buildResponsesRequest()` maps this to OpenAI web search or OpenRouter online mode.
const webSearch = true;

/*
  Step 1: Define tools the model can call.
  Each tool is a JSON Schema describing the function name, purpose, and expected arguments.
  The model never runs these — it only decides *when* to call them and *with what arguments*.
*/
const tools = [
  {
    type: "function",
    name: "get_weather",
    description: "Get current weather for a given location",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name" },
      },
      required: ["location"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    type: "function",
    name: "send_email",
    description: "Send a short email message to a recipient",
    parameters: {
      type: "object",
      properties: {
        to: { type: "string", description: "Recipient email address" },
        subject: { type: "string", description: "Email subject" },
        body: { type: "string", description: "Plain-text email body" },
      },
      required: ["to", "subject", "body"],
      additionalProperties: false,
    },
    strict: true,
  },
];

/*
  Step 2: Implement the actual logic behind each tool.
  This is regular code — the model has no access to it.
  Here we just return hardcoded data and a mocked email confirmation.
*/
const requireText = (value, fieldName) => {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`"${fieldName}" must be a non-empty string.`);
  }

  return value.trim();
};

const handlers = {
  get_weather({ location }) {
    const city = requireText(location, "location");
    const weather = {
      "Kraków": { temp: -2, conditions: "snow" },
      "London": { temp: 8, conditions: "rain" },
      "Tokyo": { temp: 15, conditions: "cloudy" },
    };
    return weather[city] ?? { temp: null, conditions: "unknown" };
  },

  send_email({ to, subject, body }) {
    const recipient = requireText(to, "to");
    const emailSubject = requireText(subject, "subject");
    const emailBody = requireText(body, "body");

    return {
      success: true,
      status: "sent",
      to: recipient,
      subject: emailSubject,
      body: emailBody,
    };
  },
};

/* Step 3: Send messages + tool definitions to the Responses API */
const requestResponse = async (input) => {
  const body = buildResponsesRequest({
    model,
    input,
    tools,
    webSearch,
  });

  const response = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS,
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message ?? `Request failed (${response.status})`);
  return data;
};

const MAX_TOOL_STEPS = 5;

/*
  Step 4: Run the tool-calling workflow.

  This is not a full autonomous agent.
  It is a small tool-using workflow:

  USER question → model response → optional tool call → tool result → model response

  If the model asks for tools, we execute them and continue.
  If the model answers normally, we return that final text.
*/
const chat = async (conversation) => {
  let currentConversation = conversation;
  let stepsRemaining = MAX_TOOL_STEPS;

  while (stepsRemaining > 0) {
    stepsRemaining -= 1;

    const response = await requestResponse(currentConversation);
    const toolCalls = getToolCalls(response);

    if (toolCalls.length === 0) {
      return getFinalText(response);
    }

    currentConversation = await buildNextConversation(currentConversation, toolCalls, handlers);
  }

  throw new Error(`Tool calling did not finish within ${MAX_TOOL_STEPS} steps.`);
};

const query = "Use web search to check the current weather in Kraków. Then send a short email with the answer to student@example.com.";
logQuestion(query);

const answer = await chat([{ role: "user", content: query }]);
logAnswer(answer);
