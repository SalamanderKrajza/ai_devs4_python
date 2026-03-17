import {
  AI_API_KEY,
  EXTRA_API_HEADERS,
  RESPONSES_API_ENDPOINT,
  resolveModelForProvider
} from "../config.js";
import { extractResponseText, toMessage } from "./helpers.js";

const MODEL = resolveModelForProvider("gpt-5.2");

async function chat(input, history = []) {
  const response = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify({
      model: MODEL,
      input: [...history, toMessage("user", input)],
      reasoning: { effort: "medium" }
    })
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    const message = data?.error?.message ?? `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  const text = extractResponseText(data);

  if (!text) {
    throw new Error("Missing text output in API response");
  }

  return {
    text,
    reasoningTokens: data?.usage?.output_tokens_details?.reasoning_tokens ?? 0
  };
}

async function main() {
  const firstQuestion = "What is 25 * 48?";
  const firstAnswer = await chat(firstQuestion);

  const secondQuestion = "Divide that by 4.";
  const secondQuestionContext = [
    {
      type: "message",
      role: "user",
      content: firstQuestion
    },
    {
      type: "message",
      role: "assistant",
      content: firstAnswer.text
    }
  ];
  const secondAnswer = await chat(secondQuestion, secondQuestionContext);

  console.log("Q:", firstQuestion);
  console.log("A:", firstAnswer.text, `(${firstAnswer.reasoningTokens} reasoning tokens)`);
  console.log("Q:", secondQuestion);
  console.log("A:", secondAnswer.text, `(${secondAnswer.reasoningTokens} reasoning tokens)`);
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exit(1);
});