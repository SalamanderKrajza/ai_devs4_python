import {
  AI_API_KEY,
  EXTRA_API_HEADERS,
  RESPONSES_API_ENDPOINT,
  resolveModelForProvider
} from "../config.js";
import { extractResponseText } from "./helpers.js";

const MODEL = resolveModelForProvider("gpt-5.4");

async function extractPerson(text) {
  const response = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify({
      model: MODEL,
      input: `Extract person information from: "${text}"`,
      text: { format: personSchema }
    })
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    const message = data?.error?.message ?? `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  const outputText = extractResponseText(data);

  if (!outputText) {
    throw new Error("Missing text output in API response");
  }

  return JSON.parse(outputText);
}

const personSchema = {
  type: "json_schema",
  name: "person",
  strict: true,
  schema: {
    type: "object",
    properties: {
      name: {
        type: ["string", "null"],
        description: "Full name of the person. Use null if not mentioned."
      },
      age: {
        type: ["number", "null"],
        description: "Age in years. Use null if not mentioned or unclear."
      },
      occupation: {
        type: ["string", "null"],
        description: "Job title or profession. Use null if not mentioned."
      },
      skills: {
        type: "array",
        items: { type: "string" },
        description: "List of skills, technologies, or competencies. Empty array if none mentioned."
      }
    },
    required: ["name", "age", "occupation", "skills"],
    additionalProperties: false
  }
};

async function main() {
  const text = "John is 30 years old and works as a software engineer. He is skilled in JavaScript, Python, and React.";
  const person = await extractPerson(text);

  console.log("Name:", person.name ?? "unknown");
  console.log("Age:", person.age ?? "unknown");
  console.log("Occupation:", person.occupation ?? "unknown");
  console.log("Skills:", person.skills.length ? person.skills.join(", ") : "none");
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exit(1);
});
