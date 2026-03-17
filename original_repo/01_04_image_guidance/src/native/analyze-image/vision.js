import { api } from "../../config.js";
import {
  AI_API_KEY,
  EXTRA_API_HEADERS,
  RESPONSES_API_ENDPOINT
} from "../../../../config.js";
import { extractResponseText } from "../../helpers/response.js";
import { recordUsage } from "../../helpers/stats.js";

export const vision = async ({ imageBase64, mimeType, question }) => {
  const response = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify({
      model: api.visionModel,
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: question },
            { type: "input_image", image_url: `data:${mimeType};base64,${imageBase64}` }
          ]
        }
      ]
    })
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(data?.error?.message || `Vision request failed (${response.status})`);
  }

  recordUsage(data.usage);
  return extractResponseText(data) || "No response";
};
