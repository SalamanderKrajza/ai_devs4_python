export const buildSearchPrompt = ({ concept }) => `Use web search to verify and expand on this concept.
Search thoroughly and provide accurate, factual information.
Return JSON only, matching the schema.

Requirements:
- Write a concise summary grounded in search results
- Include 2-4 key points with specific facts
- List sources with titles and URLs from the search

Concept: ${concept.canonical}${concept.searchQuery ? `
Search query: ${concept.searchQuery}` : ""}${concept.aliases?.length ? `
Also known as: ${concept.aliases.join(", ")}` : ""}`;
