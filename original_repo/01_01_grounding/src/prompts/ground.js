export const buildGroundPrompt = ({ paragraph, groundingItems, index, total }) => `Convert this single paragraph into semantic HTML.
Highlight concepts by wrapping exact surfaceForms with:
<span class="grounded" data-grounding="...">phrase</span>

Rules:
- Only wrap phrases that appear verbatim in the paragraph
- Use the provided dataAttr value verbatim for data-grounding
- Prefer the longest matching surfaceForm when multiple overlaps exist
- Avoid wrapping the same concept multiple times
- Do not add new facts. Preserve the original wording
- Use appropriate HTML tags (h1-h6 for headers, p for paragraphs, ul/ol for lists)

Return JSON only.

Document context: paragraph ${index + 1} of ${total}

Grounding items for this paragraph:
${JSON.stringify(groundingItems, null, 2)}

--- Paragraph ---
${paragraph}`;
