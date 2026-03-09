import { MAX_HEADER, MAX_BODY } from "../pipeline/concept-filter.js";

export const splitParagraphs = (markdown) => {
  return markdown
    .replace(/\r\n/g, "\n")
    .split(/\n\s*\n+/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0);
};

export const chunk = (items, size) => {
  const count = Math.ceil(items.length / size);

  return Array.from({ length: count }, (_, index) =>
    items.slice(index * size, index * size + size)
  );
};

export const truncate = (text, max) => {
  if (text.length <= max) {
    return text;
  }

  return `${text.slice(0, max - 3)}...`;
};

export const getParagraphType = (paragraph) => {
  const isHeader = /^#{1,6}\s+/.test(paragraph);
  return isHeader ? "header" : "body";
};

export const getTargetCount = (paragraphType) => {
  return paragraphType === "header" ? `0-${MAX_HEADER}` : `2-${MAX_BODY}`;
};
