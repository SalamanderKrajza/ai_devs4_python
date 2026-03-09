import { readFile, writeFile } from "node:fs/promises";

import { paths, models } from "../config.js";
import { callResponses, parseJsonOutput } from "../api.js";
import { ensureDir } from "../utils/file.js";
import { truncate, chunk, splitParagraphs } from "../utils/text.js";
import { groundSchema } from "../schemas/index.js";
import { buildGroundPrompt } from "../prompts/index.js";
import { buildConceptEntries } from "./extract.js";

const CONCURRENCY = 5;

const escapeAttribute = (value) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const buildGroundingItems = (conceptsData, dedupeData, searchData) => {
  const conceptEntries = buildConceptEntries(conceptsData)
    .map((concept, id) => ({ id, ...concept }))
    .filter((concept) => concept.needsSearch);

  const entryById = new Map(conceptEntries.map((entry) => [entry.id, entry]));

  return dedupeData.groups.map((group) => {
    const members = group.ids.map((id) => entryById.get(id)).filter(Boolean);

    const surfaceForms = members.flatMap((member) => member.surfaceForms || []);
    const paragraphIndices = new Set(members.map((m) => m.paragraphIndex));

    const searchResult = searchData.resultsByCanonical[group.canonical];
    const sources = (searchResult?.sources || [])
      .map((source) => ({
        title: source.title ?? null,
        url: source.url
      }))
      .filter((source) => Boolean(source.url));

    const summary = truncate(searchResult?.summary ?? "", 420);
    const dataAttr = escapeAttribute(
      JSON.stringify({
        summary,
        sources
      })
    );

    return {
      label: group.canonical,
      surfaceForms: Array.from(new Set(surfaceForms)).sort(
        (a, b) => b.length - a.length
      ),
      paragraphIndices: Array.from(paragraphIndices),
      dataAttr
    };
  });
};

const groundSingleParagraph = async (paragraph, relevantItems, index, total) => {
  if (!relevantItems.length) {
    // No grounding items for this paragraph, just convert to basic HTML
    return convertToBasicHtml(paragraph);
  }

  const input = buildGroundPrompt({
    paragraph,
    groundingItems: relevantItems.map(({ label, surfaceForms, dataAttr }) => ({
      label,
      surfaceForms,
      dataAttr
    })),
    index,
    total
  });

  const data = await callResponses({
    model: models.ground,
    input,
    textFormat: groundSchema,
    reasoning: { effort: "medium" }
  });

  const result = parseJsonOutput(data, `ground: paragraph ${index + 1}`);
  return result.html;
};

const convertToBasicHtml = (paragraph) => {
  const trimmed = paragraph.trim();
  
  // Header detection
  const headerMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
  if (headerMatch) {
    const level = headerMatch[1].length;
    const text = headerMatch[2];
    return `<h${level}>${escapeHtml(text)}</h${level}>`;
  }
  
  // List detection
  if (trimmed.match(/^[-*]\s+/m)) {
    const items = trimmed
      .split(/\n/)
      .filter((line) => line.match(/^[-*]\s+/))
      .map((line) => `<li>${escapeHtml(line.replace(/^[-*]\s+/, ""))}</li>`)
      .join("\n");
    return `<ul>\n${items}\n</ul>`;
  }
  
  // Default to paragraph
  return `<p>${escapeHtml(trimmed)}</p>`;
};

const escapeHtml = (text) =>
  text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

export const generateAndApplyTemplate = async (
  markdown,
  conceptsData,
  dedupeData,
  searchData
) => {
  const groundingItems = buildGroundingItems(
    conceptsData,
    dedupeData,
    searchData
  );

  const paragraphs = splitParagraphs(markdown);
  const total = paragraphs.length;

  console.log(`   Processing ${total} paragraphs (${CONCURRENCY} parallel)`);

  const batches = chunk(
    paragraphs.map((p, i) => ({ paragraph: p, index: i })),
    CONCURRENCY
  );

  const htmlParts = new Array(total);

  for (const [batchIndex, batch] of batches.entries()) {
    const indices = batch.map((item) => item.index + 1).join(", ");
    console.log(`  [batch ${batchIndex + 1}/${batches.length}] Paragraphs: ${indices}`);

    const results = await Promise.all(
      batch.map(({ paragraph, index }) => {
        // Find grounding items relevant to this paragraph
        const relevantItems = groundingItems.filter((item) =>
          item.paragraphIndices.includes(index)
        );
        return groundSingleParagraph(paragraph, relevantItems, index, total);
      })
    );

    for (let i = 0; i < batch.length; i++) {
      const { index } = batch[i];
      htmlParts[index] = results[i];
      console.log(`    ✓ [${index + 1}] grounded`);
    }
  }

  const htmlChunk = htmlParts.join("\n\n");

  const template = await readFile(paths.template, "utf8");

  if (!template.includes("<!--CONTENT-->")) {
    throw new Error("Template is missing <!--CONTENT--> placeholder.");
  }

  const filled = template.replace("<!--CONTENT-->", htmlChunk);

  await ensureDir(paths.output);
  await writeFile(paths.grounded, filled, "utf8");

  return paths.grounded;
};
