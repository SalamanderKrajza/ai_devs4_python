import { paths, models, cli } from "../config.js";
import { callResponses, parseJsonOutput } from "../api.js";
import { ensureDir, readJsonIfExists, safeWriteJson } from "../utils/file.js";
import { hashObject, hashText } from "../utils/hash.js";
import { chunk, getParagraphType, getTargetCount } from "../utils/text.js";
import { extractSchema } from "../schemas/index.js";
import { buildExtractPrompt } from "../prompts/index.js";
import { filterConcepts } from "./concept-filter.js";

const CONCURRENCY = 5;

const updateConceptCounts = (conceptsData) => {
  const conceptCount = conceptsData.paragraphs.reduce(
    (total, paragraph) => total + paragraph.concepts.length,
    0
  );

  conceptsData.paragraphCount = conceptsData.paragraphs.length;
  conceptsData.conceptCount = conceptCount;
};

const updateAndPersist = async (conceptsData, entryByIndex, sourceHash, currentIndices) => {
  for (const index of entryByIndex.keys()) {
    if (!currentIndices.has(index)) {
      entryByIndex.delete(index);
    }
  }

  conceptsData.paragraphs = Array.from(entryByIndex.values()).sort(
    (a, b) => a.index - b.index
  );
  updateConceptCounts(conceptsData);
  conceptsData.sourceHash = sourceHash;
  conceptsData.model = models.extract;
  conceptsData.conceptsHash = computeConceptsHash(conceptsData);
  await safeWriteJson(paths.concepts, conceptsData);
};

const computeConceptsHash = (conceptsData) => {
  const payload = conceptsData.paragraphs.map((paragraph) => ({
    index: paragraph.index,
    hash: paragraph.hash,
    concepts: paragraph.concepts.map((concept) => ({
      label: concept.label,
      category: concept.category,
      needsSearch: concept.needsSearch,
      searchQuery: concept.searchQuery,
      surfaceForms: concept.surfaceForms
    }))
  }));

  return hashObject(payload);
};

export const buildConceptEntries = (conceptsData) => {
  return conceptsData.paragraphs.flatMap((paragraph) =>
    paragraph.concepts.map((concept) => ({
      ...concept,
      paragraphIndex: paragraph.index
    }))
  );
};

const extractSingleParagraph = async (item, total) => {
  const paragraphType = getParagraphType(item.paragraph);
  const targetCount = getTargetCount(paragraphType);

  const input = buildExtractPrompt({
    paragraph: item.paragraph,
    paragraphType,
    targetCount,
    index: item.index,
    total
  });

  const data = await callResponses({
    model: models.extract,
    input,
    textFormat: extractSchema,
    reasoning: { effort: "medium" }
  });

  const result = parseJsonOutput(data, `extract: paragraph ${item.index + 1}`);

  const filtered = filterConcepts({
    concepts: result.concepts || [],
    paragraph: item.paragraph,
    paragraphType
  });

  return {
    index: item.index,
    hash: item.hash,
    text: item.paragraph,
    concepts: filtered,
    rawCount: result.concepts?.length || 0
  };
};

export const extractConcepts = async (paragraphs, sourceFile) => {
  await ensureDir(paths.output);

  const sourceHash = hashText(paragraphs.join("\n\n"));
  const existing = await readJsonIfExists(paths.concepts);
  const shouldReuse = existing && existing.sourceFile === sourceFile && !cli.force;
  const sameSourceHash = existing?.sourceHash === sourceHash;
  const sameModel = existing?.model === models.extract;

  if (shouldReuse && sameSourceHash && sameModel) {
    return existing;
  }

  const conceptsData = shouldReuse
    ? existing
    : {
        sourceFile,
        model: models.extract,
        paragraphs: []
      };

  const entryByIndex = new Map(
    conceptsData.paragraphs.map((paragraph) => [paragraph.index, paragraph])
  );

  const pending = [];

  for (const [index, paragraph] of paragraphs.entries()) {
    const paragraphHash = hashText(paragraph);
    const cached = entryByIndex.get(index);

    if (cached && cached.hash === paragraphHash && !cli.force) {
      console.log(`  [${index + 1}/${paragraphs.length}] Cached`);
      continue;
    }

    pending.push({ index, paragraph, hash: paragraphHash });
  }

  const currentIndices = new Set(paragraphs.map((_, i) => i));

  if (!pending.length) {
    await updateAndPersist(conceptsData, entryByIndex, sourceHash, currentIndices);
    return conceptsData;
  }

  const batches = chunk(pending, CONCURRENCY);
  console.log(`  Processing ${pending.length} paragraphs (${CONCURRENCY} parallel)`);

  for (const [batchIndex, batch] of batches.entries()) {
    const indices = batch.map((item) => item.index + 1).join(", ");
    console.log(`  [batch ${batchIndex + 1}/${batches.length}] Paragraphs: ${indices}`);

    const results = await Promise.all(
      batch.map((item) => extractSingleParagraph(item, paragraphs.length))
    );

    for (const result of results) {
      const dropped = result.rawCount - result.concepts.length;

      entryByIndex.set(result.index, {
        index: result.index,
        hash: result.hash,
        text: result.text,
        concepts: result.concepts
      });

      const dropInfo = dropped > 0 ? ` (filtered ${dropped})` : "";
      console.log(`    ✓ [${result.index + 1}] ${result.concepts.length} concepts${dropInfo}`);
    }

    await updateAndPersist(conceptsData, entryByIndex, sourceHash, currentIndices);
  }

  return conceptsData;
};
