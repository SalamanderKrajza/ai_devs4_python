import { paths, models, cli } from "../config.js";
import { callResponses, parseJsonOutput } from "../api.js";
import { readJsonIfExists, safeWriteJson } from "../utils/file.js";
import { hashObject } from "../utils/hash.js";
import { dedupeSchema } from "../schemas/index.js";
import { buildDedupePrompt } from "../prompts/index.js";
import { buildConceptEntries } from "./extract.js";

export const dedupeConcepts = async (conceptsData) => {
  const existing = await readJsonIfExists(paths.dedupe);
  const sameSource =
    existing && existing.sourceFile === conceptsData.sourceFile;
  const sameCounts =
    existing?.paragraphCount === conceptsData.paragraphCount &&
    existing?.conceptCount === conceptsData.conceptCount;
  const sameSourceHash = existing?.sourceHash === conceptsData.sourceHash;
  const sameConceptsHash = existing?.conceptsHash === conceptsData.conceptsHash;

  if (sameSource && sameCounts && sameSourceHash && sameConceptsHash && !cli.force) {
    console.log("   Using cached dedupe data");
    if (!existing.dedupeHash) {
      existing.dedupeHash = hashObject(existing.groups || []);
      await safeWriteJson(paths.dedupe, existing);
    }
    return existing;
  }

  const conceptEntries = buildConceptEntries(conceptsData)
    .map((concept, id) => ({ id, ...concept }))
    .filter((concept) => concept.needsSearch);

  if (!conceptEntries.length) {
    const empty = {
      sourceFile: conceptsData.sourceFile,
      model: models.extract,
      sourceHash: conceptsData.sourceHash,
      conceptsHash: conceptsData.conceptsHash,
      paragraphCount: conceptsData.paragraphCount,
      conceptCount: conceptsData.conceptCount,
      dedupeHash: hashObject([]),
      groups: []
    };
    await safeWriteJson(paths.dedupe, empty);
    return empty;
  }

  const input = buildDedupePrompt({ conceptEntries });

  const data = await callResponses({
    model: models.extract,
    input,
    textFormat: dedupeSchema,
    reasoning: { effort: "medium" }
  });

  const result = parseJsonOutput(data, "concept dedupe");

  const dedupeData = {
    sourceFile: conceptsData.sourceFile,
    model: models.extract,
    sourceHash: conceptsData.sourceHash,
    conceptsHash: conceptsData.conceptsHash,
    paragraphCount: conceptsData.paragraphCount,
    conceptCount: conceptsData.conceptCount,
    dedupeHash: hashObject(result.groups),
    groups: result.groups
  };

  await safeWriteJson(paths.dedupe, dedupeData);
  return dedupeData;
};
