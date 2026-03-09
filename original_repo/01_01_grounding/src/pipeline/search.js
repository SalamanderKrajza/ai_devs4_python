import { paths, models, cli } from "../config.js";
import { callResponses, parseJsonOutput, getWebSources } from "../api.js";
import { readJsonIfExists, safeWriteJson } from "../utils/file.js";
import { chunk } from "../utils/text.js";
import { searchSchema } from "../schemas/index.js";
import { buildSearchPrompt } from "../prompts/index.js";
import { buildConceptEntries } from "./extract.js";
import { AI_PROVIDER } from "../../../config.js";

const CONCURRENCY = 5;
const OPENROUTER_ONLINE_SUFFIX = ":online";
const OPENAI_SEARCH_INCLUDE = ["web_search_call.action.sources"];

const resolveSearchModel = () => {
  if (AI_PROVIDER !== "openrouter") {
    return models.search;
  }

  return models.search.endsWith(OPENROUTER_ONLINE_SUFFIX)
    ? models.search
    : `${models.search}${OPENROUTER_ONLINE_SUFFIX}`;
};

const buildSearchRequest = ({ model, input }) => {
  if (AI_PROVIDER === "openrouter") {
    // OpenRouter responses endpoint rejects OpenAI-specific include values.
    // Web search is enabled by using an `:online` model slug.
    return {
      model,
      input,
      textFormat: searchSchema
    };
  }

  return {
    model,
    input,
    tools: [{ type: "web_search" }],
    include: OPENAI_SEARCH_INCLUDE,
    textFormat: searchSchema
  };
};

const searchSingleConcept = async (concept, model) => {
  const input = buildSearchPrompt({ concept });
  const request = buildSearchRequest({ model, input });

  const data = await callResponses(request);

  const result = parseJsonOutput(data, `search: ${concept.canonical}`);
  const sources = getWebSources(data);

  return {
    canonical: concept.canonical,
    ...result,
    rawSources: sources
  };
};

export const searchConcepts = async (conceptsData, dedupeData) => {
  const searchModel = resolveSearchModel();

  if (AI_PROVIDER === "openrouter") {
    console.warn(`   Using OpenRouter provider with web plugin via model: ${searchModel}`);
  }

  const existing = await readJsonIfExists(paths.search);
  const shouldReuse =
    existing && existing.sourceFile === conceptsData.sourceFile && !cli.force;
  const sameSourceHash = existing?.sourceHash === conceptsData.sourceHash;
  const sameDedupeHash = existing?.dedupeHash === dedupeData.dedupeHash;
  const sameModel = existing?.model === searchModel;

  const shouldReset = !sameSourceHash || !sameDedupeHash || !sameModel;

  if (shouldReuse && shouldReset) {
    console.log("   Search cache invalidated (source, dedupe, or model changed)");
  }

  const base = shouldReuse && !shouldReset
    ? existing
    : {
        sourceFile: conceptsData.sourceFile,
        model: searchModel,
        sourceHash: conceptsData.sourceHash,
        dedupeHash: dedupeData.dedupeHash,
        resultsByCanonical: {}
      };

  if (shouldReuse && !shouldReset) {
    if (!base.sourceHash) {
      base.sourceHash = conceptsData.sourceHash;
    }
    if (!base.dedupeHash) {
      base.dedupeHash = dedupeData.dedupeHash;
    }
  }

  const conceptEntries = buildConceptEntries(conceptsData)
    .map((concept, id) => ({ id, ...concept }))
    .filter((concept) => concept.needsSearch);

  const conceptById = new Map(conceptEntries.map((entry) => [entry.id, entry]));

  const canonicalConcepts = dedupeData.groups.map((group) => {
    const memberEntries = group.ids
      .map((id) => conceptById.get(id))
      .filter(Boolean);

    const searchQuery =
      memberEntries.find((entry) => entry.searchQuery)?.searchQuery ??
      group.canonical;

    const surfaceForms = memberEntries.flatMap(
      (entry) => entry.surfaceForms || []
    );

    return {
      canonical: group.canonical,
      aliases: group.aliases,
      searchQuery,
      surfaceForms: Array.from(new Set(surfaceForms))
    };
  });

  const pending = canonicalConcepts.filter(
    (concept) => !base.resultsByCanonical[concept.canonical]
  );

  if (!pending.length && sameSourceHash && sameDedupeHash) {
    console.log("   Using cached search results");
    return base;
  }

  console.log(`   ${pending.length} concepts to search (${CONCURRENCY} parallel)`);
  const batches = chunk(pending, CONCURRENCY);

  for (const [batchIndex, batch] of batches.entries()) {
    if (!batch.length) continue;

    const batchLabels = batch.map((c) => c.canonical).join(", ");
    console.log(`  [batch ${batchIndex + 1}/${batches.length}] Searching: ${batchLabels}`);

    const results = await Promise.all(
      batch.map((concept) => searchSingleConcept(concept, searchModel))
    );

    for (const result of results) {
      base.resultsByCanonical[result.canonical] = {
        canonical: result.canonical,
        summary: result.summary,
        keyPoints: result.keyPoints,
        sources: result.sources
      };
      console.log(`    ✓ ${result.canonical} (${result.sources.length} sources)`);
    }

    await safeWriteJson(paths.search, base);
  }

  return base;
};
