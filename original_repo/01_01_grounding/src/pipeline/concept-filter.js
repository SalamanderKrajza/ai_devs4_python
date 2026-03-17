import { CONCEPT_CATEGORIES } from "../schemas/categories.js";

export const MAX_BODY = 5;
export const MAX_HEADER = 1;
const MAX_SURFACE_FORM_LENGTH = 100;

const stripMarkdownSyntax = (text) =>
  text.replace(/^#{1,6}\s+/, "").trim();

const normalizeSurfaceForms = (surfaceForms, paragraph) => {
  if (!Array.isArray(surfaceForms)) {
    return [];
  }

  const cleanParagraph = stripMarkdownSyntax(paragraph);
  const unique = new Set();

  surfaceForms.forEach((form) => {
    if (typeof form !== "string") {
      return;
    }

    let trimmed = form.trim();
    if (!trimmed) {
      return;
    }

    // Strip markdown syntax if present
    trimmed = stripMarkdownSyntax(trimmed);
    if (!trimmed) {
      return;
    }

    // Reject overly long surface forms (likely entire sentences)
    if (trimmed.length > MAX_SURFACE_FORM_LENGTH) {
      return;
    }

    // Check against both original and cleaned paragraph
    if (!paragraph.includes(trimmed) && !cleanParagraph.includes(trimmed)) {
      return;
    }

    unique.add(trimmed);
  });

  return Array.from(unique);
};

const normalizeConcept = (concept, paragraph) => {
  if (!concept || typeof concept !== "object") {
    return null;
  }

  const label = typeof concept.label === "string" ? concept.label.trim() : "";
  if (!label) {
    return null;
  }

  const category = typeof concept.category === "string"
    ? concept.category.trim().toLowerCase()
    : "concept";

  const normalizedCategory = CONCEPT_CATEGORIES.includes(category)
    ? category
    : "concept";

  const needsSearch = Boolean(concept.needsSearch);
  let searchQuery =
    typeof concept.searchQuery === "string" ? concept.searchQuery.trim() : null;

  if (!needsSearch) {
    searchQuery = null;
  } else if (!searchQuery) {
    searchQuery = label;
  }

  const reason = typeof concept.reason === "string" ? concept.reason.trim() : "";
  const surfaceForms = normalizeSurfaceForms(concept.surfaceForms, paragraph);

  if (!surfaceForms.length) {
    return null;
  }

  return {
    label,
    category: normalizedCategory,
    needsSearch,
    searchQuery,
    reason,
    surfaceForms
  };
};

export const filterConcepts = ({ concepts, paragraph, paragraphType }) => {
  const maxCount = paragraphType === "header" ? MAX_HEADER : MAX_BODY;

  if (!Array.isArray(concepts)) {
    return [];
  }

  const normalized = concepts
    .map((concept) => normalizeConcept(concept, paragraph))
    .filter(Boolean);

  // Dedupe by label, keep first occurrence
  const deduped = new Map();
  normalized.forEach((concept) => {
    if (!deduped.has(concept.label)) {
      deduped.set(concept.label, concept);
    }
  });

  // Sort by label length (longer = more specific) and take top N
  const sorted = Array.from(deduped.values())
    .sort((a, b) => b.label.length - a.label.length);

  return sorted.slice(0, maxCount);
};
