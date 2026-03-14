const extractTaggedValue = (text, tag) => {
  const match = text.match(new RegExp(`^${tag}:\\s*(.+)$`, "im"));
  return match?.[1]?.trim() ?? "";
};

const extractBulletSection = (text, section) => {
  const lines = text.split("\n");
  const header = `${section}:`;
  const startIndex = lines.findIndex((line) => line.trim().toUpperCase() === header);

  if (startIndex === -1) {
    return [];
  }

  const items = [];

  for (let index = startIndex + 1; index < lines.length; index++) {
    const trimmed = lines[index].trim();

    if (!trimmed) {
      continue;
    }

    if (/^[A-Z_ ]+:$/.test(trimmed)) {
      break;
    }

    if (trimmed.startsWith("- ")) {
      items.push(trimmed.slice(2).trim());
    }
  }

  return items;
};

export const parseAnalysisReport = (analysis) => {
  const rawVerdict = extractTaggedValue(analysis, "VERDICT").toUpperCase();
  const scoreText = extractTaggedValue(analysis, "SCORE");
  const score = Number.parseInt(scoreText, 10);

  return {
    verdict: rawVerdict === "RETRY" ? "retry" : "accept",
    score: Number.isFinite(score) ? score : null,
    blockingIssues: extractBulletSection(analysis, "BLOCKING_ISSUES"),
    minorIssues: extractBulletSection(analysis, "MINOR_ISSUES"),
    nextPromptHints: extractBulletSection(analysis, "NEXT_PROMPT_HINT")
  };
};
