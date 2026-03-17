/**
 * Token usage and cache hit rate tracking.
 *
 * Records cumulative stats from each Responses API call
 * and prints a summary at the end of the run.
 */

import log from "./logger.js";

const stats = {
  requests: 0,
  inputTokens: 0,
  cachedTokens: 0,
  outputTokens: 0,
  totalTokens: 0
};

export const recordUsage = (usage) => {
  if (!usage) return;

  stats.requests++;
  stats.inputTokens += usage.input_tokens || 0;
  stats.cachedTokens += usage.input_tokens_details?.cached_tokens || 0;
  stats.outputTokens += usage.output_tokens || 0;
  stats.totalTokens += usage.total_tokens || 0;
};

export const logStats = () => {
  const cacheHitRate = stats.inputTokens > 0
    ? ((stats.cachedTokens / stats.inputTokens) * 100).toFixed(1)
    : 0;

  log.info(`📊 Stats: ${stats.requests} requests | ${stats.totalTokens} tokens | Cache: ${cacheHitRate}% (${stats.cachedTokens}/${stats.inputTokens})`);
};
