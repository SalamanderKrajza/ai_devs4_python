/**
 * Token usage and cache hit rate tracking.
 */

import log from "./logger.js";

const stats = {
  requests: 0,
  inputTokens: 0,
  cachedTokens: 0,
  outputTokens: 0,
  totalTokens: 0
};

/**
 * Records usage from an API response.
 */
export const recordUsage = (usage) => {
  if (!usage) return;
  
  stats.requests++;
  stats.inputTokens += usage.input_tokens || 0;
  stats.cachedTokens += usage.input_tokens_details?.cached_tokens || 0;
  stats.outputTokens += usage.output_tokens || 0;
  stats.totalTokens += usage.total_tokens || 0;
};

/**
 * Gets current stats with calculated cache hit rate.
 */
export const getStats = () => {
  const cacheHitRate = stats.inputTokens > 0 
    ? ((stats.cachedTokens / stats.inputTokens) * 100).toFixed(1)
    : 0;
  
  return {
    ...stats,
    cacheHitRate: `${cacheHitRate}%`
  };
};

/**
 * Logs current stats summary.
 */
export const logStats = () => {
  const s = getStats();
  log.info(`📊 Stats: ${s.requests} requests | ${s.totalTokens} tokens | Cache: ${s.cacheHitRate} (${s.cachedTokens}/${s.inputTokens})`);
};

export default { recordUsage, getStats, logStats };
