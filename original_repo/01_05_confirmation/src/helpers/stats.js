/**
 * Token usage statistics tracker.
 */

let totalTokens = { input: 0, output: 0, requests: 0 };

export const recordUsage = (usage) => {
  if (!usage) return;
  totalTokens.input += usage.input_tokens || 0;
  totalTokens.output += usage.output_tokens || 0;
  totalTokens.requests += 1;
};

export const getStats = () => ({ ...totalTokens });

export const logStats = () => {
  const { input, output, requests } = totalTokens;
  console.log(`\n📊 Stats: ${requests} requests, ${input} input tokens, ${output} output tokens\n`);
};

export const resetStats = () => {
  totalTokens = { input: 0, output: 0, requests: 0 };
};
