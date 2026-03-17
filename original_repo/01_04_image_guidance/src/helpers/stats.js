/**
 * Token usage statistics tracker.
 */

let totalTokens = { input: 0, output: 0, requests: 0 };
let geminiCalls = { generations: 0, edits: 0, analyses: 0 };

export const recordUsage = (usage) => {
  if (!usage) return;
  totalTokens.input += usage.input_tokens || 0;
  totalTokens.output += usage.output_tokens || 0;
  totalTokens.requests += 1;
};

export const recordGemini = (type) => {
  if (type === "generate") geminiCalls.generations++;
  else if (type === "edit") geminiCalls.edits++;
  else if (type === "analyze") geminiCalls.analyses++;
};

export const getStats = () => ({ 
  openai: { ...totalTokens },
  gemini: { ...geminiCalls }
});

export const logStats = () => {
  const { input, output, requests } = totalTokens;
  const { generations, edits, analyses } = geminiCalls;
  console.log(`\n📊 OpenAI Stats: ${requests} requests, ${input} input tokens, ${output} output tokens`);
  console.log(`🎨 Gemini Stats: ${generations} generations, ${edits} edits, ${analyses} analyses\n`);
};

export const resetStats = () => {
  totalTokens = { input: 0, output: 0, requests: 0 };
  geminiCalls = { generations: 0, edits: 0, analyses: 0 };
};
