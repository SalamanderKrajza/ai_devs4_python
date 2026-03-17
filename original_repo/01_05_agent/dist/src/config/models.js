/**
 * Model registry — context windows, output limits, and pruning thresholds.
 */
const DEFAULT_PRUNING = {
    threshold: 0.85,
    targetUtilization: 0.5,
    minRecentTurns: 3,
    maxToolOutputChars: 10_000,
    enableSummarization: true,
};
const MODELS = {
    'gpt-5.4': {
        id: 'gpt-5.4',
        provider: 'openai',
        contextWindow: 400_000,
        maxOutputTokens: 128_000,
        pruning: {
            ...DEFAULT_PRUNING,
            minRecentTurns: 5,
        },
    },
    'gpt-5.2': {
        id: 'gpt-5.2',
        provider: 'openai',
        contextWindow: 400_000,
        maxOutputTokens: 128_000,
        pruning: {
            ...DEFAULT_PRUNING,
            minRecentTurns: 5,
        },
    },
    'gemini-3-pro-preview': {
        id: 'gemini-3-pro-preview',
        provider: 'gemini',
        contextWindow: 1_048_576,
        maxOutputTokens: 65_536,
        pruning: {
            ...DEFAULT_PRUNING,
            threshold: 0.90,
            targetUtilization: 0.6,
            minRecentTurns: 10,
        },
    },
    'gemini-3-flash-preview': {
        id: 'gemini-3-flash-preview',
        provider: 'gemini',
        contextWindow: 1_048_576,
        maxOutputTokens: 65_536,
        pruning: {
            ...DEFAULT_PRUNING,
            threshold: 0.90,
            targetUtilization: 0.6,
            minRecentTurns: 10,
        },
    },
};
/** Resolve model definition. Falls back to conservative defaults for unknown models. */
export function getModelDefinition(modelId) {
    return MODELS[modelId] ?? {
        id: modelId,
        provider: 'openai',
        contextWindow: 128_000,
        maxOutputTokens: 16_000,
        pruning: DEFAULT_PRUNING,
    };
}
