/**
 * Token estimation — character-based heuristic.
 *
 * ~4 chars per token for English text is a reasonable approximation.
 * We use a conservative multiplier (3.5) so estimates run slightly high,
 * giving us a safety margin before hitting actual limits.
 */
const CHARS_PER_TOKEN = 3.5;
export function estimateTokens(text) {
    return Math.ceil(text.length / CHARS_PER_TOKEN);
}
function itemTextLength(item) {
    switch (item.type) {
        case 'message': {
            if (typeof item.content === 'string')
                return item.content.length;
            return item.content
                .filter(p => p.type === 'text')
                .reduce((sum, p) => sum + (p.type === 'text' ? p.text.length : 0), 0);
        }
        case 'function_call':
            return item.name.length + JSON.stringify(item.arguments).length;
        case 'function_call_output':
            return item.output.length;
        case 'reasoning':
            return (item.summary?.length ?? 0);
        default:
            return 0;
    }
}
export function estimateItemTokens(item) {
    return Math.ceil(itemTextLength(item) / CHARS_PER_TOKEN);
}
/** Estimate total tokens for a list of items + optional system prompt. */
export function estimateConversationTokens(items, systemPrompt) {
    let chars = systemPrompt?.length ?? 0;
    for (const item of items) {
        chars += itemTextLength(item);
        chars += 20; // overhead per item (role tags, separators, etc.)
    }
    return Math.ceil(chars / CHARS_PER_TOKEN);
}
