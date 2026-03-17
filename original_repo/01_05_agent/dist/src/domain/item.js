// Factory function
export function createItem(id, agentId, sequence, input) {
    const base = { id, agentId, sequence, createdAt: new Date() };
    return { ...base, ...input };
}
// Type guards
export const isMessage = (item) => item.type === 'message';
export const isFunctionCall = (item) => item.type === 'function_call';
export const isFunctionCallOutput = (item) => item.type === 'function_call_output';
export const isReasoning = (item) => item.type === 'reasoning';
