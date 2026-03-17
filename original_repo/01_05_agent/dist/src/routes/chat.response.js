export function toChatResponse(agent, items, status, waitingFor) {
    const output = [];
    for (const item of items) {
        if (item.type === 'message' && item.role === 'assistant') {
            const text = typeof item.content === 'string'
                ? item.content
                : item.content
                    .filter(part => part.type === 'text')
                    .map(part => part.type === 'text' ? part.text : '')
                    .join('');
            if (text)
                output.push({ type: 'text', text });
            continue;
        }
        if (item.type === 'function_call') {
            output.push({
                type: 'function_call',
                callId: item.callId,
                name: item.name,
                arguments: item.arguments,
            });
        }
    }
    return {
        id: agent.id,
        sessionId: agent.sessionId,
        status,
        model: agent.config.model,
        output,
        waitingFor,
    };
}
export function filterResponseItems(items, responseStartSequence) {
    return items.filter(item => item.sequence > responseStartSequence);
}
