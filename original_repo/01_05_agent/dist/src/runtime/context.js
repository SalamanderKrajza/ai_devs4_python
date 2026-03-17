export function createContext(events, repositories, tools, mcp) {
    return { events, repositories, tools, mcp };
}
export function createExecutionContext(traceId, rootAgentId, parentAgentId, depth = 0) {
    return { traceId, rootAgentId, parentAgentId, depth };
}
