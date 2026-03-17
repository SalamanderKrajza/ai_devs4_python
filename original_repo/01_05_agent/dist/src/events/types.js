// Helper to create context
export function createEventContext(traceId, sessionId, agentId, rootAgentId, depth, parentAgentId, batchId) {
    return {
        traceId,
        timestamp: Date.now(),
        sessionId,
        agentId,
        rootAgentId,
        parentAgentId,
        depth,
        batchId,
    };
}
