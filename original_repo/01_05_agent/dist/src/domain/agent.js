export function createAgent(id, input) {
    return {
        id,
        sessionId: input.sessionId,
        traceId: input.traceId,
        rootAgentId: input.rootAgentId ?? id,
        parentId: input.parentId,
        sourceCallId: input.sourceCallId,
        depth: input.depth ?? 0,
        task: input.task,
        config: input.config,
        status: 'pending',
        waitingFor: [],
        turnCount: 0,
        createdAt: new Date(),
    };
}
export function startAgent(agent, traceId) {
    if (agent.status !== 'pending') {
        return { ok: false, error: `Cannot start agent in status: ${agent.status}` };
    }
    return {
        ok: true,
        agent: {
            ...agent,
            status: 'running',
            startedAt: new Date(),
            traceId: agent.traceId ?? traceId, // Set traceId if not already set
        },
    };
}
export function prepareAgentForNextTurn(agent) {
    if (agent.status === 'running') {
        return { ok: false, error: 'Session is already running.' };
    }
    if (agent.status === 'waiting') {
        return { ok: false, error: 'Session is waiting for a tool or human response.' };
    }
    if (agent.status === 'pending') {
        return { ok: true, agent };
    }
    return {
        ok: true,
        agent: {
            ...agent,
            status: 'pending',
            waitingFor: [],
            result: undefined,
            error: undefined,
            startedAt: undefined,
            completedAt: undefined,
        },
    };
}
export function waitForMany(agent, waiting) {
    if (agent.status !== 'running') {
        return { ok: false, error: `Cannot wait agent in status: ${agent.status}` };
    }
    return {
        ok: true,
        agent: { ...agent, status: 'waiting', waitingFor: waiting },
    };
}
export function deliverOne(agent, callId) {
    if (agent.status !== 'waiting') {
        return { ok: false, error: `Cannot deliver to agent in status: ${agent.status}` };
    }
    const remaining = agent.waitingFor.filter(w => w.callId !== callId);
    if (remaining.length === agent.waitingFor.length) {
        return { ok: false, error: `Agent not waiting for callId: ${callId}` };
    }
    const newStatus = remaining.length === 0 ? 'running' : 'waiting';
    return {
        ok: true,
        agent: { ...agent, status: newStatus, waitingFor: remaining },
    };
}
export function isWaitComplete(agent) {
    return agent.status === 'waiting' && agent.waitingFor.length === 0;
}
export function completeAgent(agent, result) {
    if (agent.status !== 'running') {
        return { ok: false, error: `Cannot complete agent in status: ${agent.status}` };
    }
    return {
        ok: true,
        agent: { ...agent, status: 'completed', result, completedAt: new Date() },
    };
}
export function failAgent(agent, error) {
    return { ...agent, status: 'failed', error, completedAt: new Date() };
}
export function cancelAgent(agent) {
    return { ...agent, status: 'cancelled', completedAt: new Date() };
}
export function incrementTurn(agent) {
    return { ...agent, turnCount: agent.turnCount + 1 };
}
export function addUsage(agent, usage) {
    const current = agent.usage ?? { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
    return {
        ...agent,
        usage: {
            inputTokens: current.inputTokens + usage.inputTokens,
            outputTokens: current.outputTokens + usage.outputTokens,
            totalTokens: current.totalTokens + usage.totalTokens,
            cachedTokens: (current.cachedTokens ?? 0) + (usage.cachedTokens ?? 0),
        },
    };
}
