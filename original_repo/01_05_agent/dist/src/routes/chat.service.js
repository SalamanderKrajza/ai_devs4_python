import { runAgent, runAgentStream, deliverResult as deliverAgentResult } from '../runtime/index.js';
import { filterResponseItems, toChatResponse } from './chat.response.js';
import { getAgentLastSequence, setupChatTurn } from './chat.turn.js';
function createExecution(agent, req, userId, traceId) {
    return {
        traceId,
        rootAgentId: agent.rootAgentId,
        depth: agent.depth,
        userId,
        userInput: req.input,
        agentName: req.agent,
    };
}
async function loadVisibleItems(result, ctx, agentId, responseStartSequence) {
    const items = result.status === 'completed'
        ? result.items
        : await ctx.repositories.items.listByAgent(agentId);
    return filterResponseItems(items, responseStartSequence);
}
export async function prepareChat(req, ctx, userId) {
    return setupChatTurn(req, ctx, userId);
}
export async function executePreparedChat(prepared, req, ctx, userId) {
    const { agent, traceId, responseStartSequence } = prepared;
    const result = await runAgent(agent.id, ctx, {
        maxTurns: 10,
        execution: createExecution(agent, req, userId, traceId),
    });
    if (!result.ok) {
        return { ok: false, error: result.status === 'cancelled' ? 'Cancelled' : result.error };
    }
    const visibleItems = await loadVisibleItems(result, ctx, agent.id, responseStartSequence);
    return {
        ok: true,
        response: toChatResponse(result.agent, visibleItems, result.status, result.status === 'waiting' ? result.waitingFor : undefined),
    };
}
export async function processChat(req, ctx, userId) {
    const setup = await prepareChat(req, ctx, userId);
    if (!setup.ok) {
        return { ok: false, error: setup.error };
    }
    return executePreparedChat(setup.data, req, ctx, userId);
}
export async function deliverResult(agentId, callId, result, ctx) {
    const responseStartSequence = await getAgentLastSequence(agentId, ctx);
    const runResult = await deliverAgentResult(agentId, callId, result, ctx);
    if (!runResult.ok) {
        return { ok: false, error: runResult.status === 'cancelled' ? 'Cancelled' : runResult.error };
    }
    const visibleItems = await loadVisibleItems(runResult, ctx, agentId, responseStartSequence);
    return {
        ok: true,
        response: toChatResponse(runResult.agent, visibleItems, runResult.status, runResult.status === 'waiting' ? runResult.waitingFor : undefined),
    };
}
export async function* streamPreparedChat(prepared, req, ctx, userId) {
    const { agent, traceId } = prepared;
    yield* runAgentStream(agent.id, ctx, {
        maxTurns: 10,
        execution: createExecution(agent, req, userId, traceId),
    });
}
export async function* processChatStream(req, ctx, userId) {
    const setup = await prepareChat(req, ctx, userId);
    if (!setup.ok) {
        yield { type: 'error', error: setup.error };
        return;
    }
    yield* streamPreparedChat(setup.data, req, ctx, userId);
}
