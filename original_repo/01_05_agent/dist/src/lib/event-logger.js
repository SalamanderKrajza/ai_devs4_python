/**
 * Event-driven lifecycle logging
 *
 * Subscribes to the agent event emitter and produces structured,
 * human-readable logs for every lifecycle transition.
 */
import { logger } from './logger.js';
const log = logger.child({ name: 'agent' });
function fmtTokens(u) {
    if (!u)
        return '';
    const cached = u.cachedTokens ? ` (${u.cachedTokens} cached)` : '';
    return `${u.inputTokens} in, ${u.outputTokens} out${cached}`;
}
const truncate = (s, max = 120) => s.length > max ? s.slice(0, max) + '…' : s;
export function subscribeEventLogger(events) {
    return events.onAny((event) => {
        const { ctx } = event;
        const base = {
            traceId: ctx.traceId,
            agentId: ctx.agentId,
            sessionId: ctx.sessionId,
            depth: ctx.depth,
        };
        switch (event.type) {
            case 'agent.started':
                log.info({ ...base, model: event.model, agentName: event.agentName }, `started — ${event.agentName ?? 'agent'} (${event.model})`);
                break;
            case 'agent.completed': {
                const secs = (event.durationMs / 1000).toFixed(1);
                log.info({ ...base, durationMs: event.durationMs, usage: event.usage }, `completed — ${secs}s, ${fmtTokens(event.usage)}`);
                break;
            }
            case 'agent.failed':
                log.error({ ...base, error: event.error }, `failed — ${event.error}`);
                break;
            case 'agent.cancelled':
                log.warn(base, 'cancelled');
                break;
            case 'agent.waiting':
                log.info({ ...base, waitingFor: event.waitingFor }, `waiting for ${event.waitingFor.length} tool(s)`);
                break;
            case 'agent.resumed':
                log.info({ ...base, deliveredCallId: event.deliveredCallId, remaining: event.remaining }, `resumed — ${event.remaining} remaining`);
                break;
            case 'turn.started':
                log.info({ ...base, turn: event.turnCount }, `turn ${event.turnCount}`);
                break;
            case 'turn.completed': {
                const tokens = fmtTokens(event.usage);
                log.info({ ...base, turn: event.turnCount, usage: event.usage }, `turn ${event.turnCount} done${tokens ? ` — ${tokens}` : ''}`);
                break;
            }
            case 'generation.completed': {
                const tokens = fmtTokens(event.usage);
                const secs = (event.durationMs / 1000).toFixed(1);
                log.info({ ...base, model: event.model, durationMs: event.durationMs, usage: event.usage }, `generation ${event.model} — ${secs}s${tokens ? `, ${tokens}` : ''}`);
                break;
            }
            case 'tool.called':
                log.info({ ...base, callId: event.callId, tool: event.name }, `${event.name} called`);
                break;
            case 'tool.completed': {
                const secs = (event.durationMs / 1000).toFixed(1);
                log.info({ ...base, callId: event.callId, tool: event.name, durationMs: event.durationMs, output: truncate(event.output) }, `${event.name} ok — ${secs}s`);
                break;
            }
            case 'tool.failed':
                log.warn({ ...base, callId: event.callId, tool: event.name, error: event.error }, `${event.name} failed — ${event.error}`);
                break;
            default: {
                // batch.*, stream.* — log generically
                const { type, ctx: _ctx, ...data } = event;
                log.info({ ...base, ...data }, type);
            }
        }
    });
}
