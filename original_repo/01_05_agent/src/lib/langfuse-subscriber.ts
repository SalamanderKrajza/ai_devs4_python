/**
 * Langfuse event subscriber
 *
 * Listens to agent events and creates Langfuse observations.
 * Stateful only for agent observations (need to track refs for
 * nesting and end-of-life). Generations and tools are fire-and-forget
 * since their events are self-contained.
 */
import type { AgentEventEmitter } from '../events/index.js'
import {
  isTracingEnabled,
  traceAgent,
  traceGeneration,
  traceTool,
  getSpanContext,
  type SpanContext,
} from './tracing.js'
import { logger } from './logger.js'

const log = logger.child({ name: 'langfuse' })

type AgentObs = NonNullable<ReturnType<typeof traceAgent>>

export function subscribeLangfuse(events: AgentEventEmitter): () => void {
  if (!isTracingEnabled()) return () => {}

  // Agent observation refs — needed for nesting and lifecycle management
  const agentObsMap = new Map<string, AgentObs>()
  const agentSpanCtxMap = new Map<string, SpanContext>()

  const unsubs: Array<() => void> = []

  // ── Agent lifecycle ──────────────────────────────────────────────────────

  unsubs.push(events.on('agent.started', (event) => {
    const parentCtx = event.ctx.parentAgentId
      ? agentSpanCtxMap.get(event.ctx.parentAgentId)
      : undefined

    const obs = traceAgent(event.agentName ?? 'agent', {
      input: event.task,
      metadata: {
        agentId: event.ctx.agentId,
        model: event.model,
        depth: event.ctx.depth,
        sessionId: event.ctx.sessionId,
      },
      parentSpanContext: parentCtx,
      startTime: new Date(event.ctx.timestamp),
    })

    if (!obs) return

    agentObsMap.set(event.ctx.agentId, obs)
    const spanCtx = getSpanContext(obs)
    if (spanCtx) agentSpanCtxMap.set(event.ctx.agentId, spanCtx)

    // Set trace-level attributes for the root agent
    if (event.ctx.depth === 0) {
      obs.updateTrace({
        name: event.agentName ?? 'agent',
        sessionId: event.ctx.sessionId,
        userId: event.userId,
        input: event.userInput,
      })
    }
  }))

  unsubs.push(events.on('agent.completed', (event) => {
    const obs = agentObsMap.get(event.ctx.agentId)
    if (!obs) return

    obs.update({ output: event.result ?? 'Completed' })

    if (event.ctx.depth === 0) {
      obs.updateTrace({ output: event.result })
    }

    obs.end(new Date(event.ctx.timestamp))
    agentObsMap.delete(event.ctx.agentId)
    agentSpanCtxMap.delete(event.ctx.agentId)
  }))

  unsubs.push(events.on('agent.failed', (event) => {
    const obs = agentObsMap.get(event.ctx.agentId)
    if (!obs) return

    obs.update({ level: 'ERROR', statusMessage: event.error })
    obs.end(new Date(event.ctx.timestamp))
    agentObsMap.delete(event.ctx.agentId)
    agentSpanCtxMap.delete(event.ctx.agentId)
  }))

  unsubs.push(events.on('agent.cancelled', (event) => {
    const obs = agentObsMap.get(event.ctx.agentId)
    if (!obs) return

    obs.update({ output: 'Cancelled' })
    obs.end(new Date(event.ctx.timestamp))
    agentObsMap.delete(event.ctx.agentId)
    agentSpanCtxMap.delete(event.ctx.agentId)
  }))

  unsubs.push(events.on('agent.waiting', (event) => {
    const obs = agentObsMap.get(event.ctx.agentId)
    if (!obs) return

    const waitNames = event.waitingFor.map(w => w.name).join(', ')
    obs.update({ output: `Waiting for: ${waitNames}` })
    obs.end(new Date(event.ctx.timestamp))
    agentObsMap.delete(event.ctx.agentId)
    agentSpanCtxMap.delete(event.ctx.agentId)
  }))

  // ── Generation (LLM call) ───────────────────────────────────────────────

  unsubs.push(events.on('generation.completed', (event) => {
    const parentCtx = agentSpanCtxMap.get(event.ctx.agentId)

    const obs = traceGeneration(event.model, {
      model: event.model,
      input: event.input,
      parentSpanContext: parentCtx,
      startTime: new Date(event.startTime),
    })

    if (!obs) return

    obs.update({
      output: event.output,
      usageDetails: event.usage ? {
        input: event.usage.inputTokens,
        output: event.usage.outputTokens,
        total: event.usage.totalTokens,
      } : undefined,
    })
    obs.end(new Date(event.startTime + event.durationMs))
  }))

  // ── Tool execution ──────────────────────────────────────────────────────

  unsubs.push(events.on('tool.completed', (event) => {
    const parentCtx = agentSpanCtxMap.get(event.ctx.agentId)

    const obs = traceTool(event.name, {
      input: event.arguments,
      parentSpanContext: parentCtx,
      startTime: new Date(event.startTime),
    })

    if (!obs) return

    obs.update({ output: { result: event.output } })
    obs.end(new Date(event.startTime + event.durationMs))
  }))

  unsubs.push(events.on('tool.failed', (event) => {
    const parentCtx = agentSpanCtxMap.get(event.ctx.agentId)

    const obs = traceTool(event.name, {
      input: event.arguments,
      parentSpanContext: parentCtx,
      startTime: new Date(event.startTime),
    })

    if (!obs) return

    obs.update({ level: 'ERROR', statusMessage: event.error })
    obs.end(new Date(event.startTime + event.durationMs))
  }))

  log.info('langfuse subscriber attached')

  return () => {
    for (const unsub of unsubs) unsub()
    agentObsMap.clear()
    agentSpanCtxMap.clear()
  }
}
