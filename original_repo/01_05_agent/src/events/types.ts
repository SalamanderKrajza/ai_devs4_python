/**
 * Event types with rich context for observability
 *
 * Every event carries an EventContext with trace correlation ids.
 * Payloads are self-contained — subscribers never need to reach
 * into the runner for additional data.
 */
import type { AgentId, SessionId, TraceId, BatchId, CallId, WaitingFor, TokenUsage } from '../domain/index.js'

// Re-export TokenUsage for convenience
export type { TokenUsage } from '../domain/index.js'

// Context carried by all events
export interface EventContext {
  traceId: TraceId
  timestamp: number
  sessionId: SessionId
  agentId: AgentId
  rootAgentId: AgentId
  parentAgentId?: AgentId
  depth: number
  batchId?: BatchId
}

// All events
export type AgentEvent =
  // Agent lifecycle
  | { type: 'agent.started'; ctx: EventContext; model: string; task: string; agentName?: string; userId?: string; userInput?: unknown }
  | { type: 'agent.waiting'; ctx: EventContext; waitingFor: WaitingFor[] }
  | { type: 'agent.resumed'; ctx: EventContext; deliveredCallId: CallId; remaining: number }
  | { type: 'agent.completed'; ctx: EventContext; durationMs: number; usage?: TokenUsage; result?: string }
  | { type: 'agent.failed'; ctx: EventContext; error: string }
  | { type: 'agent.cancelled'; ctx: EventContext }
  
  // Turn lifecycle
  | { type: 'turn.started'; ctx: EventContext; turnCount: number }
  | { type: 'turn.completed'; ctx: EventContext; turnCount: number; usage?: TokenUsage }
  
  // Generation (LLM call) — self-contained with full input/output
  | { type: 'generation.completed'; ctx: EventContext; model: string; instructions: string; input: unknown[]; output: unknown; usage?: TokenUsage; durationMs: number; startTime: number }

  // Tool execution — self-contained with arguments and result
  | { type: 'tool.called'; ctx: EventContext; callId: CallId; name: string; arguments: Record<string, unknown> }
  | { type: 'tool.completed'; ctx: EventContext; callId: CallId; name: string; arguments: Record<string, unknown>; output: string; durationMs: number; startTime: number }
  | { type: 'tool.failed'; ctx: EventContext; callId: CallId; name: string; arguments: Record<string, unknown>; error: string; durationMs: number; startTime: number }
  
  // Batch (parallel operations)
  | { type: 'batch.started'; ctx: EventContext; batchId: BatchId; callIds: CallId[] }
  | { type: 'batch.progress'; ctx: EventContext; batchId: BatchId; completed: number; total: number }
  | { type: 'batch.completed'; ctx: EventContext; batchId: BatchId; durationMs: number }
  
  // Streaming
  | { type: 'stream.delta'; ctx: EventContext; delta: string }
  | { type: 'stream.done'; ctx: EventContext }

export type AgentEventType = AgentEvent['type']

// Helper to create context
export function createEventContext(
  traceId: TraceId,
  sessionId: SessionId,
  agentId: AgentId,
  rootAgentId: AgentId,
  depth: number,
  parentAgentId?: AgentId,
  batchId?: BatchId
): EventContext {
  return {
    traceId,
    timestamp: Date.now(),
    sessionId,
    agentId,
    rootAgentId,
    parentAgentId,
    depth,
    batchId,
  }
}
