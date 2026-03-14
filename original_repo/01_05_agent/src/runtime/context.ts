/**
 * Runtime context — dependencies for agent execution
 */
import type { AgentEventEmitter } from '../events/index.js'
import type { Repositories } from '../repositories/index.js'
import type { ToolRegistry } from '../tools/index.js'
import type { McpManager } from '../mcp/index.js'
import type { TraceId, AgentId } from '../domain/index.js'

export interface RuntimeContext {
  events: AgentEventEmitter
  repositories: Repositories
  tools: ToolRegistry
  mcp: McpManager
}

// Per-request execution context (passed through agent hierarchy)
export interface ExecutionContext {
  traceId: TraceId
  rootAgentId: AgentId
  parentAgentId?: AgentId
  depth: number
  /** Propagated to events for trace-level attribution */
  userId?: string
  /** Propagated to events for trace-level input */
  userInput?: unknown
  /** Human-readable agent name (e.g. "alice") */
  agentName?: string
}

export function createContext(
  events: AgentEventEmitter,
  repositories: Repositories,
  tools: ToolRegistry,
  mcp: McpManager,
): RuntimeContext {
  return { events, repositories, tools, mcp }
}

export function createExecutionContext(
  traceId: TraceId,
  rootAgentId: AgentId,
  parentAgentId?: AgentId,
  depth: number = 0,
): ExecutionContext {
  return { traceId, rootAgentId, parentAgentId, depth }
}
