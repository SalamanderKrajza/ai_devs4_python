/**
 * Agent — runtime instance with hierarchy and waiting support
 */
import type { AgentId, SessionId, CallId, TraceId, AgentStatus, WaitingFor, TokenUsage } from './types.js'

export interface AgentConfig {
  model: string
  temperature?: number
  maxTokens?: number
  tools?: ToolDefinition[]
}

export type ToolDefinition = FunctionTool | WebSearchTool

export interface FunctionTool {
  type: 'function'
  name: string
  description: string
  parameters: Record<string, unknown>
}

export interface WebSearchTool {
  type: 'web_search'
}

export interface Agent {
  id: AgentId
  sessionId: SessionId
  traceId?: TraceId  // Set on first run, preserved across resumes
  
  // Hierarchy
  rootAgentId: AgentId
  parentId?: AgentId
  sourceCallId?: CallId
  depth: number

  task: string
  config: AgentConfig

  status: AgentStatus
  waitingFor: WaitingFor[]
  result?: unknown
  error?: string
  turnCount: number
  
  // Cumulative usage
  usage?: TokenUsage

  createdAt: Date
  startedAt?: Date
  completedAt?: Date
}

export interface CreateAgentInput {
  sessionId: SessionId
  traceId?: TraceId
  rootAgentId?: AgentId
  parentId?: AgentId
  sourceCallId?: CallId
  depth?: number
  task: string
  config: AgentConfig
}

export function createAgent(id: AgentId, input: CreateAgentInput): Agent {
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
  }
}

// State transitions
export type TransitionResult = { ok: true; agent: Agent } | { ok: false; error: string }

export function startAgent(agent: Agent, traceId?: TraceId): TransitionResult {
  if (agent.status !== 'pending') {
    return { ok: false, error: `Cannot start agent in status: ${agent.status}` }
  }
  return {
    ok: true,
    agent: { 
      ...agent, 
      status: 'running', 
      startedAt: new Date(),
      traceId: agent.traceId ?? traceId,  // Set traceId if not already set
    },
  }
}

export function waitForMany(agent: Agent, waiting: WaitingFor[]): TransitionResult {
  if (agent.status !== 'running') {
    return { ok: false, error: `Cannot wait agent in status: ${agent.status}` }
  }
  return {
    ok: true,
    agent: { ...agent, status: 'waiting', waitingFor: waiting },
  }
}

export function deliverOne(agent: Agent, callId: CallId): TransitionResult {
  if (agent.status !== 'waiting') {
    return { ok: false, error: `Cannot deliver to agent in status: ${agent.status}` }
  }
  
  const remaining = agent.waitingFor.filter(w => w.callId !== callId)
  
  if (remaining.length === agent.waitingFor.length) {
    return { ok: false, error: `Agent not waiting for callId: ${callId}` }
  }
  
  const newStatus = remaining.length === 0 ? 'running' : 'waiting'
  
  return {
    ok: true,
    agent: { ...agent, status: newStatus, waitingFor: remaining },
  }
}

export function isWaitComplete(agent: Agent): boolean {
  return agent.status === 'waiting' && agent.waitingFor.length === 0
}

export function completeAgent(agent: Agent, result: unknown): TransitionResult {
  if (agent.status !== 'running') {
    return { ok: false, error: `Cannot complete agent in status: ${agent.status}` }
  }
  return {
    ok: true,
    agent: { ...agent, status: 'completed', result, completedAt: new Date() },
  }
}

export function failAgent(agent: Agent, error: string): Agent {
  return { ...agent, status: 'failed', error, completedAt: new Date() }
}

export function cancelAgent(agent: Agent): Agent {
  return { ...agent, status: 'cancelled', completedAt: new Date() }
}

export function incrementTurn(agent: Agent): Agent {
  return { ...agent, turnCount: agent.turnCount + 1 }
}

export function addUsage(agent: Agent, usage: TokenUsage): Agent {
  const current = agent.usage ?? { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
  return {
    ...agent,
    usage: {
      inputTokens: current.inputTokens + usage.inputTokens,
      outputTokens: current.outputTokens + usage.outputTokens,
      totalTokens: current.totalTokens + usage.totalTokens,
      cachedTokens: (current.cachedTokens ?? 0) + (usage.cachedTokens ?? 0),
    },
  }
}
