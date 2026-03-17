/**
 * Core domain types
 */

// IDs
export type UserId = string
export type SessionId = string
export type AgentId = string
export type ItemId = string
export type CallId = string
export type TraceId = string
export type BatchId = string

// Statuses
export type SessionStatus = 'active' | 'archived'

export type AgentStatus =
  | 'pending'
  | 'running'
  | 'waiting'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type ItemType =
  | 'message'
  | 'function_call'
  | 'function_call_output'
  | 'reasoning'

export type MessageRole = 'user' | 'assistant' | 'system'

// Content
export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; data: string; mimeType: string }
  | { type: 'image'; uri: string; mimeType: string }

export type Content = string | ContentPart[]

// Waiting
export type WaitType = 'tool' | 'agent' | 'human'

export interface WaitingFor {
  callId: CallId
  type: WaitType
  name: string
  description?: string
}

// Usage tracking
export interface TokenUsage {
  inputTokens: number
  outputTokens: number
  totalTokens: number
  cachedTokens?: number
}
