/**
 * Item — polymorphic conversation entries per agent
 */
import type { ItemId, AgentId, CallId, MessageRole, Content } from './types.js'

interface BaseItem {
  id: ItemId
  agentId: AgentId
  sequence: number
  turnNumber?: number  // Which turn created this item
  createdAt: Date
}

export interface MessageItem extends BaseItem {
  type: 'message'
  role: MessageRole
  content: Content
}

export interface FunctionCallItem extends BaseItem {
  type: 'function_call'
  callId: CallId
  name: string
  arguments: Record<string, unknown>
}

export interface FunctionCallOutputItem extends BaseItem {
  type: 'function_call_output'
  callId: CallId
  output: string
  isError?: boolean
}

export interface ReasoningItem extends BaseItem {
  type: 'reasoning'
  summary?: string
  signature?: string
}

export type Item =
  | MessageItem
  | FunctionCallItem
  | FunctionCallOutputItem
  | ReasoningItem

// Input types for creating items (without base fields)
export type ItemInput =
  | { type: 'message'; role: MessageRole; content: Content; turnNumber?: number }
  | { type: 'function_call'; callId: CallId; name: string; arguments: Record<string, unknown>; turnNumber?: number }
  | { type: 'function_call_output'; callId: CallId; output: string; isError?: boolean; turnNumber?: number }
  | { type: 'reasoning'; summary?: string; signature?: string; turnNumber?: number }

// Factory function
export function createItem(
  id: ItemId,
  agentId: AgentId,
  sequence: number,
  input: ItemInput
): Item {
  const base = { id, agentId, sequence, createdAt: new Date() }
  return { ...base, ...input } as Item
}

// Type guards
export const isMessage = (item: Item): item is MessageItem => item.type === 'message'
export const isFunctionCall = (item: Item): item is FunctionCallItem => item.type === 'function_call'
export const isFunctionCallOutput = (item: Item): item is FunctionCallOutputItem => item.type === 'function_call_output'
export const isReasoning = (item: Item): item is ReasoningItem => item.type === 'reasoning'
