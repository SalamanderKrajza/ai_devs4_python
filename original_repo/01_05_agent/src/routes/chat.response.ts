import type { Agent, Item, WaitingFor } from '../domain/index.js'

export interface ChatResponse {
  id: string
  sessionId: string
  status: 'completed' | 'waiting' | 'failed'
  model: string
  output: OutputItem[]
  waitingFor?: WaitingFor[]
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number }
}

export type OutputItem =
  | { type: 'text'; text: string }
  | { type: 'function_call'; callId: string; name: string; arguments: Record<string, unknown> }

export function toChatResponse(
  agent: Agent,
  items: Item[],
  status: ChatResponse['status'],
  waitingFor?: WaitingFor[],
): ChatResponse {
  const output: OutputItem[] = []

  for (const item of items) {
    if (item.type === 'message' && item.role === 'assistant') {
      const text = typeof item.content === 'string'
        ? item.content
        : item.content
            .filter(part => part.type === 'text')
            .map(part => part.type === 'text' ? part.text : '')
            .join('')
      if (text) output.push({ type: 'text', text })
      continue
    }

    if (item.type === 'function_call') {
      output.push({
        type: 'function_call',
        callId: item.callId,
        name: item.name,
        arguments: item.arguments,
      })
    }
  }

  return {
    id: agent.id,
    sessionId: agent.sessionId,
    status,
    model: agent.config.model,
    output,
    waitingFor,
  }
}

export function filterResponseItems(items: Item[], responseStartSequence: number): Item[] {
  return items.filter(item => item.sequence > responseStartSequence)
}
