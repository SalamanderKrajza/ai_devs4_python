/**
 * Provider abstraction for LLM APIs
 */
import type { Content, ToolDefinition } from '../domain/index.js'

export interface ProviderRequest {
  model: string
  instructions?: string
  input: ProviderInputItem[]
  tools?: ToolDefinition[]
  stream?: boolean
  temperature?: number
  maxTokens?: number
  signal?: AbortSignal
}

export type ProviderInputItem =
  | { type: 'message'; role: 'user' | 'assistant' | 'system'; content: Content }
  | { type: 'function_call'; callId: string; name: string; arguments: Record<string, unknown> }
  | { type: 'function_result'; callId: string; name: string; output: string }
  | { type: 'reasoning'; text: string; signature?: string; provider?: string }

export interface ProviderResponse {
  id: string
  model: string
  output: ProviderOutputItem[]
  usage?: ProviderUsage
  finishReason?: FinishReason
}

export interface ProviderUsage {
  inputTokens: number
  outputTokens: number
  totalTokens: number
  cachedTokens?: number
}

export type FinishReason = 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'error'

export type ProviderOutputItem =
  | { type: 'text'; text: string }
  | { type: 'function_call'; callId: string; name: string; arguments: Record<string, unknown> }
  | { type: 'reasoning'; text: string; signature?: string; provider?: string }

// Stream events - unified across providers
export type ProviderStreamEvent =
  // Text content streaming
  | { type: 'text_delta'; delta: string }
  | { type: 'text_done'; text: string }
  // Function call streaming (OpenAI streams args, Gemini sends complete)
  | { type: 'function_call_delta'; callId: string; name: string; argumentsDelta: string }
  | { type: 'function_call_done'; callId: string; name: string; arguments: Record<string, unknown> }
  // Reasoning/thinking (OpenAI o-series, Gemini thinking)
  | { type: 'reasoning_delta'; delta: string }
  | { type: 'reasoning_done'; text: string }
  // Lifecycle
  | { type: 'done'; response: ProviderResponse }
  | { type: 'error'; error: string; code?: string }

export interface Provider {
  name: string
  generate(request: ProviderRequest): Promise<ProviderResponse>
  stream(request: ProviderRequest): AsyncIterable<ProviderStreamEvent>
}

/** Check if error is from abort */
export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError'
}

/** Throw if signal is aborted */
export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    const err = new Error('Operation aborted')
    err.name = 'AbortError'
    throw err
  }
}
