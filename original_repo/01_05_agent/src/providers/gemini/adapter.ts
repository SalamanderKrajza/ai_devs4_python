/**
 * Gemini provider adapter (Interactions API)
 */
import { GoogleGenAI } from '@google/genai'
import type { Interactions } from '@google/genai'
import type { Provider, ProviderRequest, ProviderResponse, ProviderStreamEvent, ProviderOutputItem, ProviderInputItem } from '../types.js'
import { throwIfAborted } from '../types.js'

export interface GeminiConfig {
  apiKey: string
  defaultModel?: string
}

// Interactions API types (from SDK)
type Content = Interactions.Content
type Tool = Interactions.Tool
interface InteractionData {
  id: string
  status: string
  outputs?: Content[]
  usage?: { total_input_tokens?: number; total_output_tokens?: number; total_tokens?: number; cached_content_token_count?: number }
}

type InteractionSSEEvent = 
  | { event_type: 'interaction.start'; interaction?: InteractionData }
  | { event_type: 'content.start'; content?: Content; index?: number }
  | { event_type: 'content.delta'; delta?: { type: string; text?: string; id?: string; name?: string; arguments?: Record<string, unknown> }; index?: number }
  | { event_type: 'content.stop'; index?: number }
  | { event_type: 'interaction.complete'; interaction?: InteractionData }
  | { event_type: 'interaction.status_update'; interaction_id?: string; status?: string }
  | { event_type: 'error'; error?: { message?: string; code?: string } }

// ─────────────────────────────────────────────────────────────────────────────
// Input mapping (our types → Gemini Interactions)
// ─────────────────────────────────────────────────────────────────────────────

function toGeminiInput(items: ProviderInputItem[]): Content[] {
  const contents: Content[] = []

  for (const item of items) {
    switch (item.type) {
      case 'message': {
        if (typeof item.content === 'string') {
          contents.push({ type: 'text', text: item.content })
        } else {
          for (const part of item.content) {
            if (part.type === 'text') {
              contents.push({ type: 'text', text: part.text })
            }
            // Note: Images handled via 'image' content type if needed
          }
        }
        break
      }

      case 'function_call': {
        contents.push({
          type: 'function_call',
          id: item.callId,
          name: item.name,
          arguments: item.arguments,
        } as Content)
        break
      }

      case 'function_result': {
        contents.push({
          type: 'function_result',
          call_id: item.callId,
          name: item.name,
          result: item.output,
        } as Content)
        break
      }

      case 'reasoning': {
        // Only pass back thoughts with signatures that originated from Gemini
        // This enables multi-provider sessions - other providers' reasoning is skipped
        if (item.signature && item.provider === 'gemini') {
          contents.push({
            type: 'thought',
            signature: item.signature,
            // Optionally include summary if present
            ...(item.text && {
              summary: [{ type: 'text', text: item.text }],
            }),
          } as Content)
        }
        // Non-Gemini reasoning is silently skipped - this is intentional
        break
      }
    }
  }

  return contents
}

function toGeminiTools(tools?: ProviderRequest['tools']): Tool[] | undefined {
  if (!tools?.length) return undefined

  return tools.map((t): Tool => {
    if (t.type === 'web_search') {
      return { type: 'google_search' }
    }
    return {
      type: 'function' as const,
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    }
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Output mapping (Gemini Interactions → our types)
// ─────────────────────────────────────────────────────────────────────────────

function fromGeminiOutputs(outputs?: Content[]): ProviderOutputItem[] {
  if (!outputs) return []

  const result: ProviderOutputItem[] = []

  for (const content of outputs) {
    if (content.type === 'text' && 'text' in content) {
      const text = (content as { text?: string }).text
      if (text) result.push({ type: 'text', text })
    }

    if (content.type === 'function_call') {
      const fc = content as { type: 'function_call'; id: string; name: string; arguments: Record<string, unknown> }
      result.push({
        type: 'function_call',
        callId: fc.id ?? crypto.randomUUID(),
        name: fc.name,
        arguments: fc.arguments ?? {},
      })
    }

    if (content.type === 'thought') {
      const thought = content as { type: 'thought'; signature?: string; summary?: Array<{ type: string; text?: string }> }
      const summaryText = thought.summary
        ?.filter((s): s is { type: 'text'; text: string } => s.type === 'text' && !!s.text)
        .map(s => s.text)
        .join('\n')
      // Store signature for multi-turn validation, even if summary is empty
      if (summaryText || thought.signature) {
        result.push({
          type: 'reasoning',
          text: summaryText ?? '',
          signature: thought.signature,
          provider: 'gemini',
        })
      }
    }
  }

  return result
}

function getFinishReason(outputs?: Content[]): 'stop' | 'tool_calls' {
  if (outputs?.some(o => o.type === 'function_call')) return 'tool_calls'
  return 'stop'
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream state for accumulating content
// ─────────────────────────────────────────────────────────────────────────────

interface StreamState {
  contents: Map<number, { type: string; text: string; id?: string; name?: string; args: string }>
}

function createStreamState(): StreamState {
  return { contents: new Map() }
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider implementation
// ─────────────────────────────────────────────────────────────────────────────

export function createGeminiProvider(config: GeminiConfig): Provider {
  const client = new GoogleGenAI({ apiKey: config.apiKey })

  return {
    name: 'gemini',

    async generate(request: ProviderRequest): Promise<ProviderResponse> {
      throwIfAborted(request.signal)

      const interaction = await client.interactions.create({
        model: request.model,
        input: toGeminiInput(request.input),
        system_instruction: request.instructions,
        tools: toGeminiTools(request.tools),
        generation_config: {
          temperature: request.temperature,
          max_output_tokens: request.maxTokens,
        },
      })

      return {
        id: interaction.id,
        model: interaction.model ?? request.model,
        output: fromGeminiOutputs(interaction.outputs),
        finishReason: getFinishReason(interaction.outputs),
        usage: interaction.usage && {
          inputTokens: interaction.usage.total_input_tokens ?? 0,
          outputTokens: interaction.usage.total_output_tokens ?? 0,
          totalTokens: interaction.usage.total_tokens ?? 0,
          cachedTokens: (interaction.usage as Record<string, unknown>).cached_content_token_count as number ?? 0,
        },
      }
    },

    async *stream(request: ProviderRequest): AsyncIterable<ProviderStreamEvent> {
      throwIfAborted(request.signal)

      const stream = await client.interactions.create({
        model: request.model,
        input: toGeminiInput(request.input),
        system_instruction: request.instructions,
        tools: toGeminiTools(request.tools),
        generation_config: {
          temperature: request.temperature,
          max_output_tokens: request.maxTokens,
        },
        stream: true,
      })

      const state = createStreamState()
      let lastInteraction: InteractionData | undefined

      for await (const event of stream as AsyncIterable<InteractionSSEEvent>) {
        switch (event.event_type) {
          case 'interaction.start':
            if (event.interaction) lastInteraction = event.interaction
            break

          case 'content.start': {
            const idx = event.index ?? 0
            const content = event.content
            if (content) {
              const meta = content as { type: string; id?: string; name?: string }
              state.contents.set(idx, {
                type: content.type,
                text: '',
                id: meta.id,
                name: meta.name,
                args: '',
              })
            }
            break
          }

          case 'content.delta': {
            const idx = event.index ?? 0
            const delta = event.delta
            const current = state.contents.get(idx)

            if (delta?.type === 'text' && delta.text) {
              if (current) current.text += delta.text
              yield { type: 'text_delta', delta: delta.text }
            }

            if (delta?.type === 'function_call') {
              if (current) {
                if (delta.id) current.id = delta.id
                if (delta.name) current.name = delta.name
                if (delta.arguments) {
                  const argsStr = JSON.stringify(delta.arguments)
                  current.args = argsStr
                  yield {
                    type: 'function_call_delta',
                    callId: current.id ?? '',
                    name: current.name ?? '',
                    argumentsDelta: argsStr,
                  }
                }
              }
            }

            // Handle thought_summary deltas (reasoning content)
            if (delta?.type === 'thought_summary') {
              const thoughtDelta = delta as { type: 'thought_summary'; content?: { type: string; text?: string } }
              const text = thoughtDelta.content?.text
              if (text) {
                if (current) current.text += text
                yield { type: 'reasoning_delta', delta: text }
              }
            }
            break
          }

          case 'content.stop': {
            const idx = event.index ?? 0
            const content = state.contents.get(idx)

            if (content?.type === 'text' && content.text) {
              yield { type: 'text_done', text: content.text }
            }

            if (content?.type === 'function_call' && content.name) {
              yield {
                type: 'function_call_done',
                callId: content.id ?? crypto.randomUUID(),
                name: content.name,
                arguments: content.args ? JSON.parse(content.args) : {},
              }
            }

            if (content?.type === 'thought' && content.text) {
              yield { type: 'reasoning_done', text: content.text }
            }
            break
          }

          case 'interaction.complete':
            if (event.interaction) lastInteraction = event.interaction
            break

          case 'error':
            yield {
              type: 'error',
              error: event.error?.message ?? 'Unknown error',
              code: event.error?.code,
            }
            break
        }
      }

      // Final done event
      if (lastInteraction) {
        yield {
          type: 'done',
          response: {
            id: lastInteraction.id ?? crypto.randomUUID(),
            model: request.model,
            output: fromGeminiOutputs(lastInteraction.outputs),
            finishReason: getFinishReason(lastInteraction.outputs),
            usage: lastInteraction.usage && {
              inputTokens: lastInteraction.usage.total_input_tokens ?? 0,
              outputTokens: lastInteraction.usage.total_output_tokens ?? 0,
              totalTokens: lastInteraction.usage.total_tokens ?? 0,
              cachedTokens: lastInteraction.usage.cached_content_token_count ?? 0,
            },
          },
        }
      }
    },
  }
}
