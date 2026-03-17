/**
 * LLM-based summarization — compress dropped conversation items into a checkpoint.
 */

import type { Item } from '../domain/index.js'
import type { Provider } from '../providers/types.js'

const INITIAL_PROMPT = `You are a conversation summarizer. Create a structured checkpoint of the conversation so far.

Format your summary as:

## Goal
What the user is trying to accomplish.

## Progress
- Done: what has been completed
- In Progress: what is currently being worked on
- Blocked: any blockers or issues

## Key Decisions
Important decisions made during the conversation.

## Critical Context
Any facts, file paths, variable names, or technical details that must be preserved.

Be concise but preserve ALL technical details (paths, names, values, errors). Never drop specifics.`

const UPDATE_PROMPT = `You are a conversation summarizer. Update the existing summary with new information.

Rules:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new conversation
- UPDATE the Progress section (move completed items to Done)
- Keep it structured and concise
- Never drop technical specifics (paths, names, values, errors)`

export function serializeItems(items: Item[]): string {
  const lines: string[] = []

  for (const item of items) {
    switch (item.type) {
      case 'message':
        if (typeof item.content === 'string') {
          lines.push(`[${item.role}]: ${item.content}`)
        } else {
          const text = item.content
            .filter(p => p.type === 'text')
            .map(p => p.type === 'text' ? p.text : '')
            .join('\n')
          if (text) lines.push(`[${item.role}]: ${text}`)
        }
        break
      case 'function_call':
        lines.push(`[tool call]: ${item.name}(${JSON.stringify(item.arguments).slice(0, 500)})`)
        break
      case 'function_call_output': {
        const preview = item.output.length > 500
          ? `${item.output.slice(0, 500)}... [truncated]`
          : item.output
        lines.push(`[tool result]: ${preview}`)
        break
      }
      case 'reasoning':
        if (item.summary) lines.push(`[thinking]: ${item.summary}`)
        break
    }
  }

  return lines.join('\n')
}

export async function generateSummary(
  provider: Provider,
  model: string,
  droppedItems: Item[],
  previousSummary?: string,
): Promise<string> {
  const serialized = serializeItems(droppedItems)

  const isUpdate = !!previousSummary
  const instructions = isUpdate ? UPDATE_PROMPT : INITIAL_PROMPT

  const input = isUpdate
    ? `Previous summary:\n${previousSummary}\n\nNew conversation to incorporate:\n${serialized}`
    : serialized

  const response = await provider.generate({
    model,
    instructions,
    input: [{ type: 'message', role: 'user', content: input }],
    temperature: 0.3,
    maxTokens: 2000,
  })

  return response.output
    .filter((o): o is { type: 'text'; text: string } => o.type === 'text')
    .map(o => o.text)
    .join('\n') || ''
}
