/**
 * Context pruning — truncate large outputs, drop old turns, keep recent.
 *
 * Operates on domain Items in-memory. DB history stays intact (audit trail).
 */

import type { Item } from '../domain/index.js'
import { isFunctionCallOutput, isFunctionCall, isMessage } from '../domain/index.js'
import { estimateConversationTokens } from './tokens.js'
import type { PruningThresholds } from '../config/models.js'

export interface PruningResult {
  items: Item[]
  estimatedTokens: number
  droppedCount: number
  truncatedCount: number
  /** Items that were dropped (for summarization) */
  droppedItems: Item[]
}

// ─────────────────────────────────────────────────────────────────────────────
// Truncation
// ─────────────────────────────────────────────────────────────────────────────

export function truncateToolOutput(output: string, maxChars: number): string {
  if (output.length <= maxChars) return output
  const half = Math.floor(maxChars / 2)
  const dropped = output.length - maxChars
  return `${output.slice(0, half)}\n\n[... ${dropped} characters truncated ...]\n\n${output.slice(-half)}`
}

function truncateLargeOutputs(items: Item[], maxChars: number): { items: Item[]; truncatedCount: number } {
  let truncatedCount = 0

  const result = items.map(item => {
    if (!isFunctionCallOutput(item)) return item
    if (item.output.length <= maxChars) return item

    truncatedCount++
    return { ...item, output: truncateToolOutput(item.output, maxChars) }
  })

  return { items: result, truncatedCount }
}

// ─────────────────────────────────────────────────────────────────────────────
// Turn identification
// ─────────────────────────────────────────────────────────────────────────────

interface Turn {
  startIndex: number
  endIndex: number
  items: Item[]
}

/**
 * Group items into turns. A turn starts with a user message and includes
 * everything until the next user message (assistant reply, tool calls, outputs).
 */
function identifyTurns(items: Item[]): Turn[] {
  const turns: Turn[] = []
  let currentStart = 0

  for (let i = 0; i < items.length; i++) {
    const item = items[i]
    if (isMessage(item) && item.role === 'user' && i > 0) {
      turns.push({
        startIndex: currentStart,
        endIndex: i - 1,
        items: items.slice(currentStart, i),
      })
      currentStart = i
    }
  }

  // Last turn
  if (currentStart < items.length) {
    turns.push({
      startIndex: currentStart,
      endIndex: items.length - 1,
      items: items.slice(currentStart),
    })
  }

  return turns
}

// ─────────────────────────────────────────────────────────────────────────────
// Core pruning
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Prune conversation items to fit within token budget.
 *
 * Strategy:
 *  1. Truncate individual large tool outputs
 *  2. If still over budget, drop oldest complete turns (keep last N)
 *  3. Always preserve: first turn + last minRecentTurns
 */
export function pruneConversation(
  items: Item[],
  systemPrompt: string | undefined,
  contextWindow: number,
  config: PruningThresholds,
): PruningResult {
  const targetTokens = Math.floor(contextWindow * config.targetUtilization)

  // Step 1: truncate large outputs
  const { items: truncated, truncatedCount } = truncateLargeOutputs(items, config.maxToolOutputChars)

  let estimate = estimateConversationTokens(truncated, systemPrompt)
  if (estimate <= targetTokens) {
    return { items: truncated, estimatedTokens: estimate, droppedCount: 0, truncatedCount, droppedItems: [] }
  }

  // Step 2: drop old turns
  const turns = identifyTurns(truncated)

  if (turns.length <= config.minRecentTurns) {
    // Can't drop anything — already at minimum
    return { items: truncated, estimatedTokens: estimate, droppedCount: 0, truncatedCount, droppedItems: [] }
  }

  // Keep first turn (setup context) + last N turns
  const keepFirst = turns[0]
  const keepRecent = turns.slice(-config.minRecentTurns)
  const droppable = turns.slice(1, -config.minRecentTurns)

  // Drop oldest turns until under budget
  const droppedItems: Item[] = []
  let droppedCount = 0
  const surviving = [...droppable]

  while (surviving.length > 0) {
    const dropped = surviving.shift()!
    droppedItems.push(...dropped.items)
    droppedCount += dropped.items.length

    const remainingItems = [
      ...keepFirst.items,
      ...surviving.flatMap(t => t.items),
      ...keepRecent.flatMap(t => t.items),
    ]

    estimate = estimateConversationTokens(remainingItems, systemPrompt)
    if (estimate <= targetTokens) {
      return { items: remainingItems, estimatedTokens: estimate, droppedCount, truncatedCount, droppedItems }
    }
  }

  // Dropped everything droppable, still over — return what we have
  const finalItems = [
    ...keepFirst.items,
    ...keepRecent.flatMap(t => t.items),
  ]
  estimate = estimateConversationTokens(finalItems, systemPrompt)

  return { items: finalItems, estimatedTokens: estimate, droppedCount, truncatedCount, droppedItems }
}

/**
 * Check whether pruning is needed for the given items + model config.
 */
export function needsPruning(
  items: Item[],
  systemPrompt: string | undefined,
  contextWindow: number,
  threshold: number,
): boolean {
  const estimate = estimateConversationTokens(items, systemPrompt)
  return estimate > contextWindow * threshold
}
