/**
 * Ask user tool — pause the agent and wait for human input.
 *
 * Transitions the agent to 'waiting' with type 'human'.
 * The HTTP layer returns 202 with the question in waitingFor.
 * A human (or external system) delivers the answer via
 * POST /api/chat/agents/:agentId/deliver — could be seconds or hours later.
 */
import type { Tool } from '../types.js'

export interface AskUserArgs {
  question: string
}

export const askUserTool: Tool = {
  type: 'human',
  definition: {
    type: 'function',
    name: 'ask_user',
    description:
      'Ask the user a question and wait for their response. ' +
      'Use this when you need clarification, confirmation, or additional ' +
      'information that only the user can provide. The agent will pause ' +
      'until the user responds.',
    parameters: {
      type: 'object',
      properties: {
        question: {
          type: 'string',
          description: 'The question to ask the user',
        },
      },
      required: ['question'],
    },
  },
  handler: async (args) => {
    // Validation only — the runner defers this to waitingFor
    const { question } = args as unknown as AskUserArgs
    if (!question) {
      return { ok: false, error: '"question" is required' }
    }
    return { ok: true, output: question }
  },
}
