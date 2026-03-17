/**
 * Send message tool — non-blocking context push to another agent.
 *
 * Writes a system message into the target agent's conversation history.
 * The target sees it on their next turn.  Does NOT block the sender.
 *
 * NOTE: The handler receives a plain args object.  The actual item
 * creation requires RuntimeContext (repositories), which the handler
 * doesn't have.  The runner intercepts this tool by name and performs
 * the write, then returns the handler's output to the LLM.
 */
import type { Tool } from '../types.js'

export interface SendMessageArgs {
  to: string
  message: string
}

export const sendMessageTool: Tool = {
  type: 'sync',
  definition: {
    type: 'function',
    name: 'send_message',
    description:
      'Send a non-blocking message to another running agent. ' +
      'The message appears in the target agent\'s context on their next turn. ' +
      'Use this to share information without waiting for a response.',
    parameters: {
      type: 'object',
      properties: {
        to: {
          type: 'string',
          description: 'The agent ID to send the message to',
        },
        message: {
          type: 'string',
          description: 'The message content to deliver',
        },
      },
      required: ['to', 'message'],
    },
  },
  handler: async (args) => {
    // Validation only — actual write handled by runner (needs RuntimeContext)
    const { to, message } = args as unknown as SendMessageArgs
    if (!to || !message) {
      return { ok: false, error: 'Both "to" and "message" are required' }
    }
    return { ok: true, output: JSON.stringify({ to, message }) }
  },
}
