/**
 * Delegate tool — spawn a child agent to handle a sub-task.
 *
 * The handler itself is a no-op placeholder; the actual spawning logic
 * lives in the runner's agent branch (handleTurnResponse).  The runner
 * recognises tool.type === 'agent' and executes the delegation inline.
 *
 * The handler is still called by the runner to validate arguments and
 * resolve the target agent template name + task description before
 * the child agent is created.
 */
import type { Tool } from '../types.js'

export interface DelegateArgs {
  agent: string
  task: string
}

export const delegateTool: Tool = {
  type: 'agent',
  definition: {
    type: 'function',
    name: 'delegate',
    description:
      'Delegate a task to another agent and wait for the result. ' +
      'Use this when a specialised agent can handle part of the work ' +
      '(e.g. web research, file operations).',
    parameters: {
      type: 'object',
      properties: {
        agent: {
          type: 'string',
          description: 'Name of the agent template to run (e.g. "bob")',
        },
        task: {
          type: 'string',
          description: 'A clear description of what the child agent should accomplish',
        },
      },
      required: ['agent', 'task'],
    },
  },
  handler: async (args) => {
    // Validation only — real execution handled by runner
    const { agent, task } = args as unknown as DelegateArgs
    if (!agent || !task) {
      return { ok: false, error: 'Both "agent" and "task" are required' }
    }
    return { ok: true, output: JSON.stringify({ agent, task }) }
  },
}
