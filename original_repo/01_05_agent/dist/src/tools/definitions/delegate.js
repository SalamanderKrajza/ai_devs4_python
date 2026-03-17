export const delegateTool = {
    type: 'agent',
    definition: {
        type: 'function',
        name: 'delegate',
        description: 'Delegate a task to another agent and wait for the result. ' +
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
        const { agent, task } = args;
        if (!agent || !task) {
            return { ok: false, error: 'Both "agent" and "task" are required' };
        }
        return { ok: true, output: JSON.stringify({ agent, task }) };
    },
};
