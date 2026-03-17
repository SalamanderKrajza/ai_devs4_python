export const askUserTool = {
    type: 'human',
    definition: {
        type: 'function',
        name: 'ask_user',
        description: 'Ask the user a question and wait for their response. ' +
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
        const { question } = args;
        if (!question) {
            return { ok: false, error: '"question" is required' };
        }
        return { ok: true, output: question };
    },
};
