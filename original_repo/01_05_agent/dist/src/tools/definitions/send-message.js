export const sendMessageTool = {
    type: 'sync',
    definition: {
        type: 'function',
        name: 'send_message',
        description: 'Send a non-blocking message to another running agent. ' +
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
        const { to, message } = args;
        if (!to || !message) {
            return { ok: false, error: 'Both "to" and "message" are required' };
        }
        return { ok: true, output: JSON.stringify({ to, message }) };
    },
};
