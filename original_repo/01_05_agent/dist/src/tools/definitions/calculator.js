function calculate({ operation, a, b }) {
    switch (operation) {
        case 'add': return a + b;
        case 'subtract': return a - b;
        case 'multiply': return a * b;
        case 'divide':
            if (b === 0)
                throw new Error('Division by zero');
            return a / b;
        default:
            throw new Error(`Unknown operation: ${operation}`);
    }
}
export const calculatorTool = {
    type: 'sync',
    definition: {
        type: 'function',
        name: 'calculator',
        description: 'Perform basic math operations: add, subtract, multiply, divide',
        parameters: {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    enum: ['add', 'subtract', 'multiply', 'divide'],
                    description: 'The math operation to perform',
                },
                a: {
                    type: 'number',
                    description: 'First operand',
                },
                b: {
                    type: 'number',
                    description: 'Second operand',
                },
            },
            required: ['operation', 'a', 'b'],
        },
    },
    handler: async (args) => {
        const { operation, a, b } = args;
        const result = calculate({ operation, a, b });
        return { ok: true, output: String(result) };
    },
};
