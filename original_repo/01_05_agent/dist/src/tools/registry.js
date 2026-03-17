export function createToolRegistry() {
    const tools = new Map();
    return {
        register(tool) {
            tools.set(tool.definition.name, tool);
        },
        get(name) {
            return tools.get(name);
        },
        list() {
            return Array.from(tools.values()).map(t => t.definition);
        },
        async execute(name, args, signal) {
            const tool = tools.get(name);
            if (!tool) {
                return { ok: false, error: `Tool not found: ${name}` };
            }
            if (signal?.aborted) {
                return { ok: false, error: 'Operation aborted' };
            }
            try {
                return await tool.handler(args, signal);
            }
            catch (err) {
                const message = err instanceof Error ? err.message : 'Tool execution failed';
                return { ok: false, error: message };
            }
        },
    };
}
