/**
 * Tool registry — manages available tools for agents
 */
import type { Tool, ToolRegistry, ToolResult } from './types.js'
import type { FunctionTool } from '../domain/index.js'

export function createToolRegistry(): ToolRegistry {
  const tools = new Map<string, Tool>()

  return {
    register(tool) {
      tools.set(tool.definition.name, tool)
    },

    get(name) {
      return tools.get(name)
    },

    list(): FunctionTool[] {
      return Array.from(tools.values()).map(t => t.definition)
    },

    async execute(name, args, signal): Promise<ToolResult> {
      const tool = tools.get(name)
      if (!tool) {
        return { ok: false, error: `Tool not found: ${name}` }
      }

      if (signal?.aborted) {
        return { ok: false, error: 'Operation aborted' }
      }

      try {
        return await tool.handler(args, signal)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Tool execution failed'
        return { ok: false, error: message }
      }
    },
  }
}
