/**
 * Tool execution types
 */
import type { FunctionTool } from '../domain/index.js'

export type ToolResult =
  | { ok: true; output: string }
  | { ok: false; error: string }

export type ToolHandler = (
  args: Record<string, unknown>,
  signal?: AbortSignal
) => Promise<ToolResult>

/** Tool execution type: sync (immediate), async (background), agent (sub-agent), human (confirmation) */
export type ToolType = 'sync' | 'async' | 'agent' | 'human'

export interface Tool {
  type: ToolType
  definition: FunctionTool
  handler: ToolHandler
}

export interface ToolRegistry {
  register(tool: Tool): void
  get(name: string): Tool | undefined
  list(): FunctionTool[]
  execute(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<ToolResult>
}
