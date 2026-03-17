import { readFile, readdir, access } from 'node:fs/promises'
import { join, basename } from 'node:path'
import matter from 'gray-matter'
import type { ToolDefinition, FunctionTool } from '../domain/index.js'
import type { ToolRegistry } from '../tools/types.js'
import type { McpManager } from '../mcp/index.js'
import { config } from '../lib/config.js'

export interface AgentTemplate {
  name: string
  model?: string
  tools: string[]
  systemPrompt: string
}

export interface LoadedAgent {
  name: string
  config: {
    model: string
    systemPrompt: string
    tools: ToolDefinition[]
  }
}

const AGENT_EXTENSION = '.agent.md'
const MCP_SEPARATOR = '__'

export async function loadAgentTemplate(filePath: string): Promise<AgentTemplate> {
  const content = await readFile(filePath, 'utf-8')
  const { data, content: systemPrompt } = matter(content)

  const name = data.name ?? basename(filePath, AGENT_EXTENSION)
  const model = typeof data.model === 'string' ? data.model : undefined
  const tools = Array.isArray(data.tools) ? data.tools : []

  return {
    name,
    model,
    tools,
    systemPrompt: systemPrompt.trim(),
  }
}

/**
 * Resolve tool names to ToolDefinition[].
 * 
 * Supported formats:
 *  - "calculator"          → built-in tool from registry
 *  - "web_search"          → native web search tool
 *  - "files__fs_read"      → MCP tool (server__toolName)
 */
export async function resolveToolDefinitions(
  toolNames: string[],
  registry: ToolRegistry,
  mcp: McpManager,
): Promise<ToolDefinition[]> {
  const registeredTools = registry.list()
  const definitions: ToolDefinition[] = []

  // Collect MCP tool names we need to resolve
  const mcpToolNames = toolNames.filter(name => name.includes(MCP_SEPARATOR))
  const mcpToolsCache = mcpToolNames.length > 0
    ? await mcp.listTools()
    : []

  for (const name of toolNames) {
    // Native web search
    if (name === 'web_search') {
      definitions.push({ type: 'web_search' })
      continue
    }

    // MCP tool (server__toolName)
    if (name.includes(MCP_SEPARATOR)) {
      const mcpTool = mcpToolsCache.find(t => t.prefixedName === name)
      if (mcpTool) {
        definitions.push({
          type: 'function',
          name: mcpTool.prefixedName,
          description: mcpTool.description ?? '',
          parameters: mcpTool.inputSchema,
        })
      }
      continue
    }

    // Built-in tool from registry
    const tool = registeredTools.find((t: FunctionTool) => t.name === name)
    if (tool) {
      definitions.push(tool)
    }
  }

  return definitions
}

function templateToLoadedAgent(
  template: AgentTemplate,
  tools: ToolDefinition[],
): LoadedAgent {
  return {
    name: template.name,
    config: {
      model: template.model ?? config.defaultModel,
      systemPrompt: template.systemPrompt,
      tools,
    },
  }
}

/**
 * Resolve a single agent by name — reads from disk on every call.
 */
export async function resolveAgent(
  name: string,
  workspacePath: string,
  registry: ToolRegistry,
  mcp: McpManager,
): Promise<LoadedAgent | undefined> {
  const filePath = join(workspacePath, 'agents', `${name}${AGENT_EXTENSION}`)

  try {
    await access(filePath)
  } catch {
    return undefined
  }

  const template = await loadAgentTemplate(filePath)
  const tools = await resolveToolDefinitions(template.tools, registry, mcp)
  return templateToLoadedAgent(template, tools)
}

/**
 * List all available agent names from workspace.
 */
export async function listAgentNames(workspacePath: string): Promise<string[]> {
  const agentsDir = join(workspacePath, 'agents')

  try {
    const files = await readdir(agentsDir)
    return files
      .filter(f => f.endsWith(AGENT_EXTENSION))
      .map(f => basename(f, AGENT_EXTENSION))
  } catch {
    return []
  }
}
