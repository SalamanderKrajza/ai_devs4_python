/**
 * Runtime initialization
 */
import { createEventEmitter } from '../events/index.js'
import { createSQLiteRepositories } from '../repositories/index.js'
import { createContext, type RuntimeContext } from '../runtime/index.js'
import { registerProvider, createOpenAIProvider, createGeminiProvider, listProviders } from '../providers/index.js'
import { createToolRegistry, calculatorTool, delegateTool, sendMessageTool, askUserTool } from '../tools/index.js'
import { createMcpManager, type McpManager } from '../mcp/index.js'
import { listAgentNames, resolveAgent, type LoadedAgent } from '../workspace/index.js'
import { config } from './config.js'
import { logger } from './logger.js'
import { subscribeEventLogger } from './event-logger.js'
import { initTracing, shutdownTracing } from './tracing.js'
import { subscribeLangfuse } from './langfuse-subscriber.js'

const log = logger.child({ name: 'runtime' })

let runtime: RuntimeContext | undefined
let mcpManager: McpManager | undefined

const normalizeOpenRouterModel = (model: string, tools?: Array<{ type: string }>) => {
  const baseModel = model.includes('/')
    ? model
    : model.startsWith('gpt-')
      ? `openai/${model}`
      : model

  const usesWebSearch = tools?.some(tool => tool.type === 'web_search') ?? false
  return usesWebSearch && !baseModel.endsWith(':online')
    ? `${baseModel}:online`
    : baseModel
}

export async function initRuntime(): Promise<RuntimeContext> {
  if (runtime) return runtime

  // Initialize Langfuse tracing (no-op when keys are absent)
  initTracing()

  // Register providers
  if (config.openaiApiKey) {
    registerProvider(createOpenAIProvider({ apiKey: config.openaiApiKey }))
  }
  if (config.openRouterApiKey) {
    registerProvider(createOpenAIProvider({
      name: 'openrouter',
      apiKey: config.openRouterApiKey,
      baseUrl: 'https://openrouter.ai/api/v1',
      defaultHeaders: config.openRouterHeaders,
      webSearchMode: 'model_suffix',
      transformModel: (model, request) => normalizeOpenRouterModel(model, request.tools),
    }))
  }
  if (config.geminiApiKey) {
    registerProvider(createGeminiProvider({ apiKey: config.geminiApiKey }))
  }

  const providers = listProviders()
  if (providers.length === 0) {
    log.warn('no AI providers configured — set OPENAI_API_KEY or OPENROUTER_API_KEY')
  } else {
    log.info({ providers }, 'providers registered')
  }

  const repositories = await createSQLiteRepositories({
    url: config.databaseUrl,
    authToken: config.databaseAuthToken,
  })

  log.info({ driver: 'sqlite', url: config.databaseUrl }, 'database ready')

  // Create tool registry with default tools
  const tools = createToolRegistry()
  tools.register(calculatorTool)
  tools.register(delegateTool)
  tools.register(sendMessageTool)
  tools.register(askUserTool)
  log.info({ tools: tools.list().map(t => t.name) }, 'tools registered')

  // Initialize MCP clients
  const baseUrl = `http://${config.host}:${config.port}`
  mcpManager = await createMcpManager(process.cwd(), baseUrl)
  const mcpServers = mcpManager.servers()
  if (mcpServers.length > 0) {
    const mcpTools = await mcpManager.listTools()
    log.info({ servers: mcpServers, tools: mcpTools.map(t => t.prefixedName) }, 'mcp ready')
  }

  // List available agent templates
  const agentNames = await listAgentNames(config.workspacePath)
  log.info({ agents: agentNames }, 'workspace loaded')

  const events = createEventEmitter()
  subscribeEventLogger(events)
  subscribeLangfuse(events)

  runtime = createContext(
    events,
    repositories,
    tools,
    mcpManager,
  )

  return runtime
}

export function getRuntime(): RuntimeContext {
  if (!runtime) throw new Error('Runtime not initialized')
  return runtime
}

export function hasRuntime(): boolean {
  return runtime !== undefined
}

/**
 * Resolve agent template by name — reads fresh from disk each time.
 */
export async function getAgent(name: string): Promise<LoadedAgent | undefined> {
  if (!runtime) return undefined
  return resolveAgent(name, config.workspacePath, runtime.tools, runtime.mcp)
}

/**
 * Graceful shutdown — close MCP connections.
 */
export async function shutdownRuntime(): Promise<void> {
  if (mcpManager) {
    await mcpManager.close()
  }
  await shutdownTracing()
}
