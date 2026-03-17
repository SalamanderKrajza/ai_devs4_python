import type { ChatRequest } from './chat.schema.js'
import type { RuntimeContext } from '../runtime/index.js'
import { prepareAgentForNextTurn, type Agent, type Content, type Session, type ToolDefinition, type UserId } from '../domain/index.js'
import { getAgent } from '../lib/runtime.js'
import { config } from '../lib/config.js'

interface ResolvedAgentConfig {
  model: string
  task: string
  tools?: ToolDefinition[]
}

interface AgentInput {
  task: Agent['task']
  config: Agent['config']
}

export interface SetupChatTurnResult {
  agent: Agent
  traceId: string
  responseStartSequence: number
}

function toContent(content: string | Array<{ type: string; text?: string; data?: string; uri?: string; mimeType?: string }>): Content {
  if (typeof content === 'string') return content
  return content as Content
}

async function resolveAgentConfig(req: ChatRequest): Promise<ResolvedAgentConfig> {
  const template = req.agent ? await getAgent(req.agent) : undefined

  return {
    model: req.model ?? template?.config.model ?? config.defaultModel,
    task: req.instructions ?? template?.config.systemPrompt ?? 'You are a helpful assistant.',
    tools: req.tools ?? template?.config.tools as ToolDefinition[] | undefined,
  }
}

function mergeTools(ctx: RuntimeContext, requestTools?: ChatRequest['tools']): ChatRequest['tools'] {
  const registeredTools = ctx.tools.list()
  const requestToolNames = new Set(requestTools?.filter(tool => tool.type === 'function').map(tool => tool.name) ?? [])

  const merged = [...(requestTools ?? [])]
  for (const tool of registeredTools) {
    if (!requestToolNames.has(tool.name)) {
      merged.push(tool)
    }
  }

  return merged.length > 0 ? merged : undefined
}

function buildAgentInput(
  req: ChatRequest,
  agentConfig: ResolvedAgentConfig,
  tools: ChatRequest['tools'],
): AgentInput {
  return {
    task: agentConfig.task,
    config: {
      model: agentConfig.model,
      temperature: req.temperature,
      maxTokens: req.maxTokens,
      tools,
    },
  }
}

async function loadSession(req: ChatRequest, ctx: RuntimeContext, userId: UserId): Promise<Session> {
  return req.sessionId
    ? await ctx.repositories.sessions.getById(req.sessionId) ?? await ctx.repositories.sessions.create(userId)
    : await ctx.repositories.sessions.create(userId)
}

async function selectSessionRootAgent(session: Session, ctx: RuntimeContext): Promise<Agent | undefined> {
  if (session.rootAgentId) {
    const rootAgent = await ctx.repositories.agents.getById(session.rootAgentId)
    if (rootAgent) return rootAgent
  }

  const agents = await ctx.repositories.agents.listBySession(session.id)
  return [...agents].reverse().find(agent => agent.parentId === undefined) ?? agents.at(-1)
}

async function ensureSessionRootAgent(
  session: Session,
  ctx: RuntimeContext,
  agentInput: AgentInput,
): Promise<{ agent: Agent; created: boolean }> {
  const rootAgent = await selectSessionRootAgent(session, ctx)
  if (rootAgent) {
    if (session.rootAgentId !== rootAgent.id) {
      await ctx.repositories.sessions.update({ ...session, rootAgentId: rootAgent.id })
    }
    return { agent: rootAgent, created: false }
  }

  const agent = await ctx.repositories.agents.create({
    sessionId: session.id,
    ...agentInput,
  })
  await ctx.repositories.sessions.update({ ...session, rootAgentId: agent.id })
  return { agent, created: true }
}

export async function getAgentLastSequence(agentId: Agent['id'], ctx: RuntimeContext): Promise<number> {
  const items = await ctx.repositories.items.listByAgent(agentId)
  return items.at(-1)?.sequence ?? 0
}

// Reuse the session root agent so the conversation has one canonical history.
export async function setupChatTurn(
  req: ChatRequest,
  ctx: RuntimeContext,
  userId: UserId,
): Promise<{ ok: true; data: SetupChatTurnResult } | { ok: false; error: string }> {
  const traceId = crypto.randomUUID()
  const agentConfig = await resolveAgentConfig(req)
  const tools = mergeTools(ctx, agentConfig.tools)
  const agentInput = buildAgentInput(req, agentConfig, tools)
  const session = await loadSession(req, ctx, userId)
  const rootAgent = await ensureSessionRootAgent(session, ctx, agentInput)

  const preparedAgent = prepareAgentForNextTurn({
    ...rootAgent.agent,
    ...agentInput,
  })
  if (!preparedAgent.ok) {
    return { ok: false, error: preparedAgent.error }
  }

  let agent = preparedAgent.agent
  if (!rootAgent.created) {
    agent = await ctx.repositories.agents.update(agent)
  }

  const responseStartSequence = await getAgentLastSequence(agent.id, ctx)
  const input = typeof req.input === 'string'
    ? [{ type: 'message' as const, role: 'user' as const, content: req.input }]
    : req.input

  for (const item of input) {
    if (item.type === 'message') {
      await ctx.repositories.items.create(agent.id, {
        type: 'message',
        role: item.role,
        content: toContent(item.content),
      })
    }
  }

  return { ok: true, data: { agent, traceId, responseStartSequence } }
}
