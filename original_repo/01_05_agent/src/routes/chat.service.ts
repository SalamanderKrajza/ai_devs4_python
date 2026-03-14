import type { ChatRequest } from './chat.schema.js'
import type { RuntimeContext, ExecutionContext } from '../runtime/index.js'
import type { ProviderStreamEvent } from '../providers/index.js'
import type { Agent, Item, WaitingFor, AgentId, CallId, Content, ToolDefinition, UserId } from '../domain/index.js'
import { runAgent, runAgentStream, deliverResult as deliverAgentResult } from '../runtime/index.js'
import type { ToolResult } from '../tools/index.js'
import { getAgent } from '../lib/runtime.js'
import { config } from '../lib/config.js'

export interface ChatResponse {
  id: string
  sessionId: string
  status: 'completed' | 'waiting' | 'failed'
  model: string
  output: OutputItem[]
  waitingFor?: WaitingFor[]
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number }
}

export type OutputItem =
  | { type: 'text'; text: string }
  | { type: 'function_call'; callId: string; name: string; arguments: Record<string, unknown> }

export type ChatResult =
  | { ok: true; response: ChatResponse }
  | { ok: false; error: string }

function mapToResponse(
  agent: Agent, 
  items: Item[], 
  status: 'completed' | 'waiting' | 'failed',
  waitingFor?: WaitingFor[]
): ChatResponse {
  const output: OutputItem[] = []

  for (const item of items) {
    if (item.type === 'message' && item.role === 'assistant') {
      const text = typeof item.content === 'string'
        ? item.content
        : item.content.filter(p => p.type === 'text').map(p => p.type === 'text' ? p.text : '').join('')
      if (text) output.push({ type: 'text', text })
    } else if (item.type === 'function_call') {
      output.push({ type: 'function_call', callId: item.callId, name: item.name, arguments: item.arguments })
    }
  }

  return { 
    id: agent.id, 
    sessionId: agent.sessionId, 
    status,
    model: agent.config.model, 
    output,
    waitingFor,
  }
}

// Convert request content to domain Content type
function toContent(content: string | Array<{ type: string; text?: string; data?: string; uri?: string; mimeType?: string }>): Content {
  if (typeof content === 'string') return content
  return content as Content
}

// Merge registered tools with request tools (request tools take precedence)
function mergeTools(ctx: RuntimeContext, requestTools?: ChatRequest['tools']): ChatRequest['tools'] {
  const registeredTools = ctx.tools.list()
  const requestToolNames = new Set(requestTools?.filter(t => t.type === 'function').map(t => t.name) ?? [])
  
  // Add registered tools that aren't overridden by request
  const merged = [...(requestTools ?? [])]
  for (const tool of registeredTools) {
    if (!requestToolNames.has(tool.name)) {
      merged.push(tool)
    }
  }
  
  return merged.length > 0 ? merged : undefined
}

// Resolve agent config from template name or use request values
async function resolveAgentConfig(req: ChatRequest) {
  const template = req.agent ? await getAgent(req.agent) : undefined
  
  return {
    model: req.model ?? template?.config.model ?? config.defaultModel,
    task: req.instructions ?? template?.config.systemPrompt ?? 'You are a helpful assistant.',
    tools: req.tools ?? template?.config.tools as ToolDefinition[] | undefined,
  }
}

// Create agent, session, and input items from a chat request
async function setupChatAgent(req: ChatRequest, ctx: RuntimeContext, userId: UserId) {
  const traceId = crypto.randomUUID()
  const agentConfig = await resolveAgentConfig(req)

  const session = req.sessionId
    ? await ctx.repositories.sessions.getById(req.sessionId) ?? await ctx.repositories.sessions.create(userId)
    : await ctx.repositories.sessions.create(userId)

  const tools = mergeTools(ctx, agentConfig.tools)

  const agent = await ctx.repositories.agents.create({
    sessionId: session.id,
    task: agentConfig.task,
    config: { model: agentConfig.model, temperature: req.temperature, maxTokens: req.maxTokens, tools },
  })

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

  return { agent, traceId }
}

export async function processChat(req: ChatRequest, ctx: RuntimeContext, userId: UserId): Promise<ChatResult> {
  const { agent, traceId } = await setupChatAgent(req, ctx, userId)

  const result = await runAgent(agent.id, ctx, {
    maxTurns: 10,
    execution: {
      traceId,
      rootAgentId: agent.id,
      depth: 0,
      userId,
      userInput: req.input,
      agentName: req.agent,
    },
  })

  if (!result.ok) {
    return { ok: false, error: result.status === 'cancelled' ? 'Cancelled' : result.error }
  }

  const items = result.status === 'completed' 
    ? result.items 
    : await ctx.repositories.items.listByAgent(agent.id)

  return { 
    ok: true, 
    response: mapToResponse(
      result.agent, 
      items, 
      result.status,
      result.status === 'waiting' ? result.waitingFor : undefined
    ) 
  }
}

export async function deliverResult(
  agentId: AgentId,
  callId: CallId,
  result: ToolResult,
  ctx: RuntimeContext
): Promise<ChatResult> {
  const runResult = await deliverAgentResult(agentId, callId, result, ctx)

  if (!runResult.ok) {
    return { ok: false, error: runResult.status === 'cancelled' ? 'Cancelled' : runResult.error }
  }

  const items = runResult.status === 'completed'
    ? runResult.items
    : await ctx.repositories.items.listByAgent(agentId)

  return {
    ok: true,
    response: mapToResponse(
      runResult.agent,
      items,
      runResult.status,
      runResult.status === 'waiting' ? runResult.waitingFor : undefined
    )
  }
}

export async function* processChatStream(req: ChatRequest, ctx: RuntimeContext, userId: UserId): AsyncIterable<ProviderStreamEvent> {
  const { agent, traceId } = await setupChatAgent(req, ctx, userId)

  yield* runAgentStream(agent.id, ctx, {
    maxTurns: 10,
    execution: {
      traceId,
      rootAgentId: agent.id,
      depth: 0,
      userId,
      userInput: req.input,
      agentName: req.agent,
    },
  })
}
