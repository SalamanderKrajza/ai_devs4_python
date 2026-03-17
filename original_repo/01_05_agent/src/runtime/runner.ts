/**
 * Agent runner — non-blocking execution with resume support
 */
import type { Agent, AgentId, Item, WaitingFor, CallId, TokenUsage, Session } from '../domain/index.js'
import { 
  startAgent, completeAgent, failAgent, cancelAgent, 
  incrementTurn, waitForMany, deliverOne, addUsage,
  isMessage, isFunctionCall, isFunctionCallOutput
} from '../domain/index.js'
import type { RuntimeContext, ExecutionContext } from './context.js'
import { createExecutionContext } from './context.js'
import { createEventContext } from '../events/index.js'
import type { Provider, ProviderStreamEvent, ProviderInputItem, ProviderOutputItem, ProviderResponse } from '../providers/index.js'
import { isAbortError, resolveProvider } from '../providers/index.js'
import type { ToolResult } from '../tools/index.js'
import type { DelegateArgs } from '../tools/definitions/delegate.js'
import type { SendMessageArgs } from '../tools/definitions/send-message.js'
import { getModelDefinition } from '../config/models.js'
import { needsPruning, pruneConversation } from '../utils/pruning.js'
import { generateSummary } from '../utils/summarization.js'
import { getAgent } from '../lib/runtime.js'
import { logger } from '../lib/logger.js'

const log = logger.child({ name: 'runner' })

const MAX_AGENT_DEPTH = 5

const truncate = (s: string, max = 200) => s.length > max ? s.slice(0, max) + '…' : s

/** Build a messages-style array for Langfuse generation input */
function formatGenInput(instructions: string, input: ProviderInputItem[]) {
  return [
    { role: 'system', content: instructions },
    ...input.map(item => {
      if (item.type === 'message') return { role: item.role, content: item.content }
      if (item.type === 'function_call') return { role: 'assistant', function_call: { name: item.name, arguments: item.arguments } }
      if (item.type === 'function_result') return { role: 'tool', name: item.name, content: item.output }
      return item
    }),
  ]
}

/** Extract text from provider output for Langfuse */
function formatGenOutput(output: ProviderOutputItem[]) {
  const text = output.filter(o => o.type === 'text').map(o => o.text).join('')
  const calls = output.filter(o => o.type === 'function_call')
  if (calls.length === 0) return text || null
  return {
    text: text || undefined,
    tool_calls: calls.map(c => ({ name: c.name, arguments: c.arguments })),
  }
}

export interface RunOptions {
  maxTurns?: number
  signal?: AbortSignal
  execution?: ExecutionContext
}

export type RunResult =
  | { ok: true; status: 'completed'; agent: Agent; items: Item[] }
  | { ok: true; status: 'waiting'; agent: Agent; waitingFor: WaitingFor[] }
  | { ok: false; status: 'failed'; error: string }
  | { ok: false; status: 'cancelled' }

type TurnResult =
  | { continue: true; agent: Agent; usage?: TokenUsage }
  | { continue: false; agent: Agent; usage?: TokenUsage }
  | { continue: false; waiting: WaitingFor[]; agent: Agent; usage?: TokenUsage }
  | { continue: false; error: string }

function makeCtx(exec: ExecutionContext, agent: Agent, batchId?: string) {
  return createEventContext(
    exec.traceId,
    agent.sessionId,
    agent.id,
    exec.rootAgentId,
    exec.depth,
    exec.parentAgentId,
    batchId
  )
}

/** Map stored Items to provider input format */
function mapItemsToProviderInput(items: Item[]): ProviderInputItem[] {
  const result: ProviderInputItem[] = []

  for (const item of items) {
    if (isMessage(item)) {
      result.push({
        type: 'message',
        role: item.role,
        content: item.content,
      })
    } else if (isFunctionCall(item)) {
      result.push({
        type: 'function_call',
        callId: item.callId,
        name: item.name,
        arguments: item.arguments,
      })
    } else if (isFunctionCallOutput(item)) {
      const call = items.find(i => isFunctionCall(i) && i.callId === item.callId)
      result.push({
        type: 'function_result',
        callId: item.callId,
        name: call && isFunctionCall(call) ? call.name : '',
        output: item.output,
      })
    }
    // Skip reasoning items for now
  }

  return result
}

/** Store provider output as Items */
async function storeProviderOutput(
  agentId: AgentId,
  output: ProviderOutputItem[],
  runtime: RuntimeContext,
  turnNumber?: number
): Promise<Item[]> {
  const stored: Item[] = []

  // Collect text parts for assistant message
  const textParts = output.filter(o => o.type === 'text')
  if (textParts.length > 0) {
    const content = textParts.map(t => t.text).join('')
    const item = await runtime.repositories.items.create(agentId, {
      type: 'message',
      role: 'assistant',
      content,
      turnNumber,
    })
    stored.push(item)
  }

  // Store function calls
  for (const o of output) {
    if (o.type === 'function_call') {
      const item = await runtime.repositories.items.create(agentId, {
        type: 'function_call',
        callId: o.callId,
        name: o.name,
        arguments: o.arguments,
        turnNumber,
      })
      stored.push(item)
    } else if (o.type === 'reasoning') {
      const item = await runtime.repositories.items.create(agentId, {
        type: 'reasoning',
        summary: o.text,
        turnNumber,
      })
      stored.push(item)
    }
  }

  return stored
}

// ---- Shared turn helpers ----

interface TurnInput {
  provider: Provider
  model: string
  input: ProviderInputItem[]
  session: Session
}

/** Resolve provider, load items, prune context, build provider input */
async function prepareTurnInput(
  agent: Agent,
  runtime: RuntimeContext,
  exec: ExecutionContext,
  session: Session,
): Promise<{ ok: true; data: TurnInput } | { ok: false; error: string }> {
  const turnNumber = agent.turnCount

  const resolved = resolveProvider(agent.config.model)
  if (!resolved) {
    return { ok: false, error: `Unknown model or provider: ${agent.config.model}` }
  }
  const { provider, model } = resolved
  const modelDef = getModelDefinition(model)

  const items = await runtime.repositories.items.listByAgent(agent.id)

  let prunedItems = items
  if (needsPruning(items, agent.task, modelDef.contextWindow, modelDef.pruning.threshold)) {
    const pruneResult = pruneConversation(items, agent.task, modelDef.contextWindow, modelDef.pruning)

    log.info({
      agentId: agent.id, traceId: exec.traceId, turn: turnNumber,
      dropped: pruneResult.droppedCount, truncated: pruneResult.truncatedCount,
      tokens: pruneResult.estimatedTokens,
    }, 'context pruned')

    if (modelDef.pruning.enableSummarization && pruneResult.droppedItems.length > 0) {
      try {
        const summary = await generateSummary(provider, model, pruneResult.droppedItems, session.summary)
        session = { ...session, summary, updatedAt: new Date() }
        await runtime.repositories.sessions.update(session)
      } catch (err) {
        log.warn({ agentId: agent.id, traceId: exec.traceId, turn: turnNumber, err }, 'summarization failed')
      }
    }

    prunedItems = pruneResult.items
  }

  const input = mapItemsToProviderInput(prunedItems)
  if (session.summary) {
    input.unshift({
      type: 'message',
      role: 'system',
      content: `[Context Summary — Earlier conversation was compacted]\n\n${session.summary}`,
    })
  }

  return { ok: true, data: { provider, model, input, session } }
}

/** Store output, process function calls, determine next step */
async function handleTurnResponse(
  response: ProviderResponse,
  agent: Agent,
  runtime: RuntimeContext,
  exec: ExecutionContext,
  turnNumber: number,
  signal?: AbortSignal,
): Promise<TurnResult> {
  const usage = response.usage

  await storeProviderOutput(agent.id, response.output, runtime, turnNumber)

  const functionCalls = response.output.filter(o => o.type === 'function_call')

  if (functionCalls.length === 0) {
    const completeResult = completeAgent(agent, response.output)
    if (!completeResult.ok) {
      return { continue: false, error: completeResult.error }
    }
    return { continue: false, agent: completeResult.agent, usage }
  }

  const waitingFor: WaitingFor[] = []
  let hasSyncTools = false

  for (const fc of functionCalls) {
    if (fc.type !== 'function_call') continue

    const toolCtx = { agentId: agent.id, traceId: exec.traceId, callId: fc.callId, tool: fc.name }
    const argsPreview = truncate(JSON.stringify(fc.arguments))
    const tool = runtime.tools.get(fc.name)

    if (!tool) {
      // Check if this is an MCP tool (server__toolName)
      if (runtime.mcp.parseName(fc.name)) {
        log.info({ ...toolCtx, args: argsPreview }, `${fc.name}`)
        runtime.events.emit({ type: 'tool.called', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments })
        const start = Date.now()
        try {
          const output = await runtime.mcp.callTool(fc.name, fc.arguments, signal)
          const ms = Date.now() - start
          log.info({ ...toolCtx, ms, output: truncate(output) }, `${fc.name} ok ${ms}ms`)
          await runtime.repositories.items.create(agent.id, {
            type: 'function_call_output',
            callId: fc.callId,
            output,
            isError: false,
            turnNumber,
          })
          runtime.events.emit({ type: 'tool.completed', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments, output: truncate(output, 1000), durationMs: ms, startTime: start })
          hasSyncTools = true
        } catch (err) {
          const ms = Date.now() - start
          log.warn({ ...toolCtx, ms, err }, `${fc.name} failed ${ms}ms`)
          await runtime.repositories.items.create(agent.id, {
            type: 'function_call_output',
            callId: fc.callId,
            output: (err as Error).message,
            isError: true,
            turnNumber,
          })
          runtime.events.emit({ type: 'tool.failed', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments, error: (err as Error).message, durationMs: ms, startTime: start })
          hasSyncTools = true
        }
        continue
      }

      // No registered handler — treat as external tool, wait for result
      log.info({ ...toolCtx, args: argsPreview, deferred: true }, `${fc.name} → deferred`)
      waitingFor.push({
        callId: fc.callId,
        type: 'tool',
        name: fc.name,
      })
      continue
    }

    // ── Agent tool: spawn a child agent ──────────────────────────────
    if (tool.type === 'agent') {
      const delegateResult = await handleDelegation(fc.callId, fc.arguments, agent, runtime, exec, turnNumber, signal)

      if (delegateResult.type === 'completed') {
        hasSyncTools = true
      } else if (delegateResult.type === 'waiting') {
        waitingFor.push(delegateResult.wait)
      } else {
        // error — store as failed function output, continue
        await runtime.repositories.items.create(agent.id, {
          type: 'function_call_output',
          callId: fc.callId,
          output: delegateResult.error,
          isError: true,
          turnNumber,
        })
        hasSyncTools = true
      }
      continue
    }

    // ── Human tool: defer for external delivery ─────────────────────
    if (tool.type === 'human') {
      log.info({ ...toolCtx, args: argsPreview, type: 'human', deferred: true }, `${fc.name} → deferred`)
      waitingFor.push({
        callId: fc.callId,
        type: 'human',
        name: fc.name,
        description: typeof fc.arguments.question === 'string'
          ? fc.arguments.question
          : tool.definition.description,
      })
      continue
    }

    // ── Sync tool ────────────────────────────────────────────────────
    if (tool.type === 'sync') {
      // Intercept send_message: needs RuntimeContext for cross-agent write
      if (fc.name === 'send_message') {
        runtime.events.emit({ type: 'tool.called', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments })
        const smStart = Date.now()
        const smResult = await handleSendMessage(fc.callId, fc.arguments, agent, runtime, turnNumber)
        const smMs = Date.now() - smStart
        if (!smResult.ok) {
          await runtime.repositories.items.create(agent.id, {
            type: 'function_call_output',
            callId: fc.callId,
            output: smResult.error,
            isError: true,
            turnNumber,
          })
          runtime.events.emit({ type: 'tool.failed', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments, error: smResult.error, durationMs: smMs, startTime: smStart })
        } else {
          runtime.events.emit({ type: 'tool.completed', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments, output: 'Message delivered', durationMs: smMs, startTime: smStart })
        }
        hasSyncTools = true
        continue
      }

      log.info({ ...toolCtx, args: argsPreview }, `${fc.name}`)
      runtime.events.emit({ type: 'tool.called', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments })
      const start = Date.now()
      const result = await runtime.tools.execute(fc.name, fc.arguments, signal)
      const ms = Date.now() - start
      const output = result.ok ? result.output : result.error
      log.info({ ...toolCtx, ms, ok: result.ok, output: truncate(output) }, `${fc.name} ok ${ms}ms`)
      await runtime.repositories.items.create(agent.id, {
        type: 'function_call_output',
        callId: fc.callId,
        output,
        isError: !result.ok,
        turnNumber,
      })
      if (result.ok) {
        runtime.events.emit({ type: 'tool.completed', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments, output: truncate(output, 1000), durationMs: ms, startTime: start })
      } else {
        runtime.events.emit({ type: 'tool.failed', ctx: makeCtx(exec, agent), callId: fc.callId, name: fc.name, arguments: fc.arguments, error: output, durationMs: ms, startTime: start })
      }
      hasSyncTools = true
      continue
    }

    // ── Async / unknown tool type: defer ─────────────────────────────
    const waitType = 'tool' as const
    log.info({ ...toolCtx, args: argsPreview, type: waitType, deferred: true }, `${fc.name} → deferred`)
    waitingFor.push({
      callId: fc.callId,
      type: waitType,
      name: fc.name,
      description: tool.definition.description,
    })
  }

  if (waitingFor.length > 0) {
    return { continue: false, waiting: waitingFor, agent, usage }
  }

  return { continue: true, agent, usage }
}

// ── Delegation: spawn child agent and run it ──────────────────────────────

type DelegationResult =
  | { type: 'completed' }
  | { type: 'waiting'; wait: WaitingFor }
  | { type: 'error'; error: string }

async function handleDelegation(
  callId: CallId,
  args: Record<string, unknown>,
  parent: Agent,
  runtime: RuntimeContext,
  exec: ExecutionContext,
  turnNumber: number,
  signal?: AbortSignal,
): Promise<DelegationResult> {
  const { agent: agentName, task } = args as unknown as DelegateArgs

  if (!agentName || !task) {
    return { type: 'error', error: 'delegate requires "agent" and "task"' }
  }

  // Depth guard
  if (exec.depth + 1 > MAX_AGENT_DEPTH) {
    return { type: 'error', error: `Max agent depth (${MAX_AGENT_DEPTH}) exceeded` }
  }

  // Resolve child agent template
  const template = await getAgent(agentName)
  if (!template) {
    return { type: 'error', error: `Agent template not found: ${agentName}` }
  }

  log.info({
    parentId: parent.id, traceId: exec.traceId, callId,
    childAgent: agentName, depth: exec.depth + 1,
  }, `delegate → ${agentName}`)

  // Create child agent
  const child = await runtime.repositories.agents.create({
    sessionId: parent.sessionId,
    traceId: exec.traceId,
    rootAgentId: exec.rootAgentId,
    parentId: parent.id,
    sourceCallId: callId,
    depth: exec.depth + 1,
    task: template.config.systemPrompt,
    config: {
      model: template.config.model,
      tools: template.config.tools,
    },
  })

  // Add task as user message for the child
  await runtime.repositories.items.create(child.id, {
    type: 'message',
    role: 'user',
    content: task,
    turnNumber: 0,
  })

  // Run child agent recursively
  const childExec = createExecutionContext(
    exec.traceId,
    exec.rootAgentId,
    parent.id,
    exec.depth + 1,
  )
  childExec.agentName = agentName

  const childResult = await runAgent(child.id, runtime, {
    maxTurns: 10,
    signal,
    execution: childExec,
  })

  // Child completed synchronously — store result on parent
  if (childResult.ok && childResult.status === 'completed') {
    const output = extractAgentResult(childResult.items)
    await runtime.repositories.items.create(parent.id, {
      type: 'function_call_output',
      callId,
      output,
      isError: false,
      turnNumber,
    })
    log.info({
      parentId: parent.id, childId: child.id, traceId: exec.traceId,
      output: truncate(output),
    }, `delegate ← ${agentName} completed`)
    return { type: 'completed' }
  }

  // Child is waiting (e.g. needs human input) — parent also waits
  if (childResult.ok && childResult.status === 'waiting') {
    log.info({
      parentId: parent.id, childId: child.id, traceId: exec.traceId,
      waitingFor: childResult.waitingFor,
    }, `delegate ← ${agentName} waiting`)
    return {
      type: 'waiting',
      wait: {
        callId,
        type: 'agent',
        name: `delegate:${agentName}`,
        description: `Waiting for agent "${agentName}" to complete`,
      },
    }
  }

  // Child failed or cancelled
  const errorMsg = childResult.status === 'cancelled'
    ? `Agent "${agentName}" was cancelled`
    : 'error' in childResult
      ? childResult.error
      : `Agent "${agentName}" failed`

  return { type: 'error', error: errorMsg }
}

/** Extract the final text output from a completed agent's items */
function extractAgentResult(items: Item[]): string {
  // Find the last assistant message
  const assistantMessages = items
    .filter(isMessage)
    .filter(i => i.role === 'assistant')

  if (assistantMessages.length === 0) {
    return '(no output)'
  }

  const last = assistantMessages[assistantMessages.length - 1]
  return typeof last.content === 'string'
    ? last.content
    : last.content
        .filter(p => p.type === 'text')
        .map(p => p.type === 'text' ? p.text : '')
        .join('')
}

// ── Send message: write to target agent's conversation ────────────────────

async function handleSendMessage(
  callId: CallId,
  args: Record<string, unknown>,
  sender: Agent,
  runtime: RuntimeContext,
  turnNumber: number,
): Promise<{ ok: true } | { ok: false; error: string }> {
  const { to, message } = args as unknown as SendMessageArgs

  if (!to || !message) {
    return { ok: false, error: 'send_message requires "to" and "message"' }
  }

  // Verify target agent exists
  const target = await runtime.repositories.agents.getById(to)
  if (!target) {
    return { ok: false, error: `Target agent not found: ${to}` }
  }

  // Write message into target agent's items
  await runtime.repositories.items.create(target.id, {
    type: 'message',
    role: 'system',
    content: `[Message from agent ${sender.id}]\n\n${message}`,
  })

  // Store success output on sender
  await runtime.repositories.items.create(sender.id, {
    type: 'function_call_output',
    callId,
    output: `Message delivered to agent ${to}`,
    isError: false,
    turnNumber,
  })

  log.info({
    senderId: sender.id, targetId: to,
    message: truncate(message),
  }, 'send_message delivered')

  return { ok: true }
}

/** Load agent and session, create execution context */
async function loadAgentLoop(
  agentId: AgentId,
  runtime: RuntimeContext,
  options: RunOptions,
): Promise<{ ok: true; agent: Agent; session: Session; exec: ExecutionContext } | { ok: false; error: string }> {
  const agent = await runtime.repositories.agents.getById(agentId)
  if (!agent) {
    return { ok: false, error: `Agent not found: ${agentId}` }
  }

  const session = await runtime.repositories.sessions.getById(agent.sessionId)
  if (!session) {
    return { ok: false, error: `Session not found: ${agent.sessionId}` }
  }

  const exec = options.execution ?? createExecutionContext(
    crypto.randomUUID(),
    agent.rootAgentId,
    agent.parentId,
    agent.depth,
  )

  return { ok: true, agent, session, exec }
}

// ---- Turn execution ----

async function executeTurn(
  agent: Agent, 
  runtime: RuntimeContext, 
  exec: ExecutionContext,
  session: Session,
  signal?: AbortSignal
): Promise<TurnResult> {
  if (signal?.aborted) {
    return { continue: false, error: 'Operation aborted' }
  }

  const prepared = await prepareTurnInput(agent, runtime, exec, session)
  if (!prepared.ok) {
    return { continue: false, error: prepared.error }
  }
  const { provider, model, input } = prepared.data

  const genStart = Date.now()

  const response = await provider.generate({
    model,
    instructions: agent.task,
    input,
    tools: agent.config.tools,
    temperature: agent.config.temperature,
    maxTokens: agent.config.maxTokens,
    signal,
  })

  runtime.events.emit({
    type: 'generation.completed',
    ctx: makeCtx(exec, agent),
    model,
    instructions: agent.task,
    input: formatGenInput(agent.task, input),
    output: formatGenOutput(response.output),
    usage: response.usage,
    durationMs: Date.now() - genStart,
    startTime: genStart,
  })

  return handleTurnResponse(response, agent, runtime, exec, agent.turnCount, signal)
}

/**
 * Stream a single turn — yields events as they come from provider
 */
async function* executeTurnStream(
  agent: Agent,
  runtime: RuntimeContext,
  exec: ExecutionContext,
  session: Session,
  signal?: AbortSignal
): AsyncGenerator<ProviderStreamEvent, TurnResult, unknown> {
  if (signal?.aborted) {
    return { continue: false, error: 'Operation aborted' }
  }

  const prepared = await prepareTurnInput(agent, runtime, exec, session)
  if (!prepared.ok) {
    return { continue: false, error: prepared.error }
  }
  const { provider, model, input } = prepared.data

  const genStart = Date.now()

  let response: ProviderResponse | undefined

  for await (const event of provider.stream({
    model,
    instructions: agent.task,
    input,
    tools: agent.config.tools,
    temperature: agent.config.temperature,
    maxTokens: agent.config.maxTokens,
    signal,
  })) {
    yield event

    if (event.type === 'done') {
      response = event.response
    }

    if (event.type === 'error') {
      return { continue: false, error: event.error }
    }
  }

  if (!response) {
    return { continue: false, error: 'Stream ended without response' }
  }

  runtime.events.emit({
    type: 'generation.completed',
    ctx: makeCtx(exec, agent),
    model,
    instructions: agent.task,
    input: formatGenInput(agent.task, input),
    output: formatGenOutput(response.output),
    usage: response.usage,
    durationMs: Date.now() - genStart,
    startTime: genStart,
  })

  return handleTurnResponse(response, agent, runtime, exec, agent.turnCount, signal)
}

// ---- Agent execution loops ----

export async function runAgent(
  agentId: AgentId,
  runtime: RuntimeContext,
  options: RunOptions = {}
): Promise<RunResult> {
  const { maxTurns = 10, signal } = options

  if (signal?.aborted) {
    return { ok: false, status: 'cancelled' }
  }

  const loaded = await loadAgentLoop(agentId, runtime, options)
  if (!loaded.ok) {
    return { ok: false, status: 'failed', error: loaded.error }
  }
  let { agent } = loaded
  const { session, exec } = loaded

  // Start agent if pending (pass traceId to persist it)
  if (agent.status === 'pending') {
    const startResult = startAgent(agent, exec.traceId)
    if (!startResult.ok) {
      return { ok: false, status: 'failed', error: startResult.error }
    }
    agent = await runtime.repositories.agents.update(startResult.agent)
    runtime.events.emit({
      type: 'agent.started',
      ctx: makeCtx(exec, agent),
      model: agent.config.model,
      task: agent.task,
      agentName: exec.agentName,
      userId: exec.userId,
      userInput: exec.userInput,
    })
  }

  // If already waiting, just return current state
  if (agent.status === 'waiting') {
    return { ok: true, status: 'waiting', agent, waitingFor: agent.waitingFor }
  }

  try {
    while (agent.status === 'running' && agent.turnCount < maxTurns) {
      if (signal?.aborted) {
        agent = cancelAgent(agent)
        await runtime.repositories.agents.update(agent)
        runtime.events.emit({ type: 'agent.cancelled', ctx: makeCtx(exec, agent) })
        return { ok: false, status: 'cancelled' }
      }

      runtime.events.emit({
        type: 'turn.started',
        ctx: makeCtx(exec, agent),
        turnCount: agent.turnCount,
      })

      const turnResult = await executeTurn(agent, runtime, exec, session, signal)

      // Handle error
      if ('error' in turnResult) {
        agent = failAgent(agent, turnResult.error)
        await runtime.repositories.agents.update(agent)
        runtime.events.emit({
          type: 'agent.failed',
          ctx: makeCtx(exec, agent),
          error: turnResult.error,
        })
        return { ok: false, status: 'failed', error: turnResult.error }
      }

      // Accumulate usage and increment turn
      let updatedAgent = turnResult.agent
      if (turnResult.usage) {
        updatedAgent = addUsage(updatedAgent, turnResult.usage)
      }
      agent = incrementTurn(updatedAgent)
      await runtime.repositories.agents.update(agent)

      runtime.events.emit({
        type: 'turn.completed',
        ctx: makeCtx(exec, agent),
        turnCount: agent.turnCount,
        usage: turnResult.usage,
      })

      // Handle waiting (non-blocking return)
      if ('waiting' in turnResult) {
        const waitResult = waitForMany(agent, turnResult.waiting)
        if (!waitResult.ok) {
          return { ok: false, status: 'failed', error: waitResult.error }
        }
        agent = await runtime.repositories.agents.update(waitResult.agent)
        runtime.events.emit({
          type: 'agent.waiting',
          ctx: makeCtx(exec, agent),
          waitingFor: turnResult.waiting,
        })
        return { ok: true, status: 'waiting', agent, waitingFor: turnResult.waiting }
      }

      // Continue or done
      if (!turnResult.continue) {
        break
      }
    }

    // Completed
    const durationMs = agent.startedAt 
      ? Date.now() - agent.startedAt.getTime() 
      : 0
    const items = await runtime.repositories.items.listByAgent(agentId)
    const resultText = extractAgentResult(items)

    runtime.events.emit({
      type: 'agent.completed',
      ctx: makeCtx(exec, agent),
      durationMs,
      usage: agent.usage,
      result: resultText,
    })

    return { ok: true, status: 'completed', agent, items }

  } catch (err) {
    if (isAbortError(err)) {
      agent = cancelAgent(agent)
      await runtime.repositories.agents.update(agent)
      runtime.events.emit({ type: 'agent.cancelled', ctx: makeCtx(exec, agent) })
      return { ok: false, status: 'cancelled' }
    }
    throw err
  }
}

/**
 * Deliver a result to a waiting agent.
 *
 * When a child agent completes, the runner automatically propagates
 * the result upward to the parent via this function (see runAgent
 * and handleDelegation).  External systems (humans, async tools) also
 * call this through the HTTP deliver endpoint.
 */
export async function deliverResult(
  agentId: AgentId,
  callId: CallId,
  result: ToolResult,
  runtime: RuntimeContext,
  execution?: ExecutionContext
): Promise<RunResult> {
  let agent = await runtime.repositories.agents.getById(agentId)
  if (!agent) {
    return { ok: false, status: 'failed', error: `Agent not found: ${agentId}` }
  }

  if (agent.status !== 'waiting') {
    return { ok: false, status: 'failed', error: `Agent not waiting: ${agent.status}` }
  }

  // Add result as function_call_output
  await runtime.repositories.items.create(agent.id, {
    type: 'function_call_output',
    callId,
    output: result.ok ? result.output : result.error,
    isError: !result.ok,
  })

  // Remove from waiting list
  const deliverRes = deliverOne(agent, callId)
  if (!deliverRes.ok) {
    return { ok: false, status: 'failed', error: deliverRes.error }
  }
  agent = await runtime.repositories.agents.update(deliverRes.agent)

  const exec = execution ?? createExecutionContext(
    crypto.randomUUID(),
    agent.rootAgentId,
    agent.parentId,
    agent.depth
  )

  runtime.events.emit({
    type: 'agent.resumed',
    ctx: makeCtx(exec, agent),
    deliveredCallId: callId,
    remaining: agent.waitingFor.length,
  })

  // If still waiting, return
  if (agent.waitingFor.length > 0) {
    return { ok: true, status: 'waiting', agent, waitingFor: agent.waitingFor }
  }

  // All delivered, continue execution
  const runResult = await runAgent(agentId, runtime, { execution: exec })

  // ── Auto-propagation: if this agent completed and has a parent, deliver upward ──
  if (
    runResult.ok &&
    runResult.status === 'completed' &&
    agent.parentId &&
    agent.sourceCallId
  ) {
    const output = extractAgentResult(runResult.items)
    log.info({
      childId: agent.id, parentId: agent.parentId,
      sourceCallId: agent.sourceCallId,
    }, 'auto-propagating child result to parent')

    return deliverResult(
      agent.parentId,
      agent.sourceCallId,
      { ok: true, output },
      runtime,
      createExecutionContext(
        exec.traceId,
        agent.rootAgentId,
        undefined, // parent has no parent in this context
        agent.depth - 1,
      ),
    )
  }

  return runResult
}

export async function* runAgentStream(
  agentId: AgentId,
  runtime: RuntimeContext,
  options: RunOptions = {}
): AsyncIterable<ProviderStreamEvent> {
  const { maxTurns = 10, signal } = options

  if (signal?.aborted) {
    yield { type: 'error', error: 'Operation aborted', code: 'ABORTED' }
    return
  }

  const loaded = await loadAgentLoop(agentId, runtime, options)
  if (!loaded.ok) {
    yield { type: 'error', error: loaded.error }
    return
  }
  let { agent } = loaded
  const { session, exec } = loaded

  try {
    // Start agent with traceId
    const startResult = startAgent(agent, exec.traceId)
    if (!startResult.ok) {
      yield { type: 'error', error: startResult.error }
      return
    }
    agent = await runtime.repositories.agents.update(startResult.agent)

    runtime.events.emit({
      type: 'agent.started',
      ctx: makeCtx(exec, agent),
      model: agent.config.model,
      task: agent.task,
      agentName: exec.agentName,
      userId: exec.userId,
      userInput: exec.userInput,
    })

    // Turn loop
    for (let turn = 0; turn < maxTurns; turn++) {
      runtime.events.emit({
        type: 'turn.started',
        ctx: makeCtx(exec, agent),
        turnCount: turn + 1,
      })

      // Stream the turn
      const turnGen = executeTurnStream(agent, runtime, exec, session, signal)
      let turnResult: TurnResult | undefined

      // Yield events and get final result
      while (true) {
        const { value, done } = await turnGen.next()
        if (done) {
          turnResult = value
          break
        }
        yield value
      }

      if (!turnResult) {
        yield { type: 'error', error: 'Turn ended without result' }
        return
      }

      if ('error' in turnResult) {
        yield { type: 'error', error: turnResult.error }
        return
      }

      // Accumulate usage and increment turn
      let updatedAgent = turnResult.agent
      if (turnResult.usage) {
        updatedAgent = addUsage(updatedAgent, turnResult.usage)
      }
      agent = incrementTurn(updatedAgent)
      await runtime.repositories.agents.update(agent)

      runtime.events.emit({
        type: 'turn.completed',
        ctx: makeCtx(exec, agent),
        turnCount: agent.turnCount,
        usage: turnResult.usage,
      })

      // Handle waiting
      if ('waiting' in turnResult) {
        const waitResult = waitForMany(agent, turnResult.waiting)
        if (!waitResult.ok) {
          yield { type: 'error', error: waitResult.error }
          return
        }
        agent = await runtime.repositories.agents.update(waitResult.agent)
        runtime.events.emit({
          type: 'agent.waiting',
          ctx: makeCtx(exec, agent),
          waitingFor: turnResult.waiting,
        })
        // Stream ends here when waiting - caller handles resume
        return
      }

      if (!turnResult.continue) {
        break
      }
    }

    // Completed
    runtime.events.emit({
      type: 'agent.completed',
      ctx: makeCtx(exec, agent),
      durationMs: agent.startedAt ? Date.now() - agent.startedAt.getTime() : 0,
      usage: agent.usage,
    })

  } catch (err) {
    if (isAbortError(err)) {
      agent = cancelAgent(agent)
      await runtime.repositories.agents.update(agent)
      runtime.events.emit({ type: 'agent.cancelled', ctx: makeCtx(exec, agent) })
      yield { type: 'error', error: 'Operation cancelled', code: 'CANCELLED' }
      return
    }
    throw err
  }
}
