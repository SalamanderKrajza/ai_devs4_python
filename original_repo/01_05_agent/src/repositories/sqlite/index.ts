/**
 * SQLite repository implementation using Drizzle ORM
 */
import { drizzle } from 'drizzle-orm/libsql'
import { eq, and, sql } from 'drizzle-orm'
import { createClient } from '@libsql/client'
import { users, sessions, agents, items } from './schema.js'
import type { Repositories, UserRepository, SessionRepository, AgentRepository, ItemRepository, CreateItemInput } from '../types.js'
import type { User, UserId, CreateUserInput, Session, SessionId, Agent, AgentId, CreateAgentInput, Item, ItemId, CallId } from '../../domain/index.js'
import { createUser, createSession, createAgent as createAgentDomain } from '../../domain/index.js'

type Database = ReturnType<typeof drizzle>

// ─────────────────────────────────────────────────────────────────────────────
// Mappers: DB rows ↔ Domain objects
// ─────────────────────────────────────────────────────────────────────────────

function toUser(row: typeof users.$inferSelect): User {
  return {
    id: row.id,
    email: row.email,
    apiKeyHash: row.apiKeyHash,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt ?? undefined,
  }
}

function toSession(row: typeof sessions.$inferSelect): Session {
  return {
    id: row.id,
    userId: row.userId ?? undefined,
    rootAgentId: row.rootAgentId ?? undefined,
    title: row.title ?? undefined,
    summary: row.summary ?? undefined,
    status: row.status,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt ?? undefined,
  }
}

function toAgent(row: typeof agents.$inferSelect): Agent {
  return {
    id: row.id,
    sessionId: row.sessionId,
    rootAgentId: row.rootAgentId,
    parentId: row.parentId ?? undefined,
    sourceCallId: row.sourceCallId ?? undefined,
    depth: row.depth,
    task: row.task,
    config: row.config,
    status: row.status,
    waitingFor: row.waitingFor,
    result: row.result ?? undefined,
    error: row.error ?? undefined,
    turnCount: row.turnCount,
    createdAt: row.createdAt,
    startedAt: row.startedAt ?? undefined,
    completedAt: row.completedAt ?? undefined,
  }
}

function toItem(row: typeof items.$inferSelect): Item {
  const base = {
    id: row.id,
    agentId: row.agentId,
    sequence: row.sequence,
    createdAt: row.createdAt,
  }

  switch (row.type) {
    case 'message':
      return {
        ...base,
        type: 'message',
        role: row.role as 'user' | 'assistant' | 'system',
        content: row.content as string | Array<{ type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string } | { type: 'image'; uri: string; mimeType: string }>,
      }
    case 'function_call':
      return {
        ...base,
        type: 'function_call',
        callId: row.callId!,
        name: row.name!,
        arguments: row.arguments as Record<string, unknown>,
      }
    case 'function_call_output':
      return {
        ...base,
        type: 'function_call_output',
        callId: row.callId!,
        output: row.output!,
        isError: row.isError ?? undefined,
      }
    case 'reasoning':
      return {
        ...base,
        type: 'reasoning',
        summary: row.summary ?? undefined,
        signature: row.signature ?? undefined,
      }
    default:
      throw new Error(`Unknown item type: ${row.type}`)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// User Repository
// ─────────────────────────────────────────────────────────────────────────────

function createUserRepo(db: Database): UserRepository {
  return {
    async create(input: CreateUserInput) {
      const user = createUser(crypto.randomUUID(), input)
      await db.insert(users).values({
        id: user.id,
        email: user.email,
        apiKeyHash: user.apiKeyHash,
        createdAt: user.createdAt,
      })
      return user
    },

    async getById(id) {
      const rows = await db.select().from(users).where(eq(users.id, id)).limit(1)
      return rows[0] ? toUser(rows[0]) : undefined
    },

    async getByEmail(email) {
      const rows = await db.select().from(users).where(eq(users.email, email)).limit(1)
      return rows[0] ? toUser(rows[0]) : undefined
    },

    async getByApiKeyHash(apiKeyHash) {
      const rows = await db.select().from(users).where(eq(users.apiKeyHash, apiKeyHash)).limit(1)
      return rows[0] ? toUser(rows[0]) : undefined
    },

    async update(user) {
      await db.update(users)
        .set({
          email: user.email,
          apiKeyHash: user.apiKeyHash,
          updatedAt: new Date(),
        })
        .where(eq(users.id, user.id))
      return user
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Session Repository
// ─────────────────────────────────────────────────────────────────────────────

function createSessionRepo(db: Database): SessionRepository {
  return {
    async create(userId, title) {
      const session = createSession(crypto.randomUUID(), userId, title)
      await db.insert(sessions).values({
        id: session.id,
        userId: session.userId,
        title: session.title,
        status: session.status,
        createdAt: session.createdAt,
      })
      return session
    },

    async getById(id) {
      const rows = await db.select().from(sessions).where(eq(sessions.id, id)).limit(1)
      return rows[0] ? toSession(rows[0]) : undefined
    },

    async listByUser(userId) {
      const rows = await db.select().from(sessions)
        .where(eq(sessions.userId, userId))
        .orderBy(sessions.createdAt)
      return rows.map(toSession)
    },

    async update(session) {
      await db.update(sessions)
        .set({
          rootAgentId: session.rootAgentId,
          title: session.title,
          status: session.status,
          updatedAt: new Date(),
        })
        .where(eq(sessions.id, session.id))
      return session
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent Repository
// ─────────────────────────────────────────────────────────────────────────────

function createAgentRepo(db: Database): AgentRepository {
  return {
    async create(input) {
      const agent = createAgentDomain(crypto.randomUUID(), input)
      await db.insert(agents).values({
        id: agent.id,
        sessionId: agent.sessionId,
        rootAgentId: agent.rootAgentId,
        parentId: agent.parentId,
        sourceCallId: agent.sourceCallId,
        depth: agent.depth,
        task: agent.task,
        config: agent.config,
        status: agent.status,
        waitingFor: agent.waitingFor,
        turnCount: agent.turnCount,
        createdAt: agent.createdAt,
      })
      return agent
    },

    async getById(id) {
      const rows = await db.select().from(agents).where(eq(agents.id, id)).limit(1)
      return rows[0] ? toAgent(rows[0]) : undefined
    },

    async update(agent) {
      await db.update(agents)
        .set({
          status: agent.status,
          waitingFor: agent.waitingFor,
          result: agent.result as string | null,
          error: agent.error,
          turnCount: agent.turnCount,
          startedAt: agent.startedAt,
          completedAt: agent.completedAt,
        })
        .where(eq(agents.id, agent.id))
      return agent
    },

    async listBySession(sessionId) {
      const rows = await db.select().from(agents)
        .where(eq(agents.sessionId, sessionId))
        .orderBy(agents.createdAt)
      return rows.map(toAgent)
    },

    async listByParent(parentId) {
      const rows = await db.select().from(agents)
        .where(eq(agents.parentId, parentId))
        .orderBy(agents.createdAt)
      return rows.map(toAgent)
    },

    async findWaitingForCall(callId) {
      // Push JSON filtering into SQLite — avoids fetching all waiting agents
      const rows = await db.select().from(agents)
        .where(and(
          eq(agents.status, 'waiting'),
          sql`EXISTS (SELECT 1 FROM json_each(${agents.waitingFor}) WHERE json_extract(value, '$.callId') = ${callId})`,
        ))
        .limit(1)
      return rows[0] ? toAgent(rows[0]) : undefined
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Item Repository
// ─────────────────────────────────────────────────────────────────────────────

function createItemRepo(db: Database): ItemRepository {
  return {
    async create(agentId: AgentId, input: CreateItemInput): Promise<Item> {
      // Atomically get next sequence via MAX (single aggregate, no full scan)
      const [row] = await db.select({ maxSeq: sql<number>`coalesce(max(${items.sequence}), 0)` })
        .from(items)
        .where(eq(items.agentId, agentId))
      const sequence = (row?.maxSeq ?? 0) + 1

      const id = crypto.randomUUID()
      const createdAt = new Date()

      const base = { id, agentId, sequence, createdAt, type: input.type }

      if (input.type === 'message') {
        await db.insert(items).values({
          ...base,
          role: input.role,
          content: input.content,
        })
        return { ...base, type: 'message', role: input.role, content: input.content }
      }

      if (input.type === 'function_call') {
        await db.insert(items).values({
          ...base,
          callId: input.callId,
          name: input.name,
          arguments: input.arguments,
        })
        return { ...base, type: 'function_call', callId: input.callId, name: input.name, arguments: input.arguments }
      }

      if (input.type === 'function_call_output') {
        await db.insert(items).values({
          ...base,
          callId: input.callId,
          output: input.output,
          isError: input.isError,
        })
        return { ...base, type: 'function_call_output', callId: input.callId, output: input.output, isError: input.isError }
      }

      if (input.type === 'reasoning') {
        await db.insert(items).values({
          ...base,
          summary: input.summary,
          signature: input.signature,
        })
        return { ...base, type: 'reasoning', summary: input.summary, signature: input.signature }
      }

      throw new Error(`Unknown item type: ${(input as { type: string }).type}`)
    },

    async getById(id) {
      const rows = await db.select().from(items).where(eq(items.id, id)).limit(1)
      return rows[0] ? toItem(rows[0]) : undefined
    },

    async listByAgent(agentId) {
      const rows = await db.select().from(items)
        .where(eq(items.agentId, agentId))
        .orderBy(items.sequence)
      return rows.map(toItem)
    },

    async getOutputByCallId(callId) {
      const rows = await db.select().from(items)
        .where(and(
          eq(items.callId, callId),
          eq(items.type, 'function_call_output')
        ))
        .limit(1)
      return rows[0] ? toItem(rows[0]) : undefined
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

export interface SQLiteConfig {
  url: string
  authToken?: string
}

export async function createSQLiteRepositories(config: SQLiteConfig): Promise<Repositories> {
  const client = createClient({
    url: config.url,
    authToken: config.authToken
  })
  const db = drizzle(client)

  // SQLite best-practice pragmas
  await client.execute('PRAGMA journal_mode=WAL')
  await client.execute('PRAGMA synchronous=NORMAL')
  await client.execute('PRAGMA foreign_keys=ON')
  await client.execute('PRAGMA busy_timeout=5000')

  return {
    users: createUserRepo(db),
    sessions: createSessionRepo(db),
    agents: createAgentRepo(db),
    items: createItemRepo(db),
    async ping() {
      try {
        await db.run(sql`SELECT 1`)
        return true
      } catch {
        return false
      }
    },
  }
}
