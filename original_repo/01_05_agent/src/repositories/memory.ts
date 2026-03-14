/**
 * In-memory repository implementations
 */
import type { Repositories, UserRepository, SessionRepository, AgentRepository, ItemRepository, CreateItemInput } from './types.js'
import type { User, UserId, CreateUserInput, Session, SessionId, Agent, AgentId, CreateAgentInput, Item, ItemId, CallId } from '../domain/index.js'
import { createUser, createSession, createAgent, createItem, isFunctionCallOutput } from '../domain/index.js'

function createUserRepo(): UserRepository {
  const store = new Map<UserId, User>()

  return {
    async create(input: CreateUserInput) {
      const user = createUser(crypto.randomUUID(), input)
      store.set(user.id, user)
      return user
    },
    async getById(id) {
      return store.get(id)
    },
    async getByEmail(email) {
      for (const user of store.values()) {
        if (user.email === email) return user
      }
      return undefined
    },
    async getByApiKeyHash(apiKeyHash) {
      for (const user of store.values()) {
        if (user.apiKeyHash === apiKeyHash) return user
      }
      return undefined
    },
    async update(user) {
      store.set(user.id, user)
      return user
    },
  }
}

function createSessionRepo(): SessionRepository {
  const store = new Map<SessionId, Session>()

  return {
    async create(userId, title) {
      const session = createSession(crypto.randomUUID(), userId, title)
      store.set(session.id, session)
      return session
    },
    async getById(id) {
      return store.get(id)
    },
    async listByUser(userId) {
      return Array.from(store.values())
        .filter(s => s.userId === userId)
        .sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime())
    },
    async update(session) {
      store.set(session.id, session)
      return session
    },
  }
}

function createAgentRepo(): AgentRepository {
  const store = new Map<AgentId, Agent>()

  return {
    async create(input) {
      const agent = createAgent(crypto.randomUUID(), input)
      store.set(agent.id, agent)
      return agent
    },
    async getById(id) {
      return store.get(id)
    },
    async update(agent) {
      store.set(agent.id, agent)
      return agent
    },
    async listBySession(sessionId) {
      return Array.from(store.values())
        .filter(a => a.sessionId === sessionId)
        .sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime())
    },
    async listByParent(parentId) {
      return Array.from(store.values())
        .filter(a => a.parentId === parentId)
        .sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime())
    },
    async findWaitingForCall(callId) {
      for (const agent of store.values()) {
        if (agent.status === 'waiting' && agent.waitingFor.some(w => w.callId === callId)) {
          return agent
        }
      }
      return undefined
    },
  }
}

function createItemRepo(): ItemRepository {
  const store = new Map<ItemId, Item>()
  const sequences = new Map<AgentId, number>()

  function nextSeq(agentId: AgentId): number {
    const current = sequences.get(agentId) ?? 0
    sequences.set(agentId, current + 1)
    return current + 1
  }

  return {
    async create(agentId: AgentId, input: CreateItemInput): Promise<Item> {
      const item = createItem(
        crypto.randomUUID(),
        agentId,
        nextSeq(agentId),
        input
      )
      store.set(item.id, item)
      return item
    },
    async getById(id) {
      return store.get(id)
    },
    async listByAgent(agentId) {
      return Array.from(store.values())
        .filter(i => i.agentId === agentId)
        .sort((a, b) => a.sequence - b.sequence)
    },
    async getOutputByCallId(callId) {
      for (const item of store.values()) {
        if (isFunctionCallOutput(item) && item.callId === callId) {
          return item
        }
      }
      return undefined
    },
  }
}

export function createMemoryRepositories(): Repositories {
  return {
    users: createUserRepo(),
    sessions: createSessionRepo(),
    agents: createAgentRepo(),
    items: createItemRepo(),
    async ping() { return true },
  }
}
