/**
 * Repository interfaces
 */
import type {
  User,
  UserId,
  CreateUserInput,
  Session,
  SessionId,
  Agent,
  AgentId,
  CreateAgentInput,
  Item,
  MessageItem,
  FunctionCallItem,
  FunctionCallOutputItem,
  ReasoningItem,
  ItemId,
  CallId,
} from '../domain/index.js'

export interface UserRepository {
  create(input: CreateUserInput): Promise<User>
  getById(id: UserId): Promise<User | undefined>
  getByEmail(email: string): Promise<User | undefined>
  getByApiKeyHash(apiKeyHash: string): Promise<User | undefined>
  update(user: User): Promise<User>
}

export interface SessionRepository {
  create(userId?: UserId, title?: string): Promise<Session>
  getById(id: SessionId): Promise<Session | undefined>
  listByUser(userId: UserId): Promise<Session[]>
  update(session: Session): Promise<Session>
}

export interface AgentRepository {
  create(input: CreateAgentInput): Promise<Agent>
  getById(id: AgentId): Promise<Agent | undefined>
  update(agent: Agent): Promise<Agent>
  listBySession(sessionId: SessionId): Promise<Agent[]>
  listByParent(parentId: AgentId): Promise<Agent[]>
  findWaitingForCall(callId: CallId): Promise<Agent | undefined>
}

// Type-safe item creation inputs
export type CreateMessageInput = Omit<MessageItem, 'id' | 'agentId' | 'sequence' | 'createdAt'>
export type CreateFunctionCallInput = Omit<FunctionCallItem, 'id' | 'agentId' | 'sequence' | 'createdAt'>
export type CreateFunctionCallOutputInput = Omit<FunctionCallOutputItem, 'id' | 'agentId' | 'sequence' | 'createdAt'>
export type CreateReasoningInput = Omit<ReasoningItem, 'id' | 'agentId' | 'sequence' | 'createdAt'>

export type CreateItemInput = 
  | CreateMessageInput 
  | CreateFunctionCallInput 
  | CreateFunctionCallOutputInput 
  | CreateReasoningInput

export interface ItemRepository {
  create(agentId: AgentId, input: CreateItemInput): Promise<Item>
  getById(id: ItemId): Promise<Item | undefined>
  listByAgent(agentId: AgentId): Promise<Item[]>
  getOutputByCallId(callId: CallId): Promise<Item | undefined>
}

export interface Repositories {
  users: UserRepository
  sessions: SessionRepository
  agents: AgentRepository
  items: ItemRepository
  /** Lightweight connectivity check — resolves true if the store is reachable. */
  ping(): Promise<boolean>
}
