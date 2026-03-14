/**
 * Session — user-facing conversation container
 */
import type { SessionId, UserId, AgentId, SessionStatus } from './types.js'

export interface Session {
  id: SessionId
  userId?: UserId
  rootAgentId?: AgentId
  title?: string
  summary?: string
  status: SessionStatus
  createdAt: Date
  updatedAt?: Date
}

export function createSession(id: SessionId, userId?: UserId, title?: string): Session {
  return {
    id,
    userId,
    title,
    status: 'active',
    createdAt: new Date(),
  }
}
