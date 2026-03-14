/**
 * User — multi-tenant authentication entity
 */
import type { UserId } from './types.js'

export interface User {
  id: UserId
  email: string
  apiKeyHash: string
  createdAt: Date
  updatedAt?: Date
}

export interface CreateUserInput {
  email: string
  apiKeyHash: string
}

export function createUser(id: UserId, input: CreateUserInput): User {
  return {
    id,
    email: input.email,
    apiKeyHash: input.apiKeyHash,
    createdAt: new Date(),
  }
}
