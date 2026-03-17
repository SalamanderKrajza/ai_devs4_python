import { sqliteTable, text, integer, index, uniqueIndex } from 'drizzle-orm/sqlite-core';
// ─────────────────────────────────────────────────────────────────────────────
// Users
// ─────────────────────────────────────────────────────────────────────────────
export const users = sqliteTable('users', {
    id: text('id').primaryKey(),
    email: text('email').notNull(),
    apiKeyHash: text('api_key_hash').notNull(),
    createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
    updatedAt: integer('updated_at', { mode: 'timestamp' }),
}, (table) => [
    uniqueIndex('users_email_idx').on(table.email),
    uniqueIndex('users_api_key_hash_idx').on(table.apiKeyHash),
]);
// ─────────────────────────────────────────────────────────────────────────────
// Sessions
// ─────────────────────────────────────────────────────────────────────────────
export const sessions = sqliteTable('sessions', {
    id: text('id').primaryKey(),
    userId: text('user_id'),
    rootAgentId: text('root_agent_id'),
    title: text('title'),
    summary: text('summary'),
    status: text('status', { enum: ['active', 'archived'] }).notNull().default('active'),
    createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
    updatedAt: integer('updated_at', { mode: 'timestamp' }),
}, (table) => [
    index('sessions_user_idx').on(table.userId),
]);
// ─────────────────────────────────────────────────────────────────────────────
// Agents
// ─────────────────────────────────────────────────────────────────────────────
export const agents = sqliteTable('agents', {
    id: text('id').primaryKey(),
    sessionId: text('session_id').notNull().references(() => sessions.id),
    // Hierarchy
    rootAgentId: text('root_agent_id').notNull(),
    parentId: text('parent_id'),
    sourceCallId: text('source_call_id'),
    depth: integer('depth').notNull().default(0),
    // Task & config
    task: text('task').notNull(),
    config: text('config', { mode: 'json' }).notNull().$type(),
    // State
    status: text('status', {
        enum: ['pending', 'running', 'waiting', 'completed', 'failed', 'cancelled']
    }).notNull().default('pending'),
    waitingFor: text('waiting_for', { mode: 'json' }).notNull().default('[]').$type(),
    result: text('result', { mode: 'json' }),
    error: text('error'),
    turnCount: integer('turn_count').notNull().default(0),
    // Timestamps
    createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
    startedAt: integer('started_at', { mode: 'timestamp' }),
    completedAt: integer('completed_at', { mode: 'timestamp' }),
}, (table) => [
    index('agents_session_idx').on(table.sessionId),
    index('agents_parent_idx').on(table.parentId),
    index('agents_status_idx').on(table.status),
]);
// ─────────────────────────────────────────────────────────────────────────────
// Items (polymorphic conversation entries)
// ─────────────────────────────────────────────────────────────────────────────
export const items = sqliteTable('items', {
    id: text('id').primaryKey(),
    agentId: text('agent_id').notNull().references(() => agents.id),
    sequence: integer('sequence').notNull(),
    type: text('type', {
        enum: ['message', 'function_call', 'function_call_output', 'reasoning']
    }).notNull(),
    // Message fields
    role: text('role', { enum: ['user', 'assistant', 'system'] }),
    content: text('content', { mode: 'json' }).$type(),
    // Function call fields
    callId: text('call_id'),
    name: text('name'),
    arguments: text('arguments', { mode: 'json' }).$type(),
    // Function call output fields
    output: text('output'),
    isError: integer('is_error', { mode: 'boolean' }),
    // Reasoning fields
    summary: text('summary'),
    signature: text('signature'),
    // Timestamps
    createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
}, (table) => [
    index('items_agent_seq_idx').on(table.agentId, table.sequence),
    index('items_call_id_idx').on(table.callId),
]);
