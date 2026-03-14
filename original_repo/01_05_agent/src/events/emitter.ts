/**
 * Event emitter — thin wrapper over Node's EventEmitter
 */
import { EventEmitter } from 'node:events'
import type { AgentEvent, AgentEventType } from './types.js'
import { logger } from '../lib/logger.js'

const log = logger.child({ name: 'events' })

export type EventHandler<T extends AgentEvent = AgentEvent> = (event: T) => void

export interface AgentEventEmitter {
  emit(event: AgentEvent): void
  on<T extends AgentEventType>(type: T, handler: EventHandler<Extract<AgentEvent, { type: T }>>): () => void
  onAny(handler: EventHandler): () => void
}

export function createEventEmitter(): AgentEventEmitter {
  const emitter = new EventEmitter()

  const safeCall = (handler: EventHandler, event: AgentEvent) => {
    try {
      handler(event)
    } catch (err) {
      log.error({ event: event.type, err }, 'handler error')
    }
  }

  return {
    emit(event) {
      emitter.emit(event.type, event)
      emitter.emit('*', event)
    },

    on(type, handler) {
      const wrapped = (e: AgentEvent) => safeCall(handler as EventHandler, e)
      emitter.on(type, wrapped)
      return () => emitter.off(type, wrapped)
    },

    onAny(handler) {
      const wrapped = (e: AgentEvent) => safeCall(handler, e)
      emitter.on('*', wrapped)
      return () => emitter.off('*', wrapped)
    },
  }
}
