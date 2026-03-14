/**
 * Langfuse observability via OpenTelemetry
 *
 * When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set,
 * initializes a NodeSDK with the Langfuse span processor.
 * All helper functions gracefully return undefined when disabled.
 */
import { NodeSDK } from '@opentelemetry/sdk-node'
import { resourceFromAttributes } from '@opentelemetry/resources'
import { LangfuseSpanProcessor } from '@langfuse/otel'
import { startObservation } from '@langfuse/tracing'
import type { SpanContext } from '@opentelemetry/api'
import { config } from './config.js'
import { logger } from './logger.js'

const log = logger.child({ name: 'tracing' })

let sdk: NodeSDK | undefined

export function isTracingEnabled(): boolean {
  return !!(config.langfusePublicKey && config.langfuseSecretKey)
}

export function initTracing(): void {
  if (!isTracingEnabled()) {
    log.info('langfuse disabled — set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable')
    return
  }

  const processor = new LangfuseSpanProcessor({
    publicKey: config.langfusePublicKey,
    secretKey: config.langfuseSecretKey,
    baseUrl: config.langfuseBaseUrl,
    environment: config.nodeEnv,
  })

  sdk = new NodeSDK({
    spanProcessors: [processor],
    resource: resourceFromAttributes({ 'service.name': 'agent' }),
    autoDetectResources: false,
  })
  sdk.start()

  log.info(
    { baseUrl: config.langfuseBaseUrl ?? 'https://cloud.langfuse.com' },
    'langfuse tracing enabled',
  )
}

export async function shutdownTracing(): Promise<void> {
  if (!sdk) return
  try {
    await sdk.shutdown()
    log.info('langfuse tracing flushed and shut down')
  } catch (err) {
    log.error({ err }, 'langfuse shutdown error')
  }
}

// ── Observation factories ───────────────────────────────────────────────────

export type { SpanContext }

export function getSpanContext(
  obs?: { otelSpan: { spanContext(): SpanContext } },
): SpanContext | undefined {
  return obs?.otelSpan.spanContext()
}

export function traceAgent(
  name: string,
  opts: {
    input?: unknown
    metadata?: Record<string, unknown>
    parentSpanContext?: SpanContext
    startTime?: Date
  },
) {
  if (!isTracingEnabled()) return undefined
  return startObservation(
    name,
    { input: opts.input, metadata: opts.metadata },
    { asType: 'agent', parentSpanContext: opts.parentSpanContext, startTime: opts.startTime },
  )
}

export function traceGeneration(
  name: string,
  opts: {
    model: string
    input?: unknown
    modelParameters?: Record<string, string | number>
    metadata?: Record<string, unknown>
    parentSpanContext?: SpanContext
    startTime?: Date
  },
) {
  if (!isTracingEnabled()) return undefined
  return startObservation(
    name,
    {
      input: opts.input,
      model: opts.model,
      modelParameters: opts.modelParameters,
      metadata: opts.metadata,
    },
    { asType: 'generation', parentSpanContext: opts.parentSpanContext, startTime: opts.startTime },
  )
}

export function traceTool(
  name: string,
  opts: {
    input?: unknown
    metadata?: Record<string, unknown>
    parentSpanContext?: SpanContext
    startTime?: Date
  },
) {
  if (!isTracingEnabled()) return undefined
  return startObservation(
    name,
    { input: opts.input, metadata: opts.metadata },
    { asType: 'tool', parentSpanContext: opts.parentSpanContext, startTime: opts.startTime },
  )
}
