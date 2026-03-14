import { z } from 'zod'

const contentPartSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('text'), text: z.string() }),
  z.object({ type: z.literal('image'), data: z.string().optional(), uri: z.string().optional(), mimeType: z.string() }),
])

const inputItemSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('message'), role: z.enum(['user', 'assistant', 'system']), content: z.union([z.string(), z.array(contentPartSchema)]) }),
  z.object({ type: z.literal('function_result'), callId: z.string(), name: z.string(), output: z.string() }),
])

const toolSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('function'),
    name: z.string(),
    description: z.string(),
    parameters: z.record(z.string(), z.unknown()),
  }),
  z.object({ type: z.literal('web_search') }),
])

export const chatRequestSchema = z.object({
  // Use a predefined agent template (loads instructions and tools from workspace)
  agent: z.string().optional(),
  // Or provide custom config (these override agent template if both specified)
  model: z.string().optional(),
  instructions: z.string().optional(),
  input: z.union([z.string(), z.array(inputItemSchema)]),
  tools: z.array(toolSchema).optional(),
  stream: z.boolean().default(false),
  temperature: z.number().min(0).max(2).optional(),
  maxTokens: z.number().positive().optional(),
  sessionId: z.string().optional(),
})

export type ChatRequest = z.infer<typeof chatRequestSchema>

// Deliver result to waiting agent
export const deliverRequestSchema = z.object({
  callId: z.string(),
  output: z.string(),
  isError: z.boolean().default(false),
})

export type DeliverRequest = z.infer<typeof deliverRequestSchema>
