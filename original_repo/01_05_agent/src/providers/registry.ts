/**
 * Provider registry — simple map of name -> provider
 */
import type { Provider } from './types.js'

const providers = new Map<string, Provider>()

export function registerProvider(provider: Provider): void {
  providers.set(provider.name, provider)
}

export function getProvider(name: string): Provider | undefined {
  return providers.get(name)
}

export function listProviders(): string[] {
  return Array.from(providers.keys())
}

/** Parse "openrouter:gpt-5.4" -> { provider: "openrouter", model: "gpt-5.4" } */
export function parseModelString(modelString: string): { providerName: string; model: string } | undefined {
  const idx = modelString.indexOf(':')
  if (idx === -1) return undefined
  return {
    providerName: modelString.slice(0, idx),
    model: modelString.slice(idx + 1),
  }
}

/** Resolve model string to provider instance */
export function resolveProvider(modelString: string): { provider: Provider; model: string } | undefined {
  const parsed = parseModelString(modelString)
  if (!parsed) return undefined

  const provider = providers.get(parsed.providerName)
  if (!provider) return undefined

  return { provider, model: parsed.model }
}
