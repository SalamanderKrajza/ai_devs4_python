import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod/v4';
const __dir = path.dirname(fileURLToPath(import.meta.url));
const ROOT_ENV_FILE = path.resolve(__dir, '../../../.env');
const LOCAL_ENV_FILE = path.resolve(__dir, '../../.env');
const loadEnvFile = (file) => {
    if (!existsSync(file) || typeof process.loadEnvFile !== 'function')
        return;
    try {
        process.loadEnvFile(file);
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to load .env file (${file}): ${message}`);
    }
};
// Local project .env first — process.loadEnvFile never overrides already-set vars,
// so local keys take priority; root fills in anything not set locally.
loadEnvFile(LOCAL_ENV_FILE);
loadEnvFile(ROOT_ENV_FILE);
const envSchema = z.object({
    PORT: z.coerce.number().int().min(1).max(65535).default(3000),
    HOST: z.string().min(1).default('127.0.0.1'),
    // Request limits
    BODY_LIMIT: z.coerce.number().int().positive().default(1024 * 1024), // 1MB
    TIMEOUT_MS: z.coerce.number().int().positive().default(60_000), // 60s
    // CORS
    CORS_ORIGIN: z.string().min(1).default('*'),
    // Provider keys — at least one should be set (warned at runtime)
    OPENAI_API_KEY: z.string().optional(),
    OPENROUTER_API_KEY: z.string().optional(),
    GEMINI_API_KEY: z.string().optional(),
    AI_PROVIDER: z.enum(['openai', 'openrouter']).optional(),
    OPENROUTER_HTTP_REFERER: z.string().optional(),
    OPENROUTER_APP_NAME: z.string().optional(),
    // Default model — must be "provider:model"
    DEFAULT_MODEL: z.string().regex(/^[a-z]+:.+$/, 'Must be "provider:model" format').optional(),
    // Agent
    AGENT_MAX_TURNS: z.coerce.number().int().positive().default(10),
    // Database
    DATABASE_URL: z.string().min(1).default('file:.data/agent.db'),
    DATABASE_AUTH_TOKEN: z.string().optional(),
    // Workspace
    WORKSPACE_PATH: z.string().min(1).default('./workspace'),
    // Logging / env
    LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace']).default('info'),
    NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
    // Rate limiting
    RATE_LIMIT_RPM: z.coerce.number().int().positive().default(60),
    // Graceful shutdown
    SHUTDOWN_TIMEOUT_MS: z.coerce.number().int().positive().default(10_000),
    // Langfuse observability (optional)
    LANGFUSE_PUBLIC_KEY: z.string().optional(),
    LANGFUSE_SECRET_KEY: z.string().optional(),
    LANGFUSE_BASE_URL: z.string().optional(),
});
const parsed = envSchema.parse(process.env);
const hasOpenAIKey = Boolean(parsed.OPENAI_API_KEY?.trim());
const hasOpenRouterKey = Boolean(parsed.OPENROUTER_API_KEY?.trim());
// Production guard: refuse wildcard CORS
if (parsed.NODE_ENV === 'production' && parsed.CORS_ORIGIN === '*') {
    throw new Error('CORS_ORIGIN=* is not allowed in production. Set explicit origins.');
}
const defaultProvider = (() => {
    if (parsed.AI_PROVIDER) {
        if (parsed.AI_PROVIDER === 'openai' && !hasOpenAIKey) {
            throw new Error('AI_PROVIDER=openai requires OPENAI_API_KEY');
        }
        if (parsed.AI_PROVIDER === 'openrouter' && !hasOpenRouterKey) {
            throw new Error('AI_PROVIDER=openrouter requires OPENROUTER_API_KEY');
        }
        return parsed.AI_PROVIDER;
    }
    if (hasOpenAIKey)
        return 'openai';
    if (hasOpenRouterKey)
        return 'openrouter';
    return 'openai';
})();
export const config = {
    port: parsed.PORT,
    host: parsed.HOST,
    bodyLimit: parsed.BODY_LIMIT,
    timeoutMs: parsed.TIMEOUT_MS,
    corsOrigin: parsed.CORS_ORIGIN,
    openaiApiKey: parsed.OPENAI_API_KEY,
    openRouterApiKey: parsed.OPENROUTER_API_KEY,
    geminiApiKey: parsed.GEMINI_API_KEY,
    defaultProvider,
    openRouterHeaders: {
        ...(parsed.OPENROUTER_HTTP_REFERER
            ? { 'HTTP-Referer': parsed.OPENROUTER_HTTP_REFERER }
            : {}),
        ...(parsed.OPENROUTER_APP_NAME
            ? { 'X-Title': parsed.OPENROUTER_APP_NAME }
            : {}),
    },
    defaultModel: parsed.DEFAULT_MODEL ?? `${defaultProvider}:gpt-5.4`,
    agentMaxTurns: parsed.AGENT_MAX_TURNS,
    databaseUrl: parsed.DATABASE_URL,
    databaseAuthToken: parsed.DATABASE_AUTH_TOKEN,
    workspacePath: parsed.WORKSPACE_PATH,
    logLevel: parsed.LOG_LEVEL,
    nodeEnv: parsed.NODE_ENV,
    rateLimitRpm: parsed.RATE_LIMIT_RPM,
    shutdownTimeoutMs: parsed.SHUTDOWN_TIMEOUT_MS,
    langfusePublicKey: parsed.LANGFUSE_PUBLIC_KEY,
    langfuseSecretKey: parsed.LANGFUSE_SECRET_KEY,
    langfuseBaseUrl: parsed.LANGFUSE_BASE_URL,
};
