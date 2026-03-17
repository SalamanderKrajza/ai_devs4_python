import { createMiddleware } from 'hono/factory';
import { err } from '../errors/index.js';
/**
 * Hash API key using SHA-256 for secure storage/lookup.
 */
export async function hashApiKey(apiKey) {
    const encoder = new TextEncoder();
    const data = encoder.encode(apiKey);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}
/**
 * Bearer token authentication middleware.
 * Validates Authorization header against user API keys in database.
 * Sets ctx.var.user on successful authentication.
 */
export const bearerAuth = createMiddleware(async (ctx, next) => {
    const authHeader = ctx.req.header('Authorization');
    if (!authHeader) {
        throw err.unauthorized('Missing Authorization header');
    }
    if (!authHeader.startsWith('Bearer ')) {
        throw err.unauthorized('Invalid Authorization format. Expected: Bearer <token>');
    }
    const apiKey = authHeader.slice(7); // Remove 'Bearer ' prefix
    if (!apiKey) {
        throw err.unauthorized('Missing token');
    }
    const runtime = ctx.get('runtime');
    if (!runtime) {
        throw err.internal('Runtime not initialized');
    }
    const apiKeyHash = await hashApiKey(apiKey);
    const user = await runtime.repositories.users.getByApiKeyHash(apiKeyHash);
    if (!user) {
        throw err.unauthorized('Invalid API key');
    }
    ctx.set('user', user);
    await next();
});
