import { createMiddleware } from 'hono/factory';
import { config } from '../lib/config.js';
import { err } from '../errors/index.js';
const windows = new Map();
const CLEANUP_INTERVAL_MS = 60_000;
// Periodically prune expired windows to prevent memory leak
setInterval(() => {
    const now = Date.now();
    for (const [key, window] of windows) {
        if (now >= window.resetAt) {
            windows.delete(key);
        }
    }
}, CLEANUP_INTERVAL_MS).unref();
/**
 * Fixed-window rate limiter keyed by authenticated user ID.
 * Must be registered AFTER bearerAuth so ctx.get('user') is available.
 */
export const rateLimiter = createMiddleware(async (ctx, next) => {
    const user = ctx.get('user');
    const key = user.id;
    const now = Date.now();
    const windowMs = 60_000; // 1-minute window
    const limit = config.rateLimitRpm;
    let window = windows.get(key);
    if (!window || now >= window.resetAt) {
        window = { count: 0, resetAt: now + windowMs };
        windows.set(key, window);
    }
    window.count++;
    const remaining = Math.max(0, limit - window.count);
    const resetSeconds = Math.ceil((window.resetAt - now) / 1000);
    ctx.header('X-RateLimit-Limit', String(limit));
    ctx.header('X-RateLimit-Remaining', String(remaining));
    ctx.header('X-RateLimit-Reset', String(resetSeconds));
    if (window.count > limit) {
        ctx.header('Retry-After', String(resetSeconds));
        throw err.rateLimited(`Rate limit exceeded. Try again in ${resetSeconds}s`);
    }
    await next();
});
