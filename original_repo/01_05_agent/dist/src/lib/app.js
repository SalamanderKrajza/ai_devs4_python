import { Hono } from 'hono';
import { secureHeaders } from 'hono/secure-headers';
import { cors } from 'hono/cors';
import { bodyLimit } from 'hono/body-limit';
import { timeout } from 'hono/timeout';
import { requestId } from 'hono/request-id';
import { createMiddleware } from 'hono/factory';
import { config } from './config.js';
import { logger } from './logger.js';
import { getRuntime, hasRuntime } from './runtime.js';
import { errorHandler, notFoundHandler, err } from '../errors/index.js';
import { bearerAuth, rateLimiter } from '../middleware/index.js';
import { routes } from '../routes/index.js';
import { mcp as mcpCallbackRoutes } from '../routes/mcp.js';
const httpLog = logger.child({ name: 'http' });
export const app = new Hono();
app.use(requestId());
// Structured HTTP request logging
app.use(createMiddleware(async (c, next) => {
    const start = Date.now();
    await next();
    const ms = Date.now() - start;
    const status = c.res.status;
    const method = c.req.method;
    const path = c.req.path;
    const requestId = c.get('requestId');
    const logFn = status >= 500 ? httpLog.error : status >= 400 ? httpLog.warn : httpLog.info;
    logFn.call(httpLog, { requestId, method, path, status, ms }, `${method} ${path}`);
}));
app.use(secureHeaders());
app.use(cors({
    origin: config.corsOrigin === '*'
        ? '*'
        : config.corsOrigin.split(',').map(o => o.trim()),
    allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization'],
    maxAge: 86400,
}));
app.use('/api/*', bodyLimit({
    maxSize: config.bodyLimit,
    onError: () => { throw err.payloadTooLarge(); },
}));
app.use('/api/*', timeout(config.timeoutMs));
// Inject runtime into request context
const injectRuntime = createMiddleware(async (c, next) => {
    if (hasRuntime()) {
        c.set('runtime', getRuntime());
    }
    await next();
});
// API routes — authenticated, then rate-limited
app.use('/api/*', injectRuntime);
app.use('/api/*', bearerAuth);
app.use('/api/*', rateLimiter);
// MCP OAuth callback — public (browser redirect, no auth header)
app.use('/mcp/*', injectRuntime);
app.route('/mcp', mcpCallbackRoutes);
app.onError(errorHandler);
app.notFound(notFoundHandler);
app.get('/health', async (c) => {
    const checks = { runtime: hasRuntime() };
    if (hasRuntime()) {
        checks.db = await getRuntime().repositories.ping();
    }
    const healthy = Object.values(checks).every(Boolean);
    const status = healthy ? 'ok' : 'degraded';
    return c.json({ status, checks }, healthy ? 200 : 503);
});
app.route('/api', routes);
