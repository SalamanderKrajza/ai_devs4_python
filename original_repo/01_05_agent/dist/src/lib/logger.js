/**
 * Structured logging with pino
 *
 * JSON in production, pretty-printed in development.
 * Child loggers carry context through the request / agent lifecycle.
 *
 * In development: logs to both stdout (pretty) and ./agent.log (JSON).
 */
import pino from 'pino';
import { config } from './config.js';
const targets = [];
if (config.nodeEnv !== 'production') {
    // Pretty stdout
    targets.push({
        target: 'pino-pretty',
        options: {
            colorize: true,
            translateTime: 'HH:MM:ss.l',
            ignore: 'pid,hostname',
        },
        level: config.logLevel,
    });
    // JSON file
    targets.push({
        target: 'pino/file',
        options: { destination: './agent.log', mkdir: true },
        level: config.logLevel,
    });
}
else {
    targets.push({
        target: 'pino/file',
        options: { destination: 1 }, // stdout
        level: config.logLevel,
    });
}
export const logger = pino({
    level: config.logLevel,
    transport: { targets },
    redact: ['apiKey', 'apiKeyHash', 'req.headers.authorization'],
});
