import { serve } from '@hono/node-server';
import { app } from './lib/app.js';
import { config } from './lib/config.js';
import { logger } from './lib/logger.js';
import { initRuntime, shutdownRuntime } from './lib/runtime.js';
import { printExamples } from './examples.js';
async function main() {
    await initRuntime();
    const server = serve({
        fetch: app.fetch,
        port: config.port,
        hostname: config.host,
    }, (info) => {
        const base = `http://${info.address}:${info.port}`;
        logger.info({ host: info.address, port: info.port }, `server started on ${base}`);
        // THIS IS JUST FOR DEMO PURPOSES!!!!!!!!!!!!!!
        const token = process.env.SEED_API_KEY ?? '0f47acce-3aa7-4b58-9389-21b2940ecc70';
        printExamples(base, token);
    });
    let shuttingDown = false;
    const shutdown = async () => {
        if (shuttingDown)
            return;
        shuttingDown = true;
        logger.info('shutting down — stopping new connections');
        // Hard deadline: force exit if drain takes too long
        const forceTimer = setTimeout(() => {
            logger.error('shutdown timeout exceeded — forcing exit');
            process.exit(1);
        }, config.shutdownTimeoutMs);
        forceTimer.unref();
        // Stop accepting new connections, wait for in-flight to drain
        server.close(async () => {
            try {
                await shutdownRuntime();
                logger.info('shutdown complete');
                process.exit(0);
            }
            catch (err) {
                logger.error({ err }, 'error during shutdown');
                process.exit(1);
            }
        });
    };
    process.on('SIGINT', () => void shutdown());
    process.on('SIGTERM', () => void shutdown());
}
main().catch(err => {
    logger.fatal({ err }, 'fatal startup error');
    process.exit(1);
});
