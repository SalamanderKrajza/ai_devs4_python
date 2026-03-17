import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'drizzle-kit';
const ROOT_ENV_FILE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../.env');
if (existsSync(ROOT_ENV_FILE) && typeof process.loadEnvFile === 'function') {
    try {
        process.loadEnvFile(ROOT_ENV_FILE);
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to load root .env file (${ROOT_ENV_FILE}): ${message}`);
    }
}
export default defineConfig({
    dialect: 'sqlite',
    schema: './src/repositories/sqlite/schema.ts',
    out: './drizzle',
    dbCredentials: {
        url: process.env.DATABASE_URL ?? 'file:.data/agent.db',
    },
});
