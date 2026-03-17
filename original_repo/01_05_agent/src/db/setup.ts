/**
 * Idempotent DB setup — creates tables (via drizzle-kit push) and seeds
 * only when the schema hasn't been applied yet.
 *
 * Run: npx tsx src/db/setup.ts
 */
import { existsSync, mkdirSync } from 'node:fs'
import { execSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createClient } from '@libsql/client'

const __dir = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dir, '../..')
const ROOT_ENV_FILE = path.resolve(PROJECT_ROOT, '../.env')
const LOCAL_ENV_FILE = path.resolve(PROJECT_ROOT, '.env')

const loadEnvFile = (file: string) => {
  if (!existsSync(file) || typeof process.loadEnvFile !== 'function') return
  try {
    process.loadEnvFile(file)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`Failed to load .env file (${file}): ${message}`)
  }
}

loadEnvFile(ROOT_ENV_FILE)
loadEnvFile(LOCAL_ENV_FILE)

const DATABASE_URL = process.env.DATABASE_URL ?? 'file:.data/agent.db'

const dbPath = DATABASE_URL.replace(/^file:/, '')
const dbDir = path.dirname(path.resolve(PROJECT_ROOT, dbPath))
mkdirSync(dbDir, { recursive: true })

async function tablesExist(): Promise<boolean> {
  const client = createClient({ url: DATABASE_URL })
  try {
    const result = await client.execute(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='users'`,
    )
    return result.rows.length > 0
  } finally {
    client.close()
  }
}

async function setup() {
  if (await tablesExist()) {
    console.log('DB already initialised — skipping setup.')
    return
  }

  console.log('First run detected — pushing schema & seeding…')
  execSync('npx drizzle-kit push', { cwd: PROJECT_ROOT, stdio: 'inherit' })
  execSync('npx tsx src/db/seed.ts', { cwd: PROJECT_ROOT, stdio: 'inherit' })
  console.log('Setup complete.')
}

setup().catch((err) => {
  console.error('Setup failed:', err)
  process.exit(1)
})
