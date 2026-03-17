/**
 * Database seed script
 * Run: npm run db:seed
 */
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { drizzle } from 'drizzle-orm/libsql'
import { createClient } from '@libsql/client'
import { users } from '../repositories/sqlite/schema.js'
import { hashApiKey } from '../middleware/index.js'

const __dir = path.dirname(fileURLToPath(import.meta.url))
const ROOT_ENV_FILE = path.resolve(__dir, '../../../.env')
const LOCAL_ENV_FILE = path.resolve(__dir, '../../.env')

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

async function seed() {
  console.log('Seeding database...')

  const client = createClient({ url: DATABASE_URL })
  const db = drizzle(client)

  // Seed users
  const seedUsers = [
    { email: 'alice@aidevs.pl', apiKey: process.env.SEED_API_KEY ?? '0f47acce-3aa7-4b58-9389-21b2940ecc70' },
  ]

  for (const { email, apiKey } of seedUsers) {
    const apiKeyHash = await hashApiKey(apiKey)
    
    await db.insert(users).values({
      id: crypto.randomUUID(),
      email,
      apiKeyHash,
      createdAt: new Date(),
    }).onConflictDoNothing()

    console.log(`  ✓ User: ${email}`)
  }

  console.log('Seed completed.')
  process.exit(0)
}

seed().catch((err) => {
  console.error('Seed failed:', err)
  process.exit(1)
})
