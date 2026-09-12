import type { FastifyInstance } from 'fastify'
import { mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { bindCapability } from '../capabilities/bind.js'
import { openDatabase, type SqliteDatabase } from '../../db/sqlite.js'
import { sepilotdHome } from '../../storage/home.js'

export interface UsageSnapshot {
  totalSessions: number
  totalMessages: number
  totalToolCalls: number
}

function openUsageDb(): SqliteDatabase {
  const dir = join(sepilotdHome(), 'memory')
  mkdirSync(dir, { recursive: true })
  return openDatabase(join(dir, 'usage.db'))
}

// usage.db 가 비어 있거나 usage_records 테이블이 아직 안 만들어진 경우를
// 안전하게 다루기 위한 헬퍼. runtime.usageTracker 가 붙어 있지 않은 fallback
// 경로에서만 사용한다.
function snapshotFromUsageRecords(db: SqliteDatabase): UsageSnapshot {
  try {
    const exists = db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type='table' AND name='usage_records'`,
      )
      .get()
    if (!exists) {
      return { totalSessions: 0, totalMessages: 0, totalToolCalls: 0 }
    }
    const sessions = (db
      .prepare(`SELECT COUNT(DISTINCT session_id) AS c FROM usage_records`)
      .get() as { c: number }).c
    const toolCalls = (db
      .prepare(
        `SELECT COUNT(*) AS c FROM usage_records WHERE tool_name IS NOT NULL`,
      )
      .get() as { c: number }).c
    const messages = (db
      .prepare(
        `SELECT COUNT(*) AS c FROM usage_records WHERE tool_name IS NULL`,
      )
      .get() as { c: number }).c
    return {
      totalSessions: sessions,
      totalMessages: messages,
      totalToolCalls: toolCalls,
    }
  } catch {
    return { totalSessions: 0, totalMessages: 0, totalToolCalls: 0 }
  }
}

export async function registerUsageSnapshotRoute(
  app: FastifyInstance,
): Promise<void> {
  const runtime = app.runtime

  await bindCapability(
    app,
    {
      name: 'usage',
      version: '1',
      methods: [{ method: 'GET', path: '/usage/snapshot' }],
    },
    async (a) => {
      a.get('/usage/snapshot', async (): Promise<UsageSnapshot> => {
        if (runtime?.usageTracker) {
          return runtime.usageTracker.getSnapshot()
        }

        const db = openUsageDb()
        try {
          return snapshotFromUsageRecords(db)
        } finally {
          db.close()
        }
      })
    },
  )
}
