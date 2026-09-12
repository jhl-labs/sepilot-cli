import { appendFile, readFile, mkdir, rename } from 'node:fs/promises'
import { dirname } from 'node:path'
import { createHash } from 'node:crypto'
import type { IAuditLogger, AuditEvent, Timestamp } from '@sepilotd/core'
import { createLogger } from '../logger.js'
import { isNodeFsError } from '../utils/fs-error.js'

const log = createLogger('security:audit-logger')

export interface AuditChainVerificationResult {
  ok: boolean
  firstBreakLine?: number
  reason?: 'invalid_json' | 'prev_hash_mismatch'
}

export class JsonlAuditLogger implements IAuditLogger {
  private logPath: string
  private prevHash: string = 'genesis'
  private seq: number = 0

  constructor(logPath: string) {
    this.logPath = logPath
  }

  async init(): Promise<void> {
    await mkdir(dirname(this.logPath), { recursive: true })
    // Resume from the last entry of the existing log so the
    // hash-chain (`prev_hash`) and the sequence counter stay
    // continuous across restarts. The whole point of the audit
    // log is tamper-evident: if the last line cannot be parsed we
    // rotate the file aside instead of silently restarting at
    // `seq=0, prev_hash='genesis'` — that would let a partial-
    // write or a deliberate truncation hide behind a fresh chain.
    let data: string
    try {
      data = await readFile(this.logPath, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return
      }
      log.error('failed to read existing audit log', {
        path: this.logPath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }

    const lines = data.trim().split('\n').filter(Boolean)
    if (lines.length === 0) {
      return
    }
    const lastLine = lines[lines.length - 1]!
    try {
      const last = JSON.parse(lastLine) as { seq?: number }
      this.prevHash = this.hashEntry(lastLine)
      this.seq = (last.seq ?? 0) + 1
    } catch (err) {
      const aside = `${this.logPath}.broken-${Date.now()}`
      log.error('audit log tail unparseable; rotating aside', {
        path: this.logPath,
        rotated: aside,
        error: err instanceof Error ? err.message : String(err),
      })
      await rename(this.logPath, aside)
      // After rotation we restart with a fresh chain — but the
      // operator now has a log entry pointing at the salvaged
      // file plus the broken file itself for forensic review.
    }
  }

  async log(event: AuditEvent): Promise<void> {
    const entry = {
      seq: this.seq++,
      ...event,
      prev_hash: this.prevHash,
    }
    const line = JSON.stringify(entry)
    this.prevHash = this.hashEntry(line)
    await appendFile(this.logPath, line + '\n', 'utf-8')
  }

  async query(filter: { since?: Timestamp; event?: string; limit?: number }): Promise<AuditEvent[]> {
    let data: string
    try {
      data = await readFile(this.logPath, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return []
      }
      log.error('failed to read audit log for query', {
        path: this.logPath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }
    let entries: AuditEvent[] = data
      .trim()
      .split('\n')
      .filter(Boolean)
      .map((l) => JSON.parse(l) as AuditEvent)

    if (filter.since) {
      entries = entries.filter((e) => e.timestamp >= filter.since!)
    }
    if (filter.event) {
      entries = entries.filter((e) => e.event === filter.event)
    }
    if (filter.limit) {
      entries = entries.slice(-filter.limit)
    }
    return entries
  }

  async verifyChain(): Promise<AuditChainVerificationResult> {
    let data: string
    try {
      data = await readFile(this.logPath, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return { ok: true }
      }
      log.error('failed to read audit log for chain verification', {
        path: this.logPath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }

    let expectedPrevHash = 'genesis'
    const lines = data.trim().split('\n').filter(Boolean)
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]!
      let entry: { prev_hash?: unknown }
      try {
        entry = JSON.parse(line) as { prev_hash?: unknown }
      } catch {
        return { ok: false, firstBreakLine: index + 1, reason: 'invalid_json' }
      }

      if (!hashMatches(entry.prev_hash, expectedPrevHash)) {
        return { ok: false, firstBreakLine: index + 1, reason: 'prev_hash_mismatch' }
      }
      expectedPrevHash = this.hashEntry(line)
    }

    return { ok: true }
  }

  private hashEntry(line: string): string {
    return createHash('sha256').update(line).digest('hex')
  }
}

function hashMatches(actual: unknown, expected: string): boolean {
  if (typeof actual !== 'string') return false
  if (actual === expected) return true
  return actual.length === 16 && expected.startsWith(actual)
}
