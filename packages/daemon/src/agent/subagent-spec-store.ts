import { AutonomyLevel } from '@sepilotd/core'
import { z } from 'zod'
import { openDomainDb } from '../storage/domain-db.js'
import type { SubagentChildSpec } from './subagent-dispatcher.js'

const receipt = z.object({
  path: z.string().min(1),
  branch: z.string().min(1),
  baseCommit: z.string().min(1),
  base: z.enum(['head', 'working-tree']).optional(),
  seedCommit: z.string().optional(),
  repoRoot: z.string().min(1),
  retained: z.boolean(),
  reason: z.string(),
  patchPath: z.string().optional(),
  mergeCommand: z.string().optional(),
})
const specSchema = z.object({
  sessionId: z.string().min(1),
  parentSessionId: z.string().optional(),
  category: z.object({ id: z.string() }).passthrough().optional(),
  agentId: z.string().optional(),
  toolNames: z.array(z.string()),
  providerId: z.string().min(1),
  model: z.string().min(1),
  maxIterations: z.number().int().positive(),
  systemPrompt: z.string(),
  forkSnapshot: z
    .array(
      z.object({
        role: z.enum(['user', 'assistant', 'tool']),
        content: z.string(),
        toolCallId: z.string().optional(),
        name: z.string().optional(),
        toolCalls: z
          .array(z.object({ id: z.string(), name: z.string(), arguments: z.record(z.unknown()) }))
          .optional(),
      }),
    )
    .optional(),
  parentExecutionPolicy: z
    .object({
      autonomy: z.nativeEnum(AutonomyLevel),
      requireToolApproval: z.boolean(),
      allowedToolNames: z.array(z.string()),
    })
    .optional(),
  cwd: z.string().optional(),
  workspaceRoot: z.string().optional(),
  isolation: z.literal('worktree').optional(),
  worktreeBase: z.enum(['head', 'working-tree']).optional(),
  worktree: z.object({ receipt, cwd: z.string() }).optional(),
  runContract: z.object({}).passthrough().optional(),
  scopeTags: z.array(z.string()).optional(),
  title: z.string(),
  updatedAt: z.string(),
  external: z
    .object({ cli: z.enum(['claude', 'codex', 'opencode']), model: z.string().optional() })
    .optional(),
  externalThreadId: z.string().optional(),
  reviewOutcomes: z.boolean().optional(),
  lastStatus: z.enum(['completed', 'failed', 'truncated']).optional(),
  lastOutput: z.string().optional(),
  budget: z
    .object({ requestedIterations: z.number(), appliedIterations: z.number(), hardCap: z.number() })
    .optional(),
})

export interface SubagentSpecStore {
  save(spec: SubagentChildSpec): void
  load(sessionId: string): SubagentChildSpec | null
  delete(sessionId: string): void
}

export function createSubagentSpecStore(): SubagentSpecStore {
  const db = openDomainDb({ name: 'subagents' })
  db.exec(
    'CREATE TABLE IF NOT EXISTS child_specs (session_id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)',
  )
  return {
    save(spec) {
      const parsed = specSchema.parse(spec)
      db.prepare(
        'INSERT INTO child_specs (session_id, version, payload) VALUES (?, 1, ?) ON CONFLICT(session_id) DO UPDATE SET version = 1, payload = excluded.payload',
      ).run(spec.sessionId, JSON.stringify(parsed))
    },
    load(sessionId) {
      const row = db
        .prepare('SELECT version, payload FROM child_specs WHERE session_id = ?')
        .get(sessionId) as { version: number; payload: string } | undefined
      if (!row) return null
      let payload: unknown
      try {
        payload = JSON.parse(row.payload)
      } catch {
        throw new Error('SUBAGENT_STATE_INVALID: saved execution contract is invalid')
      }
      const result = row.version === 1 ? specSchema.safeParse(payload) : null
      if (!result?.success || result.data.sessionId !== sessionId)
        throw new Error('SUBAGENT_STATE_INVALID: saved execution contract is invalid')
      return result.data as unknown as SubagentChildSpec
    },
    delete(sessionId) {
      db.prepare('DELETE FROM child_specs WHERE session_id = ?').run(sessionId)
    },
  }
}
