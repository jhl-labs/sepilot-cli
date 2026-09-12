import { execFile } from 'node:child_process'
import { readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { promisify } from 'node:util'

const run = promisify(execFile)
const SWARM_TMUX_SESSION = /^sepilotd_swarm_([^_]+)_/
const SWARM_WORKTREE_PREFIX = 'sepilotd-swarm-wt-'
const SWARM_BRANCH_PREFIX = 'sepilotd/'

export interface ReaperWorktree {
  path: string
  branch?: string
  originRepo?: string
  createdByDaemon?: boolean
}

export interface ReapOrphanSwarmResourcesDeps {
  listTmuxSessions: () => Promise<string[]>
  removeTmuxSession: (name: string) => Promise<void>
  listWorktrees: () => Promise<ReaperWorktree[]>
  removeWorktree: (worktree: ReaperWorktree) => Promise<void>
  activeRunIds: Set<string>
}

export interface ReapOrphanSwarmResourcesResult {
  sessions: string[]
  worktrees: string[]
}

function normalizeRunId(runId: string): string {
  return runId.replace(/^swarm_/, '')
}

function activeRunIdSet(runIds: Set<string>): Set<string> {
  return new Set([...runIds].flatMap((runId) => [runId, normalizeRunId(runId)]))
}

export function runIdFromSwarmTmuxSession(name: string): string | null {
  return SWARM_TMUX_SESSION.exec(name)?.[1] ?? null
}

export function runIdFromSwarmWorktree(worktree: ReaperWorktree): string | null {
  if (!worktree.branch?.startsWith(SWARM_BRANCH_PREFIX)) return null
  return worktree.branch.slice(SWARM_BRANCH_PREFIX.length)
}

export async function reapOrphanSwarmResources(
  deps: ReapOrphanSwarmResourcesDeps,
): Promise<ReapOrphanSwarmResourcesResult> {
  const active = activeRunIdSet(deps.activeRunIds)
  const sessions: string[] = []
  for (const session of await deps.listTmuxSessions()) {
    const runId = runIdFromSwarmTmuxSession(session)
    if (!runId || active.has(runId) || active.has(normalizeRunId(runId))) continue
    await deps.removeTmuxSession(session)
    sessions.push(session)
  }

  const worktrees: string[] = []
  for (const worktree of await deps.listWorktrees()) {
    const runId = runIdFromSwarmWorktree(worktree)
    if (!runId || active.has(runId) || active.has(normalizeRunId(runId))) continue
    await deps.removeWorktree(worktree)
    worktrees.push(worktree.path)
  }

  return { sessions, worktrees }
}

export async function listTmuxSessionNames(): Promise<string[]> {
  try {
    const { stdout } = await run('tmux', ['list-sessions', '-F', '#{session_name}'])
    return stdout.split('\n').map((line) => line.trim()).filter(Boolean)
  } catch {
    return []
  }
}

export async function removeTmuxSession(name: string): Promise<void> {
  try {
    await run('tmux', ['kill-session', '-t', name])
  } catch {
    /* already gone */
  }
}

async function gitOutput(cwd: string, args: string[]): Promise<string | null> {
  try {
    const { stdout } = await run('git', args, { cwd })
    const text = stdout.trim()
    return text || null
  } catch {
    return null
  }
}

async function inferOriginRepo(worktreePath: string): Promise<string | undefined> {
  const commonDir = await gitOutput(worktreePath, [
    'rev-parse',
    '--path-format=absolute',
    '--git-common-dir',
  ])
  if (!commonDir) return undefined
  return basename(commonDir) === '.git' ? dirname(commonDir) : dirname(commonDir)
}

export async function listDaemonSwarmWorktrees(root = tmpdir()): Promise<ReaperWorktree[]> {
  const entries = await readdir(root, { withFileTypes: true }).catch(() => [])
  const worktrees: ReaperWorktree[] = []
  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith(SWARM_WORKTREE_PREFIX)) continue
    const path = join(root, entry.name)
    const branch = await gitOutput(path, ['rev-parse', '--abbrev-ref', 'HEAD'])
    if (!branch?.startsWith(SWARM_BRANCH_PREFIX)) continue
    worktrees.push({
      path,
      branch,
      originRepo: await inferOriginRepo(path),
      createdByDaemon: true,
    })
  }
  return worktrees
}

export async function removeDaemonSwarmWorktree(worktree: ReaperWorktree): Promise<void> {
  let removedByGit = false
  if (worktree.originRepo) {
    try {
      await run('git', ['worktree', 'remove', '--force', worktree.path], {
        cwd: worktree.originRepo,
      })
      removedByGit = true
    } catch {
      /* fall back to rm below */
    }
  }
  if (!removedByGit) {
    await rm(worktree.path, { recursive: true, force: true })
  }
  if (removedByGit && worktree.originRepo && worktree.branch) {
    try {
      await run('git', ['branch', '-D', worktree.branch], {
        cwd: worktree.originRepo,
      })
    } catch {
      /* branch may already be gone */
    }
  }
}
