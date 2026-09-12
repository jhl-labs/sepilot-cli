import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

export interface Worktree {
  path: string
  createdByDaemon: boolean
  branch?: string
  originRepo?: string
}

export class WorktreeManager {
  private isGitRepo(cwd: string): boolean {
    try {
      execFileSync('git', ['rev-parse', '--is-inside-work-tree'], { cwd, stdio: 'ignore' })
      return true
    } catch {
      return false
    }
  }

  create(originCwd: string, runId: string): Worktree {
    if (!existsSync(originCwd)) {
      throw new Error(`cwd does not exist: ${originCwd}`)
    }
    if (!this.isGitRepo(originCwd)) {
      return { path: originCwd, createdByDaemon: false }
    }
    const wtRoot = mkdtempSync(join(tmpdir(), 'sepilotd-swarm-wt-'))
    const branch = `sepilotd/${runId}`
    try {
      execFileSync('git', ['worktree', 'add', '-b', branch, wtRoot, 'HEAD'], { cwd: originCwd })
    } catch (e) {
      try {
        rmSync(wtRoot, { recursive: true, force: true })
      } catch {
        /* ignore */
      }
      throw e
    }
    return { path: wtRoot, createdByDaemon: true, branch, originRepo: originCwd }
  }

  remove(wt: Worktree): void {
    if (!wt.createdByDaemon || !wt.originRepo) return
    let removed = false
    try {
      execFileSync('git', ['worktree', 'remove', '--force', wt.path], { cwd: wt.originRepo })
      removed = true
    } catch {
      try {
        rmSync(wt.path, { recursive: true, force: true })
      } catch {
        /* ignore */
      }
    }
    // Best-effort branch cleanup. Only meaningful when worktree removal succeeded
    // (otherwise the branch is still checked out by the worktree).
    if (removed && wt.branch) {
      try {
        execFileSync('git', ['branch', '-D', wt.branch], { cwd: wt.originRepo, stdio: 'ignore' })
      } catch {
        /* ignore — branch may not exist */
      }
    }
  }
}
