import { execFile } from 'node:child_process'
import { mkdtemp, realpath, rmdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'
import { promisify } from 'node:util'

const exec = promisify(execFile)
const git = async (cwd: string, args: string[]) =>
  (await exec('git', args, { cwd, timeout: 30_000, maxBuffer: 2 * 1024 * 1024 })).stdout.trim()

function contains(root: string, path: string): boolean {
  const rel = relative(root, path)
  return !isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${sep}`)
}

export interface SubagentWorktreeReceipt {
  path: string
  branch: string
  baseCommit: string
  retained: boolean
  reason: string
}

/** An explicit isolated checkout is a mapped capability, never a fallback to the parent cwd. */
export async function createSubagentWorktree(input: {
  cwd: string
  workspaceRoot?: string
  sessionId: string
}): Promise<{ cwd: string; receipt: SubagentWorktreeReceipt; finish(retainForRecovery?: boolean): Promise<void> }> {
  const cwd = await realpath(input.cwd)
  const repo = await realpath(await git(cwd, ['rev-parse', '--show-toplevel']))
  if (input.workspaceRoot && !contains(await realpath(input.workspaceRoot), repo)) {
    throw new Error('Worktree isolation would copy files outside the parent workspace boundary')
  }
  if (!contains(repo, cwd)) throw new Error('Working directory is outside the repository')
  const baseCommit = await git(repo, ['rev-parse', 'HEAD'])
  const parent = await mkdtemp(join(tmpdir(), 'sepilot-subagent-'))
  const path = join(parent, 'checkout')
  const branch = `sepilot/subagent-${input.sessionId}`
  try {
    await git(repo, ['worktree', 'add', '-b', branch, path, baseCommit])
  } catch (error) {
    // Remove only an empty directory; never destroy a partially created checkout.
    await rmdir(parent).catch(() => {})
    throw error
  }
  const receipt: SubagentWorktreeReceipt = {
    path, branch, baseCommit, retained: true,
    reason: 'Owned checkout; starts at committed HEAD (uncommitted parent edits are not copied).',
  }
  return {
    cwd: resolve(path, relative(repo, cwd)),
    receipt,
    async finish(retainForRecovery) {
      if (retainForRecovery) { receipt.reason = 'Interrupted or incomplete child; checkout retained for explicit checkpoint recovery.'; return }
      try {
        // Include ignored files: generated artifacts may be valuable even when
        // git considers them disposable. Never force-remove or delete commits.
        const status = await git(path, ['status', '--porcelain=v1', '--untracked-files=all', '--ignored=matching'])
        const head = await git(path, ['rev-parse', 'HEAD'])
        if (status || head !== baseCommit) {
          receipt.reason = 'Retained changes, artifacts or commits; inspect and merge explicitly.'
          return
        }
        await git(repo, ['worktree', 'remove', path])
        receipt.retained = false
        receipt.reason = 'Removed unchanged checkout; parent workspace was not modified.'
        await git(repo, ['branch', '-d', branch]).catch(() => {})
        await rmdir(parent).catch(() => {})
      } catch {
        receipt.reason = 'Cleanup could not prove safety; checkout retained for inspection.'
      }
    },
  }
}
