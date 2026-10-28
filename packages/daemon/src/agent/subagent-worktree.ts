import { execFile } from 'node:child_process'
import {
  copyFile,
  lstat,
  mkdir,
  mkdtemp,
  readlink,
  realpath,
  rmdir,
  symlink,
  writeFile,
} from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { promisify } from 'node:util'

const exec = promisify(execFile)
const GIT_MAX_BUFFER = 64 * 1024 * 1024
const git = async (cwd: string, args: string[]) =>
  (await exec('git', args, { cwd, timeout: 60_000, maxBuffer: GIT_MAX_BUFFER })).stdout.trim()
const gitRaw = async (cwd: string, args: string[]) =>
  (await exec('git', args, { cwd, timeout: 60_000, maxBuffer: GIT_MAX_BUFFER, encoding: 'buffer' }))
    .stdout as Buffer

const MAX_SEED_UNTRACKED_FILES = 10_000
const MAX_SEED_UNTRACKED_BYTES = 512 * 1024 * 1024

const SEED_COMMIT_IDENTITY = [
  '-c', 'user.name=sepilot',
  '-c', 'user.email=subagent@sepilot.invalid',
  '-c', 'commit.gpgsign=false',
]

function contains(root: string, path: string): boolean {
  const rel = relative(root, path)
  return !isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${sep}`)
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9_./:@%+=-]+$/.test(value) ? value : `'${value.replace(/'/g, `'\\''`)}'`
}

export type SubagentWorktreeBase = 'head' | 'working-tree'

export interface SubagentWorktreeReceipt {
  path: string
  branch: string
  baseCommit: string
  base?: SubagentWorktreeBase
  seedCommit?: string
  repoRoot?: string
  retained: boolean
  reason: string
  patchPath?: string
  mergeCommand?: string
}

export interface SubagentWorktreeHandle {
  cwd: string
  receipt: SubagentWorktreeReceipt
  finish(retainForRecovery?: boolean): Promise<void>
}

async function seedParentWorkingTree(repo: string, path: string, scratch: string): Promise<boolean> {
  let seeded = false
  const trackedDiff = await gitRaw(repo, ['diff', '--binary', '--no-color', '--no-ext-diff', 'HEAD'])
  if (trackedDiff.length > 0) {
    const patch = join(scratch, 'seed.patch')
    await writeFile(patch, trackedDiff)
    await git(path, ['apply', '--binary', '--whitespace=nowarn', patch])
    seeded = true
  }
  const untracked = (await gitRaw(repo, ['ls-files', '--others', '--exclude-standard', '-z']))
    .toString('utf8')
    .split('\0')
    .filter(Boolean)
  if (untracked.length > MAX_SEED_UNTRACKED_FILES) {
    throw new Error(
      `Worktree base working-tree would copy ${untracked.length} untracked files (limit ${MAX_SEED_UNTRACKED_FILES}); ignore generated files or use the head base`,
    )
  }
  let bytes = 0
  for (const rel of untracked) {
    const source = resolve(repo, rel)
    const target = resolve(path, rel)
    if (!contains(repo, source) || !contains(path, target)) continue
    const info = await lstat(source).catch(() => null)
    if (!info) continue
    if (info.isSymbolicLink()) {
      await mkdir(dirname(target), { recursive: true })
      await symlink(await readlink(source), target)
    } else if (info.isFile()) {
      bytes += info.size
      if (bytes > MAX_SEED_UNTRACKED_BYTES) {
        throw new Error('Worktree base working-tree would copy more than 512 MiB of untracked files; use the head base')
      }
      await mkdir(dirname(target), { recursive: true })
      await copyFile(source, target)
    } else {
      continue
    }
    seeded = true
  }
  if (!seeded) return false
  await git(path, ['add', '-A'])
  await git(path, [
    ...SEED_COMMIT_IDENTITY,
    'commit', '--no-verify', '-q', '-m', 'sepilot: seed parent working tree for subagent',
  ])
  return true
}

function createFinisher(state: {
  repo: string
  path: string
  scratch: string
  receipt: SubagentWorktreeReceipt
  startCommit: string
}): SubagentWorktreeHandle['finish'] {
  const { repo, path, scratch, receipt, startCommit } = state
  return async (retainForRecovery) => {
    receipt.retained = true
    delete receipt.patchPath
    delete receipt.mergeCommand
    if (retainForRecovery) {
      receipt.reason = 'Interrupted or incomplete child; checkout retained for explicit checkpoint recovery.'
      return
    }
    try {
      const status = await git(path, ['status', '--porcelain=v1', '--untracked-files=all', '--ignored=matching'])
      const head = await git(path, ['rev-parse', 'HEAD'])
      if (status || head !== startCommit) {
        await describeRetainedChanges({ repo, path, scratch, receipt, startCommit, head, status })
        return
      }
      await git(repo, ['worktree', 'remove', path])
      receipt.retained = false
      receipt.reason = 'Removed unchanged checkout; parent workspace was not modified.'
      await git(repo, ['branch', startCommit === receipt.baseCommit ? '-d' : '-D', receipt.branch]).catch(() => {})
      await rmdir(scratch).catch(() => {})
    } catch {
      receipt.reason = 'Cleanup could not prove safety; checkout retained for inspection.'
    }
  }
}

async function describeRetainedChanges(input: {
  repo: string
  path: string
  scratch: string
  receipt: SubagentWorktreeReceipt
  startCommit: string
  head: string
  status: string
}): Promise<void> {
  const { repo, path, scratch, receipt, startCommit, head, status } = input
  const trackedOrUntracked = status
    .split('\n')
    .filter((line) => line && !line.startsWith('!!'))
  const onlyCommits = trackedOrUntracked.length === 0 && head !== startCommit
  if (onlyCommits && startCommit === receipt.baseCommit) {
    receipt.mergeCommand = `git -C ${shellQuote(repo)} merge ${shellQuote(receipt.branch)}`
    receipt.reason = 'Retained child commits; merge the branch explicitly.'
    return
  }
  await git(path, ['add', '-A'])
  const patch = await gitRaw(path, ['diff', '--cached', '--binary', '--no-color', '--no-ext-diff', startCommit])
  if (patch.length === 0) {
    receipt.reason = 'Retained ignored artifacts only; inspect the checkout, nothing to merge.'
    return
  }
  const patchPath = join(scratch, 'changes.patch')
  await writeFile(patchPath, patch)
  receipt.patchPath = patchPath
  receipt.mergeCommand = `git -C ${shellQuote(repo)} apply --binary ${shellQuote(patchPath)}`
  receipt.reason = receipt.base === 'working-tree'
    ? 'Retained changes made on top of the parent working tree; apply the patch to the parent checkout.'
    : 'Retained changes, artifacts or commits; apply the patch to the parent checkout.'
}

export async function createSubagentWorktree(input: {
  cwd: string
  workspaceRoot?: string
  sessionId: string
  base?: SubagentWorktreeBase
}): Promise<SubagentWorktreeHandle> {
  const base = input.base ?? 'head'
  const cwd = await realpath(input.cwd)
  const repo = await realpath(await git(cwd, ['rev-parse', '--show-toplevel']))
  if (input.workspaceRoot && !contains(await realpath(input.workspaceRoot), repo)) {
    throw new Error('Worktree isolation would copy files outside the parent workspace boundary')
  }
  if (!contains(repo, cwd)) throw new Error('Working directory is outside the repository')
  const baseCommit = await git(repo, ['rev-parse', 'HEAD'])
  const scratch = await mkdtemp(join(tmpdir(), 'sepilot-subagent-'))
  const path = join(scratch, 'checkout')
  const branch = `sepilot/subagent-${input.sessionId}`
  try {
    await git(repo, ['worktree', 'add', '-b', branch, path, baseCommit])
  } catch (error) {
    await rmdir(scratch).catch(() => {})
    throw error
  }
  let startCommit = baseCommit
  if (base === 'working-tree') {
    try {
      if (await seedParentWorkingTree(repo, path, scratch)) {
        startCommit = await git(path, ['rev-parse', 'HEAD'])
      }
    } catch (error) {
      await git(repo, ['worktree', 'remove', '--force', path]).catch(() => {})
      await git(repo, ['branch', '-D', branch]).catch(() => {})
      throw error
    }
  }
  const receipt: SubagentWorktreeReceipt = {
    path, branch, baseCommit, base, repoRoot: repo, retained: true,
    ...(startCommit !== baseCommit ? { seedCommit: startCommit } : {}),
    reason: base === 'working-tree'
      ? 'Owned checkout; starts at the parent working tree (tracked diff and untracked non-ignored files).'
      : 'Owned checkout; starts at committed HEAD (uncommitted parent edits are not copied).',
  }
  return {
    cwd: resolve(path, relative(repo, cwd)),
    receipt,
    finish: createFinisher({ repo, path, scratch, receipt, startCommit }),
  }
}

export async function reopenSubagentWorktree(input: {
  receipt: SubagentWorktreeReceipt
  cwd: string
}): Promise<SubagentWorktreeHandle> {
  const previous = input.receipt
  if (!previous.retained || !previous.repoRoot) {
    throw new Error('The subagent checkout was removed; dispatch a new subagent instead')
  }
  const path = await realpath(previous.path)
  if (path !== resolve(previous.path)) throw new Error('The retained subagent checkout was redirected')
  const top = await realpath(await git(path, ['rev-parse', '--show-toplevel']))
  if (top !== path) throw new Error('The retained subagent checkout is no longer a git worktree')
  const repo = await realpath(previous.repoRoot)
  const common = await realpath(resolve(path, await git(path, ['rev-parse', '--git-common-dir'])))
  const parentCommon = await realpath(resolve(repo, await git(repo, ['rev-parse', '--git-common-dir'])))
  const gitDir = await realpath(resolve(path, await git(path, ['rev-parse', '--git-dir'])))
  if (common !== parentCommon || gitDir === common) throw new Error('The retained checkout does not belong to its recorded repository')
  if (await git(path, ['symbolic-ref', '--short', 'HEAD']) !== previous.branch) throw new Error('The retained checkout changed branches')
  await git(path, ['merge-base', '--is-ancestor', previous.seedCommit ?? previous.baseCommit, 'HEAD'])
  const cwd = await realpath(input.cwd)
  if (!contains(path, cwd)) throw new Error('The retained working directory is outside its checkout')
  const receipt: SubagentWorktreeReceipt = {
    ...previous,
    path,
    retained: true,
    reason: 'Re-entered retained checkout for a follow-up message.',
  }
  delete receipt.patchPath
  delete receipt.mergeCommand
  return {
    cwd,
    receipt,
    finish: createFinisher({
      repo: previous.repoRoot,
      path,
      scratch: dirname(path),
      receipt,
      startCommit: previous.seedCommit ?? previous.baseCommit,
    }),
  }
}
