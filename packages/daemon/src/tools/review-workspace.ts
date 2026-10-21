import { lstat, realpath } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { throwIfAborted } from '../abort.js'
import type { TerminalRunner } from './terminal.js'
import type { ToolExecutionContext } from './registry.js'

export class ReviewWorkspaceError extends Error {
  constructor(message: string, readonly code: string) { super(message) }
}

export async function prepareReviewWorkspace(
  root: string,
  candidate: string,
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner?: TerminalRunner,
): Promise<{ path: string; created: boolean; reused: boolean; mode: 'review'; ref: string; revision: string; detached: true }> {
  if (!runner) throw new ReviewWorkspaceError('Review checkout requires an active workspace-only sandbox runner.', 'SANDBOX_UNAVAILABLE')
  const ref = typeof input.ref === 'string' ? input.ref.trim() : ''
  if (!ref || ref.startsWith('-') || ref.length > 300 || /[\u0000-\u001f\u007f]/u.test(ref)) throw new ReviewWorkspaceError('Review mode requires a local branch/ref or exact commit revision, not a command option.', 'INVALID_INPUT_PERMANENT')
  const metadata = await lstat(join(root, '.git')).catch(() => null)
  if (!metadata?.isDirectory() || metadata.isSymbolicLink()) throw new ReviewWorkspaceError('Review mode requires a repository with its own .git directory inside the active workspace. External worktree metadata and symlinks are not allowed.', 'WORKSPACE_BOUNDARY_PERMANENT')
  const git = async (cwd: string, args: string[]): Promise<string> => {
    throwIfAborted(context.signal, 'Review preparation aborted')
    const result = await runner.run({
      executable: 'git', args: ['-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false', '-c', 'diff.external=', '-C', cwd, ...args],
      cwd: root, workspaceRoot: root, cwdBoundary: 'strict_workspace', timeoutMs: 30_000, signal: context.signal,
    })
    if (!result.executionPosture.sandbox.active || !result.executionPosture.filesystem.isolated || result.executionPosture.network.mode !== 'none') throw new ReviewWorkspaceError('Review preparation did not obtain an isolated, network-disabled workspace sandbox.', 'SANDBOX_UNAVAILABLE')
    if (result.status !== 'success') throw new ReviewWorkspaceError('Review checkout preparation failed: ' + result.stderr.trim().slice(0, 1000), result.code ?? 'REVIEW_PREPARE_PERMANENT')
    return result.stdout.trim()
  }
  const revision = await git(root, ['rev-parse', '--verify', '--end-of-options', ref + '^{commit}'])
  if (!/^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$/u.test(revision)) throw new ReviewWorkspaceError('The requested local ref did not resolve to an exact commit. Fetch it using a separately authorized tool before retrying.', 'REVISION_UNAVAILABLE_PERMANENT')
  const existing = await lstat(candidate).catch(error => {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw error
  })
  if (existing) {
    if (existing.isSymbolicLink() || !existing.isDirectory()) throw new ReviewWorkspaceError('Refusing a symlink or non-directory review destination.', 'SYMLINK_PERMANENT')
    if (input.onExisting !== 'reuse') throw new ReviewWorkspaceError('The review destination exists; explicit reuse is required.', 'EEXIST_PERMANENT')
    const top = await git(candidate, ['rev-parse', '--show-toplevel'])
    const common = await git(candidate, ['rev-parse', '--git-common-dir'])
    const head = await git(candidate, ['rev-parse', '--verify', 'HEAD'])
    const branch = await git(candidate, ['rev-parse', '--abbrev-ref', 'HEAD'])
    const status = await git(candidate, ['status', '--porcelain', '--untracked-files=all'])
    if (top !== candidate || resolve(candidate, common) !== join(root, '.git') || head !== revision || branch !== 'HEAD' || status) throw new ReviewWorkspaceError('Reuse requires this repository’s clean detached review worktree at the exact requested revision. Existing files and branches were left unchanged.', 'REVIEW_CHECKOUT_MISMATCH_PERMANENT')
  } else {
    await context.workspaceMutation?.recordWrite(candidate)
    await git(root, ['worktree', 'add', '--detach', '--', candidate, revision])
  }
  throwIfAborted(context.signal, 'Review preparation aborted')
  const prepared = await realpath(candidate)
  if (prepared !== candidate || await git(candidate, ['rev-parse', '--verify', 'HEAD']) !== revision) throw new ReviewWorkspaceError('The prepared checkout does not match the requested path and commit.', 'REVIEW_CHECKOUT_MISMATCH_PERMANENT')
  return { path: prepared, created: !existing, reused: !!existing, mode: 'review', ref, revision, detached: true }
}
