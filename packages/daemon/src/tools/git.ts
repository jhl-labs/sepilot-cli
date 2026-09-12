import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { throwIfAborted } from '../abort.js'
import { strictWorkspacePathViolation } from '../security/policy-engine.js'
import type { TerminalRunner } from './terminal.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import { enforceOutputLimit } from './output-limit.js'
import { resolveToolPath } from './path-utils.js'

const execFileAsync = promisify(execFile)
const GIT_OUTPUT_RESUME_HINT = 'scope with a path, ref, or commit (for example, git.diff commit=HEAD path=src/file.ts or git.log ref=HEAD~5)'
const STRICT_GIT_TIMEOUT_MS = 30_000
const STRICT_GIT_GLOBAL_ARGS = [
  '--no-optional-locks',
  '-c', 'core.fsmonitor=false',
  '-c', 'core.hooksPath=/dev/null',
  '-c', 'diff.external=',
]
const MAX_GIT_DATE_FILTER_LENGTH = 128

export interface GitToolOptions {
  /** Runner that proves workspace-only filesystem isolation for strict turns. */
  strictWorkspaceRunner?: TerminalRunner
  /** Injectable daemon-host clock for deterministic local-day boundaries. */
  now?: () => Date
}

function resolveRepoPath(input: Record<string, unknown>, contextCwd?: string): string {
  return typeof input.repoPath === 'string' && input.repoPath.trim()
    ? resolveToolPath(input.repoPath, contextCwd)
    : resolveToolPath(contextCwd ?? process.cwd())
}

function optionalGitText(input: Record<string, unknown>, key: string): string {
  return typeof input[key] === 'string' ? input[key].trim() : ''
}

function resolvedGitPath(
  input: Record<string, unknown>,
  contextCwd?: string,
): string {
  const path = optionalGitText(input, 'path')
  return path ? resolveToolPath(path, resolveRepoPath(input, contextCwd)) : ''
}

function sameGitRepository(
  observed: Record<string, unknown>,
  requested: Record<string, unknown>,
  contextCwd?: string,
): boolean {
  return resolveRepoPath(observed, contextCwd) === resolveRepoPath(requested, contextCwd)
}

function normalizedGitLogLimit(input: Record<string, unknown>): number {
  return typeof input.limit === 'number' && Number.isFinite(input.limit)
    ? Math.max(1, Math.min(100, Math.trunc(input.limit)))
    : 10
}

function hostLocalTodayRange(options: GitToolOptions): { since: string; until: string } {
  const now = new Date((options.now?.() ?? new Date()).getTime())
  const start = new Date(now.getTime())
  start.setHours(0, 0, 0, 0)
  return {
    since: start.toISOString(),
    until: now.toISOString(),
  }
}

function gitLogObservationCovers(
  observed: Record<string, unknown>,
  requested: Record<string, unknown>,
  contextCwd?: string,
): boolean {
  if (!sameGitRepository(observed, requested, contextCwd)) return false
  if (resolvedGitPath(observed, contextCwd) !== resolvedGitPath(requested, contextCwd)) return false
  if (optionalGitText(observed, 'ref') !== optionalGitText(requested, 'ref')) return false
  if ((observed.all === true) !== (requested.all === true)) return false
  if (optionalGitText(observed, 'period') !== optionalGitText(requested, 'period')) return false
  if (optionalGitText(observed, 'since') !== optionalGitText(requested, 'since')) return false
  if (optionalGitText(observed, 'until') !== optionalGitText(requested, 'until')) return false
  if (normalizedGitLogLimit(observed) < normalizedGitLogLimit(requested)) return false
  return observed.detailed === true || requested.detailed !== true
}

function limitedGitOutput(output: string, toolName: string): string {
  return enforceOutputLimit(output, {
    toolName,
    resumeHint: GIT_OUTPUT_RESUME_HINT,
  }).output
}

function gitDateFilter(
  input: Record<string, unknown>,
  key: 'since' | 'until',
): { value?: string; error?: string } {
  const raw = input[key]
  if (raw === undefined) return {}
  if (typeof raw !== 'string') {
    return { error: `${key} must be a string.` }
  }
  const value = raw.trim()
  if (
    value.length === 0
    || value.length > MAX_GIT_DATE_FILTER_LENGTH
    || /[\0\r\n]/u.test(value)
  ) {
    return {
      error: `${key} must be a non-empty Git date expression of at most ${MAX_GIT_DATE_FILTER_LENGTH} characters without control-line characters.`,
    }
  }
  return { value }
}

function invalidGitLogInput(output: string, start: number): ToolResult {
  return {
    output,
    status: 'error',
    code: 'INVALID_INPUT_PERMANENT',
    durationMs: Date.now() - start,
  }
}

async function runGit(
  args: string[],
  repoPath: string,
  context: Parameters<ToolDefinitionRuntime['execute']>[1],
  options: GitToolOptions,
  scopedPath?: string,
  signal?: AbortSignal,
): Promise<Pick<ToolResult, 'output' | 'status' | 'code' | 'executionPosture'>> {
  const workspaceRoot = context?.workspaceRoot?.trim()
  if (workspaceRoot) {
    const violation = strictWorkspacePathViolation(repoPath, workspaceRoot)
    if (violation) {
      return {
        output: violation,
        status: 'error',
        code: 'WORKSPACE_BOUNDARY_PERMANENT',
      }
    }
    if (scopedPath) {
      const pathViolation = strictWorkspacePathViolation(scopedPath, workspaceRoot)
      if (pathViolation) {
        return {
          output: pathViolation,
          status: 'error',
          code: 'WORKSPACE_BOUNDARY_PERMANENT',
        }
      }
    }

    if (!options.strictWorkspaceRunner) {
      return {
        output: 'Strict workspace Git inspection requires an active workspace-only sandbox, but no supported sandbox runner is available on this host.',
        status: 'error',
        code: 'SANDBOX_UNAVAILABLE',
      }
    }

    const result = await options.strictWorkspaceRunner.run({
      executable: 'git',
      args: [...STRICT_GIT_GLOBAL_ARGS, '-C', repoPath, ...args],
      cwd: workspaceRoot,
      timeoutMs: STRICT_GIT_TIMEOUT_MS,
      signal,
      cwdBoundary: 'strict_workspace',
    })
    if (!result.executionPosture.sandbox.active || !result.executionPosture.filesystem.isolated) {
      return {
        output: result.stderr.trim() || 'Strict workspace Git sandbox did not become active; the command was not run on the host.',
        status: 'error',
        code: result.code ?? 'SANDBOX_UNAVAILABLE',
        executionPosture: result.executionPosture,
      }
    }
    return {
      output: (result.stdout || result.stderr || '').trim() || '[no output]',
      status: result.status,
      ...(result.code ? { code: result.code } : {}),
      executionPosture: result.executionPosture,
    }
  }

  try {
    throwIfAborted(signal, 'Git command aborted')
    const { stdout, stderr } = await execFileAsync(
      'git',
      ['-C', repoPath, ...args],
      {
        signal,
        maxBuffer: 10 * 1024 * 1024,
      },
    )
    return {
      output: (stdout || stderr || '').trim() || '[no output]',
      status: 'success',
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return {
      output: message,
      status: 'error',
    }
  }
}

export function createGitStatusTool(options: GitToolOptions = {}): ToolDefinitionRuntime {
  return {
    name: 'git.status',
    description: 'Return structured git working tree status for a repository.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'git' },
    observationCoverage: {
      covers: (observed, requested, context) =>
        sameGitRepository(observed, requested, context.cwd),
    },
    inputSchema: {
      type: 'object',
      properties: {
        repoPath: { type: 'string', description: 'Repository path. Defaults to the active session cwd.' },
        short: { type: 'boolean', description: 'Use porcelain short format. Defaults to true.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const repoPath = resolveRepoPath(input, context?.cwd)
      const args = [
        'status',
        '--branch',
        (input.short as boolean | undefined) !== false ? '--short' : '',
      ].filter(Boolean)
      const result = await runGit(args, repoPath, context, options, undefined, context?.signal)
      return {
        output: limitedGitOutput(result.output, 'git.status'),
        status: result.status,
        ...(result.code ? { code: result.code } : {}),
        ...(result.executionPosture ? { executionPosture: result.executionPosture } : {}),
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createGitDiffTool(options: GitToolOptions = {}): ToolDefinitionRuntime {
  return {
    name: 'git.diff',
    description: 'Return a git diff for the repository, optionally scoped to a path or reference.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'git' },
    observationCoverage: {
      covers: (observed, requested, context) =>
        sameGitRepository(observed, requested, context.cwd)
        && resolvedGitPath(observed, context.cwd) === resolvedGitPath(requested, context.cwd)
        && optionalGitText(observed, 'ref') === optionalGitText(requested, 'ref')
        && optionalGitText(observed, 'commit') === optionalGitText(requested, 'commit')
        && (observed.staged === true) === (requested.staged === true),
    },
    inputSchema: {
      type: 'object',
      properties: {
        repoPath: { type: 'string', description: 'Repository path. Defaults to the active session cwd.' },
        path: { type: 'string', description: 'Optional path within the repository.' },
        ref: { type: 'string', description: 'Optional git reference to compare the current working tree against. For the patch introduced by one commit, use commit instead.' },
        commit: { type: 'string', description: 'Optional commit whose introduced patch should be returned, including for a root commit. Cannot be combined with ref or staged.' },
        staged: { type: 'boolean', description: 'Show staged diff instead of working tree diff.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const repoPath = resolveRepoPath(input, context?.cwd)
      const ref = optionalGitText(input, 'ref')
      const commit = optionalGitText(input, 'commit')
      if (commit && (ref || input.staged === true)) {
        return {
          output: 'git.diff commit cannot be combined with ref or staged.',
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      const args = commit
        ? ['show', '--format=', '--no-color', '--no-ext-diff', '--end-of-options', commit]
        : ['diff', '--no-color', '--no-ext-diff']
      if (!commit && input.staged === true) args.push('--staged')
      if (!commit && ref) {
        // Prevent a ref-shaped value such as --output=/outside/file from
        // becoming a git option with a filesystem side effect.
        args.push('--end-of-options', ref)
      }
      if (typeof input.path === 'string' && input.path.trim()) {
        args.push('--', input.path)
      }
      const scopedPath = typeof input.path === 'string' && input.path.trim()
        ? resolveToolPath(input.path, repoPath)
        : undefined
      const result = await runGit(args, repoPath, context, options, scopedPath, context?.signal)
      return {
        output: limitedGitOutput(result.output, 'git.diff'),
        status: result.status,
        ...(result.code ? { code: result.code } : {}),
        ...(result.executionPosture ? { executionPosture: result.executionPosture } : {}),
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createGitLogTool(options: GitToolOptions = {}): ToolDefinitionRuntime {
  return {
    name: 'git.log',
    description: 'Return git commits with optional ref, path, all-ref, and date-range filters. Use period="today" for the daemon host\'s current local calendar day instead of calling terminal.run or date. For introductions or summaries, set detailed=true to get author, date, body, and changed-file statistics in one call instead of running git show per commit.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'git' },
    observationCoverage: {
      covers: (observed, requested, context) =>
        gitLogObservationCovers(observed, requested, context.cwd),
    },
    inputSchema: {
      type: 'object',
      properties: {
        repoPath: { type: 'string', description: 'Repository path. Defaults to the active session cwd.' },
        limit: { type: 'number', description: 'Maximum number of commits to return. Defaults to 10.' },
        detailed: { type: 'boolean', description: 'Include full hash, author, date, body, and file statistics for every commit. Defaults to false.' },
        ref: { type: 'string', description: 'Optional starting reference.' },
        path: { type: 'string', description: 'Optional path within the repository.' },
        all: { type: 'boolean', description: 'Include commits reachable from all refs. Cannot be combined with ref.' },
        period: {
          type: 'string',
          enum: ['today'],
          description: 'Named local-time period. today means midnight through now in the daemon host timezone. Cannot be combined with since or until.',
        },
        since: { type: 'string', description: 'Optional Git-compatible lower date bound, such as an ISO-8601 timestamp or "2 weeks ago".' },
        until: { type: 'string', description: 'Optional Git-compatible upper date bound, such as an ISO-8601 timestamp or "now".' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const repoPath = resolveRepoPath(input, context?.cwd)
      const ref = typeof input.ref === 'string' ? input.ref.trim() : ''
      if (input.all === true && ref) {
        return invalidGitLogInput('git.log all=true cannot be combined with ref.', start)
      }
      const period = input.period
      if (period !== undefined && period !== 'today') {
        return invalidGitLogInput('git.log period must be "today" when provided.', start)
      }
      if (period === 'today' && (input.since !== undefined || input.until !== undefined)) {
        return invalidGitLogInput('git.log period cannot be combined with since or until.', start)
      }
      const since = gitDateFilter(input, 'since')
      if (since.error) return invalidGitLogInput(since.error, start)
      const until = gitDateFilter(input, 'until')
      if (until.error) return invalidGitLogInput(until.error, start)
      const limit = typeof input.limit === 'number' && Number.isFinite(input.limit)
        ? Math.max(1, Math.min(100, Math.trunc(input.limit)))
        : 10
      const args = input.detailed === true
        ? [
            'log',
            '--decorate',
            `-${limit}`,
            '--date=iso-strict',
            '--format=commit %H%d%nAuthor: %an <%ae>%nDate: %ad%n%n%s%n%n%b',
            // Machine-readable per-file counts prevent models from mistaking
            // the visual width in --stat for insertions. Keep the aggregate
            // and create/delete summary so one call still answers overview
            // and file-lifecycle questions.
            '--numstat',
            '--shortstat',
            '--summary',
          ]
        : ['log', '--decorate', '--oneline', `-${limit}`]
      if (input.all === true) {
        args.push('--all')
      }
      if (period === 'today') {
        // Resolve the named period in the daemon before entering a sandbox.
        // A workspace-only runner can have a UTC /etc/localtime or TZ even
        // when the daemon host is in another zone, so relative Git dates such
        // as "midnight" do not uphold the tool's host-local-day contract.
        const range = hostLocalTodayRange(options)
        args.push(`--since=${range.since}`, `--until=${range.until}`)
      } else {
        if (since.value) args.push(`--since=${since.value}`)
        if (until.value) args.push(`--until=${until.value}`)
      }
      if (ref) {
        args.push('--end-of-options', ref)
      }
      if (typeof input.path === 'string' && input.path.trim()) {
        args.push('--', input.path)
      }
      const scopedPath = typeof input.path === 'string' && input.path.trim()
        ? resolveToolPath(input.path, repoPath)
        : undefined
      const result = await runGit(args, repoPath, context, options, scopedPath, context?.signal)
      return {
        output: limitedGitOutput(result.output, 'git.log'),
        status: result.status,
        ...(result.code ? { code: result.code } : {}),
        ...(result.executionPosture ? { executionPosture: result.executionPosture } : {}),
        durationMs: Date.now() - start,
      }
    },
  }
}
