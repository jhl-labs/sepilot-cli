import { realpath, stat } from 'node:fs/promises'
import { isAbsolute, relative, sep } from 'node:path'

export interface WorkspaceBoundSession {
  cwd?: string
  workspaceIsolation?: 'policy' | 'strict'
}

export interface WorkspaceBoundCheckpoint {
  cwd?: string
  workspaceRoot?: string
  workspaceIsolation?: 'policy' | 'strict'
}

/**
 * A resumable execution inherits the immutable workspace capability captured
 * when it started. It is invalid as soon as the session binding differs.
 */
export function checkpointMatchesSessionWorkspace(
  session: WorkspaceBoundSession,
  checkpoint: WorkspaceBoundCheckpoint,
): boolean {
  if (session.workspaceIsolation === 'policy' || checkpoint.workspaceIsolation === 'policy') {
    return session.workspaceIsolation === checkpoint.workspaceIsolation
      // A narrower graph (for example read-only scouting) may impose a root
      // within a normal CLI session. Resuming retains that stricter root.
      && (!checkpoint.workspaceRoot || checkpoint.workspaceRoot === session.cwd)
      && session.cwd === checkpoint.cwd
  }
  // A fresh CLI chat sends `cwd` without an explicit workspaceRoot. The
  // session binds that directory as its workspace, while graph checkpoints
  // retain it in `cwd`. Treat both checkpoint shapes as the same immutable
  // capability; otherwise every first approval in a CLI-created session is
  // rejected as WORKSPACE_BINDING_CHANGED.
  return session.cwd === (checkpoint.workspaceRoot ?? checkpoint.cwd)
}

/** Separate a CLI launch directory from a requested strict filesystem boundary.
 * Persist this decision: neither a later transport nor an omitted root may
 * downgrade a bound session. Legacy sessions deliberately remain strict.
 */
export async function resolveChatRequestWorkspace(input: {
  cwd?: unknown
  workspaceRoot?: unknown
  session?: WorkspaceBoundSession | null
  projectDirectory?: string
  surface?: string | null
}): Promise<{ cwd?: string; workspaceRoot?: string; workspaceIsolation: 'policy' | 'strict' }> {
  const explicitRoot = normalizeOptionalCwd(input.workspaceRoot)
  const bound = input.session?.cwd ? input.session : undefined
  const isolation = bound
    ? bound.workspaceIsolation ?? 'strict'
    : input.surface === 'cli' && !explicitRoot ? 'policy' : 'strict'
  const cwd = normalizeOptionalCwd(input.cwd) ?? explicitRoot ?? bound?.cwd ?? input.projectDirectory
  if (isolation === 'policy') {
    if (explicitRoot) {
      throw new InvalidCwdError(explicitRoot, 'Start a new session to change its workspace isolation')
    }
    const resolved = await resolveRequestWorkspace(cwd, undefined)
    if (bound?.cwd && resolved.cwd !== bound.cwd) {
      throw new InvalidCwdError(resolved.cwd ?? '', 'cwd does not match the working directory bound to this session')
    }
    return { ...resolved, workspaceIsolation: isolation }
  }
  const root = explicitRoot ?? bound?.cwd ?? input.projectDirectory ?? cwd
  return {
    ...await resolveRequestWorkspace(cwd, root, bound?.cwd),
    workspaceIsolation: isolation,
  }
}

export class InvalidCwdError extends Error {
  readonly code = 'INVALID_CWD'

  constructor(readonly cwd: string, message: string) {
    super(message)
    this.name = 'InvalidCwdError'
  }
}

function normalizeOptionalCwd(raw: unknown): string | undefined {
  if (typeof raw !== 'string') {
    return undefined
  }
  const trimmed = raw.trim()
  return trimmed ? trimmed : undefined
}

export async function resolveRequestCwd(raw: unknown): Promise<string | undefined> {
  const cwd = normalizeOptionalCwd(raw)
  if (!cwd) {
    return undefined
  }
  if (!isAbsolute(cwd)) {
    throw new InvalidCwdError(cwd, `cwd must be an absolute path: ${cwd}`)
  }

  try {
    const realCwd = await realpath(cwd)
    const cwdStat = await stat(realCwd)
    if (!cwdStat.isDirectory()) {
      throw new InvalidCwdError(cwd, `cwd is not a directory: ${cwd}`)
    }
    return realCwd
  } catch (error) {
    if (error instanceof InvalidCwdError) {
      throw error
    }
    throw new InvalidCwdError(
      cwd,
      `Working directory does not exist or is not accessible: ${cwd}`,
    )
  }
}

/**
 * Resolve the immutable workspace capability root and ensure the requested
 * working directory cannot escape it. Both values are real paths at this
 * transport boundary, so later tool arguments cannot redefine the root.
 */
export async function resolveRequestWorkspace(
  rawCwd: unknown,
  rawWorkspaceRoot: unknown,
  rawBoundWorkspaceRoot?: unknown,
): Promise<{ cwd?: string; workspaceRoot?: string }> {
  const requestedWorkspaceRoot = await resolveRequestCwd(rawWorkspaceRoot)
  const boundWorkspaceRoot = await resolveRequestCwd(rawBoundWorkspaceRoot)
  if (
    requestedWorkspaceRoot
    && boundWorkspaceRoot
    && requestedWorkspaceRoot !== boundWorkspaceRoot
  ) {
    throw new InvalidCwdError(
      requestedWorkspaceRoot,
      'workspaceRoot does not match the workspace bound to this session',
    )
  }
  const workspaceRoot = boundWorkspaceRoot ?? requestedWorkspaceRoot
  const cwd = await resolveRequestCwd(rawCwd) ?? workspaceRoot
  if (workspaceRoot && cwd) {
    const relativePath = relative(workspaceRoot, cwd)
    if (
      relativePath !== ''
      && (
        relativePath === '..'
        || relativePath.startsWith(`..${sep}`)
        || isAbsolute(relativePath)
      )
    ) {
      throw new InvalidCwdError(
        cwd,
        `cwd is outside the strict workspace root: ${cwd} (workspace: ${workspaceRoot})`,
      )
    }
  }
  return { cwd, workspaceRoot }
}

export function invalidCwdResponse(error: InvalidCwdError): {
  error: { code: string; message: string }
} {
  return {
    error: {
      code: error.code,
      message: error.message,
    },
  }
}
