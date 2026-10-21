import { stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, isAbsolute, relative, resolve } from 'node:path'

function normalizeOptionalPath(raw: unknown): string | undefined {
  if (typeof raw !== 'string') return undefined
  const trimmed = raw.trim()
  return trimmed ? trimmed : undefined
}

export function expandHomePath(path: string): string {
  if (path === '~') return homedir()
  if (path.startsWith('~/')) return resolve(homedir(), path.slice(2))
  return path
}

export function resolveToolPath(
  rawPath: string,
  baseCwd: string | undefined = process.cwd(),
): string {
  const expanded = expandHomePath(rawPath)
  if (isAbsolute(expanded)) {
    return expanded
  }
  return resolve(baseCwd ?? process.cwd(), expanded)
}

export function resolveSessionScratchPath(
  rawPath: string,
  baseDir: string | undefined,
  sessionId: string,
): { path: string; root: string } {
  const safeSessionId = sessionId.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 80) || 'session'
  const root = resolve(baseDir ?? process.cwd(), '.sepilot-scratch', safeSessionId)
  const expanded = expandHomePath(rawPath.trim())
  const scratchInput = isAbsolute(expanded) ? basename(expanded) : expanded
  const candidate = resolve(root, scratchInput || 'artifact')
  const relativePath = relative(root, candidate)
  if (relativePath.startsWith('../') || relativePath === '..' || isAbsolute(relativePath)) {
    throw new Error('temporary fs.write path must stay within the session scratch directory')
  }
  return { path: candidate, root }
}

export function resolveToolCwd(rawCwd: unknown, contextCwd?: string): string {
  const fallback = normalizeOptionalPath(contextCwd)
  const requested = normalizeOptionalPath(rawCwd)
  const baseCwd = fallback ? resolveToolPath(fallback) : process.cwd()
  return requested ? resolveToolPath(requested, baseCwd) : baseCwd
}

export async function directoryExists(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory()
  } catch {
    return false
  }
}

export type ToolCwdKind = 'directory' | 'file' | 'missing'

export async function classifyToolCwd(path: string): Promise<ToolCwdKind> {
  try {
    const info = await stat(path)
    return info.isDirectory() ? 'directory' : 'file'
  } catch {
    return 'missing'
  }
}

export function rejectToolCwd(
  cwd: string,
  kind: Exclude<ToolCwdKind, 'directory'>,
  options: { scopeParameter?: string } = {},
): { output: string; code: string } {
  if (kind === 'file') {
    const correction = options.scopeParameter
      ? ` Pass cwd=${dirname(cwd)} and ${options.scopeParameter}=${basename(cwd)} to scope to that file.`
      : ` Pass cwd=${dirname(cwd)} to work in its directory.`
    return {
      output: `cwd must be a directory, but ${cwd} is a file.${correction}`,
      code: 'CWD_IS_FILE_PERMANENT',
    }
  }
  return {
    output: `Working directory does not exist or is not accessible: ${cwd}`,
    code: 'CWD_NOT_FOUND_PERMANENT',
  }
}
