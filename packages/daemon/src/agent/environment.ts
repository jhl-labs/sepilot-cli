import { readFile, realpath } from 'node:fs/promises'
import { platform } from 'node:os'
import { basename, isAbsolute, join, relative, sep } from 'node:path'
import { DAEMON_VERSION } from '../version.js'

export interface EnvironmentInfo {
  platform: string
  shell?: string
  nodeVersion: string
  daemonVersion: string
  gitBranch?: string
  today: string
  timeZone?: string
  /** Clock snapshot when this turn's environment was collected, not a tool observation time. */
  capturedAt?: string
  localTime?: string
}

export interface GatherEnvironmentOptions {
  cwd?: string
  workspaceRoot?: string
  now?: Date
  /** IANA timezone override for deterministic callers/tests. Defaults to the host timezone. */
  timeZone?: string
}

export async function gatherEnvironmentInfo(
  options: GatherEnvironmentOptions = {},
): Promise<EnvironmentInfo> {
  const now = options.now ?? new Date()
  const timeZone = options.timeZone
    ?? Intl.DateTimeFormat().resolvedOptions().timeZone
  const info: EnvironmentInfo = {
    platform: platform(),
    nodeVersion: process.version,
    daemonVersion: DAEMON_VERSION,
    today: toIsoDate(now, timeZone),
    capturedAt: now.toISOString(),
    localTime: new Intl.DateTimeFormat('en-GB', {
      timeZone, hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
    }).format(now),
    ...(timeZone ? { timeZone } : {}),
  }
  const shellEnv = process.env.SHELL
  if (shellEnv) {
    info.shell = basename(shellEnv)
  }
  if (options.cwd) {
    const branch = await detectGitBranch(options.cwd, options.workspaceRoot)
    if (branch) {
      info.gitBranch = branch
    }
  }
  return info
}

export function formatEnvironmentBlock(
  info: EnvironmentInfo,
  cwd: string | undefined,
): string {
  const lines: string[] = ['Environment:']
  if (cwd) {
    lines.push(`- Working directory: ${cwd}`)
  }
  lines.push(`- Platform: ${info.platform}`)
  if (info.shell) {
    lines.push(`- Shell: ${info.shell}`)
  }
  lines.push(`- Node.js: ${info.nodeVersion}`)
  lines.push(`- sepilotd: ${info.daemonVersion}`)
  if (info.gitBranch) {
    lines.push(`- Git branch: ${info.gitBranch}`)
  }
  lines.push(`- Today: ${info.today}${info.timeZone ? ` (host timezone: ${info.timeZone})` : ''}`)
  if (info.capturedAt) {
    lines.push(`- Turn clock snapshot: ${info.capturedAt}${info.localTime ? `; host local time ${info.today} ${info.localTime}` : ''}`)
    lines.push('- This clock anchors relative dates and remaining time today; it is not a source retrieval timestamp. Host timezone is not necessarily the user timezone. Honor an explicitly supplied user timezone, and refresh time through an available tool for time-sensitive work after a long run.')
  }
  return lines.join('\n')
}

async function detectGitBranch(
  cwd: string,
  workspaceRoot?: string,
): Promise<string | undefined> {
  try {
    const resolvedCwd = await realpath(cwd)
    if (workspaceRoot) {
      const resolvedWorkspace = await realpath(workspaceRoot)
      if (
        resolvedWorkspace !== workspaceRoot
        || resolvedCwd !== cwd
        || !isPathInside(resolvedWorkspace, resolvedCwd)
      ) {
        return undefined
      }
    }
    const resolvedHead = await realpath(join(resolvedCwd, '.git', 'HEAD'))
    const relativeHead = relative(resolvedCwd, resolvedHead)
    if (
      relativeHead !== ''
      && (
        relativeHead === '..'
        || relativeHead.startsWith(`..${sep}`)
        || isAbsolute(relativeHead)
      )
    ) {
      return undefined
    }
    const head = (await readFile(resolvedHead, 'utf-8')).trim()
    if (head.startsWith('ref: refs/heads/')) {
      return head.slice('ref: refs/heads/'.length)
    }
    if (/^[0-9a-f]{7,40}$/.test(head)) {
      return `(detached: ${head.slice(0, 7)})`
    }
    return undefined
  } catch {
    return undefined
  }
}

function isPathInside(root: string, target: string): boolean {
  const rel = relative(root, target)
  return rel === '' || (
    rel !== '..'
    && !rel.startsWith(`..${sep}`)
    && !isAbsolute(rel)
  )
}

function toIsoDate(d: Date, timeZone?: string): string {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(d)
  const part = (type: Intl.DateTimeFormatPartTypes): string =>
    parts.find((entry) => entry.type === type)?.value ?? ''
  return `${part('year')}-${part('month')}-${part('day')}`
}
