import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import type { DesktopAgentSession } from '@sepilotd/api-client'
import { desktopAgentSshSchema, remoteAgentCwd } from './ssh-launch.js'

const run = promisify(execFile)
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`

export function reviewCommand(session: DesktopAgentSession, kind: 'status' | 'diff') {
  const args = ['--no-optional-locks', '-c', 'core.fsmonitor=false', '-C', session.cwd,
    ...(kind === 'status' ? ['status', '--porcelain=v1', '-z', '--untracked-files=normal'] : ['diff', 'HEAD', '--no-ext-diff', '--no-textconv', '--']),
  ]
  if (!session.ssh) return { command: 'git', args }
  const ssh = desktopAgentSshSchema.parse(session.ssh)
  remoteAgentCwd(session.cwd)
  return { command: 'ssh', args: [
    '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ForwardAgent=no',
    '-o', 'ClearAllForwardings=yes', '-o', 'RemoteCommand=none', '-o', 'ConnectTimeout=8',
    ...(ssh.port ? ['-p', String(ssh.port)] : []), ...(ssh.user ? ['-l', ssh.user] : []),
    ssh.host, ['git', ...args].map(quote).join(' '),
  ] }
}

export function parseGitStatus(value: string) {
  const tokens = value.split('\0')
  const files: Array<{ status: string; path: string; previousPath?: string }> = []
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]!
    if (token.length < 4) continue
    const status = token.slice(0, 2)
    files.push({ status, path: token.slice(3),
      ...(/[RC]/.test(status) ? { previousPath: tokens[++i] ?? '' } : {}),
    })
  }
  return files
}

export async function readWorkspaceReview(session: DesktopAgentSession) {
  const read = async (kind: 'status' | 'diff') => {
    const invocation = reviewCommand(session, kind)
    try {
      const result = await run(invocation.command, invocation.args, {
        timeout: 12_000, maxBuffer: 512 * 1024, encoding: 'utf8', windowsHide: true,
        env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
      })
      return { text: result.stdout, error: null as string | null }
    } catch (error) {
      // Do not return partial stdout as complete evidence.
      const detail = error as { code?: string | number; stderr?: string }
      return { text: '', error: detail.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER'
        ? 'Output exceeds the review limit. Inspect this workspace on its host.'
        : (detail.stderr?.trim().slice(0, 1000) || 'Workspace review unavailable or timed out.') }
    }
  }
  const [status, diff] = await Promise.all([read('status'), read('diff')])
  return { observedAt: new Date().toISOString(), cwd: session.cwd, host: session.ssh?.host ?? 'local',
    files: parseGitStatus(status.text), diff: diff.text, statusError: status.error, diffError: diff.error,
    scope: 'working-tree-against-head' as const, tests: 'not-observed' as const,
  }
}
