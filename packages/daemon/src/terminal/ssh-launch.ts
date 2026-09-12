import { z } from 'zod'
import type { DesktopAgentSshTarget } from '@sepilotd/api-client'

export const desktopAgentSshSchema = z
  .object({
    host: z
      .string()
      .trim()
      .min(1)
      .max(253)
      .regex(/^[a-zA-Z0-9:][a-zA-Z0-9._:-]*$/),
    user: z
      .string()
      .trim()
      .min(1)
      .max(128)
      .regex(/^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$/)
      .optional(),
    port: z.number().int().min(1).max(65535).optional(),
  })
  .strict()

export function remoteAgentCwd(cwd: string): string {
  if (!cwd.startsWith('/') || cwd.length > 4096 || /[\x00-\x1f\x7f]/.test(cwd))
    throw new Error('SSH 작업 폴더는 원격 호스트의 절대 POSIX 경로여야 합니다.')
  return cwd
}
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`

/** Only structured targets and an allowlisted agent cross the remote shell boundary. */
export function sshAgentLaunch(command: string, cwd: string, target: DesktopAgentSshTarget) {
  const ssh = desktopAgentSshSchema.parse(target)
  if (!['claude', 'codex', 'gemini', 'opencode'].includes(command))
    throw new Error('지원하지 않는 CLI입니다.')
  const script = `cd ${quote(remoteAgentCwd(cwd))} && exec ${command}`
  const args = [
    '-o',
    'RequestTTY=force',
    '-o',
    'StrictHostKeyChecking=ask',
    '-o',
    'RemoteCommand=none',
    '-o',
    'ForwardAgent=no',
    '-o',
    'ClearAllForwardings=yes',
    '-o',
    'ServerAliveInterval=30',
    '-o',
    'ServerAliveCountMax=3',
    '-e',
    'none',
  ]
  if (ssh.port) args.push('-p', String(ssh.port))
  if (ssh.user) args.push('-l', ssh.user)
  args.push(ssh.host, `exec "\${SHELL:-/bin/sh}" -lc ${quote(script)}`)
  return { command: 'ssh', args, cwd: process.cwd() }
}
