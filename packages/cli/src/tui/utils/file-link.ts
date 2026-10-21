import { homedir } from 'node:os'
import { isAbsolute, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const CONTROL_CHARACTER_RE = /[\u0000-\u001f\u007f]/u

export function supportsTerminalFileLinks(
  stdoutIsTty = process.stdout.isTTY === true,
  term = process.env.TERM,
): boolean {
  return stdoutIsTty && term?.toLowerCase() !== 'dumb'
}

function absoluteFilePath(path: string, cwd: string): string {
  if (path.startsWith('~/')) return resolve(homedir(), path.slice(2))
  return isAbsolute(path) ? path : resolve(cwd, path)
}

export function terminalFileLink(
  path: string,
  options: {
    cwd?: string
    enabled?: boolean
  } = {},
): string {
  if (CONTROL_CHARACTER_RE.test(path)) {
    return path.replace(/[\u0000-\u001f\u007f]/gu, '�')
  }
  if (!(options.enabled ?? supportsTerminalFileLinks())) return path

  const uri = pathToFileURL(absoluteFilePath(path, options.cwd ?? process.cwd())).href
  return `\u001b]8;;${uri}\u0007${path}\u001b]8;;\u0007`
}
