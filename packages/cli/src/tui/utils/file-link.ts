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

/**
 * OSC 8 lets the terminal own Ctrl/Command+click without enabling mouse
 * reporting. That preserves native scroll, selection, copy, and search in the
 * inline shell while still making tool file paths directly openable.
 */
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
  // BEL-terminated OSC 8. Every hyperlink-capable terminal accepts BEL, and
  // Ink's output tokenizer only understands this form: with the ESC \ (ST)
  // terminator it drops the visible text of a link that ends a row.
  return `\u001b]8;;${uri}\u0007${path}\u001b]8;;\u0007`
}
