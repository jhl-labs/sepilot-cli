import { spawn } from 'node:child_process'
import { extractMarkdownCodeBlocks } from '../renderer/markdown.js'
import type { Message } from '../types.js'

export interface CopyableCodeBlock {
  language: string | null
  content: string
}

interface ClipboardCommand {
  command: string
  args: string[]
}

export function findLatestCopyableCodeBlock(
  messages: Pick<Message, 'role' | 'content'>[],
  streamingContent?: string | null,
): CopyableCodeBlock | null {
  if (streamingContent) {
    const streamingBlocks = extractMarkdownCodeBlocks(streamingContent)
    if (streamingBlocks.length > 0) {
      return streamingBlocks[streamingBlocks.length - 1] ?? null
    }
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.role !== 'assistant' && message.role !== 'tool') {
      continue
    }
    const blocks = extractMarkdownCodeBlocks(message.content)
    if (blocks.length > 0) {
      return blocks[blocks.length - 1] ?? null
    }
  }

  return null
}

export function getClipboardCommands(
  platform = process.platform,
): ClipboardCommand[] {
  switch (platform) {
    case 'darwin':
      return [{ command: 'pbcopy', args: [] }]
    case 'win32':
      return [{ command: 'clip', args: [] }]
    default:
      return [
        { command: 'wl-copy', args: [] },
        { command: 'xclip', args: ['-selection', 'clipboard'] },
        { command: 'xsel', args: ['--clipboard', '--input'] },
      ]
  }
}

export async function copyTextToClipboard(
  text: string,
  platform = process.platform,
  options: { stdout?: { write: (chunk: string) => boolean } } = {},
): Promise<void> {
  const attempts: string[] = []

  for (const candidate of getClipboardCommands(platform)) {
    try {
      await runClipboardCommand(candidate, text)
      return
    } catch (error) {
      attempts.push(formatClipboardFailure(candidate.command, error))
    }
  }

  // OSC 52 fallback. Works over SSH and in headless / no-X11 environments
  // when the terminal supports it (most modern ones do: iTerm2, Wezterm,
  // Kitty, Alacritty, recent xterm, mosh, tmux with `set -g set-clipboard on`).
  // We can't observe whether the terminal accepted it, so this is best-effort —
  // if a paste later turns up empty, the terminal silently ignored the
  // sequence (Termius/older builds, GNOME Terminal without the patch, etc.).
  const stdout = options.stdout ?? process.stdout
  if (writeOsc52(text, stdout)) {
    return
  }

  throw new Error(
    attempts.length > 0
      ? `Clipboard copy failed. ${attempts.join(' ')}`
      : 'Clipboard copy failed. No clipboard command is configured.',
  )
}

// 1 MiB is well past the safe ceiling for OSC 52 on most terminals
// (xterm caps at 100k by default, Wezterm/Kitty at ~few MiB). Above this
// we don't bother — the user will need a native clipboard tool.
const OSC52_MAX_BYTES = 1_000_000

function writeOsc52(
  text: string,
  stdout: { write: (chunk: string) => boolean } | undefined,
): boolean {
  if (!stdout || typeof stdout.write !== 'function') {
    return false
  }
  const encoded = Buffer.from(text, 'utf-8').toString('base64')
  if (encoded.length > OSC52_MAX_BYTES) {
    return false
  }
  try {
    // ESC ] 52 ; c ; <base64> BEL — `c` selects the system clipboard
    // (vs `p` primary selection). BEL terminator is the broadly-supported
    // form; ST (ESC \) would work on stricter parsers but breaks on some.
    stdout.write(`\x1b]52;c;${encoded}\x07`)
    return true
  } catch {
    return false
  }
}

function runClipboardCommand(
  candidate: ClipboardCommand,
  text: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(candidate.command, candidate.args, {
      stdio: ['pipe', 'ignore', 'pipe'],
    })
    let stderr = ''

    child.on('error', reject)
    child.stderr?.on('data', (chunk) => {
      stderr += String(chunk)
    })
    child.stdin?.on('error', reject)
    child.stdin?.end(text)
    child.on('close', (code) => {
      if (code === 0) {
        resolve()
        return
      }
      reject(new Error(stderr.trim() || `${candidate.command} exited with code ${code}`))
    })
  })
}

function formatClipboardFailure(command: string, error: unknown): string {
  if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
    return `${command}:not-found`
  }
  return `${command}:${error instanceof Error ? error.message : String(error)}`
}
