export interface ClipboardDeps {
  platform: NodeJS.Platform
  env: Record<string, string | undefined>
  writeStdout(data: string): void
  /** Resolves true when the command accepted the complete payload. */
  spawn(command: string, args: string[], input: string): Promise<boolean>
}

export interface ClipboardResult {
  ok: boolean
  via: 'osc52' | 'command' | null
  command?: string
  error?: string
}

interface ClipboardCandidate {
  command: string
  args: string[]
}

// Terminals vary widely. Keeping the encoded sequence at 1 MiB avoids writing
// an unbounded escape payload while still supporting substantial responses.
const OSC52_MAX_ENCODED_BYTES = 1_000_000

export function encodeOsc52(text: string): string {
  return `\u001b]52;c;${Buffer.from(text, 'utf8').toString('base64')}\u0007`
}

export function clipboardCandidates(deps: ClipboardDeps): ClipboardCandidate[] {
  const candidates: ClipboardCandidate[] = []

  if (deps.platform === 'darwin') {
    candidates.push({ command: 'pbcopy', args: [] })
  }
  if (deps.platform === 'win32' || deps.env.WSL_DISTRO_NAME) {
    candidates.push({ command: 'clip.exe', args: [] })
  }
  if (deps.env.WAYLAND_DISPLAY) {
    candidates.push({ command: 'wl-copy', args: [] })
  }
  if (deps.env.DISPLAY) {
    candidates.push(
      { command: 'xclip', args: ['-selection', 'clipboard'] },
      { command: 'xsel', args: ['--clipboard', '--input'] },
    )
  }

  return candidates
}

export async function copyToClipboard(
  text: string,
  deps: ClipboardDeps,
): Promise<ClipboardResult> {
  let osc52Failure: string | null = null

  if (deps.env.SEPILOT_TUI_OSC52 !== '0') {
    const sequence = encodeOsc52(text)
    if (Buffer.byteLength(sequence, 'utf8') <= OSC52_MAX_ENCODED_BYTES) {
      try {
        deps.writeStdout(sequence)
        return { ok: true, via: 'osc52' }
      } catch (error) {
        osc52Failure = `OSC52 write failed: ${errorMessage(error)}`
      }
    } else {
      osc52Failure = 'OSC52 payload exceeds the 1 MB safety limit'
    }
  }

  const candidates = clipboardCandidates(deps)
  const attempted: string[] = []
  for (const candidate of candidates) {
    attempted.push(candidate.command)
    try {
      if (await deps.spawn(candidate.command, candidate.args, text)) {
        return { ok: true, via: 'command', command: candidate.command }
      }
    } catch {
      // A missing helper and a non-zero exit both move to the next candidate.
      // The injected contract intentionally exposes only success/failure.
    }
  }

  const commandFailure =
    attempted.length > 0
      ? `copy commands failed (${attempted.join(', ')})`
      : 'no supported copy command for this platform'
  return {
    ok: false,
    via: null,
    error: `clipboard: ${[osc52Failure, commandFailure].filter(Boolean).join('; ')}`,
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
