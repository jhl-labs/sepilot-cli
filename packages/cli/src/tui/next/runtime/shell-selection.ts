export type ShellKind = 'next' | 'legacy'

/** The inline shell is default; legacy remains an explicit recovery path. */
export function selectShell(env: Record<string, string | undefined>): ShellKind {
  return env.SEPILOT_TUI?.trim().toLowerCase() === 'legacy' ? 'legacy' : 'next'
}

export interface ShellTerminalPolicy {
  useAltScreen: boolean
  useMouseProxy: boolean
}

export function shellTerminalPolicy(
  shell: ShellKind,
  options: { isTty: boolean; mouseProxyEnabled: boolean },
): ShellTerminalPolicy {
  const legacyTty = shell === 'legacy' && options.isTty
  return {
    useAltScreen: legacyTty,
    useMouseProxy: legacyTty && options.mouseProxyEnabled,
  }
}
