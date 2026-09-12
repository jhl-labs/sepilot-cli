export type AutonomyLevel = 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'

export interface AutonomyOption {
  id: AutonomyLevel
  icon: string
  label: string
  description: string
  aliases?: readonly string[]
}

export const AUTONOMY_OPTIONS: AutonomyOption[] = [
  {
    id: 'readonly',
    icon: '🔒',
    label: 'Read Only',
    description: 'Allow observations and agent bookkeeping, but no workspace or external mutations.',
    aliases: ['read-only', 'readonly'],
  },
  {
    id: 'accept-edits',
    icon: '✎',
    label: 'Accept Edits',
    description: 'Auto-accept file edits. Shell and browser actions still require approval.',
    aliases: ['accept', 'edits', 'accept-edits'],
  },
  {
    id: 'workspace-write',
    icon: '▣',
    label: 'Workspace Write',
    description: 'Auto-accept edits inside the active workspace. Shell and browser actions still require approval.',
    aliases: ['workspace', 'workspace-write', 'worktree', 'ww'],
  },
  {
    id: 'supervised',
    icon: '🛡',
    label: 'Supervised',
    description: 'Ask for approval before risky tools and edits.',
    aliases: ['supervised', 'normal', 'default'],
  },
  {
    id: 'autonomous',
    icon: '⚡',
    label: 'Autonomous',
    description: 'Run policy-allowed tools without prompts. Approval-required tools are blocked.',
    aliases: ['autonomous', 'auto', 'bypass', 'yolo'],
  },
] as const

export function getAutonomyOption(level: AutonomyLevel): AutonomyOption {
  return AUTONOMY_OPTIONS.find((option) => option.id === level)
    ?? AUTONOMY_OPTIONS[0]!
}

export function nextAutonomyLevel(level: AutonomyLevel): AutonomyLevel {
  const currentIndex = AUTONOMY_OPTIONS.findIndex((option) => option.id === level)
  if (currentIndex === -1) {
    return AUTONOMY_OPTIONS[0]!.id
  }
  return AUTONOMY_OPTIONS[(currentIndex + 1) % AUTONOMY_OPTIONS.length]!.id
}

export function findAutonomyOption(query: string): AutonomyOption | null {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return null
  }

  return AUTONOMY_OPTIONS.find((option) => {
    if (option.id === normalizedQuery) return true
    if (option.label.toLowerCase() === normalizedQuery) return true
    if (option.aliases?.some((alias) => alias === normalizedQuery)) return true
    if (option.id.includes(normalizedQuery)) return true
    if (option.label.toLowerCase().includes(normalizedQuery)) return true
    if (option.aliases?.some((alias) => alias.includes(normalizedQuery))) return true
    return false
  }) ?? null
}

export function formatAutonomyBadge(level: AutonomyLevel): string {
  // The status bar redraws in place every frame, so this must use only
  // unambiguous-width glyphs. Emoji like 🛡 / ⚡ / 🔒 (no VS16) measure 1 or 2
  // columns depending on terminal and font; when Ink's count disagrees with
  // what the terminal actually drew, its in-place erase is off by a column
  // and stale characters bleed into neighbouring rows. Colour + label carry
  // the signal in the status bar; the picker overlay still shows the icons.
  return getAutonomyOption(level).label
}

export type AutonomyColor = 'info' | 'success' | 'warning' | 'error'

const AUTONOMY_COLOR_MAP: Record<AutonomyLevel, AutonomyColor> = {
  'readonly': 'info',
  'accept-edits': 'success',
  'workspace-write': 'success',
  'supervised': 'warning',
  'autonomous': 'error',
}

export function getAutonomyColor(level: AutonomyLevel): AutonomyColor {
  return AUTONOMY_COLOR_MAP[level] ?? 'info'
}
