const MEMORY_CONTROL_SHORTCUTS = new Set([
  'audit',
  'audits',
  'backlog',
  'backlogs',
  'cleanup',
  'close',
  'current',
  'file',
  'files',
  'health',
  'hide',
  'info',
  'lifecycle',
  'maintain',
  'maintenance',
  'maintenance-status',
  'markdown',
  'off',
  'open-loop',
  'open-loops',
  'openloops',
])

export type ComposerShortcut =
  | { kind: 'help' }
  | { kind: 'shell-unsupported' }
  | { kind: 'memory-open' }
  | { kind: 'memory-search'; query: string }
  | { kind: 'memory-command'; command: string }
  | { kind: 'memory-save'; content: string }

export interface ComposerShortcutHintCopy {
  help: string
  shellUnsupported: string
  memoryOpen: string
  memorySearch: string
  memoryCommand: string
  memorySave: string
}

export interface ComposerShortcutHintOptions {
  overrideHint?: string | null
}

function firstToken(value: string): string {
  return value.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? ''
}

export function resolveComposerShortcut(value: string): ComposerShortcut | null {
  const text = value.trim()
  if (!text) return null

  if (text === '?') {
    return { kind: 'help' }
  }

  if (text === '!!' || text.startsWith('!')) {
    return { kind: 'shell-unsupported' }
  }

  if (!text.startsWith('#')) {
    return null
  }

  const content = text.slice(1).trim()
  if (!content || content === '?' || ['current', 'info'].includes(firstToken(content))) {
    return { kind: 'memory-open' }
  }

  if (content.startsWith('?')) {
    const query = content.slice(1).trim()
    return query ? { kind: 'memory-search', query } : { kind: 'memory-open' }
  }

  if (MEMORY_CONTROL_SHORTCUTS.has(firstToken(content))) {
    return { kind: 'memory-command', command: content }
  }

  return { kind: 'memory-save', content }
}

export function getComposerShortcutHint(
  value: string,
  copy: ComposerShortcutHintCopy,
  options: ComposerShortcutHintOptions = {},
): string | null {
  if (options.overrideHint) {
    return options.overrideHint
  }

  const shortcut = resolveComposerShortcut(value)
  if (!shortcut) return null

  switch (shortcut.kind) {
    case 'help':
      return copy.help
    case 'shell-unsupported':
      return copy.shellUnsupported
    case 'memory-open':
      return copy.memoryOpen
    case 'memory-search':
      return copy.memorySearch
    case 'memory-command':
      return copy.memoryCommand
    case 'memory-save':
      return copy.memorySave
  }
}
