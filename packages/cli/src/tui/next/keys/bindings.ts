import { COMMANDS, leaderBindings } from '../commands/registry.js'

export const DEFAULT_LEADER_KEY = 'ctrl+x'

export interface KeybindingConfig {
  leaderKey: string
  /** Follow-up key to command id. */
  bindings: Map<string, string>
  warnings: string[]
}

const KNOWN_COMMAND_IDS = new Set(COMMANDS.map((command) => command.id))
const LEADER_PATTERN = /^ctrl\+[a-z]$/i
const BINDING_PATTERN = /^leader (.)$/u

function defaultBindings(): Map<string, string> {
  return new Map(
    [...leaderBindings()].map(([key, command]) => [key, command.id] as const),
  )
}

function warningValue(value: unknown): string {
  return typeof value === 'string' ? value : String(value)
}

export function resolveKeybindings(raw: unknown): KeybindingConfig {
  const warnings: string[] = []
  let leaderKey = DEFAULT_LEADER_KEY
  let bindings = defaultBindings()

  if (raw === undefined || raw === null) return { leaderKey, bindings, warnings }
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    warnings.push('keybindings: configuration root must be an object')
    return { leaderKey, bindings, warnings }
  }

  const input = raw as { leader?: unknown; bindings?: unknown }
  if (input.leader !== undefined) {
    if (typeof input.leader === 'string' && LEADER_PATTERN.test(input.leader)) {
      leaderKey = input.leader.toLowerCase()
    } else {
      warnings.push(
        `keybindings: unsupported leader "${warningValue(input.leader)}"; using ${DEFAULT_LEADER_KEY}`,
      )
    }
  }

  if (input.bindings === undefined) return { leaderKey, bindings, warnings }
  if (
    typeof input.bindings !== 'object' ||
    input.bindings === null ||
    Array.isArray(input.bindings)
  ) {
    warnings.push('keybindings: bindings must be an object')
    return { leaderKey, bindings, warnings }
  }

  const next = new Map(bindings)
  for (const [commandId, value] of Object.entries(input.bindings)) {
    if (!KNOWN_COMMAND_IDS.has(commandId)) {
      warnings.push(`keybindings: unknown command "${commandId}"`)
      continue
    }
    if (!Array.isArray(value)) {
      warnings.push(`keybindings: bindings for ${commandId} must be an array`)
      continue
    }

    const requestedKeys: string[] = []
    for (const entry of value) {
      const match = typeof entry === 'string' ? BINDING_PATTERN.exec(entry) : null
      if (!match) {
        warnings.push(
          `keybindings: unsupported key "${warningValue(entry)}" for ${commandId}`,
        )
        continue
      }
      requestedKeys.push(match[1])
    }

    // An empty array intentionally unbinds. A non-empty but wholly malformed
    // array keeps the old binding so a typo cannot make a command unreachable.
    if (value.length > 0 && requestedKeys.length === 0) continue

    const previousKeys = [...next]
      .filter(([, boundId]) => boundId === commandId)
      .map(([key]) => key)
    const acceptedKeys: string[] = []
    for (const key of requestedKeys) {
      const existing = next.get(key)
      if (existing && existing !== commandId) {
        warnings.push(`keybindings: key "${key}" already bound to ${existing}`)
        continue
      }
      if (!acceptedKeys.includes(key)) acceptedKeys.push(key)
    }

    // Preserve the previous binding when all valid candidates collide.
    if (requestedKeys.length > 0 && acceptedKeys.length === 0) continue
    for (const key of previousKeys) next.delete(key)
    for (const key of acceptedKeys) next.set(key, commandId)
  }

  bindings = next
  return { leaderKey, bindings, warnings }
}

export function loadKeybindings(
  readFile: (path: string) => string | null,
  home: string,
): KeybindingConfig {
  const path = `${home}/.sepilotd/keybindings.json`
  let contents: string | null

  try {
    contents = readFile(path)
  } catch (error) {
    // The user-level override is optional. Node's readFileSync throws when it
    // does not exist, while injected readers may represent the same case as
    // null, so normalize both forms to the default configuration.
    if (isMissingFileError(error)) return resolveKeybindings(undefined)
    return withLoadWarning('read', path, error)
  }
  if (contents === null) return resolveKeybindings(undefined)

  try {
    return resolveKeybindings(JSON.parse(contents))
  } catch (error) {
    return withLoadWarning('parse', path, error)
  }
}

function isMissingFileError(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    error.code === 'ENOENT'
  )
}

function withLoadWarning(action: 'read' | 'parse', path: string, error: unknown): KeybindingConfig {
  const fallback = resolveKeybindings(undefined)
  const message = error instanceof Error ? error.message : String(error)
  fallback.warnings.push(`keybindings: could not ${action} ${path}: ${message}`)
  return fallback
}
