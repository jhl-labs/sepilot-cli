import { diffLines } from 'diff'
import type { ToolCallState } from '../types.js'

type StoredToolStatus = 'pending' | 'approved' | 'denied' | 'executing'

export interface DiffLine {
  prefix: '+' | '-' | ' '
  value: string
}

function splitPreservingLastLine(value: string): string[] {
  if (!value) return ['']
  const lines = value.split('\n')
  if (value.endsWith('\n')) {
    return lines.slice(0, -1)
  }
  return lines
}

export function previewTextLines(value: string): string[] {
  return splitPreservingLastLine(value)
}

export function truncatePreviewItems<T>(
  items: T[],
  maxItems: number,
): {
  visible: T[]
  omitted: number
} {
  if (maxItems <= 0) {
    return {
      visible: [],
      omitted: items.length,
    }
  }

  if (items.length <= maxItems) {
    return {
      visible: items,
      omitted: 0,
    }
  }

  const visibleCount = Math.max(0, maxItems - 1)
  return {
    visible: items.slice(0, visibleCount),
    omitted: items.length - visibleCount,
  }
}

export function formatToolInput(input?: Record<string, unknown>): string {
  return JSON.stringify(input ?? {}, null, 2)
}

export function summarizeToolInput(
  input?: Record<string, unknown>,
  toolName?: string,
): string {
  if (toolName === 'terminal.run') {
    const summary = `$ ${terminalCommandLabel(input ?? {})}`
    return summary.length > 120
      ? `${summary.slice(0, 119)}…`
      : summary
  }

  const path = typeof input?.path === 'string' ? input.path : ''
  if (path && toolName === 'fs.read') {
    const offset = typeof input?.offset === 'number'
      && Number.isInteger(input.offset)
      && input.offset > 0
      ? input.offset
      : null
    const limit = typeof input?.limit === 'number'
      && Number.isInteger(input.limit)
      && input.limit > 0
      ? input.limit
      : null
    if (offset !== null && limit !== null) {
      return `file ${path} lines ${offset}-${offset + limit - 1}`
    }
    if (offset !== null) {
      return `file ${path} from line ${offset}`
    }
  }
  if (path && toolName?.startsWith('fs.')) {
    return `file ${path}`
  }

  const serialized = JSON.stringify(input ?? {})
  return serialized.length > 120
    ? `${serialized.slice(0, 119)}…`
    : serialized
}

export function toolFilePath(
  input?: Record<string, unknown>,
  toolName?: string,
): string | null {
  if (!toolName?.startsWith('fs.')) return null
  const path = input?.path
  return typeof path === 'string' && path.trim() ? path : null
}

function shouldExpandTool(name: string, status: ToolCallState['status']): boolean {
  return name === 'fs.edit'
    || name === 'fs.write'
    || name === 'terminal.run'
    || status === 'pending'
    || status === 'error'
}

export function buildStoredToolCallState(options: {
  id: string
  name: string
  input?: Record<string, unknown>
  status: ToolCallState['status']
  output?: string
  meta?: string
  approvalRequestId?: string
  approvalState?: 'live' | 'stale'
  resumeAvailable?: boolean
  editDiff?: string
}): ToolCallState {
  const input = options.input ?? {}
  return {
    id: options.id,
    name: options.name,
    arguments: formatToolInput(input),
    input,
    status: options.status,
    output: options.output,
    meta: options.meta,
    collapsed: !shouldExpandTool(options.name, options.status),
    approvalRequestId: options.approvalRequestId,
    approvalState: options.approvalState,
    resumeAvailable: options.resumeAvailable,
    editDiff: options.editDiff,
  }
}

export function mapStoredToolStatus(
  status: StoredToolStatus,
): ToolCallState['status'] {
  switch (status) {
    case 'pending':
      return 'pending'
    case 'denied':
      return 'error'
    default:
      return 'running'
  }
}

export function toolMetaFromStoredStatus(
  status: StoredToolStatus,
): string {
  switch (status) {
    case 'pending':
      return 'Awaiting approval'
    case 'approved':
      return 'Approved for execution'
    case 'denied':
      return 'Denied by policy'
    case 'executing':
      return 'Running now'
  }
}

export function splitTerminalOutput(output?: string): {
  stdout: string
  stderr: string
} {
  if (!output) {
    return { stdout: '', stderr: '' }
  }

  const marker = '\n[stderr]\n'
  const markerIndex = output.indexOf(marker)
  if (markerIndex === -1) {
    return { stdout: output, stderr: '' }
  }

  return {
    stdout: output.slice(0, markerIndex),
    stderr: output.slice(markerIndex + marker.length),
  }
}

export function terminalCommandLabel(input: Record<string, unknown>): string {
  const executable = typeof input.executable === 'string' ? input.executable : ''
  const args = Array.isArray(input.args)
    ? input.args.map((arg) => String(arg))
    : []
  if (!executable) {
    // terminal.run also accepts a plain `command` string instead of
    // executable/args — label with it rather than "unknown".
    const command = typeof input.command === 'string' ? input.command.trim() : ''
    if (command) {
      return command
    }

    // Keep any structured args visible even when an older/incomplete event
    // omitted the executable. Returning only "unknown" hides the useful part
    // of the command and regresses the original fallback contract.
    return ['unknown', ...args].join(' ').trim()
  }
  return [executable, ...normalizeCommandLabelArgs(args, executable)].join(' ').trim()
}

function commandBasename(value: string): string {
  return (value.split(/[\\/]/).pop() ?? value).toLowerCase().replace(/\.exe$/i, '')
}

function normalizeCommandLabelArgs(args: string[], executable: string): string[] {
  if (args.length === 0) {
    return args
  }

  if (commandBasename(args[0]!) === commandBasename(executable)) {
    return args.slice(1)
  }

  if (args.length !== 1) {
    return args
  }

  const onlyArg = args[0]!.trim()
  if (!onlyArg) {
    return args
  }

  const parts = onlyArg.split(/\s+/).filter(Boolean)
  if (parts.length <= 1) {
    return args
  }

  if (commandBasename(parts[0]!) === commandBasename(executable)) {
    return parts.slice(1)
  }

  return onlyArg.startsWith('-') ? parts : args
}

export function buildDiffLines(
  previousContent: string | null | undefined,
  nextContent: string,
): DiffLine[] {
  if (previousContent === undefined) {
    return previewTextLines(nextContent).map((line) => ({
      prefix: '+',
      value: line,
    }))
  }

  return diffLines(previousContent ?? '', nextContent).flatMap((part) => {
    const prefix: DiffLine['prefix'] = part.added
      ? '+'
      : part.removed
        ? '-'
        : ' '

    return previewTextLines(part.value).map((line) => ({
      prefix,
      value: line,
    }))
  })
}
