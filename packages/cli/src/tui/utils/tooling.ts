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

export function toolDisplayName(name: string): string {
  const labels: Record<string, string> = {
    'fs.read': 'Read',
    'fs.write': 'Write',
    'fs.edit': 'Edit',
    'fs.append': 'Append to',
    'fs.list': 'List files',
    'fs.glob': 'Find files',
    'fs.search': 'Search contents',
    'apply_patch': 'Apply patch',
    'git.status': 'Git status',
    'git.diff': 'Review changes',
    'terminal.run': 'Run',
    'workspace.prepare': 'Prepare workspace',
    'process.list': 'Inspect processes',
    'process.start': 'Start process',
    'web.search': 'Search web',
    'web.fetch': 'Fetch page',
  }
  return labels[name] ?? name
}

function clipToolSummary(value: string): string {
  const line = value.replace(/\s+/gu, ' ').trim()
  return line.length > 120 ? `${line.slice(0, 119)}…` : line
}

export function summarizeToolInput(
  input?: Record<string, unknown>,
  toolName?: string,
): string {
  if (toolName === 'apply_patch' && typeof input?.patch === 'string') {
    const paths = [...new Set([...input.patch.matchAll(/^\*\*\* (?:Add|Update|Delete) File: (.+)\r?$/gmu)]
      .map((match) => match[1]!.trim()))]
    if (paths.length > 0) return clipToolSummary(`${paths.length} ${paths.length === 1 ? 'file' : 'files'} · ${paths.slice(0, 3).join(', ')}${paths.length > 3 ? ', …' : ''}`)
  }
  if (toolName === 'process.list' && typeof input?.port === 'number') return `TCP port ${input.port}`
  if (toolName === 'process.start') {
    const network = input?.network
    const ports = network && typeof network === 'object' && 'ports' in network && Array.isArray(network.ports)
      ? network.ports.filter(port => typeof port === 'number') : []
    return clipToolSummary([
      input?.reuseId ? 'reuse existing server' : null,
      `$ ${compactTerminalCommandLabel(input ?? {})}`,
      ports.length > 0 ? `localhost ${ports.join(', ')}` : null,
      input?.lifetime === 'session' ? 'session lifetime' : null,
    ].filter(Boolean).join(' · '))
  }
  if (toolName === 'terminal.run') {
    const summary = `$ ${compactTerminalCommandLabel(input ?? {})}`
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

  if (toolName === 'fs.list' || toolName === 'fs.glob' || toolName === 'fs.search') {
    return clipToolSummary([
      typeof input?.query === 'string' ? input.query : null,
      typeof input?.pattern === 'string' ? input.pattern : null,
      typeof input?.cwd === 'string' ? `in ${input.cwd}` : 'in current folder',
      input?.hidden === true ? 'including hidden files' : null,
      typeof input?.glob === 'string' ? `files ${input.glob}` : null,
    ].filter(Boolean).join(' · '))
  }

  if (!input || Object.keys(input).length === 0) return ''
  const fields = ['path', 'pattern', 'query', 'cwd', 'url', 'target', 'name']
    .flatMap((key) => typeof input[key] === 'string' ? [`${key}: ${input[key]}`] : [])
  if (fields.length > 0) return clipToolSummary(fields.join(' · '))
  const count = Object.keys(input).length
  return `${count} parameter${count === 1 ? '' : 's'} · Ctrl+O details`
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
  approvalState?: ToolCallState['approvalState']
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

const SHELL_WRAPPER_NAMES = new Set(['bash', 'sh', 'zsh', 'dash', 'ksh'])
const QUOTED_SEGMENT_KEEP_CHARS = 16

export function compactTerminalCommandLabel(input: Record<string, unknown>): string {
  const executable = typeof input.executable === 'string' ? input.executable : ''
  const args = Array.isArray(input.args) ? input.args.map((arg) => String(arg)) : []
  const scriptIndex = args.findIndex((arg) => arg === '-c' || arg === '-lc')
  const script = SHELL_WRAPPER_NAMES.has(commandBasename(executable)) && scriptIndex >= 0
    ? args[scriptIndex + 1]
    : undefined
  const label = script ?? terminalCommandLabel(input)
  return abbreviateQuotedSegments(label.replace(/\s+/g, ' ').trim())
}

export function abbreviateQuotedSegments(text: string): string {
  return text.replace(/'[^']*'|"(?:[^"\\]|\\.)*"/g, (segment) => {
    if (segment.length <= QUOTED_SEGMENT_KEEP_CHARS + 2) return segment
    const quote = segment[0]
    return `${quote}${segment.slice(1, QUOTED_SEGMENT_KEEP_CHARS - 1).trimEnd()}…${quote}`
  })
}

export function terminalCommandLabel(input: Record<string, unknown>): string {
  const executable = typeof input.executable === 'string' ? input.executable : ''
  const args = Array.isArray(input.args)
    ? input.args.map((arg) => String(arg))
    : []
  if (!executable) {
    const command = typeof input.command === 'string' ? input.command.trim() : ''
    if (command) {
      return command
    }

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

const META_SEPARATOR = ' · '

function isExecutionPostureMeta(part: string): boolean {
  return /^(?:host execution, no sandbox|[\w-]+ sandbox)(?: \(.*\))?$/.test(part)
}

export function toolExecutionPostureMeta(meta: string | undefined): string | undefined {
  return meta?.split(META_SEPARATOR).find(isExecutionPostureMeta)
}

export function rowToolMeta(meta: string | undefined, status: string): string | undefined {
  if (!meta) return undefined
  const parts = meta.split(META_SEPARATOR)
    .filter((part) => !isExecutionPostureMeta(part))
    .filter((part) => !(status === 'success' && part === 'Execution finished'))
  return parts.length > 0 ? parts.join(META_SEPARATOR) : undefined
}
