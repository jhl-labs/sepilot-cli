import type { ToolCall, ToolExecutionPosture, ToolResultMetadata, ToolSecurityEffect } from '@sepilotd/core'
import type {
  AgentEvidenceLedger,
  AgentEvidenceLedgerEntry,
  AgentState,
  GraphExecutionContext,
} from './types.js'
import { isExecutorConfirmedReadOnlyObservation } from '../read-only-observation-evidence.js'
import {
  evidenceRequirementMinSourceFiles,
  evidenceRequirementMinSourceObservations,
  evidenceRequirementMinSourceScopes,
  evidenceRequirementRequiresSearch,
  evidenceRequirementUsesRepositoryBreadth,
  sourceToolMatchesEvidenceRequirement,
} from '../evidence-requirement-policy.js'
import { searchPathMatchesReadPath } from './repository-path.js'

const MAX_LEDGER_ENTRIES_PER_BUCKET = 80
const MAX_LEDGER_SUMMARY_CHARS = 220
const MANAGED_PROCESS_RECEIPT_TOOLS = new Set(['process.start', 'process.read', 'process.follow', 'process.wait'])

function validManagedProcessReceipt(value: unknown): value is NonNullable<ToolResultMetadata['managedProcess']> {
  if (!value || typeof value !== 'object') return false
  const receipt = value as Record<string, unknown>
  return typeof receipt.id === 'string' && receipt.id.length > 0 && receipt.id.length <= 200
    && typeof receipt.startedAt === 'string' && receipt.startedAt.length <= 40 && Number.isFinite(Date.parse(receipt.startedAt))
    && (receipt.status === 'running' || receipt.status === 'exited')
    && (receipt.exitCode === null || (typeof receipt.exitCode === 'number' && Number.isInteger(receipt.exitCode)))
    && (receipt.signal === null || typeof receipt.signal === 'string')
    && (receipt.terminationReason === undefined || ['requested', 'ttl', 'daemon-shutdown'].includes(String(receipt.terminationReason)))
}

const CONTENT_READ_TOOL_NAMES = new Set([
  'fs.read',
  'webfetch',
  'browser.navigate',
  'browser.extract',
  'browser.click',
  'browser.evaluate',
  'apps.read',
  'memory.documents.get',
  'memory.daily.read',
  'office.read_active',
  'office.read_selection',
  'process.read',
  'process.follow',
])

const SEARCH_TOOL_NAMES = new Set([
  'fs.glob',
  'fs.search',
  'git.status',
  'git.diff',
  'git.log',
  'code.symbols',
  'code.dependencies',
  'code.diagnostics',
  'lsp',
  'web.search',
  'apps.search',
  'memory.search',
  'memory.graph.search',
  'memory.documents.search',
  'memory.daily.search',
  'skillhub.search',
  'browser.screenshot',
])

export function toolProvidesContentReadEvidence(tool: string): boolean {
  return CONTENT_READ_TOOL_NAMES.has(tool)
}

export function toolProvidesSearchEvidence(tool: string): boolean {
  return SEARCH_TOOL_NAMES.has(tool)
}
const BROWSER_VISUAL_AUDIT_TOOL_NAMES = new Set([
  'browser.screenshot',
  'browser.click',
  'browser.evaluate',
])

function isNonEvidentialSearchOutput(output: string): boolean {
  return /^\[no matches(?: after offset \d+)?\]$/i.test(output.trim())
}

export function isNonEvidentialReadOutput(output: string): boolean {
  const trimmed = output.trim()
  if (!trimmed) return true
  return /^\[fs\.read: offset \d+ is past (?:the scanned )?end of file/i.test(trimmed)
}

const DIRECT_FILE_MUTATION_TOOL_NAMES = new Set(['fs.write', 'fs.append', 'fs.edit', 'apply_patch'])

function terminalRunCommandText(args: Record<string, unknown> | undefined): string {
  if (!args) return ''
  if (typeof args.cmd === 'string') return args.cmd
  if (typeof args.command === 'string') return args.command
  if (Array.isArray(args.command) && args.command.every((entry) => typeof entry === 'string')) {
    return args.command.join(' ')
  }
  if (typeof args.executable !== 'string') return ''
  const argv = Array.isArray(args.args)
    ? args.args.filter((entry): entry is string => typeof entry === 'string')
    : []
  return [args.executable, ...argv].join(' ')
}

function terminalRunLooksLikeFileMutation(args: Record<string, unknown> | undefined): boolean {
  const command = terminalRunCommandText(args)
  if (!command.trim()) return false
  const normalized = command.replace(/\s+/g, ' ')
  const withoutBenignStdoutRedirection = normalized
    .replace(/\b[12]?>&\d\b/g, ' ')
    .replace(/\b[12]?>\s*\/dev\/null\b/g, ' ')
  return (
    /(^|[^0-9])(?:>|>>|1>)\s*(?![&0-9]|\/dev\/null\b)(?:"[^"]+"|'[^']+'|[^\s;&|]+)/.test(withoutBenignStdoutRedirection)
    || /(?:^|[\s;&|])tee\s+(?:-[a-zA-Z]*a[a-zA-Z]*\s+)?(?!\/dev\/null\b)(?:"[^"]+"|'[^']+'|[^\s;&|]+)/.test(normalized)
    || /(?:^|[\s;&|])(?:touch|mkdir|cp|mv|rm)\b/.test(normalized)
    || /(?:^|[\s;&|])(?:sed\s+-i|perl\s+-pi|git\s+apply|patch)\b/.test(normalized)
    || /\b(?:fs\.)?(?:writeFileSync|writeFile|appendFileSync|appendFile)\s*\(/.test(normalized)
    || /\.(?:write_text|write_bytes)\s*\(/.test(normalized)
    || /\bopen\s*\([^)]*,\s*['"][^'"]*[wa+][^'"]*['"]/.test(normalized)
    || /\b(?:npm|pnpm|yarn|bun)\s+(?:create|init)\b/.test(normalized)
    || /\bnpx\s+(?:create-|degit\b|shadcn\b)/.test(normalized)
  )
}

const SHELL_STATE_OR_CONTROL_COMMANDS = new Set([
  '.', 'source', 'eval', 'exec', 'exit', 'return', 'break', 'continue', 'trap', 'set', 'shift',
  'unset', 'read', 'readarray', 'mapfile', 'getopts', 'let', 'declare', 'typeset',
  'local', 'export', 'readonly', 'printf', 'builtin', 'command', 'enable',
  'if', 'then', 'else', 'elif', 'fi', 'for', 'while', 'until', 'do', 'done',
  'case', 'esac', 'select', 'function', 'coproc', 'time', '[[', ']]',
])

function cleanupPreservesCapturedShellStatus(cleanup: string): boolean {
  if (/[\\$=(){}&|<>#!\x60]/u.test(cleanup)) return false
  return cleanup.replace(/['"]/gu, '').split(/[;\n]/u).every(segment => {
    const commandName = segment.trim().split(/\s+/u)[0] ?? ''
    return !SHELL_STATE_OR_CONTROL_COMMANDS.has(commandName)
  })
}

export function terminalRunHasReliableSuccessStatus(
  args: Record<string, unknown> | undefined,
): boolean {
  if (!args) return true
  const executable = typeof args.executable === 'string'
    ? args.executable.split(/[\\/]/).at(-1)?.toLowerCase() ?? ''
    : ''
  const argv = Array.isArray(args.args)
    ? args.args.filter((entry): entry is string => typeof entry === 'string')
    : []
  let shellCommandIndex = -1
  if (['bash', 'sh', 'zsh', 'dash', 'ksh'].includes(executable)) {
    for (let index = 0; index < argv.length; index += 1) {
      const option = argv[index]!
      if (option === '--' || !option.startsWith('-')) break
      if (/^-[^-]*c/u.test(option)) {
        shellCommandIndex = index + 1
        break
      }
      if (/^-[^-]*[oO]$/u.test(option)) index += 1
    }
  }
  const shellOptions = shellCommandIndex >= 0 ? argv.slice(0, shellCommandIndex) : []
  const command = typeof args.command === 'string'
    ? args.command
    : typeof args.cmd === 'string'
      ? args.cmd
      : Array.isArray(args.command) && args.command.every((entry) => typeof entry === 'string')
        ? args.command.join(' ')
    : ['bash', 'sh', 'zsh', 'dash', 'ksh'].includes(executable)
      ? (shellCommandIndex >= 0 ? argv[shellCommandIndex] : argv.at(-1)) ?? ''
      : ''
  if (!command.trim()) return true

  const assignments = [...command.matchAll(
    /(?:^|[;\n])\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\$\?\s*;/gu,
  )]
  const assignment = assignments.length === 1 ? assignments[0] : undefined
  const finalExit = command.match(
    /(?:^|[;\n])\s*exit\s+("?)\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))\1\s*;?\s*$/u,
  )
  let statusCommand = command
  if (assignments.length > 0 || finalExit) {
    if (assignment?.index === undefined || finalExit?.index === undefined) return false
    const capturedName = assignment[1]
    const exitName = finalExit[2] ?? finalExit[3]
    const cleanup = command.slice(
      assignment.index + assignment[0].length,
      finalExit.index,
    )
    if (
      capturedName !== exitName
      || assignment.index === 0
      || !cleanupPreservesCapturedShellStatus(cleanup)
    ) return false
    statusCommand = command.slice(0, assignment.index)
  }

  const normalized = statusCommand.replace(/\s+/g, ' ')
  const hasFallbackBranch = /(?:^|\s)\|\|(?:\s|$)/.test(normalized)
  const hasPipeline = /(^|[^|])\|([^|]|$)/.test(normalized)
  const hasSequence = /(?:^|[^&]);|\n/u.test(statusCommand)
  const pipefailEnabled = /(?:^|[;&]\s*|\s)set\s+-[^;\n]*o\s+pipefail(?:\s|;|$)/.test(statusCommand)
    || /(?:^|\s)-(?:[a-zA-Z]*o\s+pipefail|[a-zA-Z]*O\s+pipefail)(?:\s|$)/.test(
      [executable, ...shellOptions].join(' '),
    )
  const exitOnErrorEnabled = /(?:^|[;\n]\s*)set\s+-[^;\n]*e/u.test(statusCommand)
    || /(?:^|[;\n]\s*)set\s+-o\s+errexit(?:\s|;|$)/u.test(statusCommand)
  return !hasFallbackBranch
    && (!hasPipeline || pipefailEnabled)
    && (!hasSequence || exitOnErrorEnabled)
}

function shellPathToken(match: RegExpMatchArray): string | null {
  const raw = match[1] ?? match[2] ?? match[3]
  if (!raw || raw === '/dev/null') return null
  return normalizeEvidencePath(raw)
}

function stripShellHeredocBodies(command: string): string {
  const out: string[] = []
  let terminator: string | null = null
  for (const line of command.split('\n')) {
    if (terminator) {
      if (line.trim() === terminator) terminator = null
      continue
    }
    out.push(line)
    const heredoc = line.match(/<<-?\s*['"]?([A-Za-z_][A-Za-z0-9_-]*)['"]?/)
    if (heredoc?.[1]) {
      terminator = heredoc[1]
    }
  }
  return out.join('\n')
}

function extractTerminalRunMutationPaths(args: Record<string, unknown> | undefined): string[] {
  const command = terminalRunCommandText(args)
  if (!command.trim()) return []
  const shellLinesOnly = stripShellHeredocBodies(command)
  const paths = new Set<string>()
  for (const match of shellLinesOnly.matchAll(
    /(?:^|[^0-9])(?:>|>>|1>)\s*(?:"([^"]+)"|'([^']+)'|([^\s;&|]+))/g,
  )) {
    const path = shellPathToken(match)
    if (path) paths.add(path)
  }
  for (const match of shellLinesOnly.matchAll(
    /(?:^|[\s;&|])tee\s+(?:-[a-zA-Z]*a[a-zA-Z]*\s+)?(?:"([^"]+)"|'([^']+)'|([^\s;&|]+))/g,
  )) {
    const path = shellPathToken(match)
    if (path) paths.add(path)
  }
  for (const match of command.matchAll(
    /\b(?:fs\.)?(?:writeFileSync|writeFile|appendFileSync|appendFile)\s*\(\s*(?:"([^"]+)"|'([^']+)'|`([^`]+)`)/g,
  )) {
    const path = shellPathToken(match)
    if (path) paths.add(path)
  }
  for (const match of command.matchAll(
    /\bopen\s*\(\s*(?:"([^"]+)"|'([^']+)'|`([^`]+)`)\s*,\s*['"][^'"]*[wa+][^'"]*['"]/g,
  )) {
    const path = shellPathToken(match)
    if (path) paths.add(path)
  }
  return [...paths]
}

export function toolCallMayMutateFiles(
  toolName: string,
  args: Record<string, unknown> | undefined,
): boolean {
  if (DIRECT_FILE_MUTATION_TOOL_NAMES.has(toolName)) return true
  return toolName === 'terminal.run' && terminalRunLooksLikeFileMutation(args)
}

export function toolCallRepresentsProductMutation(
  toolName: string,
  args: Record<string, unknown> | undefined,
): boolean {
  if (DIRECT_FILE_MUTATION_TOOL_NAMES.has(toolName)) return true
  return toolName === 'terminal.run'
    && args?.actionPurpose === 'mutate'
    && terminalRunLooksLikeFileMutation(args)
}

const VALIDATION_TOOL_NAMES = new Set([
  'terminal.run',
  'assistant.status',
  'gitea.actions.runs.inspect',
  'system.info',
  'usage.report',
  'process.list',
  'process.sessions',
  'process.follow',
  'service.status',
  'service.logs',
  'service.healthcheck',
  'schedule_list',
  'schedule_runs',
])

export function emptyEvidenceLedger(): AgentEvidenceLedger {
  return {
    sourceReads: [],
    sourceSearches: [],
    artifactWrites: [],
    artifactReadBacks: [],
    validationRuns: [],
    errors: [],
  }
}

export function cloneEvidenceLedger(
  ledger: AgentEvidenceLedger | undefined,
): AgentEvidenceLedger | undefined {
  if (!ledger) return undefined
  const cloneEntry = (entry: AgentEvidenceLedgerEntry): AgentEvidenceLedgerEntry => ({
    ...entry,
    ...(entry.defects ? { defects: [...entry.defects] } : {}),
    ...(entry.origin ? { origin: { ...entry.origin } } : {}),
  })
  return {
    sourceReads: ledger.sourceReads.map(cloneEntry),
    sourceSearches: ledger.sourceSearches.map(cloneEntry),
    artifactWrites: ledger.artifactWrites.map(cloneEntry),
    artifactReadBacks: ledger.artifactReadBacks.map(cloneEntry),
    validationRuns: ledger.validationRuns.map(cloneEntry),
    errors: ledger.errors.map(cloneEntry),
  }
}

export type EvidenceOrderMarker = Pick<AgentEvidenceLedgerEntry, 'ts' | 'order'>

function normalizedEvidenceOrder(marker: EvidenceOrderMarker): number | undefined {
  return typeof marker.order === 'number' && Number.isFinite(marker.order)
    ? marker.order
    : undefined
}

export function compareEvidenceOrder(
  left: EvidenceOrderMarker,
  right: EvidenceOrderMarker,
): number {
  const leftOrder = normalizedEvidenceOrder(left)
  const rightOrder = normalizedEvidenceOrder(right)
  if (leftOrder !== undefined && rightOrder !== undefined && leftOrder !== rightOrder) {
    return leftOrder - rightOrder
  }
  return left.ts - right.ts
}

export function evidenceEntryOccursAfter(
  entry: EvidenceOrderMarker,
  marker: EvidenceOrderMarker,
): boolean {
  const entryOrder = normalizedEvidenceOrder(entry)
  const markerOrder = normalizedEvidenceOrder(marker)
  if (entryOrder !== undefined && markerOrder !== undefined) {
    return entryOrder > markerOrder
  }
  return entry.ts > marker.ts
}

function allEvidenceLedgerEntries(ledger: AgentEvidenceLedger): AgentEvidenceLedgerEntry[] {
  return [
    ...ledger.sourceReads,
    ...ledger.sourceSearches,
    ...ledger.artifactWrites,
    ...ledger.artifactReadBacks,
    ...ledger.validationRuns,
    ...ledger.errors,
  ]
}

export function nextEvidenceOrder(ledger: AgentEvidenceLedger): number {
  const entries = allEvidenceLedgerEntries(ledger)
  const maxExistingOrder = entries.reduce((max, entry) => {
    const order = normalizedEvidenceOrder(entry)
    return order === undefined ? max : Math.max(max, order)
  }, 0)
  return Math.max(maxExistingOrder, entries.length) + 1
}

function compactLedgerSummary(value: string, max = MAX_LEDGER_SUMMARY_CHARS): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  if (normalized.length <= max) return normalized
  return `${normalized.slice(0, max - 3).trimEnd()}...`
}

function extractBrowserLayoutAuditWarnings(output: string): string[] {
  const lines = output.split('\n')
  const warnings: string[] = []
  let inLayoutAudit = false
  let inWarnings = false
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed === 'Browser layout audit:') {
      inLayoutAudit = true
      inWarnings = false
      continue
    }
    if (!inLayoutAudit) continue
    if (trimmed === 'Browser console/page audit:') break
    if (trimmed === '- warnings:') {
      inWarnings = true
      continue
    }
    if (!inWarnings) continue
    if (/^- examples:/.test(trimmed) || (/^-\s/.test(trimmed) && !/^\s+-\s/.test(line))) {
      break
    }
    const warning = line.match(/^\s+-\s+(.+)$/)?.[1]?.trim()
    if (!warning) continue
    if (/^none detected by DOM\/canvas layout audit$/i.test(warning)) continue
    if (/^none detected by DOM layout audit$/i.test(warning)) continue
    warnings.push(compactLedgerSummary(warning, 220))
  }
  return warnings.slice(0, 8)
}

function extractBrowserConsolePageAuditErrors(output: string): string[] {
  const lines = output.split('\n')
  const errors: string[] = []
  let inAudit = false
  let inErrors = false
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed === 'Browser console/page audit:') {
      inAudit = true
      inErrors = false
      continue
    }
    if (!inAudit) continue
    if (trimmed === '- errors:') {
      inErrors = true
      continue
    }
    if (trimmed === '- warnings:' || /^[A-Z][A-Za-z /-]+ audit:$/.test(trimmed)) {
      break
    }
    if (!inErrors) continue
    if (/^-\s/.test(trimmed) && !/^\s+-\s/.test(line)) {
      break
    }
    const error = line.match(/^\s+-\s+(.+)$/)?.[1]?.trim()
    if (!error) continue
    if (/^none detected by browser console\/page audit$/i.test(error)) continue
    errors.push(compactLedgerSummary(error, 220))
  }
  return errors.slice(0, 8)
}

function extractToolDefects(toolCall: ToolCall, output: string): string[] | undefined {
  if (!BROWSER_VISUAL_AUDIT_TOOL_NAMES.has(toolCall.name)) return undefined
  const defects = [
    ...extractBrowserLayoutAuditWarnings(output),
    ...extractBrowserConsolePageAuditErrors(output),
  ]
  return defects.length > 0 ? defects : undefined
}

function compactAuditPart(value: string, max = 96): string {
  return compactLedgerSummary(value, max)
}

function browserUiAuditKey(toolCall: ToolCall): string | undefined {
  if (!BROWSER_VISUAL_AUDIT_TOOL_NAMES.has(toolCall.name)) return undefined
  const args = toolCall.arguments ?? {}
  const url = typeof args.url === 'string' && args.url.trim()
    ? compactAuditPart(args.url.trim())
    : 'unknown-url'
  const viewportWidth = typeof args.viewportWidth === 'number' || typeof args.viewportWidth === 'string'
    ? String(args.viewportWidth)
    : ''
  const viewportHeight = typeof args.viewportHeight === 'number' || typeof args.viewportHeight === 'string'
    ? String(args.viewportHeight)
    : ''
  const viewport = viewportWidth || viewportHeight
    ? `viewport:${viewportWidth || 'auto'}x${viewportHeight || 'auto'}`
    : ''
  const path = typeof args.path === 'string' && args.path.trim()
    ? `path:${compactAuditPart(args.path.trim())}`
    : ''
  const state = toolCall.name === 'browser.click' && typeof args.selector === 'string' && args.selector.trim()
    ? `selector:${compactAuditPart(args.selector.trim())}`
    : toolCall.name === 'browser.evaluate'
      ? browserEvaluateAuditState(args.script)
      : 'screenshot'
  return [toolCall.name, url, viewport || path || 'default-view', state].join('|')
}

function stripJavaScriptComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
}

function isLiteralOnlyBrowserEvaluateScript(script: string): boolean {
  const normalized = stripJavaScriptComments(script)
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/;$/, '')
  return /^(?:return\s+)?(?:true|false|null|undefined|[-+]?\d+(?:\.\d+)?|["'`][^"'`]*["'`]|\{\s*(?:ok|success|passed|active|done|result)\s*:\s*(?:true|false|[-+]?\d+(?:\.\d+)?|["'`][^"'`]*["'`])\s*\}|\[\s*\])$/i
    .test(normalized)
}

function browserEvaluateScriptTouchesRenderedUi(script: string): boolean {
  const source = stripJavaScriptComments(script)
  if (isLiteralOnlyBrowserEvaluateScript(source)) return false
  return /\b(?:document|window|localStorage|sessionStorage|history|location|navigator|requestAnimationFrame|getComputedStyle|getBoundingClientRect|querySelector|getElementById|elementsFromPoint|dispatchEvent|MouseEvent|KeyboardEvent|PointerEvent|TouchEvent|click|focus|blur|scroll|canvas|getContext|classList|dataset|innerText|textContent|aria|role|value|checked|disabled|selected|clientWidth|clientHeight|scrollWidth|scrollHeight|offsetWidth|offsetHeight)\b/i
    .test(source)
}

function browserEvaluateScriptPerformsUiInteraction(script: string): boolean {
  const source = stripJavaScriptComments(script)
  return /(?:\.(?:click|focus|blur|scroll(?:To|By|IntoView)?|requestSubmit|submit|play|pause)\s*\(|\bdispatchEvent\s*\()/i
    .test(source)
}

function unquoteJavaScriptStringLiteral(value: string): string {
  const trimmed = value.trim()
  if (trimmed.length < 2) return trimmed
  const quote = trimmed[0]
  if ((quote !== '"' && quote !== '\'' && quote !== '`') || trimmed.at(-1) !== quote) return trimmed
  return trimmed.slice(1, -1).replace(/\\(["'`\\])/g, '$1')
}

function normalizeBrowserEvaluateScript(source: string): string {
  return stripJavaScriptComments(source)
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/;+\s*$/g, '')
}

function browserEvaluateSelectorTarget(kind: string, rawSelector: string): string {
  const selector = compactAuditPart(unquoteJavaScriptStringLiteral(rawSelector), 96)
  return kind === 'getElementById' ? `id:${selector}` : `selector:${selector}`
}

function extractBrowserEvaluateInteractionTarget(script: string): string | null {
  const source = normalizeBrowserEvaluateScript(script)
  const directCall = source.match(
    /\b(?:document\.)?(querySelector|getElementById)\s*\(\s*(["'`][^"'`]{1,160}["'`])\s*\)\s*\??\.\s*(?:click|focus|blur|scroll(?:To|By|IntoView)?|requestSubmit|submit|play|pause|dispatchEvent)\s*\(/i,
  )
  if (directCall?.[1] && directCall[2]) {
    return browserEvaluateSelectorTarget(directCall[1], directCall[2])
  }

  const referencedTargets = [...source.matchAll(
    /\b(?:document\.)?(querySelector|getElementById)\s*\(\s*(["'`][^"'`]{1,160}["'`])\s*\)/ig,
  )].map((match) => browserEvaluateSelectorTarget(match[1] ?? 'querySelector', match[2] ?? ''))
    .filter(Boolean)
  const uniqueTargets = [...new Set(referencedTargets)]
  if (uniqueTargets.length === 1) return uniqueTargets[0] ?? null
  return null
}

function browserEvaluateAuditState(script: unknown): string {
  if (typeof script !== 'string' || !browserEvaluateScriptPerformsUiInteraction(script)) return 'evaluate'
  const target = extractBrowserEvaluateInteractionTarget(script)
  if (target) return `evaluate:${target}`
  const fingerprint = compactAuditPart(normalizeBrowserEvaluateScript(script).toLowerCase(), 96)
  return fingerprint ? `evaluate:script:${fingerprint}` : 'evaluate'
}

function browserEvaluateScriptHasSpecificFallbackInteraction(script: string): boolean {
  const source = normalizeBrowserEvaluateScript(script)
  if (!source) return false
  if (/\b(?:KeyboardEvent|key\s*:)\b/i.test(source)) return true
  return /\b(?:canvas|getContext|elementsFromPoint)\b/i.test(source)
    && /\b(?:dispatchEvent|MouseEvent|PointerEvent|TouchEvent|click)\b/i.test(source)
}

function browserClickSelectorTargetsSpecificControl(selector: string): boolean {
  const normalized = selector.trim().toLowerCase().replace(/\s+/g, ' ')
  if (!normalized) return false
  if (/^(?:html|body|:root|#root|#app|main|\*)$/i.test(normalized)) return false
  if (
    /(?:^|[\s>+~])(?:button|a|input|select|textarea|canvas|svg)(?::(?:first-child|last-child|first-of-type|last-of-type|nth-child\([^)]*\)|nth-of-type\([^)]*\)))?$/i
      .test(normalized)
  ) {
    return false
  }
  return /(?:^|[\s>+~])(?:[#.][a-z0-9_-]+|[a-z][a-z0-9_-]*(?:[#.][a-z0-9_-]+|\[(?:data-testid|data-test|data-cy|aria-label|name|title|value)=["']?[^"'\]]+["']?\])|\[(?:data-testid|data-test|data-cy|aria-label|name|title|value)=["']?[^"'\]]+["']?\])(?:$|[#.\[:\s>+~])/i
    .test(normalized)
}

export function browserInteractionTargetStateIsSpecific(
  toolName: string,
  state: string | null | undefined,
): boolean {
  const normalizedState = state?.trim()
  if (!normalizedState) return false
  if (toolName === 'browser.click') {
    const selector = normalizedState.startsWith('selector:')
      ? normalizedState.slice('selector:'.length)
      : normalizedState
    return browserClickSelectorTargetsSpecificControl(selector)
  }
  if (toolName === 'browser.evaluate') {
    if (normalizedState.startsWith('evaluate:script:')) {
      return browserEvaluateScriptHasSpecificFallbackInteraction(
        normalizedState.slice('evaluate:script:'.length),
      )
    }
    if (normalizedState.startsWith('evaluate:selector:')) {
      return browserClickSelectorTargetsSpecificControl(
        normalizedState.slice('evaluate:selector:'.length),
      )
    }
    if (normalizedState.startsWith('evaluate:id:')) {
      return browserClickSelectorTargetsSpecificControl(
        `#${normalizedState.slice('evaluate:id:'.length)}`,
      )
    }
  }
  return false
}

function browserToolProvidesInteractionSmoke(toolCall: ToolCall): boolean {
  const args = toolCall.arguments ?? {}
  if (toolCall.name === 'browser.click') {
    return typeof args.selector === 'string' && browserClickSelectorTargetsSpecificControl(args.selector)
  }
  if (toolCall.name === 'browser.evaluate') {
    const script = typeof args.script === 'string' ? args.script : ''
    const target = extractBrowserEvaluateInteractionTarget(script)
    return browserEvaluateScriptTouchesRenderedUi(script)
      && browserEvaluateScriptPerformsUiInteraction(script)
      && (target
        ? browserInteractionTargetStateIsSpecific(toolCall.name, `evaluate:${target}`)
        : browserEvaluateScriptHasSpecificFallbackInteraction(script))
  }
  return false
}

function outputHasBrowserLayoutAudit(output: string): boolean {
  return /^\s*Browser layout audit:\s*$/im.test(output)
}

function outputHasBrowserConsolePageAudit(output: string): boolean {
  return /^\s*Browser console\/page audit:\s*$/im.test(output)
}

function outputHasBrowserScreenshotImageAttachment(output: string): boolean {
  return /^\s*Screenshot image attachment:\s*attached\b/im.test(output)
}

function normalizeEvidencePath(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const normalized = value
    .replace(/\\/g, '/')
    .replace(/\/+/g, '/')
    .replace(/^\.\//, '')
    .replace(/\/$/, '')
    .trim()
  return normalized ? normalized : null
}

function evidencePathsReferToSameFile(a: string, b: string): boolean {
  const left = normalizeEvidencePath(a)
  const right = normalizeEvidencePath(b)
  if (!left || !right) return false
  return left === right
    || searchPathMatchesReadPath(left, right)
    || searchPathMatchesReadPath(right, left)
    || left.endsWith(`/${right}`)
    || right.endsWith(`/${left}`)
}

export function collectRequiredEvidenceArtifactPaths(
  state: AgentState,
  context?: GraphExecutionContext,
): string[] {
  const contract = state.seedContract ?? context?.agentContext.runContract
  const paths = new Set<string>()
  for (const artifact of contract?.requiredArtifacts ?? []) {
    const path = normalizeEvidencePath(artifact.path)
    if (path) paths.add(path)
  }
  for (const section of contract?.artifactSections ?? []) {
    const path = normalizeEvidencePath(section.artifactPath)
    if (path) paths.add(path)
  }
  return [...paths]
}

function isRequiredArtifactPath(
  path: string | undefined,
  state: AgentState,
  context?: GraphExecutionContext,
): boolean {
  if (!path) return false
  return collectRequiredEvidenceArtifactPaths(state, context)
    .some((artifactPath) => evidencePathsReferToSameFile(path, artifactPath))
}

function extractApplyPatchPaths(patch: string): string[] {
  const paths = new Set<string>()
  for (const match of patch.matchAll(/^\*\*\* (?:Add|Update|Delete) File:\s*(.+)$/gm)) {
    const path = normalizeEvidencePath(match[1]?.trim())
    if (path) paths.add(path)
  }
  for (const match of patch.matchAll(/^\+\+\+\s+(?!\/dev\/null)(?:b\/)?(.+)$/gm)) {
    const path = normalizeEvidencePath(match[1]?.trim())
    if (path) paths.add(path)
  }
  return [...paths]
}

export function extractPathsFromToolCall(toolCall: ToolCall): string[] {
  const args = toolCall.arguments ?? {}
  if (
    (toolCall.name === 'fs.read' || toolCall.name === 'fs.write' || toolCall.name === 'fs.append' || toolCall.name === 'fs.edit')
    && typeof args.path === 'string'
  ) {
    const path = normalizeEvidencePath(args.path)
    return path ? [path] : []
  }
  if (toolCall.name === 'apply_patch' && typeof args.patch === 'string') {
    return extractApplyPatchPaths(args.patch)
  }
  if (toolCallRepresentsProductMutation(toolCall.name, args)) {
    return extractTerminalRunMutationPaths(args)
  }
  if (typeof args.path === 'string') {
    const path = normalizeEvidencePath(args.path)
    return path ? [path] : []
  }
  return []
}

function extractQueryFromToolCall(toolCall: ToolCall): string | undefined {
  const args = toolCall.arguments ?? {}
  const query = typeof args.query === 'string'
    ? args.query
    : typeof args.pattern === 'string'
      ? args.pattern
      : typeof args.symbol === 'string'
        ? args.symbol
        : typeof args.url === 'string'
          ? args.url
          : undefined
  if (query) return compactLedgerSummary(query, 180)
  if (Array.isArray(args.patterns)) {
    const patterns = args.patterns.filter((entry): entry is string => typeof entry === 'string')
    if (patterns.length > 0) return compactLedgerSummary(patterns.join(', '), 180)
  }
  return undefined
}

function extractCommandFromToolCall(toolCall: ToolCall): string | undefined {
  const args = toolCall.arguments ?? {}
  if (typeof args.command === 'string' && args.command.trim()) {
    return compactLedgerSummary(args.command, 220)
  }
  if (typeof args.executable === 'string' && args.executable.trim()) {
    const argv = Array.isArray(args.args)
      ? args.args.filter((entry): entry is string => typeof entry === 'string')
      : []
    return compactLedgerSummary([args.executable, ...argv].join(' '), 220)
  }
  return undefined
}

function extractWrittenBytes(toolCall: ToolCall, output: string): number | undefined {
  const match = output.match(/\bWrote\s+(\d+)\s+bytes\b/i)
  if (match) {
    const value = Number(match[1])
    return Number.isFinite(value) ? value : undefined
  }
  const appendMatch = output.match(/\bAppended\s+(\d+)\s+bytes\b/i)
  if (appendMatch) {
    const value = Number(appendMatch[1])
    return Number.isFinite(value) ? value : undefined
  }
  const content = toolCall.arguments?.content
  if (typeof content === 'string') {
    return Buffer.byteLength(content, 'utf8')
  }
  return undefined
}

export function pushBoundedEvidence(
  list: AgentEvidenceLedgerEntry[],
  entry: AgentEvidenceLedgerEntry,
): void {
  list.push(entry)
  if (list.length > MAX_LEDGER_ENTRIES_PER_BUCKET) {
    list.splice(0, list.length - MAX_LEDGER_ENTRIES_PER_BUCKET)
  }
}

function baseEntry(
  toolCall: ToolCall,
  status: 'success' | 'error',
  output: string,
  order: number,
  provenance?: {
    executionObserved?: boolean
    securityEffect?: ToolSecurityEffect
  },
): AgentEvidenceLedgerEntry {
  return {
    toolCallId: toolCall.id,
    tool: toolCall.name,
    status,
    ts: Date.now(),
    order,
    summary: compactLedgerSummary(output),
    ...(typeof toolCall.arguments.actionPurpose === 'string'
      ? { actionPurpose: toolCall.arguments.actionPurpose }
      : {}),
    ...(provenance?.securityEffect
      ? { securityEffect: provenance.securityEffect }
      : {}),
    ...(provenance?.executionObserved === true
      ? { executionObserved: true }
      : {}),
  }
}

export function updateEvidenceLedgerFromToolResult(
  state: AgentState,
  toolCall: ToolCall,
  status: 'success' | 'error',
  output: string,
  context?: GraphExecutionContext,
  executionPosture?: ToolExecutionPosture,
  provenance?: {
    executionObserved?: boolean
    securityEffect?: ToolSecurityEffect
    managedProcess?: unknown
  },
): void {
  const ledger = state.evidenceLedger ?? emptyEvidenceLedger()
  state.evidenceLedger = ledger

  const paths = extractPathsFromToolCall(toolCall)
  const query = extractQueryFromToolCall(toolCall)
  const command = extractCommandFromToolCall(toolCall)
  const defects = extractToolDefects(toolCall, output)
  const uiAuditKey = browserUiAuditKey(toolCall)
  const uiAuditHasLayout = outputHasBrowserLayoutAudit(output)
  const uiAuditHasConsole = outputHasBrowserConsolePageAudit(output)
  const uiAuditHasImageAttachment = outputHasBrowserScreenshotImageAttachment(output)
  const uiInteractionSmoke = browserToolProvidesInteractionSmoke(toolCall)
  const order = nextEvidenceOrder(ledger)
  const recordBrowserUiAuditValidation = (entry: AgentEvidenceLedgerEntry): void => {
    if (!uiAuditKey || !uiAuditHasLayout) return
    pushBoundedEvidence(ledger.validationRuns, {
      ...entry,
      verified: !entry.defects?.length,
    })
  }
  const makeEntry = (path?: string): AgentEvidenceLedgerEntry => ({
    ...baseEntry(toolCall, status, output, order, provenance),
    ...(path ? { path } : {}),
    ...(query ? { query } : {}),
    ...(command ? { command } : {}),
    ...(defects ? { defects } : {}),
    ...(uiAuditKey ? { uiAuditKey } : {}),
    ...(uiAuditHasLayout ? { uiAuditHasLayout } : {}),
    ...(uiAuditHasConsole ? { uiAuditHasConsole } : {}),
    ...(uiAuditHasImageAttachment ? { uiAuditHasImageAttachment } : {}),
    ...(uiInteractionSmoke ? { uiInteractionSmoke } : {}),
    ...(executionPosture ? { executionPosture } : {}),
    ...(toolCallRepresentsProductMutation(toolCall.name, toolCall.arguments)
      ? { bytes: extractWrittenBytes(toolCall, output) }
      : {}),
  })

  if (status === 'error') {
    const targets = paths.length > 0 ? paths : [undefined]
    for (const path of targets) {
      pushBoundedEvidence(ledger.errors, makeEntry(path))
    }
    return
  }

  if (toolCall.name === 'tool.output.read') return

  if (toolCallRepresentsProductMutation(toolCall.name, toolCall.arguments)) {
    const targets = paths.length > 0 ? paths : [undefined]
    for (const path of targets) {
      pushBoundedEvidence(ledger.artifactWrites, makeEntry(path))
    }
    return
  }

  if (MANAGED_PROCESS_RECEIPT_TOOLS.has(toolCall.name)) {
    const receipt = provenance?.managedProcess
    if (provenance?.executionObserved === true && validManagedProcessReceipt(receipt)) {
      let launch = ledger.validationRuns.find(entry =>
        entry.managedProcessId === receipt.id && entry.managedProcessStartedAt === receipt.startedAt,
      )
      if (toolCall.name === 'process.start' && !launch) {
        launch = {
          ...makeEntry(), ts: Date.parse(receipt.startedAt), verified: false,
          managedProcessId: receipt.id, managedProcessStartedAt: receipt.startedAt,
          managedProcessReliableStatus: terminalRunHasReliableSuccessStatus(toolCall.arguments),
        }
        pushBoundedEvidence(ledger.validationRuns, launch)
      }
      if (launch && (toolCall.name === 'process.start' || toolCall.arguments.id === receipt.id) && receipt.status === 'exited') {
        const changedDuringRun = ledger.artifactWrites.some(entry => entry.status === 'success'
          && ((entry.order ?? 0) > (launch.order ?? 0) || entry.ts >= launch.ts))
        Object.assign(launch, {
          toolCallId: toolCall.id, tool: toolCall.name, summary: compactLedgerSummary(output),
          verified: receipt.exitCode === 0 && receipt.signal === null && receipt.terminationReason === undefined
            && launch.managedProcessReliableStatus === true && !changedDuringRun,
        })
      }
    }
    if (toolCall.name !== 'process.start' && !isNonEvidentialReadOutput(output)) {
      pushBoundedEvidence(ledger.sourceReads, makeEntry())
    }
    return
  }

  if (
    (toolCall.name === 'fs.read' || CONTENT_READ_TOOL_NAMES.has(toolCall.name))
    && isNonEvidentialReadOutput(output)
  ) {
    return
  }

  if (toolCall.name === 'fs.read') {
    const targets = paths.length > 0 ? paths : [undefined]
    for (const path of targets) {
      const writtenArtifact = path && latestSuccessfulPathEntry(ledger.artifactWrites, path)
      if (isRequiredArtifactPath(path, state, context) || writtenArtifact) {
        pushBoundedEvidence(ledger.artifactReadBacks, { ...makeEntry(path), verified: true })
      } else {
        pushBoundedEvidence(ledger.sourceReads, makeEntry(path))
      }
    }
    return
  }

  if (CONTENT_READ_TOOL_NAMES.has(toolCall.name)) {
    const targets = paths.length > 0 ? paths : [undefined]
    for (const path of targets) {
      const entry = makeEntry(path)
      pushBoundedEvidence(ledger.sourceReads, entry)
      recordBrowserUiAuditValidation(entry)
    }
    return
  }

  if (SEARCH_TOOL_NAMES.has(toolCall.name)) {
    if (isNonEvidentialSearchOutput(output)) {
      return
    }
    const entry = makeEntry(paths[0])
    pushBoundedEvidence(ledger.sourceSearches, entry)
    recordBrowserUiAuditValidation(entry)
    return
  }

  if (VALIDATION_TOOL_NAMES.has(toolCall.name)) {
    const verified = toolCall.name !== 'terminal.run'
      || terminalRunHasReliableSuccessStatus(toolCall.arguments)
    pushBoundedEvidence(ledger.validationRuns, { ...makeEntry(paths[0]), verified })
    return
  }

  if (
    provenance?.executionObserved === true
    && provenance.securityEffect === 'observe'
  ) {
    pushBoundedEvidence(ledger.sourceReads, makeEntry(paths[0]))
  }
}

function distinctSuccessfulSourceReadPaths(
  state: AgentState,
  context?: GraphExecutionContext,
): string[] {
  const seen = new Set<string>()
  const paths: string[] = []
  for (const entry of state.evidenceLedger?.sourceReads ?? []) {
    if (entry.status !== 'success' || entry.tool !== 'fs.read' || !entry.path) continue
    if (isRequiredArtifactPath(entry.path, state, context)) continue
    const normalized = normalizeEvidencePath(entry.path)
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    paths.push(normalized)
  }
  return paths
}

function distinctSuccessfulSourceObservations(
  state: AgentState,
  context?: GraphExecutionContext,
): AgentEvidenceLedgerEntry[] {
  const entries = [
    ...(state.evidenceLedger?.sourceReads ?? []),
    ...(state.evidenceLedger?.sourceSearches ?? []),
  ].filter((entry) =>
    entry.status === 'success'
    && !(entry.path && isRequiredArtifactPath(entry.path, state, context))
  )
  const seen = new Set<string>()
  return entries.filter((entry) => {
    const identity = entry.toolCallId
      ? `call:${entry.toolCallId}`
      : `entry:${entry.tool}:${entry.order ?? entry.ts}:${entry.path ?? ''}:${entry.query ?? ''}`
    if (seen.has(identity)) return false
    seen.add(identity)
    return true
  })
}

export function collectObservedSourceReadPaths(state: AgentState): string[] {
  const seen = new Set<string>()
  const ledger = state.evidenceLedger
  for (const entry of [
    ...(ledger?.sourceReads ?? []),
    ...(ledger?.artifactReadBacks ?? []),
  ]) {
    if (entry.status !== 'success' || !entry.path) continue
    const normalized = normalizeEvidencePath(entry.path)
    if (normalized) seen.add(normalized)
  }
  return [...seen]
}

export function evidencePathMatchesObserved(
  candidate: string,
  observed: string[],
): boolean {
  const normalized = normalizeEvidencePath(candidate)
  if (!normalized) return false
  return observed.some((path) => evidencePathsReferToSameFile(path, normalized))
}

function pathHasFileExtension(value: string): boolean {
  return /\.[A-Za-z0-9][A-Za-z0-9_-]{0,15}$/.test(value)
}

function directorySegmentsForEvidencePath(path: string): string[] {
  const normalized = normalizeEvidencePath(path)
  if (!normalized) return []
  const segments = normalized.split('/').filter(Boolean)
  const last = segments.at(-1) ?? ''
  return pathHasFileExtension(last) ? segments.slice(0, -1) : segments
}

function commonLeadingSegments(paths: string[][]): string[] {
  if (paths.length === 0) return []
  const [first, ...rest] = paths
  const common = [...first]
  for (const path of rest) {
    let index = 0
    while (index < common.length && index < path.length && common[index] === path[index]) {
      index += 1
    }
    common.length = index
    if (common.length === 0) break
  }
  return common
}

function sourceEvidenceScopes(paths: string[]): string[] {
  const directories = paths
    .map(directorySegmentsForEvidencePath)
    .filter((segments) => segments.length > 0)
  const commonPrefix = commonLeadingSegments(directories)
  const scopes = new Set<string>()
  for (const segments of directories) {
    const scopeSegments = segments.length > commonPrefix.length
      ? segments.slice(0, commonPrefix.length + 1)
      : segments
    const scope = scopeSegments.join('/')
    if (scope) scopes.add(scope)
  }
  return Array.from(scopes)
}

function latestSuccessfulPathEntry(
  entries: AgentEvidenceLedgerEntry[],
  path: string,
): AgentEvidenceLedgerEntry | null {
  let latest: AgentEvidenceLedgerEntry | null = null
  for (const entry of entries) {
    if (
      entry.status === 'success'
      && entry.path
      && evidencePathsReferToSameFile(entry.path, path)
      && (!latest || compareEvidenceOrder(entry, latest) > 0)
    ) {
      latest = entry
    }
  }
  return latest
}

function hasSuccessfulArtifactReadAfter(
  state: AgentState,
  path: string,
  write: EvidenceOrderMarker,
): boolean {
  return (state.evidenceLedger?.artifactReadBacks ?? []).some((entry) =>
    entry.status === 'success'
    && entry.path
    && evidenceEntryOccursAfter(entry, write)
    && evidencePathsReferToSameFile(entry.path, path)
  )
}

function retrievalSettlesValidation(
  ledger: AgentEvidenceLedger | undefined,
  artifactPaths: readonly string[],
): boolean {
  if (artifactPaths.length > 0) return false
  if ((ledger?.artifactWrites ?? []).some((entry) => entry.status === 'success')) return false
  return (ledger?.sourceReads ?? []).some((entry) => entry.status === 'success')
    || (ledger?.sourceSearches ?? []).some((entry) => entry.status === 'success')
}

export function evaluateContractEvidenceGaps(
  state: AgentState,
  context?: GraphExecutionContext,
): string[] {
  const contract = state.seedContract ?? context?.agentContext.runContract
  if (!contract) return []

  const gaps: string[] = []
  const ledger = state.evidenceLedger
  const artifactPaths = collectRequiredEvidenceArtifactPaths(state, context)
  for (const artifactPath of artifactPaths) {
    const write = latestSuccessfulPathEntry(ledger?.artifactWrites ?? [], artifactPath)
    if (!write) {
      gaps.push(`Required artifact has no successful write evidence yet: ${artifactPath}.`)
      continue
    }
    if (!hasSuccessfulArtifactReadAfter(state, artifactPath, write)) {
      gaps.push(`Required artifact was written but not read back after the latest write: ${artifactPath}.`)
    }
  }

  gaps.push(...evaluateContractSourceEvidenceGaps(state, context))

  if (
    (contract.evidenceRequirements ?? []).some((requirement) => requirement.kind === 'validation')
    && !(ledger?.validationRuns ?? []).some(
      (entry) => entry.status === 'success' && entry.verified === true,
    )
    && !retrievalSettlesValidation(ledger, artifactPaths)
  ) {
    gaps.push('Validation evidence is required by the run contract, but no successful validation/check run is recorded.')
  }

  return gaps
}

export function evaluateContractSourceEvidenceGaps(
  state: AgentState,
  context?: GraphExecutionContext,
): string[] {
  const contract = state.seedContract ?? context?.agentContext.runContract
  if (!contract) return []

  const gaps: string[] = []
  const ledger = state.evidenceLedger
  const sourceRequirements = (contract.evidenceRequirements ?? [])
    .filter((requirement) =>
      requirement.kind === 'source' || requirement.kind === 'repository'
    )
  const sourceReadPaths = distinctSuccessfulSourceReadPaths(state, context)
  const sourceScopeCount = sourceEvidenceScopes(sourceReadPaths).length
  const observations = distinctSuccessfulSourceObservations(state, context)

  for (const requirement of sourceRequirements) {
    const matchingObservations = observations.filter((entry) =>
      sourceToolMatchesEvidenceRequirement(requirement, entry.tool)
    )
    const requiredObservations = evidenceRequirementMinSourceObservations(requirement)
    if (matchingObservations.length < requiredObservations) {
      const toolBoundary = requirement.sourceToolNames?.length
        ? ` from ${requirement.sourceToolNames.join(' or ')}`
        : ''
      gaps.push(
        `Source observation evidence is incomplete: ${matchingObservations.length}/${requiredObservations} successful observation(s)${toolBoundary}.`,
      )
    }

    if (evidenceRequirementUsesRepositoryBreadth(requirement)) {
      const requiredSourceFiles = evidenceRequirementMinSourceFiles(requirement)
      if (sourceReadPaths.length < requiredSourceFiles) {
        gaps.push(
          `Source evidence is incomplete: ${sourceReadPaths.length}/${requiredSourceFiles} successful source file read(s).`,
        )
      }
      const requiredSourceScopes = evidenceRequirementMinSourceScopes(requirement)
      if (sourceScopeCount < requiredSourceScopes) {
        gaps.push(
          `Source scope evidence is incomplete: ${sourceScopeCount}/${requiredSourceScopes} distinct source scope(s).`,
        )
      }
    }

    if (
      evidenceRequirementRequiresSearch(requirement)
      && !(ledger?.sourceSearches ?? []).some((entry) =>
        entry.status === 'success'
        && sourceToolMatchesEvidenceRequirement(requirement, entry.tool)
      )
    ) {
      gaps.push('Source evidence requires at least one successful search/inventory step, but none is recorded.')
    }
  }

  return gaps
}

export function requiredArtifactReadBackGapPaths(
  state: AgentState,
  context?: GraphExecutionContext,
): string[] {
  const ledger = state.evidenceLedger
  return collectRequiredEvidenceArtifactPaths(state, context).filter((artifactPath) => {
    const write = latestSuccessfulPathEntry(ledger?.artifactWrites ?? [], artifactPath)
    return Boolean(write && !hasSuccessfulArtifactReadAfter(state, artifactPath, write))
  })
}

function formatLedgerEntry(entry: AgentEvidenceLedgerEntry): string {
  const target = entry.path
    ? entry.path
    : entry.command
      ? entry.command
      : entry.query
        ? entry.query
        : entry.summary
          ? entry.summary
          : '(no target)'
  const provenance = entry.origin ? ` (via subagent ${entry.origin.category})` : ''
  const defects = entry.defects?.length
    ? ` [defects: ${entry.defects.slice(0, 2).join('; ')}]`
    : ''
  const uiEvidence = [
    entry.uiAuditHasLayout ? 'layout audit' : '',
    entry.uiAuditHasImageAttachment ? 'image attached' : '',
    entry.uiInteractionSmoke ? 'interaction smoke' : '',
  ].filter(Boolean)
  const uiEvidenceSuffix = uiEvidence.length > 0 ? ` [${uiEvidence.join(', ')}]` : ''
  return `${entry.tool}: ${target}${provenance}${uiEvidenceSuffix}${defects}`
}

function formatRecentEntries(
  label: string,
  entries: AgentEvidenceLedgerEntry[],
  limit = 3,
): string[] {
  const recent = entries
    .filter((entry) => entry.status === 'success')
    .slice(-limit)
    .map((entry) => `- ${formatLedgerEntry(entry)}`)
  return recent.length > 0 ? [`${label}:`, ...recent] : []
}

function formatReferenceableObservations(ledger: AgentEvidenceLedger): string[] {
  const byToolCallId = new Map<string, AgentEvidenceLedgerEntry>()
  for (const entry of [
    ...ledger.sourceReads,
    ...ledger.sourceSearches,
    ...ledger.validationRuns,
  ]) {
    if (
      entry.status !== 'success'
      || !entry.toolCallId
      || !isExecutorConfirmedReadOnlyObservation({
        tool: entry.tool,
        securityEffect: entry.securityEffect,
        executionObserved: entry.executionObserved,
        actionPurpose: entry.actionPurpose,
        executionPosture: entry.executionPosture,
      })
    ) continue
    byToolCallId.set(entry.toolCallId, entry)
  }
  const referenceable = [...byToolCallId.values()]
    .sort(compareEvidenceOrder)
    .slice(-16)
  if (referenceable.length === 0) return []
  return [
    'Criterion-referenceable read-only observations:',
    ...referenceable.map((entry) => `- [evidence ${entry.toolCallId}] ${formatLedgerEntry(entry)}`),
    'For an internally reported satisfied criterion, use the exact protocol `CRITERION <id>: MET EVIDENCE <tool-call-id,...>` with only ids shown above. Use `CRITERION <id>: UNMET` when it is not satisfied.',
  ]
}

function formatArtifactStatus(
  state: AgentState,
  context?: GraphExecutionContext,
): string[] {
  const paths = collectRequiredEvidenceArtifactPaths(state, context)
  if (paths.length === 0) return []
  const lines = ['Required artifact status:']
  for (const path of paths) {
    const write = latestSuccessfulPathEntry(state.evidenceLedger?.artifactWrites ?? [], path)
    if (!write) {
      lines.push(`- ${path}: no successful write recorded`)
      continue
    }
    const readBack = hasSuccessfulArtifactReadAfter(state, path, write)
    const bytes = typeof write.bytes === 'number' ? `, ${write.bytes} bytes` : ''
    lines.push(`- ${path}: written by ${write.tool}${bytes}; read-back ${readBack ? 'recorded' : 'missing'}`)
  }
  return lines
}

export function formatEvidenceLedgerForPrompt(
  state: AgentState,
  context?: GraphExecutionContext,
): string | null {
  const ledger = state.evidenceLedger
  const hasEntries = ledger
    ? Object.values(ledger).some((entries) => entries.length > 0)
    : false
  const gaps = evaluateContractEvidenceGaps(state, context)
  if (!hasEntries && gaps.length === 0) return null

  const safeLedger = ledger ?? emptyEvidenceLedger()
  const lines = [
    '[Evidence ledger]',
    'This is structured evidence from actual tool results and survives context compaction. Use it as observed tool evidence, not as a natural-language claim.',
    `Counts: source reads ${safeLedger.sourceReads.length}; searches/inventory ${safeLedger.sourceSearches.length}; artifact writes ${safeLedger.artifactWrites.length}; artifact read-backs ${safeLedger.artifactReadBacks.length}; validation/check runs ${safeLedger.validationRuns.length}; errors ${safeLedger.errors.length}.`,
    ...formatArtifactStatus(state, context),
    ...(gaps.length > 0
      ? ['Contract evidence gaps:', ...gaps.map((gap) => `- ${gap}`)]
      : ['Contract evidence gaps: none recorded.']),
    ...formatReferenceableObservations(safeLedger),
    ...formatRecentEntries('Recent source reads', safeLedger.sourceReads),
    ...formatRecentEntries('Recent searches/inventory', safeLedger.sourceSearches),
    ...formatRecentEntries('Recent validation/check runs', safeLedger.validationRuns, 2),
    ...(safeLedger.errors.length > 0
      ? [
          'Recent tool errors:',
          ...safeLedger.errors.slice(-2).map((entry) => `- ${formatLedgerEntry(entry)}: ${entry.summary ?? 'error'}`),
        ]
      : []),
  ]

  return lines.join('\n')
}
