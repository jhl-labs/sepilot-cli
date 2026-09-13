import { createHash, randomUUID } from 'node:crypto'
import { realpath } from 'node:fs/promises'
import { basename, dirname, isAbsolute, relative, resolve } from 'node:path'
import { AutonomyLevel } from '@sepilotd/core'
import { streamToolProgress } from './tool-progress-stream.js'
import type {
  ApprovalDecision,
  AgentEvent,
  AgentRequestedProcessStart,
  AgentRunContract,
  ContentPart,
  IAuditLogger,
  Message,
  ToolCall,
} from '@sepilotd/core'
import type { HookRegistry } from '../hook/registry.js'
import {
  isPolicyReadOnlyRequest,
  isPolicyReadOnlyTool,
  type PolicyEngine,
} from '../security/policy-engine.js'
import { getAbortError, isAbortError, throwIfAborted } from '../abort.js'
import {
  advanceRunCheckpoint as advanceRunCheckpointHelper,
  buildCompletedToolExecutionRecord,
  buildDenialToolOutput,
  buildRunningToolExecutionRecord,
  buildAutonomousApprovalBlockAuditEntry,
  buildToolExecutionAuditEntry,
  cloneMessage,
  cloneToolCall,
  collectParallelBatch as collectParallelBatchHelper,
  decorateErrorOutput,
  emitPostFileEdit,
  emitPostToolExecute as emitPostToolExecuteHook,
  emitPreToolExecute as emitPreToolExecuteHook,
  isFileEditTool,
  normalizeApprovalDecision,
  normalizeToolCallInputsInPlace,
  prepareParallelToolCall as prepareParallelToolCallHelper,
  missingRequiredToolArguments,
  yieldToolError as yieldToolErrorBase,
  yieldToolSuccessOrRecovery as yieldToolSuccessOrRecoveryBase,
  type PostToolExecuteResult,
  type PreparedParallelToolCall,
} from './tool-execution-helpers.js'
import { findUnsupportedRepositoryPathClaims } from './outcome-review.js'
import {
  contractLimitsCurrentTurnToDocumentArtifact,
  inputRequestsProspectiveDocumentPhase,
} from './task-contract.js'
import { APPROVAL_FAILURE_STATUS_METADATA_KEY } from './approval-failure.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'
import {
  formatNestedInstructions,
  getDefaultNestedInstructionInjector,
  nestedInstructionTargetPath,
  type NestedInstructionInjector,
} from './nested-instructions.js'
import {
  TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY,
  TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY,
} from './policy-failure.js'

// cloneToolCall and cloneMessage are part of this module's public API
// (engine.ts and mode-router.ts both import them from './tool-execution.js').
// Keep that import path stable while their definitions live with the other
// pure helpers.
export { cloneMessage, cloneToolCall }
import { summarizeToolOutputForAgentContext } from './tool-output.js'
import { partitionToolCallsBySequence } from './tool-call-budget.js'
import {
  resolveToolBatchFailureCouplingKey,
  toolResultBlocksFailureCoupledSiblings,
  type ToolDefinitionRuntime,
  type ToolRegistry,
  type ToolResult,
} from '../tools/registry.js'
import type { ToolExecutionRecord } from '../server/runtime/tool-executions.js'
import { logAgentDebugTrace } from '../observability/agent-trace.js'
import { resolveDefaultManagedProcessTtlMs } from '../tools/process.js'

const MAX_TOOL_IMAGE_CONTEXT_PARTS = 3
const MAX_CLIENT_IMAGE_PARTS = 1
const MAX_CLIENT_IMAGE_BASE64_CHARS = 8 * 1024 * 1024

const DEFAULT_TOOL_TIMEOUT_MS = 120_000

function contentPartsFromToolResult(result: Pick<ToolResult, 'contentParts' | 'images'>): ContentPart[] | undefined {
  const imageParts = (result.images ?? []).map((image): ContentPart => ({
    type: 'image',
    source: {
      type: 'base64',
      mediaType: image.mediaType,
      data: image.data,
    },
  }))
  const contentParts = [
    ...(result.contentParts ?? []),
    ...imageParts,
  ]
  return contentParts.length > 0 ? contentParts : undefined
}

/**
 * Keep client-visible images opt-in and bounded. Tool images normally exist
 * only for the next model turn; broadcasting every computer screenshot would
 * unnecessarily widen their audience and inflate renderer memory.
 */
function clientContentPartsFromToolResult(
  result: Pick<ToolResult, 'images'>,
): ContentPart[] | undefined {
  const parts = (result.images ?? [])
    .filter((image) =>
      image.displayToClient === true
      && /^image\/(?:png|jpeg|webp|gif)$/i.test(image.mediaType)
      && image.data.length > 0
      && image.data.length <= MAX_CLIENT_IMAGE_BASE64_CHARS
    )
    .slice(0, MAX_CLIENT_IMAGE_PARTS)
    .map((image): ContentPart => ({
      type: 'image',
      source: {
        type: 'base64',
        mediaType: image.mediaType,
        data: image.data,
      },
    }))
  return parts.length > 0 ? parts : undefined
}

/**
 * Per-tool wall-clock timeout (ms). A tool that ignores its abort signal and
 * hangs would otherwise suspend the agent generator forever and leak a pending
 * promise for the daemon's lifetime. `0`/negative disables the backstop. Tools
 * with their own (usually shorter) timeout still win because they resolve
 * first. Configurable via SEPILOTD_TOOL_TIMEOUT_MS.
 */
function resolveToolTimeoutMs(): number {
  const raw = Number(process.env.SEPILOTD_TOOL_TIMEOUT_MS)
  if (Number.isFinite(raw)) {
    return raw > 0 ? Math.floor(raw) : 0
  }
  return DEFAULT_TOOL_TIMEOUT_MS
}

// Chat-role labels that show up as a bogus tool-call `name` when a model's
// turn/channel format leaks into its tool-calling output.
const CHAT_ROLE_TOOL_NAMES = new Set([
  'assistant',
  'user',
  'system',
  'developer',
  'tool',
  'function',
  'human',
])

const FILE_MUTATION_TOOL_NAMES = new Set(['fs.write', 'fs.append', 'fs.edit', 'apply_patch'])
const CACHEABLE_ADJACENT_OBSERVATION_TOOLS = new Set([
  'fs.read',
  'fs.list',
  'fs.glob',
  'fs.search',
  'code.symbols',
  'code.dependencies',
])
const TOOL_EXECUTION_SIGNATURE_METADATA_KEY = 'toolExecutionSignature'
const TOOL_RESULT_STATUS_METADATA_KEY = 'toolResultStatus'
const TERMINAL_MUTATING_EXECUTABLES = new Set([
  'chmod',
  'chown',
  'bun',
  'cp',
  'install',
  'mkdir',
  'mv',
  'npm',
  'npx',
  'patch',
  'pnpm',
  'rm',
  'rmdir',
  'tee',
  'touch',
  'truncate',
  'yarn',
])
const TERMINAL_MUTATING_GIT_SUBCOMMANDS = new Set([
  'am',
  'apply',
  'checkout',
  'clean',
  'commit',
  'merge',
  'pull',
  'push',
  'rebase',
  'reset',
  'restore',
  'switch',
])
const TERMINAL_SHELL_MUTATION_PATTERN =
  /(^|[\s;&|()])(?:rm|mv|cp|mkdir|rmdir|touch|truncate|tee|patch|chmod|chown)\b|\bsed\b[^\n;&|]*\s-[A-Za-z]*i[A-Za-z]*\b|\bperl\b[^\n;&|]*\s-[A-Za-z]*i[A-Za-z]*\b|\b(?:writeFileSync|writeFile|appendFileSync|appendFile|rmSync|unlinkSync|renameSync|mkdirSync|openSync)\b|\bopen\s*\([^)]*,\s*['"][wax][^'"]*['"]|\b(?:write_text|write_bytes|unlink|rename|replace|mkdir|rmdir)\s*\(|\b(?:os|shutil)\.(?:remove|unlink|rename|replace|mkdir|makedirs|rmdir|removedirs|copy|copyfile|copytree|move|rmtree)\s*\(|(?:^|[^<])>>?|\bgit\b[^\n;&|]*\b(?:am|apply|checkout|clean|commit|merge|pull|push|rebase|reset|restore|switch)\b/iu

function normalizeStableJson(value: unknown): unknown {
  if (!value || typeof value !== 'object') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map(normalizeStableJson)
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, normalizeStableJson(entry)]),
  )
}

function toolCallSignature(toolCall: ToolCall): string {
  return `${toolCall.name}:${JSON.stringify(normalizeStableJson(toolCall.arguments))}`
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content
  }
  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

const APPROVAL_CONTEXT_MAX_CHARS = 240

/**
 * Tail of the current assistant turn's text, attached to
 * `approval_request` events so approval surfaces can show *why* the agent
 * wants to run the tool. Pure display-side truncation of text the model
 * already produced (no extra LLM call, no parsing); returns undefined when
 * the tool-call turn carried no prose.
 */
export function buildApprovalContext(messages: Message[]): string | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!
    if (message.role !== 'assistant') continue
    const text = messageText(message).trim()
    if (!text) return undefined
    if (text.length <= APPROVAL_CONTEXT_MAX_CHARS) return text
    return `…${text.slice(-APPROVAL_CONTEXT_MAX_CHARS)}`
  }
  return undefined
}

function getStringArgument(
  args: Record<string, unknown>,
  keys: string[],
): string | undefined {
  for (const key of keys) {
    const value = args[key]
    if (typeof value === 'string') {
      return value
    }
  }
  return undefined
}

function getStringArgumentEntry(
  args: Record<string, unknown>,
  keys: string[],
): { key: string; value: string } | undefined {
  for (const key of keys) {
    const value = args[key]
    if (typeof value === 'string') {
      return { key, value }
    }
  }
  return undefined
}

async function canonicalArtifactPath(path: string): Promise<string> {
  let ancestor = resolve(path)
  const missing: string[] = []
  for (;;) {
    try {
      return resolve(await realpath(ancestor), ...missing.toReversed())
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
      const parent = dirname(ancestor)
      if (parent === ancestor) throw error
      missing.push(basename(ancestor))
      ancestor = parent
    }
  }
}

function isWithinDirectory(path: string, directory: string): boolean {
  const rel = relative(resolve(directory), resolve(path))
  return rel === '' || (!!rel && !rel.startsWith('..') && !isAbsolute(rel))
}

async function misplacedRequestedArtifactWrite(
  runContract: AgentRunContract | undefined,
  toolCall: ToolCall,
  cwd: string | undefined,
): Promise<{ requestedPath: string; expectedPath: string; actualPath: string } | null> {
  if (!cwd || !FILE_MUTATION_TOOL_NAMES.has(toolCall.name)) {
    return null
  }
  const actualPath = typeof toolCall.arguments.path === 'string'
    ? toolCall.arguments.path
    : undefined
  if (!actualPath || !isAbsolute(actualPath)) {
    return null
  }

  const requestedPaths = runContract?.requiredArtifacts?.map((artifact) => artifact.path) ?? []
  for (const requestedPath of requestedPaths) {
    if (basename(actualPath) !== basename(requestedPath)) {
      continue
    }
    const requestedIsAbsolute = isAbsolute(requestedPath)
    const expectedPath = requestedIsAbsolute
      ? resolve(requestedPath)
      : resolve(cwd, requestedPath)
    try {
      const [actual, expected, root] = await Promise.all([
        canonicalArtifactPath(actualPath), canonicalArtifactPath(expectedPath), canonicalArtifactPath(cwd),
      ])
      if (actual === expected && (requestedIsAbsolute || isWithinDirectory(actual, root))) continue
    } catch {
      // An inaccessible or cyclic path cannot establish target identity.
    }
    return { requestedPath, expectedPath, actualPath }
  }
  return null
}

const NO_TOUCH_CONTRACT_TARGET_PATTERN = /Do not directly modify target "([^"]+)"/g

function normalizeNoTouchContractTarget(value: string): string {
  return value
    .trim()
    .replace(/^[`"']|[`"']$/g, '')
    .replace(/[.,;:]+$/g, '')
    .trim()
}

function noTouchTargetsFromRunContract(runContract: AgentRunContract | undefined): string[] {
  if (!runContract) return []
  const targets: string[] = [...(runContract.executionIntent?.protectedWriteTargets ?? [])]
  const source = [...runContract.constraints, ...runContract.outOfScope].join('\n')
  NO_TOUCH_CONTRACT_TARGET_PATTERN.lastIndex = 0
  for (const match of source.matchAll(NO_TOUCH_CONTRACT_TARGET_PATTERN)) {
    const target = normalizeNoTouchContractTarget(match[1] ?? '')
    if (!target) {
      continue
    }
    if (!targets.some((existing) => existing.toLowerCase() === target.toLowerCase())) {
      targets.push(target)
    }
  }
  const authorized = new Set(
    (runContract.executionIntent?.authorizedWriteTargets ?? [])
      .map((target) => normalizeNoTouchContractTarget(target).replace(/\\/g, '/').toLowerCase()),
  )
  return targets.filter((target) =>
    !authorized.has(normalizeNoTouchContractTarget(target).replace(/\\/g, '/').toLowerCase()))
}

function applyPatchTargetPaths(toolCall: ToolCall): string[] {
  if (toolCall.name !== 'apply_patch' || typeof toolCall.arguments.patch !== 'string') {
    return []
  }
  const paths: string[] = []
  const pattern = /^\*\*\* (?:Add|Update|Delete) File: (.+)$|^\*\*\* Move to: (.+)$/gm
  for (const match of toolCall.arguments.patch.matchAll(pattern)) {
    const path = (match[1] ?? match[2] ?? '').trim()
    if (path) {
      paths.push(path)
    }
  }
  return paths
}

function fileMutationTargetPaths(toolCall: ToolCall): string[] {
  if (!FILE_MUTATION_TOOL_NAMES.has(toolCall.name)) {
    return []
  }
  if (toolCall.name === 'apply_patch') {
    return applyPatchTargetPaths(toolCall)
  }
  const path = typeof toolCall.arguments.path === 'string'
    ? toolCall.arguments.path.trim()
    : ''
  return path ? [path] : []
}

function normalizedPathSegments(path: string): string[] {
  return path.replace(/\\/g, '/').split('/').filter(Boolean)
}

function noTouchTargetMentionedInText(text: string, target: string, cwd: string | undefined): boolean {
  const normalizedText = text.replace(/\\/g, '/')
  const normalizedTarget = normalizeNoTouchContractTarget(target).replace(/\\/g, '/')
  if (!normalizedTarget) {
    return false
  }
  const targetLooksPathLike =
    normalizedTarget.includes('/')
    || normalizedTarget.startsWith('.')
    || normalizedTarget.startsWith('~')
  if (targetLooksPathLike) {
    if (normalizedText.includes(normalizedTarget)) {
      return true
    }
    if (!normalizedTarget.startsWith('~')) {
      const targetPath = isAbsolute(normalizedTarget) || !cwd
        ? resolve(normalizedTarget)
        : resolve(cwd, normalizedTarget)
      return normalizedText.includes(targetPath.replace(/\\/g, '/'))
    }
    return false
  }
  const escaped = normalizedTarget.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return new RegExp(`(^|[\\s/"'])${escaped}(?=$|[\\s/"'])`, 'u').test(normalizedText)
}

function splitTerminalCommandWords(text: string): string[] {
  const words: string[] = []
  let current = ''
  let quote: '"' | "'" | '`' | null = null
  let escaped = false

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]!

    if (escaped) {
      current += char
      escaped = false
      continue
    }

    if (char === '\\') {
      escaped = true
      continue
    }

    if (quote) {
      if (char === quote) {
        quote = null
      } else {
        current += char
      }
      continue
    }

    if (char === '"' || char === "'" || char === '`') {
      quote = char
      continue
    }

    if (/\s/.test(char) || /[;&|<>]/.test(char)) {
      if (current) {
        words.push(current)
        current = ''
      }
      continue
    }

    current += char
  }

  if (current) {
    words.push(current)
  }

  return words
}

function stripTerminalPathToken(value: string): string {
  return value
    .trim()
    .replace(/^[`"'([{]+/u, '')
    .replace(/[`"',;:)\]}]+$/u, '')
    .trim()
}

function terminalPathCandidatesFromText(text: string): string[] {
  const candidates: string[] = []
  const add = (value: string) => {
    const candidate = stripTerminalPathToken(value)
    if (!candidate || candidate.startsWith('-')) return
    if (!candidates.includes(candidate)) {
      candidates.push(candidate)
    }
  }

  for (const word of splitTerminalCommandWords(text)) {
    add(word)
    for (const match of word.matchAll(/(?:\.{1,2}\/|\/|~\/)?[A-Za-z0-9_.@-]+(?:\/[A-Za-z0-9_.@-]+)+/g)) {
      add(match[0] ?? '')
    }
  }
  return candidates
}

function pathMatchesNoTouchTarget(path: string, target: string, cwd: string | undefined): boolean {
  const normalizedTarget = normalizeNoTouchContractTarget(target)
  if (!normalizedTarget) {
    return false
  }

  const targetLooksPathLike =
    normalizedTarget.includes('/')
    || normalizedTarget.includes('\\')
    || normalizedTarget.startsWith('.')
    || normalizedTarget.startsWith('~')
  if (targetLooksPathLike) {
    if (normalizedTarget.startsWith('~')) {
      return false
    }
    const targetPath = isAbsolute(normalizedTarget) || !cwd
      ? resolve(normalizedTarget)
      : resolve(cwd, normalizedTarget)
    const actualPath = isAbsolute(path) || !cwd ? resolve(path) : resolve(cwd, path)
    return isWithinDirectory(actualPath, targetPath)
  }

  const targetLeaf = basename(normalizedTarget.replace(/\\/g, '/'))
  return normalizedPathSegments(path).some((segment) => segment === targetLeaf)
}

function noTouchMutationViolation(
  runContract: AgentRunContract | undefined,
  toolCall: ToolCall,
  cwd: string | undefined,
): { target: string; path: string } | null {
  const noTouchTargets = noTouchTargetsFromRunContract(runContract)
  if (noTouchTargets.length === 0) {
    return null
  }

  if (toolCall.name === 'terminal.run' || toolCall.name === 'process.start') {
    return noTouchTerminalMutationViolation(noTouchTargets, toolCall, cwd)
  }

  if (toolCall.name === 'service.start') {
    return noTouchServiceStartMutationViolation(noTouchTargets, toolCall, cwd)
  }

  if (toolCall.name === 'pages.scaffold') {
    return noTouchPagesScaffoldMutationViolation(noTouchTargets, toolCall, cwd)
  }

  const targetPaths = fileMutationTargetPaths(toolCall)
  for (const target of noTouchTargets) {
    for (const path of targetPaths) {
      if (pathMatchesNoTouchTarget(path, target, cwd)) {
        return { target, path }
      }
    }
  }
  return null
}

function canonicalProcessNetwork(value: unknown): unknown {
  if (value === undefined || value === 'none') return 'none'
  if (!value || typeof value !== 'object' || Array.isArray(value)) return value
  const record = value as Record<string, unknown>
  if (record.mode === 'none') return 'none'
  if (record.mode !== 'loopback') return normalizeStableJson(value)
  const ports = Array.isArray(record.ports)
    ? [...new Set(record.ports.filter((port): port is number => (
        typeof port === 'number' && Number.isInteger(port)
      )))].sort((left, right) => left - right)
    : []
  return { mode: 'loopback', ports }
}

function effectiveProcessCwd(value: unknown, fallback: string | undefined): string | null | undefined {
  if (value !== undefined && typeof value !== 'string') return null
  const selected = typeof value === 'string' && value.trim() ? value.trim() : fallback
  if (!selected) return undefined
  return isAbsolute(selected)
    ? resolve(selected)
    : resolve(fallback ?? process.cwd(), selected)
}

function canonicalProcessLifetime(
  value: Pick<AgentRequestedProcessStart, 'ttlMs' | 'lifetime'> | Record<string, unknown>,
): { lifetime: 'bounded' | 'session'; ttlMs: number } | null {
  const lifetime = value.lifetime
  if (lifetime !== undefined && lifetime !== 'bounded' && lifetime !== 'session') return null
  const ttlMs = value.ttlMs
  if (
    ttlMs !== undefined
    && (
      typeof ttlMs !== 'number'
      || !Number.isInteger(ttlMs)
      || ttlMs < 0
    )
  ) return null
  if (lifetime === 'session' && typeof ttlMs === 'number' && ttlMs > 0) return null
  if (lifetime === 'bounded' && ttlMs === 0) return null
  const effectiveTtl = lifetime === 'session'
    ? 0
    : typeof ttlMs === 'number'
      ? ttlMs
      : resolveDefaultManagedProcessTtlMs()
  return {
    lifetime: effectiveTtl === 0 ? 'session' : 'bounded',
    ttlMs: effectiveTtl,
  }
}

/**
 * Compare the effective process.start contract rather than raw JSON shape.
 * Optional schema defaults (active cwd, bounded lifetime/default TTL, and
 * network=none) are semantically identical whether a provider emits them or
 * omits them. Command argv and every user-specified option remain exact, and
 * undeclared environment/PTY authority stays forbidden.
 */
function processStartMatchesConstraint(
  expected: AgentRequestedProcessStart,
  actual: Record<string, unknown>,
  cwd?: string,
): boolean {
  const allowedKeys = new Set(['executable', 'args', 'cwd', 'ttlMs', 'lifetime', 'network'])
  if (Object.keys(actual).some((key) => !allowedKeys.has(key))) return false
  const executable = typeof actual.executable === 'string' ? actual.executable.trim() : ''
  const args = Array.isArray(actual.args) && actual.args.every((entry) => typeof entry === 'string')
    ? actual.args
    : []
  if (
    executable !== expected.executable.trim()
    || args.length !== expected.args.length
    || args.some((entry, index) => entry !== expected.args[index])
  ) return false

  const expectedLifetime = canonicalProcessLifetime(expected)
  const actualLifetime = canonicalProcessLifetime(actual)
  if (!expectedLifetime || !actualLifetime) return false

  return JSON.stringify(normalizeStableJson({
    cwd: effectiveProcessCwd(expected.cwd, cwd),
    ...expectedLifetime,
    network: canonicalProcessNetwork(expected.network),
  })) === JSON.stringify(normalizeStableJson({
    cwd: effectiveProcessCwd(actual.cwd, cwd),
    ...actualLifetime,
    network: canonicalProcessNetwork(actual.network),
  }))
}

function executionIntentMutationViolation(
  runContract: AgentRunContract | undefined,
  toolCall: ToolCall,
  tools: ToolRegistry,
  messages: Message[],
  cwd?: string,
  workspaceRoot?: string,
): string | null {
  const intent = runContract?.executionIntent
  if (intent?.allowedTools && !intent.allowedTools.includes(toolCall.name)) {
    return `tool ${toolCall.name} is outside the exact allowed tool set (${intent.allowedTools.join(', ') || 'none'})`
  }
  if (intent?.toolSequence?.length) {
    const priorResults = messagesInCurrentAgentTurn(messages)
      .filter((message) => message.role === 'tool')
      .map((message) => ({
        tool: message.name ?? '',
        status: message.metadata?.toolResultStatus === 'success'
          ? 'success' as const
          : 'error' as const,
        executionObserved:
          message.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
      }))
    const sequenced = partitionToolCallsBySequence(
      intent.toolSequence,
      priorResults,
      [toolCall],
      intent.retryPolicy === 'forbidden',
    )
    if (sequenced.outOfOrder.length > 0) {
      if (sequenced.workflowFailed) {
        return `tool ${toolCall.name} is blocked because a required ordered step already failed and retries are forbidden`
      }
      return sequenced.nextTool
        ? `tool ${toolCall.name} is out of order; the next required tool is ${sequenced.nextTool}`
        : `tool ${toolCall.name} exceeds the completed ordered tool workflow`
    }
  }
  const requestedProcess = intent?.requestedProcessStart
  const constrainedProcess = intent?.constrainedProcessStart
  const exactProcess = requestedProcess ?? constrainedProcess
  if (requestedProcess && toolCall.name !== 'process.start') {
    return `tool ${toolCall.name} is outside the exact managed-process intent`
  }
  if (exactProcess && toolCall.name === 'process.start') {
    const expected = JSON.stringify(normalizeStableJson(exactProcess))
    if (!processStartMatchesConstraint(exactProcess, toolCall.arguments, cwd)) {
      return [
        'process.start does not match the exact user-requested process options',
        `expected ${expected}`,
      ].join('; ')
    }
  }
  const requestedCommand = intent?.requestedTerminalCommand
  if (requestedCommand) {
    if (toolCall.name !== 'terminal.run') {
      return `tool ${toolCall.name} is outside the exact single-command intent`
    }
    const executable = typeof toolCall.arguments.executable === 'string'
      ? toolCall.arguments.executable
      : ''
    const args = Array.isArray(toolCall.arguments.args)
      && toolCall.arguments.args.every((arg) => typeof arg === 'string')
      ? toolCall.arguments.args as string[]
      : []
    const argvMatches = executable === requestedCommand.executable
      && args.length === requestedCommand.args.length
      && args.every((arg, index) => arg === requestedCommand.args[index])
    const changesExecutionEnvironment = [
      'command',
      'cmd',
      'env',
      'shell',
      'stdin',
      'input',
    ].some((key) => key in toolCall.arguments)
    if (!argvMatches || changesExecutionEnvironment) {
      return [
        'terminal.run does not match the exact user-requested argv',
        `expected ${JSON.stringify([requestedCommand.executable, ...requestedCommand.args])}`,
      ].join('; ')
    }
  }
  if (intent?.workspaceMutation !== 'forbidden') return null
  if (isFileEditTool(toolCall.name)) {
    return 'workspace file mutation is forbidden by the semantic execution intent'
  }

  const requestIsReadOnly = isPolicyReadOnlyRequest({
    tool: toolCall.name,
    input: toolCall.arguments,
    cwd,
    workspaceRoot,
    registrationSource: tools.registrationSource(toolCall.name),
    security: tools.securityDescriptor(toolCall.name),
  })
  if (requestIsReadOnly) return null

  // A workspace boundary is not a ban on explicitly requested application
  // state changes. Use the registry's effect contract, never prompt wording or
  // tool-name families. Normal security policy and approval still run below.
  if (
    intent.kind !== 'inspection'
    && intent.kind !== 'conversation'
    && intent.capabilities.includes('application-state')
    && tools.securityDescriptor(toolCall.name).effect === 'external-write'
  ) return null

  if (toolCall.name === 'terminal.run') {
    if (
      intent.capabilityPolicy === 'closed'
      && !intent.capabilities.includes('terminal')
    ) {
      return 'terminal.run is outside the declared execution capabilities'
    }
    return terminalInvocationLooksMutating(toolCall)
      ? 'terminal.run was not classified as a read-only invocation'
      : null
  }
  if (toolCall.name.startsWith('process.') && intent.capabilities.includes('process')) return null
  if (toolCall.name.startsWith('service.') && intent.capabilities.includes('service')) return null
  if (toolCall.name.startsWith('browser.') && intent.capabilities.includes('browser')) return null
  if (tools.securityDescriptor(toolCall.name).effect === 'external-write') {
    return `side-effecting capability ${toolCall.name} is outside the read-only execution intent: durable application-state changes require an appropriate operational capability; workspace file permission is a separate boundary`
  }
  return `side-effecting capability ${toolCall.name} is outside the read-only execution intent`
}

function noTouchPagesScaffoldMutationViolation(
  targets: string[],
  toolCall: ToolCall,
  cwd: string | undefined,
): { target: string; path: string } | null {
  const repoPath = typeof toolCall.arguments.repoPath === 'string' && toolCall.arguments.repoPath.trim()
    ? toolCall.arguments.repoPath.trim()
    : cwd
  if (!repoPath) {
    return null
  }
  const resolvedRepoPath = isAbsolute(repoPath) || !cwd
    ? resolve(repoPath)
    : resolve(cwd, repoPath)
  const sitePath = typeof toolCall.arguments.sitePath === 'string' && toolCall.arguments.sitePath.trim()
    ? toolCall.arguments.sitePath.trim()
    : 'site'
  const siteRoot = sitePath === '.'
    ? resolvedRepoPath
    : resolve(resolvedRepoPath, sitePath.replace(/\\/g, '/'))

  for (const target of targets) {
    if (pathMatchesNoTouchTarget(resolvedRepoPath, target, cwd)) {
      return { target, path: resolvedRepoPath }
    }
    if (pathMatchesNoTouchTarget(siteRoot, target, cwd)) {
      return { target, path: siteRoot }
    }
  }
  return null
}

function noTouchServiceStartMutationViolation(
  targets: string[],
  toolCall: ToolCall,
  cwd: string | undefined,
): { target: string; path: string } | null {
  const commandViolation = noTouchTerminalMutationViolation(targets, toolCall, cwd)
  if (commandViolation) {
    return commandViolation
  }

  const volumes = Array.isArray(toolCall.arguments.volumes)
    ? toolCall.arguments.volumes
    : []
  const serviceCwd = terminalCwd(toolCall) ?? cwd
  for (const volume of volumes) {
    if (!volume || typeof volume !== 'object') {
      continue
    }
    const record = volume as Record<string, unknown>
    if (record.readonly === true || typeof record.source !== 'string') {
      continue
    }
    const source = record.source.trim()
    if (!source) {
      continue
    }
    for (const target of targets) {
      if (pathMatchesNoTouchTarget(source, target, serviceCwd)) {
        return { target, path: source }
      }
    }
  }
  return null
}

function terminalExecutableName(toolCall: ToolCall): string {
  const executable = typeof toolCall.arguments.executable === 'string'
    ? toolCall.arguments.executable
    : ''
  return basename(executable).toLowerCase().replace(/\.exe$/i, '')
}

function terminalArgs(toolCall: ToolCall): string[] {
  return Array.isArray(toolCall.arguments.args)
    ? toolCall.arguments.args.map((arg) => String(arg))
    : typeof toolCall.arguments.args === 'string'
      ? [toolCall.arguments.args]
      : []
}

function terminalCommandText(toolCall: ToolCall): string {
  const parts = [
    typeof toolCall.arguments.executable === 'string' ? toolCall.arguments.executable : '',
    ...terminalArgs(toolCall),
    typeof toolCall.arguments.command === 'string' ? toolCall.arguments.command : '',
    typeof toolCall.arguments.cmd === 'string' ? toolCall.arguments.cmd : '',
  ]
  return parts.filter(Boolean).join(' ')
}

function terminalCwd(toolCall: ToolCall): string | undefined {
  const value = toolCall.arguments.cwd
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function terminalGitSubcommand(args: string[]): string | null {
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!
    if (arg === '-C' || arg === '-c' || arg === '--git-dir' || arg === '--work-tree') {
      index += 1
      continue
    }
    if (arg.startsWith('-')) {
      continue
    }
    return arg.toLowerCase()
  }
  return null
}

function terminalInvocationLooksMutating(toolCall: ToolCall): boolean {
  const executable = terminalExecutableName(toolCall)
  const args = terminalArgs(toolCall)
  const commandText = terminalCommandText(toolCall)
  if (TERMINAL_MUTATING_EXECUTABLES.has(executable)) {
    return true
  }
  if (
    executable === 'git'
    && TERMINAL_MUTATING_GIT_SUBCOMMANDS.has(terminalGitSubcommand(args) ?? '')
  ) {
    return true
  }
  return TERMINAL_SHELL_MUTATION_PATTERN.test(commandText)
}

function noTouchTerminalMutationViolation(
  targets: string[],
  toolCall: ToolCall,
  cwd: string | undefined,
): { target: string; path: string } | null {
  if (!terminalInvocationLooksMutating(toolCall)) {
    return null
  }

  const commandText = terminalCommandText(toolCall)
  const commandCwd = terminalCwd(toolCall)
  const effectiveCwd = commandCwd ?? cwd
  const pathCandidates = terminalPathCandidatesFromText(commandText)
  for (const target of targets) {
    if (commandCwd && pathMatchesNoTouchTarget(commandCwd, target, cwd)) {
      return { target, path: commandCwd }
    }
    if (noTouchTargetMentionedInText(commandText, target, cwd)) {
      return { target, path: commandText.slice(0, 240) }
    }
    for (const candidate of pathCandidates) {
      if (pathMatchesNoTouchTarget(candidate, target, effectiveCwd)) {
        return { target, path: candidate }
      }
    }
  }
  return null
}

function requestedArtifactPaths(runContract: AgentRunContract | undefined): string[] {
  return runContract?.requiredArtifacts
    ?.map((artifact) => artifact.path)
    .filter((path): path is string => typeof path === 'string' && path.trim().length > 0)
    ?? []
}

function pathsReferToSameFile(left: string, right: string, cwd: string | undefined): boolean {
  const normalize = (path: string) => {
    const trimmed = path.trim()
    return isAbsolute(trimmed) || !cwd ? resolve(trimmed) : resolve(cwd, trimmed)
  }
  return normalize(left) === normalize(right)
}

function successfulFsWriteResult(content: string): boolean {
  return /\bWrote\s+\d+\s+bytes\s+to\s+/i.test(content)
}

function previousSuccessfulRequiredArtifactWrite(
  messages: Message[],
  path: string,
  cwd: string | undefined,
): { content: string; path: string } | null {
  const callsById = new Map<string, ToolCall>()
  let previous: { content: string; path: string } | null = null

  for (const message of messages) {
    for (const call of message.toolCalls ?? []) {
      callsById.set(call.id, call)
    }
    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const call = callsById.get(message.toolCallId)
    if (!call || call.name !== 'fs.write') {
      continue
    }
    const writtenPath = typeof call.arguments.path === 'string' ? call.arguments.path : undefined
    if (!writtenPath || !pathsReferToSameFile(writtenPath, path, cwd)) {
      continue
    }
    if (!successfulFsWriteResult(messageText(message))) {
      continue
    }
    const content = getStringArgument(call.arguments, ['content', 'contents'])
    if (content !== undefined) {
      previous = { content, path: writtenPath }
    }
  }

  return previous
}

function requiredArtifactWriteRegression(
  runContract: AgentRunContract | undefined,
  messages: Message[],
  toolCall: ToolCall,
  cwd: string | undefined,
): { path: string; previousBytes: number; nextBytes: number; reason: string } | null {
  if (toolCall.name !== 'fs.write') {
    return null
  }
  const path = typeof toolCall.arguments.path === 'string' ? toolCall.arguments.path : undefined
  const nextContent = getStringArgument(toolCall.arguments, ['content', 'contents'])
  if (!path || nextContent === undefined) {
    return null
  }
  const isRequiredArtifact = requestedArtifactPaths(runContract)
    .some((requiredPath) => pathsReferToSameFile(requiredPath, path, cwd))
  if (!isRequiredArtifact) {
    return null
  }

  const previous = previousSuccessfulRequiredArtifactWrite(messages, path, cwd)
  if (!previous) {
    return null
  }

  const previousContent = previous.content.trim()
  const next = nextContent.trim()
  if (!previousContent || !next || next.length >= previousContent.length) {
    return null
  }

  if (previousContent.includes(next)) {
    return {
      path,
      previousBytes: previous.content.length,
      nextBytes: nextContent.length,
      reason: 'the replacement content is a strict fragment of the existing required artifact',
    }
  }

  if (previous.content.length >= 4000 && nextContent.length < previous.content.length * 0.5) {
    return {
      path,
      previousBytes: previous.content.length,
      nextBytes: nextContent.length,
      reason: 'the replacement is a large shrink of an already-written required artifact',
    }
  }

  return null
}

function summarizeBlockedArtifactDraft(
  content: string,
  unsupportedPaths: readonly string[],
): string {
  const digest = createHash('sha256').update(content).digest('hex').slice(0, 12)
  const lines = content.split(/\r?\n/)
  const contexts: string[] = []
  for (const unsupportedPath of unsupportedPaths.slice(0, 5)) {
    const line = lines.find((candidate) => candidate.includes(unsupportedPath))
    if (!line) continue
    const normalized = line.trim().replace(/\s+/g, ' ')
    if (!normalized) continue
    contexts.push(`- ${unsupportedPath}: ${normalized.slice(0, 220)}`)
  }
  return [
    `Blocked draft retained in the previous ${content.length}-char tool-call arguments (sha256:${digest}).`,
    contexts.length > 0
      ? `Unsupported-claim context:\n${contexts.join('\n')}`
      : '',
    'Do not regenerate the whole draft from scratch; gather evidence, then retry the same artifact update after removing or marking unsupported claims as coverage gaps.',
  ].filter(Boolean).join('\n')
}

function sanitizeArtifactContentForUnsupportedPathClaims(
  content: string,
  unsupportedPaths: readonly string[],
): { content: string; removedLineCount: number } | null {
  if (unsupportedPaths.length === 0) {
    return null
  }
  const unsupported = unsupportedPaths.filter(Boolean)
  if (unsupported.length === 0) {
    return null
  }

  const lines = content.split(/\r?\n/)
  const kept: string[] = []
  let removedLineCount = 0
  for (const line of lines) {
    if (unsupported.some((path) => line.includes(path))) {
      removedLineCount += 1
      continue
    }
    kept.push(line)
  }
  if (removedLineCount === 0) {
    return null
  }

  const sanitized = kept.join('\n').trimEnd()
  if (sanitized.length < Math.max(400, content.length * 0.35)) {
    return null
  }

  // Return the cleaned artifact body only. The removal bookkeeping ("Coverage
  // Gaps / Removed Unsupported Path Claims ...") must not be injected into the
  // user-facing deliverable — it polluted produced docs (CLI_BACKLOG.md C4).
  // removedLineCount is still returned for the caller's telemetry/logging.
  void unsupported
  return {
    content: sanitized,
    removedLineCount,
  }
}

const PROSPECTIVE_REPOSITORY_PATH_LINE_PATTERN =
  /\b(?:planned|proposed|target|future|later|next\s+(?:phase|step|turn)|to\s+be\s+(?:added|built|created|implemented)|new\s+(?:entry|file|module|route|service))\b|(?:계획(?:한|된|할)?|제안(?:한|된)?|목표|향후|추후|다음\s*(?:단계|페이즈|턴)|신규|새로\s*(?:추가|생성|구현)|만들\s*예정)/iu
const OBSERVED_REPOSITORY_PATH_LINE_PATTERN =
  /\b(?:as[- ]is|existing|observed|verified|actual|implemented|present|today|already)\b|\bcurrently\b[^.;:\n]{0,80}\b(?:exists?|is|contains?|uses?|located|implemented)\b|(?:기존|현행|관찰(?:된|한)|확인(?:된|한)|검증(?:된|한)|실제|구현(?:된|되어)|이미\s*(?:존재|있는)|현재[^.;:\n]{0,60}(?:존재|있는|사용|위치|구현(?:된|되어)))/iu

function lineClaimsObservedRepositoryState(line: string): boolean {
  // Explicit target-state language wins over incidental temporal words such
  // as "currently planned" or "현재 단계에서 계획한". A path in such a line
  // specifies work that does not exist yet; it is not an as-is repository
  // claim. Explicit observed/existing language remains evidence-gated.
  if (PROSPECTIVE_REPOSITORY_PATH_LINE_PATTERN.test(line)) {
    return false
  }
  return OBSERVED_REPOSITORY_PATH_LINE_PATTERN.test(line)
}

function unsupportedArtifactRepositoryPathClaims(input: {
  content: string
  messages: Message[]
  extraEvidencedPaths: string[]
  prospectiveDocumentPhase: boolean
}): string[] {
  const unsupportedPaths = findUnsupportedRepositoryPathClaims({
    text: input.content,
    messages: input.messages,
    evidenceScope: 'all',
    extraEvidencedPaths: input.extraEvidencedPaths,
  })
  if (!input.prospectiveDocumentPhase) {
    return unsupportedPaths
  }

  const lines = input.content.split(/\r?\n/)
  return unsupportedPaths.filter((unsupportedPath) =>
    lines.some((line) =>
      line.includes(unsupportedPath) && lineClaimsObservedRepositoryState(line)
    )
  )
}

function requiredArtifactWriteUnsupportedPathClaims(
  runContract: AgentRunContract | undefined,
  messages: Message[],
  toolCall: ToolCall,
  cwd: string | undefined,
): {
  path: string
  unsupportedPaths: string[]
  draftSummary: string
  contentKey: string
  content: string
  extraEvidencedPaths: string[]
  prospectiveDocumentPhase: boolean
} | null {
  if (toolCall.name !== 'fs.write' && toolCall.name !== 'fs.append') {
    return null
  }
  const path = typeof toolCall.arguments.path === 'string' ? toolCall.arguments.path : undefined
  const contentEntry = getStringArgumentEntry(toolCall.arguments, ['content', 'contents'])
  if (!path || contentEntry === undefined) {
    return null
  }
  const content = contentEntry.value
  const requiredPaths = requestedArtifactPaths(runContract)
  const isRequiredArtifact = requiredPaths.some((requiredPath) =>
    pathsReferToSameFile(requiredPath, path, cwd)
  )
  if (!isRequiredArtifact) {
    return null
  }

  const currentTurnMessages = messagesInCurrentAgentTurn(messages)
  const currentUserInput = currentTurnMessages
    .find((message) => message.role === 'user')
  const currentTurnIsProspectiveDocumentPhase = Boolean(
    contractLimitsCurrentTurnToDocumentArtifact(runContract)
    || (
      currentUserInput
      && inputRequestsProspectiveDocumentPhase(messageText(currentUserInput))
    ),
  )

  const extraEvidencedPaths = [path, ...requiredPaths]
  if (cwd) {
    for (const artifactPath of [path, ...requiredPaths]) {
      extraEvidencedPaths.push(isAbsolute(artifactPath) ? resolve(artifactPath) : resolve(cwd, artifactPath))
    }
  }

  const unsupportedPaths = unsupportedArtifactRepositoryPathClaims({
    content,
    messages,
    extraEvidencedPaths,
    prospectiveDocumentPhase: currentTurnIsProspectiveDocumentPhase,
  })

  const unsupported = unsupportedPaths.slice(0, 8)
  return unsupported.length > 0
    ? {
      path,
      unsupportedPaths: unsupported,
      draftSummary: summarizeBlockedArtifactDraft(content, unsupported),
      contentKey: contentEntry.key,
      content,
      extraEvidencedPaths,
      prospectiveDocumentPhase: currentTurnIsProspectiveDocumentPhase,
    }
    : null
}

function wasExactToolCallDeniedInHistory(messages: Message[], toolCall: ToolCall): boolean {
  const currentTurnMessages = messagesInCurrentAgentTurn(messages)
  const signaturesByToolCallId = new Map<string, string>()
  for (const message of currentTurnMessages) {
    for (const historicalToolCall of message.toolCalls ?? []) {
      signaturesByToolCallId.set(historicalToolCall.id, toolCallSignature(historicalToolCall))
    }
  }

  const currentSignature = toolCallSignature(toolCall)
  return currentTurnMessages.some((message) => {
    if (message.role !== 'tool' || !message.toolCallId) {
      return false
    }
    if (message.metadata?.[APPROVAL_FAILURE_STATUS_METADATA_KEY] !== 'denied') {
      return false
    }
    if (signaturesByToolCallId.get(message.toolCallId) !== currentSignature) {
      return false
    }
    return /^\[approval:denied\]/i.test(messageText(message).trim())
  })
}

function hasDeniedApprovalInHistory(messages: Message[], toolName?: string): boolean {
  const currentTurnMessages = messagesInCurrentAgentTurn(messages)
  const toolNamesByCallId = new Map<string, string>()
  for (const message of currentTurnMessages) {
    for (const historicalToolCall of message.toolCalls ?? []) {
      toolNamesByCallId.set(historicalToolCall.id, historicalToolCall.name)
    }
  }

  return currentTurnMessages.some((message) => {
    if (
      message.role !== 'tool'
      || message.metadata?.[APPROVAL_FAILURE_STATUS_METADATA_KEY] !== 'denied'
      || !/^\[approval:denied\]/i.test(messageText(message).trim())
    ) {
      return false
    }
    if (!toolName) return true
    if (message.toolCallId && toolNamesByCallId.get(message.toolCallId) === toolName) {
      return true
    }
    return messageText(message).includes(`run ${toolName}`)
  })
}

function messagesInCurrentAgentTurn(messages: Message[]): Message[] {
  let currentTurnStart = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      currentTurnStart = index
      break
    }
  }
  if (currentTurnStart < 0) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === 'user') {
        currentTurnStart = index
        break
      }
    }
  }
  return currentTurnStart >= 0 ? messages.slice(currentTurnStart) : []
}

/**
 * Reuse only the immediately preceding successful, byte-for-byte equivalent
 * workspace observation. Any intervening tool result is an invalidation
 * boundary, so edits, commands, and even a different read force a fresh
 * observation. This bounds repeated-read loops without turning the agent
 * context into a stale filesystem cache.
 */
function reusableAdjacentObservation(
  messages: Message[],
  toolCall: ToolCall,
): string | null {
  if (!CACHEABLE_ADJACENT_OBSERVATION_TOOLS.has(toolCall.name)) return null
  const signature = toolCallSignature(toolCall)
  const currentTurnMessages = messagesInCurrentAgentTurn(messages)
  for (let index = currentTurnMessages.length - 1; index >= 0; index -= 1) {
    const message = currentTurnMessages[index]!
    if (message.role === 'user') return null
    if (message.role !== 'tool') continue
    if (
      message.name === toolCall.name
      && message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'success'
      && message.metadata?.[TOOL_EXECUTION_SIGNATURE_METADATA_KEY] === signature
    ) {
      return messageText(message)
    }
    return null
  }
  return null
}

export interface ApprovalCallbackOptions {
  /** Bypass remembered/automatic decisions and open a fresh human prompt. */
  forcePrompt?: boolean
  /**
   * Abort signal for the run that is waiting.
   *
   * Only the client that opened the prompt can answer it, so when that client
   * goes away the wait can never finish on its own — it parks until the
   * approval times out, holding the session lease the whole time. Nobody can
   * re-approve (`/approvals/resume` answers BUSY) and no new turn can start.
   * Aborting the wait unwinds the run and frees the session; the pending
   * request and its checkpoint stay behind so the approval can still be
   * answered afterwards.
   */
  signal?: AbortSignal
}

export type ApprovalCallback = (
  toolCall: ToolCall,
  requestId: string,
  options?: ApprovalCallbackOptions,
) => Promise<boolean | ApprovalDecision> | ApprovalDecision

export type AutoApprovalEvaluator = (toolCall: ToolCall) => ApprovalDecision | null

export interface PendingToolExecution {
  toolCalls: ToolCall[]
  startIndex: number
  batchSize?: number
  currentExecutionId?: string
  initialApprovalDecision?: boolean | ApprovalDecision
  skipToolCallEventForStart?: boolean
}

interface ToolExecutionCallbacks {
  persistApprovalCheckpoint?: (
    requestId: string,
    toolCalls: ToolCall[],
    currentToolIndex: number,
  ) => Promise<void>
  clearApprovalCheckpoint?: (requestId: string) => Promise<void>
  clearRunCheckpoint?: () => Promise<void>
  persistRunCheckpoint?: (pending: PendingToolExecution) => Promise<void>
  loadToolExecution?: () => Promise<ToolExecutionRecord | null>
  saveToolExecution?: (record: ToolExecutionRecord) => Promise<void>
}

export interface ToolExecutionOptions extends ToolExecutionCallbacks {
  /** Trusted mode-controller ceiling, never model arguments. Absent means visible registry only. */
  delegationToolNames?: readonly string[]
  messages: Message[]
  toolCalls: ToolCall[]
  sessionId: string
  provider: string
  model: string
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
  primaryAgentId?: string
  autoApprove?: boolean
  requireToolApproval?: boolean
  auditLogger?: IAuditLogger
  hookRegistry?: HookRegistry
  deviceName?: string
  approvalCallback?: ApprovalCallback
  evaluateAutoApproval?: AutoApprovalEvaluator
  pending?: PendingToolExecution
  emitToolCallOnStart?: boolean
  signal?: AbortSignal
  editCheckpoint?: import('../tools/registry.js').EditCheckpointHandle
  toolStats?: import('./tool-learning/store.js').ToolStatsStore
  workspaceMutation?: import('../tools/registry.js').WorkspaceMutationHandle
  pluginEvents?: import('../plugins/event-bus.js').PluginEventBus
  cwd?: string
  workspaceRoot?: string
  /** Desktop writing-canvas doc id forwarded into doc.* tools for this turn. */
  writingDocId?: string
  /** Caller scope tags forwarded into ToolExecutionContext for scope-aware tools. */
  scopeTags?: string[]
  /** Originating channel context forwarded into ToolExecutionContext for channel-aware tools. */
  channelContext?: import('../tools/registry.js').ToolExecutionContext['channelContext']
  /** Durable run contract for this turn; used for invariant checks shared across graphs. */
  runContract?: AgentRunContract
  /** Whether tool-produced visual content can be attached to the next model turn. */
  canAttachVisualContent?: boolean
  /**
   * Appends directory-scoped AGENTS.md/CLAUDE.md below cwd to the first tool
   * result touching that directory. `null` disables the injection.
   */
  nestedInstructions?: NestedInstructionInjector | null
}

export async function* runToolExecution(options: ToolExecutionOptions): AsyncGenerator<AgentEvent> {
  const {
    delegationToolNames,
    messages,
    toolCalls,
    sessionId,
    provider,
    model,
    tools,
    policy,
    autonomy,
    primaryAgentId,
    autoApprove,
    requireToolApproval,
    auditLogger,
    hookRegistry,
    deviceName,
    approvalCallback,
    evaluateAutoApproval,
    pending,
    emitToolCallOnStart = true,
    signal,
    editCheckpoint,
    toolStats,
    workspaceMutation,
    pluginEvents,
    cwd,
    workspaceRoot,
    runContract,
    writingDocId,
    scopeTags,
    channelContext,
    persistApprovalCheckpoint,
    clearApprovalCheckpoint,
    clearRunCheckpoint,
    persistRunCheckpoint,
    loadToolExecution,
    saveToolExecution,
    canAttachVisualContent = true,
    nestedInstructions = getDefaultNestedInstructionInjector(),
  } = options
  const effectiveAutoApprove = requireToolApproval === true ? false : autoApprove
  const requiresFreshToolApproval = (toolName: string): boolean =>
    requireToolApproval === true && !isPolicyReadOnlyTool(toolName)
  const requiresFreshToolCallApproval = (toolCall: ToolCall): boolean =>
    requireToolApproval === true && !isPolicyReadOnlyRequest({
      tool: toolCall.name,
      input: toolCall.arguments,
      cwd,
      workspaceRoot,
      registrationSource: tools.registrationSource(toolCall.name),
      security: tools.securityDescriptor(toolCall.name),
    })
  const checkToolPolicyNow = (toolCall: ToolCall) => policy.check(
    {
      tool: toolCall.name,
      input: toolCall.arguments,
      cwd,
      workspaceRoot,
      registrationSource: tools.registrationSource(toolCall.name),
      security: tools.securityDescriptor(toolCall.name),
    },
    autonomy,
    primaryAgentId,
    effectiveAutoApprove,
  )
  const startIndex = pending?.startIndex ?? 0
  const pendingVisualContextMessages: Message[] = []

  const appendToolResultMessage = (
    toolCall: ToolCall,
    output: string,
    contentParts?: ContentPart[],
    metadata?: Record<string, unknown>,
  ) => {
    const trustedMetadata = {
      ...metadata,
      [TOOL_EXECUTION_SIGNATURE_METADATA_KEY]: toolCallSignature(toolCall),
    }
    messages.push({
      role: 'tool',
      content: summarizeToolOutputForAgentContext(toolCall.name, output),
      toolCallId: toolCall.id,
      name: toolCall.name,
      metadata: trustedMetadata,
    })

    const imageParts = (contentParts ?? [])
      .filter((part): part is Extract<ContentPart, { type: 'image' }> => part.type === 'image')
      .slice(0, MAX_TOOL_IMAGE_CONTEXT_PARTS)
    if (!imageParts.length) return

    if (!canAttachVisualContent) {
      pendingVisualContextMessages.push({
        role: 'system',
        content: [
          `[Visual output from ${toolCall.name} was not attached]`,
          `Tool call id: ${toolCall.id}.`,
          `The current model (${provider}/${model}) cannot currently receive image input, so the screenshot image was saved by the tool but omitted from the next model request to avoid a provider error.`,
          'Do not claim the screenshot was visually inspected.',
          'Still read the tool output text: browser.screenshot/browser.click/browser.evaluate may include a DOM/canvas layout audit with overflow, clipped-content, blank-band, low-detail canvas, low text-contrast, text/control overlap, or excessive-empty-space warnings that should be treated as validation evidence.',
          'Do not spend turns searching for a visual workaround or dispatching speculative visual subagents unless the runtime has already identified a concrete vision-capable model/surface for this run.',
          'If visual inspection is required by the task and no such verified vision-capable route is already available, stop the visual-validation loop and report UNVERIFIED with this blocker after any non-visual checks you can complete.',
        ].join('\n'),
      })
      return
    }

    pendingVisualContextMessages.push({
      role: 'user',
      content: [
        {
          type: 'text',
          text: [
            `[Visual output from ${toolCall.name}]`,
            `Tool call id: ${toolCall.id}.`,
            'Inspect this screenshot before making visual/layout judgments or choosing visual coordinates.',
            'Also read any browser layout-audit warnings in the tool output; warnings about overflow, clipped content, blank bands, low-detail canvas regions, low text contrast, text/control overlap, or excessive empty space are defects to fix or explicitly report.',
            'For frontend validation, check layout, spacing, text wrapping, overflow, viewport fit, representative interactive states, and obvious visual polish issues.',
            'If the image shows broken wrapping, edge-hugging layouts, oversized controls, or empty-looking interactive surfaces, fix the UI and capture another screenshot before claiming completion.',
            'Prefer browser_snapshot/browser_click element refs when available.',
            'For pixel-only targets, use browser_mouse_click_xy or browser_mouse_drag_xy with screenshot pixel coordinates.',
            'Use browser_mouse_wheel to scroll horizontally or vertically when the target is outside the current viewport.',
          ].join('\n'),
        },
        ...imageParts,
      ],
    })
  }

  const flushPendingVisualContextMessages = () => {
    if (!pendingVisualContextMessages.length) return
    messages.push(...pendingVisualContextMessages.splice(0))
  }

  const emitPostToolExecute = (
    toolCall: ToolCall,
    result: PostToolExecuteResult,
    extras: Record<string, unknown> = {},
  ): Promise<void> =>
    emitPostToolExecuteHook({
      hookRegistry,
      sessionId,
      provider,
      model,
      toolCall,
      result,
      extras,
    })

  const logToolExecution = async (toolCall: ToolCall): Promise<void> => {
    if (!auditLogger) {
      return
    }
    await auditLogger.log(
      buildToolExecutionAuditEntry({
        toolCall,
        sessionId,
        deviceName,
        autonomy,
        autoApprove: effectiveAutoApprove,
      }),
    )
  }

  let userActionRequired = false
  const executeTool = async (
    toolCall: ToolCall,
    tool: ToolDefinitionRuntime,
    executionId: string,
    options?: {
      persistExecution?: boolean
      saveRecoveryResult?: boolean
      recoverySource?: 'journal' | 'probe'
      startedAt?: string
      /** Sink for nested events (e.g. subagent_progress) the tool emits. */
      onProgress?: (event: AgentEvent) => void
    },
  ): Promise<ToolResult> => {
    const startedAt = options?.startedAt ?? new Date().toISOString()
    throwIfAborted(signal, `Tool ${toolCall.name} aborted`)

    const preHook = await emitPreToolExecuteHook({
      signal,
      hookRegistry,
      sessionId,
      provider,
      model,
      toolCall,
      cwd,
    })
    if (preHook.blocked) {
      return {
        output: preHook.output ?? `Tool ${toolCall.name} blocked by hook`,
        status: 'error',
        durationMs: 0,
      }
    }
    if (preHook.modifiedArguments) {
      // TOCTOU guard: policy.check and the approval prompt above ran against the
      // *pre-rewrite* arguments. A `pre:tool:execute` hook that rewrites the
      // arguments (e.g. path `/tmp/x` -> `/etc/passwd`) must not slip past that
      // gate. Re-validate the rewritten arguments and fail closed when they land
      // on a policy-denied or approval-required form — the granted consent was
      // for the original arguments, not these.
      const rewritten = preHook.modifiedArguments
      const changed = JSON.stringify(rewritten) !== JSON.stringify(toolCall.arguments)
      if (changed) {
        const revalidation = policy.check(
          {
            tool: toolCall.name,
            input: rewritten,
            cwd,
            workspaceRoot,
            registrationSource: tools.registrationSource(toolCall.name),
            security: tools.securityDescriptor(toolCall.name),
          },
          autonomy,
          primaryAgentId,
          effectiveAutoApprove,
        )
        if (!revalidation.allowed) {
          return {
            output: `Tool ${toolCall.name} blocked: a pre:tool:execute hook rewrote its arguments into a policy-denied form (${revalidation.reason ?? 'not allowed'}).`,
            status: 'error',
            durationMs: 0,
            code: 'HOOK_REWRITE_POLICY_DENIED',
          }
        }
        if (
          (revalidation.requiresApproval || requiresFreshToolCallApproval({
            ...toolCall,
            arguments: rewritten,
          }))
        ) {
          return {
            output: `Tool ${toolCall.name} blocked: a pre:tool:execute hook rewrote its arguments into a form that requires approval; the granted approval was for the original arguments.`,
            status: 'error',
            durationMs: 0,
            code: 'HOOK_REWRITE_REQUIRES_APPROVAL',
          }
        }
      }
      toolCall.arguments = rewritten
    }

    if (options?.persistExecution !== false) {
      await saveToolExecution?.(
        buildRunningToolExecutionRecord({
          executionId,
          sessionId,
          toolCall,
          startedAt,
        }),
      )
    }

    await logAgentDebugTrace({
      event: 'tool.execution.start',
      source: 'tool-execution',
      sessionId,
      runId: sessionId,
      toolCallId: toolCall.id,
      executionId,
      status: 'running',
      data: {
        tool: toolCall.name,
        input: toolCall.arguments,
        cwd,
        workspaceRoot,
        startedAt,
      },
    })

    await pluginEvents?.emit('tool.execute.before', {
      sessionId,
      executionId,
      tool: toolCall.name,
      input: toolCall.arguments,
    })

    let result: ToolResult
    // Link a per-tool AbortController to the run signal so the timeout can
    // cancel just this tool (best effort) without aborting the whole turn.
    const toolController = new AbortController()
    const onRunAbort = () => toolController.abort()
    if (signal) {
      if (signal.aborted) toolController.abort()
      else signal.addEventListener('abort', onRunAbort, { once: true })
    }
    const timeoutMs = resolveToolTimeoutMs()
    let timedOut = false
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined
    let rejectTimeout: ((error: Error) => void) | undefined
    let remainingMs = timeoutMs
    let armedAt = 0
    let executionFinished = false
    const pendingChildApprovals = new Map<string, string>()
    const armTimeout = () => {
      if (executionFinished || timeoutMs <= 0 || pendingChildApprovals.size || !rejectTimeout || timedOut) return
      armedAt = Date.now()
      timeoutHandle = setTimeout(() => {
        timedOut = true
        toolController.abort()
        rejectTimeout?.(new Error(`Tool ${toolCall.name} timed out after ${timeoutMs}ms`))
      }, Math.max(0, remainingMs))
    }
    const onProgress = (event: AgentEvent) => {
      if (executionFinished) return
      // Human consent is not tool execution time. Pause the remaining budget,
      // not reset it; sibling activity and replay cannot buy more execution.
      if (event.type === 'subagent_progress') {
        const wasPaused = pendingChildApprovals.size > 0
        const inner = event.inner
        if (inner.type === 'approval_request') pendingChildApprovals.set(event.subagentId, inner.requestId)
        else if ((inner.type === 'approval_response' && pendingChildApprovals.get(event.subagentId) === inner.requestId)
          || inner.type === 'tool_call' || inner.type === 'tool_result') pendingChildApprovals.delete(event.subagentId)
        if (!wasPaused && pendingChildApprovals.size && timeoutHandle) {
          remainingMs -= Date.now() - armedAt
          clearTimeout(timeoutHandle)
          timeoutHandle = undefined
        } else if (wasPaused && !pendingChildApprovals.size) armTimeout()
      }
      options?.onProgress?.(event)
    }
    const durationSince = () => Date.now() - Date.parse(startedAt)
    try {
      const executePromise = tool.execute(toolCall.arguments, {
        executionId,
        sessionId,
        startedAt,
        cwd,
        workspaceRoot,
        writingDocId,
        signal: toolController.signal,
        editCheckpoint,
        workspaceMutation,
        scopeTags,
        channelContext,
        runContract,
        delegatedAgentPolicy: {
          autonomy,
          requireToolApproval: requireToolApproval === true,
          allowedToolNames: [...(delegationToolNames ?? tools.list().map((registeredTool) => registeredTool.name))],
        },
        ...(options?.onProgress ? { emitEvent: onProgress } : {}),
      })
      if (timeoutMs > 0) {
        // If the tool hangs, a late rejection would surface as an unhandled
        // rejection once the race has already moved on — swallow it here.
        const guarded = executePromise.catch(() => undefined)
        result = await Promise.race([
          executePromise,
          new Promise<never>((_, reject) => {
            rejectTimeout = reject
            armTimeout()
          }),
        ])
        void guarded
      } else {
        result = await executePromise
      }
    } catch (error) {
      // Only a genuine run-level abort should propagate and end the turn.
      if (!timedOut && signal?.aborted) {
        throw getAbortError(signal, `Tool ${toolCall.name} aborted`)
      }
      // Any other throw (including a per-tool timeout) is isolated into an
      // error result so sibling tool calls keep their results and the
      // assistant tool_call is not orphaned into a hard provider error next
      // turn.
      const message = error instanceof Error ? error.message : String(error)
      result = timedOut
        ? {
            output: `Tool ${toolCall.name} timed out after ${timeoutMs}ms`,
            status: 'error',
            durationMs: durationSince(),
            code: 'TIMEOUT_TRANSIENT',
          }
        : {
            output: `Tool ${toolCall.name} threw an exception: ${message}`,
            status: 'error',
            durationMs: durationSince(),
            code: 'TOOL_THREW',
          }
    } finally {
      executionFinished = true
      if (timeoutHandle) clearTimeout(timeoutHandle)
      signal?.removeEventListener('abort', onRunAbort)
    }

    // Capture the registry-owned effect after the executor boundary. Tool
    // metadata is untrusted input, so overwrite this daemon-owned field rather
    // than allowing a plugin result to self-classify as observational.
    result = {
      ...result,
      metadata: {
        ...result.metadata,
        [TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY]:
          tools.securityDescriptor(toolCall.name)?.effect ?? 'unknown',
      },
    }

    if (options?.persistExecution !== false) {
      await saveToolExecution?.(
        buildCompletedToolExecutionRecord({
          executionId,
          sessionId,
          toolCall,
          startedAt,
          completedAt: new Date().toISOString(),
          result: {
            output: result.output,
            status: result.status,
            durationMs: result.durationMs,
            recovery: options?.recoverySource,
            ...(result.executionPosture ? { executionPosture: result.executionPosture } : {}),
            ...(result.metadata ? { metadata: result.metadata } : {}),
          },
        }),
      )
    }

    toolStats?.record(sessionId, toolCall.name, {
      status: result.status,
      output: result.output,
      durationMs: result.durationMs,
    })

    await logAgentDebugTrace({
      event: 'tool.execution.end',
      source: 'tool-execution',
      sessionId,
      runId: sessionId,
      toolCallId: toolCall.id,
      executionId,
      status: result.status,
      durationMs: result.durationMs,
      data: {
        tool: toolCall.name,
        output: result.output,
        outputChars: result.output.length,
        code: result.code,
        recovery: options?.recoverySource,
        executionPosture: result.executionPosture,
        metadata: result.metadata,
      },
    })
    const observationCache = result.metadata?.observationCache
    if (observationCache && typeof observationCache === 'object') {
      await logAgentDebugTrace({
        event: 'tool.observation-cache',
        source: 'tool-execution',
        sessionId,
        runId: sessionId,
        toolCallId: toolCall.id,
        executionId,
        status: String((observationCache as Record<string, unknown>).status ?? 'unknown'),
        data: {
          tool: toolCall.name,
          ...(observationCache as Record<string, unknown>),
        },
      })
    }

    await pluginEvents?.emit('tool.execute.after', {
      sessionId,
      executionId,
      tool: toolCall.name,
      status: result.status,
      durationMs: result.durationMs,
    })

    if (result.status === 'success' && isFileEditTool(toolCall.name)) {
      await emitPostFileEdit({
        hookRegistry,
        sessionId,
        toolCall,
        output: result.output,
      })
    }

    if (result.status === 'error' && typeof result.metadata?.userActionRequired === 'string' && result.metadata.userActionRequired.trim()) userActionRequired = true

    // Surface subdirectory instruction files with the first result that
    // touches their directory, so the model never has to probe for them.
    // Appended after persistence/stats so the stored execution keeps the raw
    // tool output.
    const instructionTarget = nestedInstructions && cwd
      ? nestedInstructionTargetPath(
          toolCall.name,
          toolCall.arguments as Record<string, unknown> | undefined,
        )
      : null
    if (instructionTarget) {
      const injected = formatNestedInstructions(
        await nestedInstructions!.collect({
          sessionId,
          targetPath: instructionTarget,
          cwd: cwd!,
          workspaceRoot,
        }),
        cwd!,
      )
      if (injected) {
        result = { ...result, output: `${result.output}\n\n${injected}` }
      }
    }
    return result
  }

  const prepareParallelToolCall = (toolCall: ToolCall): PreparedParallelToolCall | null =>
    requiresFreshToolApproval(toolCall.name)
      ? null
      : prepareParallelToolCallHelper({
          toolCall,
          tools,
          policy,
          autonomy,
          cwd,
          workspaceRoot,
          primaryAgentId,
          autoApprove: effectiveAutoApprove,
        })

  const collectParallelBatch = (
    startAt: number,
    current: PreparedParallelToolCall,
  ): PreparedParallelToolCall[] =>
    collectParallelBatchHelper({
      toolCalls,
      startAt,
      current,
      tools,
      policy,
      autonomy,
      cwd,
      workspaceRoot,
      primaryAgentId,
      autoApprove: effectiveAutoApprove,
      requiresFreshApproval: requiresFreshToolApproval,
    })

  const advanceRunCheckpoint = (nextIndex: number): Promise<void> =>
    advanceRunCheckpointHelper({
      persistRunCheckpoint,
      toolCalls,
      nextIndex,
    })

  const failedCouplingGroups = new Map<string, {
    toolCallId: string
    toolName: string
  }>()
  const rememberFailureCoupling = (
    toolCall: ToolCall,
    tool: ToolDefinitionRuntime | undefined,
    result: ToolResult,
  ): void => {
    const key = resolveToolBatchFailureCouplingKey(tool, toolCall.arguments)
    if (!key || !toolResultBlocksFailureCoupledSiblings(tool, result)) return
    failedCouplingGroups.set(key, {
      toolCallId: toolCall.id,
      toolName: toolCall.name,
    })
  }

  // A completed result and the advanced startIndex are checkpointed together.
  // Rebuild failure coupling from those trusted, execution-observed messages
  // so a daemon restart between siblings does not replay a deterministic
  // template failure that the interrupted run had already observed.
  for (let priorIndex = 0; priorIndex < Math.min(startIndex, toolCalls.length); priorIndex += 1) {
    const priorCall = toolCalls[priorIndex]!
    // The executed call's stored signature was computed after normalizeInput
    // canonicalized its arguments (e.g. terminal.run injects the resolved cwd
    // when the directory exists). A resumed batch receives the pristine prior
    // call, so it must be canonicalized the same way before comparing — a raw
    // signature silently fails to match and drops the failure coupling.
    const priorTool = tools.get(priorCall.name)
    const normalizedPriorArguments = priorTool?.normalizeInput
      ? await priorTool.normalizeInput(priorCall.arguments, { cwd, workspaceRoot })
      : priorCall.arguments
    const signature = toolCallSignature({ ...priorCall, arguments: normalizedPriorArguments })
    const priorResult = messages.findLast((message) =>
      message.role === 'tool'
      && message.toolCallId === priorCall.id
      && message.name === priorCall.name
      && message.metadata?.[TOOL_EXECUTION_SIGNATURE_METADATA_KEY] === signature
      && message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'error'
      && message.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true)
    if (!priorResult) continue
    rememberFailureCoupling(priorCall, tools.get(priorCall.name), {
      output: messageText(priorResult),
      status: 'error',
      durationMs: 0,
    })
  }

  const emittedToolCallIds = new Set<string>()
  if (pending?.skipToolCallEventForStart && toolCalls[startIndex]) {
    emittedToolCallIds.add(toolCalls[startIndex]!.id)
  }
  const emitToolCall = function* (toolCall: ToolCall): Generator<AgentEvent> {
    if (emittedToolCallIds.has(toolCall.id)) return
    emittedToolCallIds.add(toolCall.id)
    yield { type: 'tool_call', toolCall }
  }

  const yieldToolError = async function* (
    toolCall: ToolCall,
    output: string,
    extras: Record<string, unknown>,
  ): AsyncGenerator<AgentEvent> {
    // A denied attempt is still part of the audit trail, even when an early
    // semantic/path guard runs before normal policy/execution instrumentation.
    if (emitToolCallOnStart) yield* emitToolCall(toolCall)
    toolStats?.record(sessionId, toolCall.name, {
      status: 'error',
      output,
      durationMs: 0,
    })
    yield* yieldToolErrorBase({
      toolCall,
      output,
      extras,
      emitPostToolExecute,
      appendToolResultMessage,
      clearApprovalCheckpoint,
    })
  }

  const cancelRemainingToolCallsAfterDenial = async function* (
    deniedIndex: number,
    deniedToolName: string,
  ): AsyncGenerator<AgentEvent> {
    for (let remainingIndex = deniedIndex + 1; remainingIndex < toolCalls.length; remainingIndex++) {
      const remainingCall = toolCalls[remainingIndex]!
      yield* yieldToolError(
        remainingCall,
        `[approval:cancelled] Tool ${remainingCall.name} was not run because the user denied ${deniedToolName}.`,
        {
          source: 'approval',
          approved: false,
          reason: 'cancelled_after_denial',
        },
      )
    }
    await advanceRunCheckpoint(toolCalls.length)
  }

  const yieldToolSuccessOrRecovery = (
    toolCall: ToolCall,
    result: PostToolExecuteResult,
    extras: Record<string, unknown>,
    eventRecoveryOverride?: 'journal' | 'probe',
  ): AsyncGenerator<AgentEvent> => {
    if (result.status === 'error' && typeof result.metadata?.userActionRequired === 'string' && result.metadata.userActionRequired.trim()) userActionRequired = true
    return yieldToolSuccessOrRecoveryBase({
      toolCall,
      result,
      extras,
      eventRecoveryOverride,
      emitPostToolExecute,
      appendToolResultMessage,
      clearApprovalCheckpoint,
    })
  }

  await normalizeToolCallInputsInPlace({
    toolCalls,
    startIndex,
    tools,
    cwd,
    workspaceRoot,
  })

  for (let index = startIndex; index < toolCalls.length; index++) {
    throwIfAborted(signal, 'Tool execution aborted')
    const toolCall = toolCalls[index]!
    if (userActionRequired) {
      yield* yieldToolError(toolCall, 'Not executed: a preceding tool requires user action before this request can continue.', { source: 'prerequisite', reason: 'user_action_required' })
      await advanceRunCheckpoint(index + 1)
      continue
    }
    const tool = tools.get(toolCall.name)
    const firstPendingTool = index === startIndex
    let toolCallEventEmitted = false
    const executionId =
      firstPendingTool && pending?.currentExecutionId ? pending.currentExecutionId : undefined

    const journaledExecution = executionId ? await loadToolExecution?.() : null

    const journaledResult =
      journaledExecution &&
      journaledExecution.executionId === executionId &&
      journaledExecution.toolCallId === toolCall.id &&
      journaledExecution.status === 'completed' &&
      journaledExecution.result
        ? journaledExecution.result
        : null

    if (executionId) {
      await persistRunCheckpoint?.({
        toolCalls,
        startIndex: index,
        currentExecutionId: executionId,
        initialApprovalDecision: firstPendingTool ? pending?.initialApprovalDecision : undefined,
        skipToolCallEventForStart: firstPendingTool
          ? pending?.skipToolCallEventForStart
          : undefined,
      })

      if (emitToolCallOnStart && !(firstPendingTool && pending?.skipToolCallEventForStart)) {
        yield* emitToolCall(toolCall)
        toolCallEventEmitted = true
      }
    }

    if (journaledResult) {
      const journalPolicy = checkToolPolicyNow(toolCall)
      if (!journalPolicy.allowed) {
        const output = `Tool ${toolCall.name} blocked: ${journalPolicy.reason}`
        yield* yieldToolError(toolCall, output, {
          source: 'policy',
          reason: journalPolicy.reason,
          recovery: 'journal',
        })
        await advanceRunCheckpoint(index + 1)
        continue
      }
      // Hook reports recovery:'journal' (we read this off disk),
      // event surfaces the journaled record's own recovery tag so
      // consumers see when the original run was probe-recovered.
      yield* yieldToolSuccessOrRecovery(
        toolCall,
        {
          output: journaledResult.output,
          status: journaledResult.status,
          durationMs: journaledResult.durationMs,
          recovery: 'journal',
          ...(journaledResult.executionPosture
            ? { executionPosture: journaledResult.executionPosture }
            : {}),
          ...(journaledResult.metadata ? { metadata: journaledResult.metadata } : {}),
        },
        { source: 'journal' },
        journaledResult.recovery ?? 'journal',
      )
      rememberFailureCoupling(toolCall, tool, journaledResult)
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const canProbeInterruptedExecution = Boolean(
      journaledExecution
      && tool?.recoverInterruptedExecution
      && journaledExecution.executionId === executionId
      && journaledExecution.toolCallId === toolCall.id
      && journaledExecution.status === 'running',
    )
    if (canProbeInterruptedExecution) {
      const recoveryPolicy = checkToolPolicyNow(toolCall)
      if (!recoveryPolicy.allowed) {
        const reason = recoveryPolicy.reason ?? 'current workspace policy denied recovery'
        yield* yieldToolError(toolCall, `Tool ${toolCall.name} recovery blocked: ${reason}`, {
          source: 'policy',
          reason,
        })
        await advanceRunCheckpoint(index + 1)
        continue
      }
    }

    const recoveredResult =
      canProbeInterruptedExecution && journaledExecution && tool?.recoverInterruptedExecution
        ? await tool.recoverInterruptedExecution(toolCall.arguments, {
            executionId: journaledExecution.executionId,
            startedAt: journaledExecution.startedAt,
            sessionId,
            cwd,
            workspaceRoot,
          })
        : null

    if (recoveredResult && journaledExecution) {
      const contentParts = contentPartsFromToolResult(recoveredResult)
      const clientContentParts = clientContentPartsFromToolResult(recoveredResult)
      const recoveredPostResult: PostToolExecuteResult = {
        output: recoveredResult.output,
        status: recoveredResult.status,
        durationMs: recoveredResult.durationMs,
        recovery: 'probe',
        ...(contentParts ? { contentParts } : {}),
        ...(clientContentParts ? { clientContentParts } : {}),
        ...(recoveredResult.executionPosture
          ? { executionPosture: recoveredResult.executionPosture }
          : {}),
        ...(recoveredResult.metadata ? { metadata: recoveredResult.metadata } : {}),
      }
      await saveToolExecution?.(
        buildCompletedToolExecutionRecord({
          executionId: journaledExecution.executionId,
          sessionId,
          toolCall,
          startedAt: journaledExecution.startedAt,
          completedAt: new Date().toISOString(),
          result: {
            output: recoveredResult.output,
            status: recoveredResult.status,
            durationMs: recoveredResult.durationMs,
            recovery: 'probe',
            ...(recoveredResult.executionPosture
              ? { executionPosture: recoveredResult.executionPosture }
              : {}),
            ...(recoveredResult.metadata ? { metadata: recoveredResult.metadata } : {}),
          },
        }),
      )
      yield* yieldToolSuccessOrRecovery(toolCall, recoveredPostResult, { source: 'recovery_probe' })
      rememberFailureCoupling(toolCall, tool, recoveredResult)
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const couplingKey = resolveToolBatchFailureCouplingKey(tool, toolCall.arguments)
    const blockedBySibling = couplingKey
      ? failedCouplingGroups.get(couplingKey)
      : undefined
    if (blockedBySibling) {
      if (!executionId) {
        await persistRunCheckpoint?.({ toolCalls, startIndex: index })
        if (emitToolCallOnStart && !toolCallEventEmitted) {
          yield* emitToolCall(toolCall)
          toolCallEventEmitted = true
        }
      }
      const output = [
        '[error: SIBLING_INPUT_TEMPLATE_FAILED_PERMANENT]',
        `Tool ${toolCall.name} was not executed because sibling call ${blockedBySibling.toolCallId} failed with a deterministic input-template error shared by this model-selected batch.`,
        'Repair the common arguments before retrying unresolved targets. This skipped result does not prove anything about the target itself.',
      ].join(' ')
      yield* yieldToolError(toolCall, output, {
        source: 'batch_failure_coupling',
        reason: 'sibling_input_template_failed',
        blockedByToolCallId: blockedBySibling.toolCallId,
      })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    if (!tool) {
      if (!executionId) {
        await persistRunCheckpoint?.({
          toolCalls,
          startIndex: index,
        })
        if (emitToolCallOnStart && !toolCallEventEmitted) {
          yield* emitToolCall(toolCall)
          toolCallEventEmitted = true
        }
      }
      // Some models (notably gpt-oss harmony output via Ollama) occasionally
      // emit a tool call whose `name` is a chat-role label ("assistant",
      // "user", ...) while the arguments are perfectly valid for a real tool.
      // Give the model an actionable hint instead of a bare "Unknown tool".
      const looksLikeChatRoleName = CHAT_ROLE_TOOL_NAMES.has(toolCall.name.trim().toLowerCase())
      const output = looksLikeChatRoleName
        ? `Tool name "${toolCall.name}" is not valid — it looks like a chat-role label, not a registered tool. Re-emit the tool call with the actual tool name (e.g. fs.glob, terminal.run, fs.read), keeping the same arguments.`
        : `Unknown tool: ${toolCall.name}`
      yield* yieldToolError(toolCall, output, { source: 'registry', reason: 'unknown_tool' })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const missingArguments = missingRequiredToolArguments(tool, toolCall.arguments)
    if (missingArguments.length > 0) {
      yield* yieldToolError(toolCall,
        `Missing required arguments for ${toolCall.name}: ${missingArguments.join(', ')}. No tool was executed. Reissue the call with these fields included.`,
        { source: 'validation', reason: 'missing_required_arguments' })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    if (tool.validateInput) {
      const validationResult = await tool.validateInput(toolCall.arguments, {
        cwd,
        workspaceRoot,
      })
      if (validationResult?.status === 'error') {
        if (!executionId) {
          await persistRunCheckpoint?.({
            toolCalls,
            startIndex: index,
          })
          if (emitToolCallOnStart && !toolCallEventEmitted) {
            yield* emitToolCall(toolCall)
            toolCallEventEmitted = true
          }
        }
        rememberFailureCoupling(toolCall, tool, validationResult)
        const output = decorateErrorOutput(validationResult)
        yield* yieldToolError(toolCall, output, {
          source: 'validation',
          reason: validationResult.code ?? 'invalid_input',
        })
        await advanceRunCheckpoint(index + 1)
        continue
      }
    }

    const noTouchViolation = noTouchMutationViolation(runContract, toolCall, cwd)
    if (noTouchViolation) {
      const output = [
        `Tool ${toolCall.name} blocked: the active run contract says not to directly modify target "${noTouchViolation.target}".`,
        `The tool call targets "${noTouchViolation.path}".`,
        'Keep direct edits scoped to the requested agent/runtime implementation, or ask the user to explicitly change this boundary.',
      ].join(' ')
      yield* yieldToolError(toolCall, output, {
        source: 'path_validation',
        reason: 'user_stated_no_touch_target',
      })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const intentViolation = executionIntentMutationViolation(
      runContract,
      toolCall,
      tools,
      messages,
      cwd,
      workspaceRoot,
    )
    if (intentViolation) {
      const output = [
        `Tool ${toolCall.name} blocked: ${intentViolation}.`,
        'Preserve user constraints and approvals. If the request was semantically misclassified, correct the execution contract through the control surface; do not request unrelated workspace write permission or bypass policy.',
      ].join(' ')
      yield* yieldToolError(toolCall, output, {
        source: 'execution_intent',
        reason: 'workspace_mutation_forbidden',
      })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const misplacedArtifact = await misplacedRequestedArtifactWrite(runContract, toolCall, cwd)
    if (misplacedArtifact) {
      const output = [
        `Tool ${toolCall.name} blocked: the user requested artifact path "${misplacedArtifact.requestedPath}",`,
        `but the tool call targets "${misplacedArtifact.actualPath}" which does not resolve to the requested workspace path.`,
        `Use "${misplacedArtifact.requestedPath}" or "${misplacedArtifact.expectedPath}" instead.`,
      ].join(' ')
      yield* yieldToolError(toolCall, output, {
        source: 'path_validation',
        reason: 'misplaced_requested_artifact',
      })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const artifactRegression = requiredArtifactWriteRegression(
      runContract,
      messages,
      toolCall,
      cwd,
    )
    if (artifactRegression) {
      const output = [
        `Tool ${toolCall.name} blocked: refusing to overwrite required artifact "${artifactRegression.path}" with regressed content.`,
        `${artifactRegression.reason}.`,
        `Previous write was ${artifactRegression.previousBytes} bytes; proposed write is ${artifactRegression.nextBytes} bytes.`,
        'Use fs.append to add new sections, fs.edit/apply_patch to update the relevant section, or rewrite the full artifact while preserving the accumulated evidence and required scope.',
      ].join(' ')
      yield* yieldToolError(toolCall, output, {
        source: 'artifact_validation',
        reason: 'required_artifact_regression',
      })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const unsupportedArtifactClaims = requiredArtifactWriteUnsupportedPathClaims(
      runContract,
      messages,
      toolCall,
      cwd,
    )
    if (unsupportedArtifactClaims) {
      const sanitized = sanitizeArtifactContentForUnsupportedPathClaims(
        unsupportedArtifactClaims.content,
        unsupportedArtifactClaims.unsupportedPaths,
      )
      if (sanitized) {
        const remainingUnsupported = unsupportedArtifactRepositoryPathClaims({
          content: sanitized.content,
          messages,
          extraEvidencedPaths: unsupportedArtifactClaims.extraEvidencedPaths,
          prospectiveDocumentPhase: unsupportedArtifactClaims.prospectiveDocumentPhase,
        })
        if (remainingUnsupported.length === 0) {
          toolCall.arguments = {
            ...toolCall.arguments,
            [unsupportedArtifactClaims.contentKey]: sanitized.content,
          }
          yield {
            type: 'thinking',
            content: [
              '[supervisor] Required artifact draft cited unsupported repository path claims.',
              `Removed ${unsupportedArtifactClaims.unsupportedPaths.length} unsupported claim(s) across ${sanitized.removedLineCount} line(s) and continued the artifact write instead of discarding the draft.`,
            ].join(' '),
          }
        } else {
          const output = [
            `Tool ${toolCall.name} blocked: required artifact "${unsupportedArtifactClaims.path}" cites repository paths without prior read/search/glob/write evidence: ${unsupportedArtifactClaims.unsupportedPaths.join(', ')}.`,
            `Do not retry ${toolCall.name} with those claims unchanged.`,
            unsupportedArtifactClaims.draftSummary,
            'Next valid step: call fs.glob for missing glob/directory inventories, fs.search for symbol or route claims, or fs.read for concrete files; otherwise remove the unsupported path claims or state the artifact is partial without asserting unobserved paths.',
          ].join(' ')
          yield* yieldToolError(toolCall, output, {
            source: 'artifact_validation',
            reason: 'unsupported_required_artifact_path_claims',
          })
          await advanceRunCheckpoint(index + 1)
          continue
        }
      } else {
      const output = [
        `Tool ${toolCall.name} blocked: required artifact "${unsupportedArtifactClaims.path}" cites repository paths without prior read/search/glob/write evidence: ${unsupportedArtifactClaims.unsupportedPaths.join(', ')}.`,
        `Do not retry ${toolCall.name} with those claims unchanged.`,
        unsupportedArtifactClaims.draftSummary,
        'Next valid step: call fs.glob for missing glob/directory inventories, fs.search for symbol or route claims, or fs.read for concrete files; otherwise remove the unsupported path claims or state the artifact is partial without asserting unobserved paths.',
      ].join(' ')
      yield* yieldToolError(toolCall, output, {
        source: 'artifact_validation',
        reason: 'unsupported_required_artifact_path_claims',
      })
      await advanceRunCheckpoint(index + 1)
      continue
      }
    }

    if (wasExactToolCallDeniedInHistory(messages, toolCall)) {
      const output = [
        `[approval:denied] Tool ${toolCall.name} blocked: this exact tool call was already denied in the current turn.`,
        'This denial ends the current turn; do not retry or work around the rejected action.',
      ].join(' ')
      yield* yieldToolError(toolCall, output, {
        source: 'approval',
        approved: false,
        decision: 'denied',
        reason: 'repeated_denied_tool_call',
      })
      await advanceRunCheckpoint(index + 1)
      yield* cancelRemainingToolCallsAfterDenial(index, toolCall.name)
      return
    }

    const checkedPolicyResult = checkToolPolicyNow(toolCall)
    const freshApprovalRequired = requiresFreshToolCallApproval(toolCall)
    const policyResult =
      checkedPolicyResult.allowed && freshApprovalRequired
        ? {
            ...checkedPolicyResult,
            requiresApproval: true,
            reason: checkedPolicyResult.requiresApproval
              ? checkedPolicyResult.reason
              : 'This turn requires explicit human approval for side-effecting tools',
          }
        : checkedPolicyResult
    await logAgentDebugTrace({
      event: 'policy.tool-decision',
      source: 'tool-execution',
      sessionId,
      runId: sessionId,
      toolCallId: toolCall.id,
      status: policyResult.allowed
        ? policyResult.requiresApproval ? 'approval-required' : 'allowed'
        : 'denied',
      data: {
        tool: toolCall.name,
        security: tools.securityDescriptor(toolCall.name),
        autonomy,
        primaryAgentId,
        workspaceBoundary: workspaceRoot ? 'strict' : 'unrestricted',
        freshApprovalRequired,
        allowed: policyResult.allowed,
        requiresApproval: policyResult.requiresApproval ?? false,
        reason: policyResult.reason,
      },
    })
    if (
      !executionId &&
      (!policyResult.allowed ||
        (policyResult.requiresApproval &&
          (autonomy !== AutonomyLevel.Autonomous || freshApprovalRequired || Boolean(approvalCallback))))
    ) {
      await persistRunCheckpoint?.({
        toolCalls,
        startIndex: index,
      })
      if (emitToolCallOnStart && !toolCallEventEmitted) {
        yield* emitToolCall(toolCall)
        toolCallEventEmitted = true
      }
    }
    if (!policyResult.allowed) {
      const output = `Tool ${toolCall.name} blocked: ${policyResult.reason}`
      yield* yieldToolError(toolCall, output, { source: 'policy', reason: policyResult.reason })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const cachedObservation = reusableAdjacentObservation(messages, toolCall)
    if (cachedObservation !== null) {
      if (!executionId) {
        await persistRunCheckpoint?.({ toolCalls, startIndex: index })
        if (emitToolCallOnStart && !toolCallEventEmitted) {
          yield* emitToolCall(toolCall)
          toolCallEventEmitted = true
        }
      }
      await logAgentDebugTrace({
        event: 'tool.execution.cache-hit',
        source: 'tool-execution',
        sessionId,
        runId: sessionId,
        toolCallId: toolCall.id,
        status: 'success',
        data: { tool: toolCall.name, input: toolCall.arguments },
      })
      yield* yieldToolSuccessOrRecovery(
        toolCall,
        {
          output: cachedObservation,
          status: 'success',
          durationMs: 0,
          metadata: { cacheHit: true },
        },
        { source: 'adjacent_observation_cache' },
      )
      await advanceRunCheckpoint(index + 1)
      continue
    }

    let approval = normalizeApprovalDecision(true)
    let approvalResolvedByRule = false
    if (
      policyResult.requiresApproval &&
      autonomy === AutonomyLevel.Autonomous &&
      !freshApprovalRequired
    ) {
      const autoDecision = evaluateAutoApproval?.(toolCall) ?? null
      if (autoDecision?.approved && autoDecision.autoApproval) {
        yield {
          type: 'auto_approval',
          toolCall,
          requestId: toolCall.id,
          decision: autoDecision.decision === 'denied' ? 'denied' : 'approved',
          rule: autoDecision.autoApproval.rule,
          scope: autoDecision.autoApproval.scope,
        }
        approval = normalizeApprovalDecision(autoDecision)
        approvalResolvedByRule = true
      } else if (autoDecision && !autoDecision.approved) {
        const output = buildDenialToolOutput({
          toolName: toolCall.name,
          decision: autoDecision.decision === 'feedback' ? 'feedback' : 'denied',
          note: autoDecision.note,
          timedOut: autoDecision.timedOut,
        })
        yield* yieldToolError(toolCall, output, {
          source: 'approval',
          approved: false,
          decision: autoDecision.decision === 'feedback' ? 'feedback' : 'denied',
          timedOut: autoDecision.timedOut,
          reason: 'remembered_denial_rule',
        })
        await advanceRunCheckpoint(index + 1)
        if (autoDecision.decision === 'feedback') {
          continue
        }
        yield* cancelRemainingToolCallsAfterDenial(index, toolCall.name)
        return
      } else if (!approvalCallback) {
        if (auditLogger) {
          await auditLogger.log(
            buildAutonomousApprovalBlockAuditEntry({
              toolCall,
              sessionId,
              deviceName,
              autonomy,
              reason: policyResult.reason,
            }),
          )
        }
        // `supervised` rules already run directly under Autonomous. Explicit
        // `ask` rules and raw shell wrappers can pause only when an interactive
        // approval callback exists; unattended runs fail closed here.
        const reason = [
          `Tool ${toolCall.name} requires human approval (policy rule 'ask' or shell wrapper)`,
          'and this run is autonomous, so it was not executed.',
          'Continue with parts of the task that do not need it, or report the exact',
          'command as the blocker.',
        ].join(' ')
        yield* yieldToolError(toolCall, reason, {
          source: 'policy',
          reason: policyResult.reason ?? 'approval required',
        })
        await advanceRunCheckpoint(index + 1)
        continue
      }
    }
    if (
      policyResult.requiresApproval &&
      !approvalResolvedByRule &&
      (autonomy !== AutonomyLevel.Autonomous || freshApprovalRequired || Boolean(approvalCallback))
    ) {
      const strictPoisoning = process.env.SEPILOTD_STRICT_DENIAL_POISONING === '1'
      // Exact semantic replays are blocked above. A denial of one fs.write or
      // terminal.run invocation must not poison every materially different
      // call that happens to share the same tool name; that made user feedback
      // unrecoverable and caused repeated "already denied" loops. Operators
      // that need turn-wide fail-stop semantics can opt into strict poisoning.
      if (strictPoisoning && hasDeniedApprovalInHistory(messages)) {
        const output = [
          `[approval:denied] Tool ${toolCall.name} blocked: an approval for ${toolCall.name} was already denied in the current turn.`,
          'Strict denial poisoning is enabled.',
          'This denial ends the current turn.',
        ].join(' ')
        yield* yieldToolError(toolCall, output, {
          source: 'approval',
          approved: false,
          decision: 'denied',
          reason: 'prior_approval_denied',
        })
        await advanceRunCheckpoint(index + 1)
        yield* cancelRemainingToolCallsAfterDenial(index, toolCall.name)
        return
      }
      const requestId = toolCall.id

      if (firstPendingTool && pending?.initialApprovalDecision !== undefined) {
        approval = normalizeApprovalDecision(pending.initialApprovalDecision)
      } else if (approvalCallback) {
        await clearRunCheckpoint?.()
        await persistApprovalCheckpoint?.(requestId, toolCalls, index)
        const approvalDecisionResult = approvalCallback(toolCall, requestId, {
          forcePrompt: freshApprovalRequired,
          signal,
        })
        // ApprovalCallback returns ApprovalDecision synchronously when a
        // remembered session/always rule short-circuits the prompt, and
        // a Promise otherwise. Discriminating up front lets us emit
        // `auto_approval` instead of `approval_request` for the
        // short-circuit path — operators see a positive signal that the
        // tool ran because of a prior decision, not because the daemon
        // silently skipped consent.
        if (approvalDecisionResult instanceof Promise) {
          const { describeRuleFor } = await import('../server/runtime/approval-rules.js')
          const suggestedRule = describeRuleFor(toolCall.name, toolCall.arguments)
          const { buildApprovalPreviewDiff } = await import('../tools/edit-diff.js')
          const previewDiff = await buildApprovalPreviewDiff(toolCall, cwd, {
            // The preview reads the target file before consent; honor the
            // operator's fs.read deny rules so a write-approval prompt
            // cannot leak content that direct reads are denied.
            canReadPath: (path) =>
              policy.check(
                {
                  tool: 'fs.read',
                  input: { path },
                  cwd,
                  workspaceRoot,
                  registrationSource: tools.registrationSource('fs.read'),
                  security: tools.securityDescriptor('fs.read'),
                },
                autonomy,
                primaryAgentId,
                effectiveAutoApprove,
              ).allowed,
          })
          const approvalContext = buildApprovalContext(messages)
          yield {
            type: 'approval_request',
            toolCall,
            requestId,
            suggestedRule,
            ...(previewDiff ? { previewDiff } : {}),
            ...(approvalContext ? { context: approvalContext } : {}),
          }
          approval = normalizeApprovalDecision(await approvalDecisionResult)
        } else {
          approval = normalizeApprovalDecision(approvalDecisionResult)
          if (approval.autoApproval && approval.decision !== 'feedback') {
            yield {
              type: 'auto_approval',
              toolCall,
              requestId,
              decision: approval.decision,
              rule: approval.autoApproval.rule,
              scope: approval.autoApproval.scope,
            }
          }
        }
      } else {
        const output = 'Tool execution requires approval (no approval handler)'
        yield* yieldToolError(toolCall, output, {
          source: 'approval',
          approved: false,
          reason: 'missing_handler',
        })
        continue
      }

      if (!approval.approved) {
        const output = buildDenialToolOutput({
          toolName: toolCall.name,
          decision: approval.decision === 'feedback' ? 'feedback' : 'denied',
          note: approval.note,
          timedOut: approval.timedOut,
        })
        yield* yieldToolError(toolCall, output, {
          source: 'approval',
          approved: false,
          decision: approval.decision,
          note: approval.note,
          timedOut: approval.timedOut,
          ...(approval.stop === true ? { stop: true } : {}),
          reason: approval.decision,
        })
        await advanceRunCheckpoint(index + 1)
        if (approval.decision === 'feedback' || approval.timedOut) {
          continue
        }
        yield* cancelRemainingToolCallsAfterDenial(index, toolCall.name)
        return
      }
    }

    // Approval may remain open for minutes. Re-resolve the canonical root and
    // target after the user responds so a symlink/junction swap cannot turn a
    // previously reviewed in-workspace action into an outside access.
    const executionPolicy = checkToolPolicyNow(toolCall)
    if (!executionPolicy.allowed) {
      const reason = executionPolicy.reason ?? 'current workspace policy denied execution'
      yield* yieldToolError(toolCall, `Tool ${toolCall.name} blocked before execution: ${reason}`, {
        source: 'policy',
        reason,
      })
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const preparedParallelCall = prepareParallelToolCall(toolCall)
    const parallelBatch =
      !executionId && preparedParallelCall
        ? collectParallelBatch(index, preparedParallelCall)
        : [preparedParallelCall].filter(
            (value): value is PreparedParallelToolCall => value !== null,
          )

    if (parallelBatch.length > 1) {
      await persistRunCheckpoint?.({
        toolCalls,
        startIndex: index,
        batchSize: parallelBatch.length,
      })

      for (const [batchIndex, batchToolCall] of parallelBatch.entries()) {
        if (
          emitToolCallOnStart &&
          !toolCallEventEmitted &&
          !(batchIndex === 0 && firstPendingTool && pending?.skipToolCallEventForStart)
        ) {
          yield* emitToolCall(batchToolCall.toolCall)
        }
      }
      toolCallEventEmitted = true

      // allSettled (not all): one tool that still throws — e.g. a run-abort
      // propagating from executeTool, or a logging/hook failure — must not
      // discard the successful siblings' results, which would orphan their
      // matching assistant tool_calls and hard-error the next turn.
      const parallelExecutionStarted = new Set<string>()
      const settled = yield* streamToolProgress(emit => Promise.allSettled(
        parallelBatch.map(async ({ toolCall: currentToolCall, tool: currentTool }) => {
          await logToolExecution(currentToolCall)
          parallelExecutionStarted.add(currentToolCall.id)
          const result = await executeTool(currentToolCall, currentTool, randomUUID(), {
            persistExecution: false,
            onProgress: emit,
          })
          await emitPostToolExecute(
            currentToolCall,
            {
              output: result.output,
              status: result.status,
              durationMs: result.durationMs,
              ...(result.executionPosture ? { executionPosture: result.executionPosture } : {}),
              ...(result.metadata ? { metadata: result.metadata } : {}),
            },
            { source: 'parallel_batch' },
          )
          return {
            toolCall: currentToolCall,
            result,
            executionObserved: true,
          }
        }),
      ))
      const abortRejection = settled.find(
        (entry): entry is PromiseRejectedResult =>
          entry.status === 'rejected' && (isAbortError(entry.reason) || Boolean(signal?.aborted)),
      )
      if (abortRejection) {
        throw getAbortError(signal, 'Tool execution aborted')
      }
      const batchResults = settled.map((entry, batchIndex) => {
        if (entry.status === 'fulfilled') return entry.value
        const failedToolCall = parallelBatch[batchIndex]!.toolCall
        const message = entry.reason instanceof Error ? entry.reason.message : String(entry.reason)
        const result: ToolResult = {
          output: `Tool ${failedToolCall.name} failed: ${message}`,
          status: 'error',
          durationMs: 0,
          code: 'TOOL_THREW',
        }
        return {
          toolCall: failedToolCall,
          result,
          executionObserved: parallelExecutionStarted.has(failedToolCall.id),
        }
      })

      for (const {
        toolCall: completedToolCall,
        result,
        executionObserved,
      } of batchResults) {
        const decorated = decorateErrorOutput(result)
        const clientContentParts = clientContentPartsFromToolResult(result)
        const eventMetadata = {
          ...result.metadata,
          ...(executionObserved
            ? { [TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY]: true }
            : {}),
        }
        yield {
          type: 'tool_result',
          toolCallId: completedToolCall.id,
          output: decorated,
          status: result.status,
          ...(clientContentParts ? { contentParts: clientContentParts } : {}),
          ...(result.executionPosture ? { executionPosture: result.executionPosture } : {}),
          ...(Object.keys(eventMetadata).length > 0 ? { metadata: eventMetadata } : {}),
        }
        appendToolResultMessage(
          completedToolCall,
          decorated,
          contentPartsFromToolResult(result),
          {
            toolResultStatus: result.status,
            ...(result.executionPosture
              ? { executionPosture: result.executionPosture }
              : {}),
            ...(executionObserved
              ? { [TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY]: true }
              : {}),
          },
        )
        await clearApprovalCheckpoint?.(completedToolCall.id)
      }

      index += parallelBatch.length - 1
      await advanceRunCheckpoint(index + 1)
      continue
    }

    const sequentialExecutionId = executionId ?? randomUUID()
    if (!executionId) {
      await persistRunCheckpoint?.({
        toolCalls,
        startIndex: index,
        currentExecutionId: sequentialExecutionId,
        initialApprovalDecision: firstPendingTool ? pending?.initialApprovalDecision : undefined,
        skipToolCallEventForStart: firstPendingTool
          ? pending?.skipToolCallEventForStart
          : undefined,
      })

      if (
        emitToolCallOnStart &&
        !toolCallEventEmitted &&
        !(firstPendingTool && pending?.skipToolCallEventForStart)
      ) {
        yield* emitToolCall(toolCall)
        toolCallEventEmitted = true
      }
    }

    await logToolExecution(toolCall)

    const result = yield* streamToolProgress(emit => executeTool(toolCall, tool, sequentialExecutionId, {
      onProgress: emit,
    }))
    rememberFailureCoupling(toolCall, tool, result)
    const decoratedSequential = decorateErrorOutput(result)
    const contentParts = contentPartsFromToolResult(result)
    const clientContentParts = clientContentPartsFromToolResult(result)
    const sequentialPostResult: PostToolExecuteResult = {
      output: decoratedSequential,
      status: result.status,
      durationMs: result.durationMs,
      ...(contentParts ? { contentParts } : {}),
      ...(clientContentParts ? { clientContentParts } : {}),
      ...(result.executionPosture ? { executionPosture: result.executionPosture } : {}),
      ...(result.metadata ? { metadata: result.metadata } : {}),
    }
    yield* yieldToolSuccessOrRecovery(toolCall, sequentialPostResult, { source: 'execution' })
    await advanceRunCheckpoint(index + 1)
  }

  flushPendingVisualContextMessages()
}
