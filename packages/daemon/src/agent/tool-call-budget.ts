import type { Message, ToolCall } from '@sepilotd/core'
import { inputPositiveCapabilityScope } from './task-contract.js'
import { TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY } from './policy-failure.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

export const DEFAULT_MAX_TOOL_CALLS_PER_BATCH = 16
const MIN_CONTEXT_AWARE_TOOL_CALLS = 4
const CONTEXT_TOKENS_PER_TOOL_CALL = 4_096

export interface ToolCallBatchLimitResult {
  accepted: ToolCall[]
  requestedCount: number
  droppedCount: number
  limit: number
  capped: boolean
}

export interface ToolCallBudgetHistoryEntry {
  tool: string
  input?: Record<string, unknown>
  executionObserved: boolean
}

export interface ToolCardinalityIdentity {
  tool: string
  aliases: readonly string[]
}

export interface NonCanonicalExactOnceToolCall {
  call: ToolCall
  canonicalTool: string
}

export interface ToolCanonicalReadIdentity {
  tool: string
  urlTemplates: readonly string[]
}

export interface NonCanonicalReadTargetToolCall {
  call: ToolCall
  canonicalTool: string
}

type CardinalityAwareTool = {
  name: string
  cardinalityAliases?: readonly string[] | (() => readonly string[])
}

type CanonicalReadTargetAwareTool = {
  name: string
  canonicalReadUrlTemplates?: readonly string[] | (() => readonly string[])
}

const TOOL_NAME_PATTERN = /\b[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+\b/giu
const EXACTLY_ONCE_PATTERN =
  /^(?:\s|[,;:()\[\]{}]|(?:은|는|을|를|도)|\b(?:call|capability|integration|invoke|use|run|then|and|with|at|the|tool)\b){0,80}(?:exactly\s+(?:once|one\s+time)|one\s+time\s+only|1\s+time\s+only|정확히\s*(?:한\s*번|한번|1\s*회)|(?:한\s*번|한번|1\s*회)만)/iu
const EXACTLY_ONCE_MARKER_PATTERN =
  /(?:exactly\s+(?:once|one\s+time)|one\s+time\s+only|1\s+time\s+only|정확히\s*(?:한\s*번|한번|1\s*회)|(?:한\s*번|한번|1\s*회)만)/iu
const GLOBAL_NAMED_TOOLS_EXACTLY_ONCE_PATTERN =
  /(?:\b(?:use|call|invoke|run)\s+only\b[^.!?\n]{0,200}\b(?:exactly\s+once\s+each|each\s+exactly\s+once|once\s+each)\b|\b(?:the\s+)?(?:following|listed|named|specified|registered|these)\b[^.!?\n]{0,48}\b(?:tools?|calls?)\b[^.!?\n]{0,160}\b(?:exactly\s+once\s+each|each\s+exactly\s+once|once\s+each)\b|(?:다음|아래|나열(?:한|된)|명시(?:한|된)|등록(?:한|된)?)\s*[^.!?。！？\n]{0,32}?(?:등록\s*)?(?:도구(?:들)?|호출(?:들)?)(?:만)?[^.!?。！？\n]{0,120}(?:각각|각\s*(?:도구|호출)(?:를|는)?)[^.!?。！？\n]{0,32}(?:정확히\s*)?(?:한\s*번|한번|1\s*회)(?:만)?)/iu
const CARDINALITY_ALIAS_PATTERN = /^[\p{L}\p{N}][\p{L}\p{N}._:/-]{1,255}$/u
const CARDINALITY_TARGET_ARGUMENT_PATTERN = /^(?:baseurl|baseuri|endpoint|host|hostname|origin|uri|url)$/iu
const CANONICAL_URL_TEMPLATE_PARAMETER_PATTERN = /^\{[A-Za-z][A-Za-z0-9_]*\}$/u
const CANONICAL_URL_TEMPLATE_PARAMETER_GLOBAL_PATTERN = /\{[A-Za-z][A-Za-z0-9_]*\}/gu
const CANONICAL_URL_TEMPLATE_PARAMETER_PREFIX = 'sepilot-canonical-param-'
const CLOSED_OTHER_TOOLS_ENGLISH_PATTERN =
  /(?:\b(?:do\s+not|don't|dont|never)\s+(?:call|use|invoke|run)\s+(?:any\s+)?(?:other|another|additional)\s+tools?|\bno\s+(?:other|additional)\s+tools?(?:\s+(?:are|should\s+be|may\s+be))?\s+(?:allowed|needed|permitted|used|called))(?:\s+(?:in|for)\s+(?:this|the)\s+(?:turn|run|task|request))?\s*(?:[.!?;]|$)/iu
const CLOSED_OTHER_TOOLS_KOREAN_PATTERN =
  /(?:(?:다른|추가|그\s*외|이외의?)\s*(?:어떤\s*)?도구|도구(?:는|를|도)?\s*(?:더|추가로))[^.!?。！？\n]{0,60}(?:호출|사용|실행)(?:은|는|을|를)?\s*하지\s*(?:마|말(?:아|라|고)?|마라|않(?:아|고|도록|는다|습니다))\s*(?:[.!?。！？;；]|$)/iu
const SINGLE_TOOL_EXACT_CALL_PATTERN =
  /(?:\b(?:call|invoke|use|run)\s+(?:(?:it|this|that|the\s+(?:tool|capability|integration))\s+)?(?:exactly\s+(?:once|one\s+time)|one\s+time\s+only|1\s+time\s+only)\b|\b(?:exactly\s+(?:once|one\s+time)|one\s+time\s+only|1\s+time\s+only)\s+(?:call|invocation|use|run)\b|(?:정확히\s*(?:한\s*번|한번|1\s*회)|(?:한\s*번|한번|1\s*회)만)\s*(?:호출|사용|실행))/iu
const REPEATED_SEMANTIC_ACTION_EXACT_ONCE_PATTERN =
  /(?:\b(?:each|every)\s+(?:(?:required|listed|named)\s+)?(?:actions?|steps?|commands?|checks?|queries|requests?)[^.!?\n]{0,32}\b(?:exactly\s+)?once\b|(?:작업|단계|명령|검증|쿼리|요청)(?:들)?(?:을|를|은|는|도)?[^.!?。！？\n]{0,32}(?:각각|각\s*(?:작업|단계|명령|검증|쿼리|요청)?(?:을|를|은|는|도)?)[^.!?。！？\n]{0,20}(?:정확히\s*)?(?:한\s*번|한번|1\s*회))/iu

/**
 * Bound model-generated fan-out before any tool is executed. The default is
 * context-aware so small-window models cannot create a result batch that is
 * larger than the next prompt, while large-window models still get useful
 * parallelism. An explicit positive limit remains an operator override.
 */
export function resolveToolCallBatchLimit(
  contextWindowTokens: number,
  configuredLimit?: number,
): number {
  if (
    typeof configuredLimit === 'number'
    && Number.isFinite(configuredLimit)
    && configuredLimit > 0
  ) {
    return Math.max(1, Math.floor(configuredLimit))
  }

  const contextAware = Math.floor(
    Math.max(1, contextWindowTokens) / CONTEXT_TOKENS_PER_TOOL_CALL,
  )
  return Math.max(
    MIN_CONTEXT_AWARE_TOOL_CALLS,
    Math.min(DEFAULT_MAX_TOOL_CALLS_PER_BATCH, contextAware),
  )
}

export function capToolCallBatch(
  toolCalls: readonly ToolCall[],
  limit: number,
): ToolCallBatchLimitResult {
  const normalizedLimit = Math.max(1, Math.floor(limit))
  const accepted = toolCalls.slice(0, normalizedLimit)
  const droppedCount = Math.max(0, toolCalls.length - accepted.length)
  return {
    accepted,
    requestedCount: toolCalls.length,
    droppedCount,
    limit: normalizedLimit,
    capped: droppedCount > 0,
  }
}

export const TOOL_BATCH_CAP_REMINDER_KIND = 'tool_batch_cap'

export function buildToolCallBatchCapMessage(
  result: Pick<ToolCallBatchLimitResult, 'accepted' | 'requestedCount' | 'droppedCount' | 'limit'>,
): Message {
  return {
    role: 'system',
    metadata: { reminderKind: TOOL_BATCH_CAP_REMINDER_KIND },
    content: [
      '[Tool batch cap]',
      `The previous response requested ${result.requestedCount} tool calls; the runtime executed ${result.accepted.length} and dropped ${result.droppedCount} to protect working-memory and provider context budgets.`,
      'Do not call more tools in this turn.',
      'Synthesize the best evidence-backed final answer now. If the retained evidence is insufficient, return INCOMPLETE and name the smallest concrete next step instead of restarting broad discovery.',
    ].join(' '),
  }
}

export function hasToolCallBatchCap(messages: readonly Message[]): boolean {
  return messages.some((message) =>
    message.role === 'system'
    && message.metadata?.reminderKind === TOOL_BATCH_CAP_REMINDER_KIND
  )
}

/**
 * Reconstruct only trusted, current-turn execution attempts for cardinality
 * budgets. A tool result without the daemon-owned execution marker represents
 * a pre-execution rejection (policy, approval, unknown tool, and similar) and
 * must not consume a user's exactly-once allowance.
 */
export function toolCallBudgetHistoryFromCurrentTurnMessages(
  messages: readonly Message[],
): ToolCallBudgetHistoryEntry[] {
  let turnStart = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      turnStart = index
      break
    }
  }
  if (turnStart < 0) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === 'user') {
        turnStart = index
        break
      }
    }
  }

  const toolCallsById = new Map<string, ToolCall>()
  const history: ToolCallBudgetHistoryEntry[] = []
  for (const message of turnStart >= 0 ? messages.slice(turnStart) : messages) {
    for (const call of message.toolCalls ?? []) {
      toolCallsById.set(call.id, call)
    }
    if (message.role !== 'tool') continue
    const sourceCall = message.toolCallId ? toolCallsById.get(message.toolCallId) : undefined
    const tool = message.name
      ?? sourceCall?.name
    if (!tool) continue
    history.push({
      tool,
      ...(sourceCall?.arguments && Object.keys(sourceCall.arguments).length > 0
        ? { input: sourceCall.arguments }
        : {}),
      executionObserved:
        message.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
    })
  }
  return history
}

/**
 * A retry prohibition is a semantic failure boundary even when a name-only
 * sequence cannot represent several actions through one tool. Only a failed,
 * actually executed action with an explicit non-observation purpose closes
 * this boundary. Preparatory reads and pre-execution rejections remain
 * repairable because neither proves that a required action was attempted.
 */
export function currentTurnHasNoRetrySemanticActionFailure(
  messages: readonly Message[],
): boolean {
  const toolCallsById = new Map<string, ToolCall>()
  let turnStart = messages.findLastIndex((message) => (
    message.role === 'user'
    && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
  ))
  if (turnStart < 0) turnStart = messages.findLastIndex((message) => message.role === 'user')

  for (const message of turnStart >= 0 ? messages.slice(turnStart) : messages) {
    for (const call of message.toolCalls ?? []) toolCallsById.set(call.id, call)
    if (
      message.role !== 'tool'
      || message.metadata?.toolResultStatus === 'success'
      || message.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] !== true
    ) continue
    const actionPurpose = message.toolCallId
      ? toolCallsById.get(message.toolCallId)?.arguments.actionPurpose
      : undefined
    if (
      actionPurpose === 'validate'
      || actionPurpose === 'mutate'
      || actionPurpose === 'unblock'
    ) return true
  }
  return false
}

/**
 * Resolve only registered, structural aliases for user-authored cardinality
 * limits. Alias callbacks are advisory runtime metadata, so a broken callback
 * must fail closed to the canonical tool name rather than break the run.
 */
export function resolveToolCardinalityIdentities(
  tools: readonly CardinalityAwareTool[],
): ToolCardinalityIdentity[] {
  return tools.map((tool) => {
    let configured: readonly string[] = []
    try {
      configured = typeof tool.cardinalityAliases === 'function'
        ? tool.cardinalityAliases()
        : tool.cardinalityAliases ?? []
    } catch {
      configured = []
    }
    const aliases = [...new Set(configured
      .map((alias) => alias.trim())
      .filter((alias) => CARDINALITY_ALIAS_PATTERN.test(alias)))]
    return { tool: tool.name, aliases }
  })
}

interface ParsedCanonicalReadUrl {
  origin: string
  pathSegments: Array<string | null>
}

function decodedUrlPathSegment(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

function parseCanonicalReadUrlTemplate(value: string): ParsedCanonicalReadUrl | null {
  const trimmed = value.trim()
  if (!trimmed || trimmed.length > 2_048) return null
  let parameterIndex = 0
  const withSentinels = trimmed.replace(
    CANONICAL_URL_TEMPLATE_PARAMETER_GLOBAL_PATTERN,
    () => `${CANONICAL_URL_TEMPLATE_PARAMETER_PREFIX}${parameterIndex++}`,
  )
  try {
    const parsed = new URL(withSentinels)
    if (
      (parsed.protocol !== 'http:' && parsed.protocol !== 'https:')
      || parsed.username
      || parsed.password
      || parsed.search
      || parsed.hash
    ) return null
    const pathSegments = parsed.pathname
      .split('/')
      .filter(Boolean)
      .map(decodedUrlPathSegment)
      .map((segment) => {
        if (segment.startsWith(CANONICAL_URL_TEMPLATE_PARAMETER_PREFIX)) {
          return /^sepilot-canonical-param-\d+$/u.test(segment) ? null : segment
        }
        return segment
      })
    if (pathSegments.some((segment) => (
      typeof segment === 'string'
      && segment.includes(CANONICAL_URL_TEMPLATE_PARAMETER_PREFIX)
    ))) return null
    const rawSegments = trimmed
      .replace(/^[a-z][a-z0-9+.-]*:\/\/[^/]+/iu, '')
      .split(/[?#]/u, 1)[0]!
      .split('/')
      .filter(Boolean)
    if (rawSegments.some((segment) => (
      segment.includes('{')
      && !CANONICAL_URL_TEMPLATE_PARAMETER_PATTERN.test(segment)
    ))) return null
    return { origin: parsed.origin.toLowerCase(), pathSegments }
  } catch {
    return null
  }
}

function canonicalReadUrlMatchesTemplate(target: string, template: string): boolean {
  const parsedTemplate = parseCanonicalReadUrlTemplate(template)
  if (!parsedTemplate) return false
  try {
    const parsedTarget = new URL(target)
    if (
      (parsedTarget.protocol !== 'http:' && parsedTarget.protocol !== 'https:')
      || parsedTarget.origin.toLowerCase() !== parsedTemplate.origin
    ) return false
    const targetSegments = parsedTarget.pathname
      .split('/')
      .filter(Boolean)
      .map(decodedUrlPathSegment)
    return targetSegments.length === parsedTemplate.pathSegments.length
      && parsedTemplate.pathSegments.every((segment, index) => (
        segment === null
          ? Boolean(targetSegments[index])
          : segment === targetSegments[index]
      ))
  } catch {
    return false
  }
}

/**
 * Resolve credential-free, integration-owned GET route templates. Broken or
 * malformed callbacks fail closed to no routing claim so registry metadata can
 * never make the agent loop unavailable.
 */
export function resolveToolCanonicalReadIdentities(
  tools: readonly CanonicalReadTargetAwareTool[],
): ToolCanonicalReadIdentity[] {
  return tools.map((tool) => {
    let configured: readonly string[] = []
    try {
      configured = typeof tool.canonicalReadUrlTemplates === 'function'
        ? tool.canonicalReadUrlTemplates()
        : tool.canonicalReadUrlTemplates ?? []
    } catch {
      configured = []
    }
    const urlTemplates = [...new Set(configured
      .map((template) => template.trim())
      .filter((template) => parseCanonicalReadUrlTemplate(template) !== null))]
    return { tool: tool.name, urlTemplates }
  })
}

function toolIsExplicitlyNamedInPositiveScope(input: string, tool: string): boolean {
  return identityOccurrences(inputPositiveCapabilityScope(input), tool).length > 0
}

function toolIsExplicitlyExcludedFromPositiveScope(input: string, tool: string): boolean {
  return identityOccurrences(input, tool).length > 0
    && identityOccurrences(inputPositiveCapabilityScope(input), tool).length === 0
}

/**
 * Reject a generic URL transport when exactly one registered integration tool
 * owns the requested GET route. The rejection is pre-execution: it does not
 * expose runtime credentials or silently convert the user's action. Explicit
 * user transport choices and explicit exclusions of the canonical tool win.
 */
export function partitionToolCallsByCanonicalReadTarget(
  input: string,
  calls: ToolCall[],
  identities: readonly ToolCanonicalReadIdentity[],
): {
  executable: ToolCall[]
  nonCanonical: NonCanonicalReadTargetToolCall[]
} {
  if (calls.length === 0 || identities.every((identity) => identity.urlTemplates.length === 0)) {
    return { executable: calls, nonCanonical: [] }
  }
  const executable: ToolCall[] = []
  const nonCanonical: NonCanonicalReadTargetToolCall[] = []
  for (const call of calls) {
    const normalizedCallTool = call.name.toLowerCase()
    const explicitMethod = typeof call.arguments.method === 'string'
      ? call.arguments.method.trim().toUpperCase()
      : ''
    if (explicitMethod && explicitMethod !== 'GET') {
      executable.push(call)
      continue
    }
    if (toolIsExplicitlyNamedInPositiveScope(input, normalizedCallTool)) {
      executable.push(call)
      continue
    }
    const targets = cardinalityTargetStrings(call.arguments)
    const matchedTools = new Set(identities
      .filter((identity) => identity.tool.toLowerCase() !== normalizedCallTool)
      .filter((identity) => !toolIsExplicitlyExcludedFromPositiveScope(input, identity.tool))
      .filter((identity) => identity.urlTemplates.some((template) =>
        targets.some((target) => canonicalReadUrlMatchesTemplate(target, template))
      ))
      .map((identity) => identity.tool.toLowerCase()))
    if (matchedTools.size !== 1) {
      executable.push(call)
      continue
    }
    nonCanonical.push({ call, canonicalTool: [...matchedTools][0]! })
  }
  return { executable, nonCanonical }
}

export function buildCanonicalReadTargetRepairMessage(
  rejected: readonly NonCanonicalReadTargetToolCall[],
): Message {
  const mappings = [...new Set(rejected.map(({ call, canonicalTool }) =>
    `${call.name} -> ${canonicalTool}`
  ))]
  const canonicalTools = [...new Set(rejected.map(({ canonicalTool }) => canonicalTool))]
  return {
    role: 'system',
    metadata: { reminderKind: 'canonical_integration_read_target' },
    content: [
      '[Canonical integration read target guard]',
      `Skipped generic URL transport call(s) for an integration-owned route: ${mappings.join(', ')}.`,
      `Call the registered canonical read tool directly: ${canonicalTools.join(', ')}.`,
      'The rejected call was not executed. The canonical tool owns runtime authentication, API versioning, timeout, and response validation for this route.',
      'Preserve an explicitly requested generic transport only when the user actually named that tool or excluded the canonical tool.',
    ].join(' '),
  }
}

function identityOccurrences(input: string, identity: string): number[] {
  const haystack = input.toLocaleLowerCase()
  const needle = identity.toLocaleLowerCase()
  const positions: number[] = []
  let cursor = 0
  while (needle && cursor < haystack.length) {
    const index = haystack.indexOf(needle, cursor)
    if (index < 0) break
    const before = index > 0 ? input[index - 1] : ''
    const after = input[index + identity.length] ?? ''
    const identifierEdge = /[\p{L}\p{N}_-]/u
    const trailingKoreanParticle = /^(?:으로|에서|에게|한테|부터|까지|처럼|보다|만|은|는|이|가|을|를|와|과|도|로|에)(?=$|[\s,.;:!?()[\]{}…，。！？；])/u
      .test(input.slice(index + identity.length))
    if (
      !identifierEdge.test(before)
      && (!identifierEdge.test(after) || trailingKoreanParticle)
    ) {
      positions.push(index)
    }
    cursor = index + Math.max(1, needle.length)
  }
  return positions
}

function aliasCarriesExactOnceMarker(
  input: string,
  alias: string,
  allIdentities: readonly string[],
): boolean {
  for (const index of identityOccurrences(input, alias)) {
    const start = index + alias.length
    let end = Math.min(input.length, start + 160)
    const punctuation = input.slice(start, end).search(/[.!?。！？;；\n]/u)
    if (punctuation >= 0) end = Math.min(end, start + punctuation)
    for (const other of allIdentities) {
      for (const otherIndex of identityOccurrences(input, other)) {
        if (otherIndex >= start && otherIndex < end) end = otherIndex
      }
    }
    if (EXACTLY_ONCE_MARKER_PATTERN.test(input.slice(start, end))) return true
  }
  return false
}

function canonicalIdentityCarriesExactOnceMarker(
  input: string,
  identity: string,
  allIdentities: readonly string[],
): boolean {
  for (const index of identityOccurrences(input, identity)) {
    const start = index + identity.length
    let end = Math.min(input.length, start + 120)
    const punctuation = input.slice(start, end).search(/[.!?。！？;；\n]/u)
    if (punctuation >= 0) end = Math.min(end, start + punctuation)
    for (const other of allIdentities) {
      for (const otherIndex of identityOccurrences(input, other)) {
        if (otherIndex >= start && otherIndex < end) end = otherIndex
      }
    }
    if (EXACTLY_ONCE_PATTERN.test(input.slice(start, end))) return true
  }
  return false
}

function toolNameCarriesExactOnceMarker(
  input: string,
  toolName: string,
  allIdentities: readonly string[],
): boolean {
  const structuralMatches = [...input.matchAll(TOOL_NAME_PATTERN)]
  const toolMatches = structuralMatches.filter((match) => (
    match[0].toLocaleLowerCase() === toolName.toLocaleLowerCase()
  ))
  for (const match of toolMatches) {
    const index = match.index
    if (index === undefined) continue
    const start = index + toolName.length
    let end = Math.min(input.length, start + 120)
    const punctuation = input.slice(start, end).search(/[.!?。！？;；\n]/u)
    if (punctuation >= 0) end = Math.min(end, start + punctuation)
    for (const otherMatch of structuralMatches) {
      const otherIndex = otherMatch.index
      if (otherIndex !== undefined && otherIndex >= start && otherIndex < end) {
        end = otherIndex
      }
    }
    for (const other of allIdentities) {
      for (const otherIndex of identityOccurrences(input, other)) {
        if (otherIndex >= start && otherIndex < end) end = otherIndex
      }
    }
    if (EXACTLY_ONCE_PATTERN.test(input.slice(start, end))) return true
  }
  return false
}

function hasAffirmativeGlobalNamedToolsExactOnceBoundary(input: string): boolean {
  const matcher = new RegExp(GLOBAL_NAMED_TOOLS_EXACTLY_ONCE_PATTERN.source, 'giu')
  for (const match of input.matchAll(matcher)) {
    const index = match.index
    if (index === undefined) continue
    const prefix = input.slice(Math.max(0, index - 64), index)
    if (
      /(?:\b(?:do\s+not|don't|dont|never)\s+(?:(?:call|use|invoke|run|calling|using|invoking|running)\s+(?:the\s+)?)?|\b(?:avoid|refrain\s+from)\s+(?:calling|using|invoking|running)\s+(?:the\s+)?)$/iu.test(prefix)
      || /(?:하지\s*(?:마|말)|금지|제외)\s*$/iu.test(prefix)
    ) continue
    const suffix = input.slice(index + match[0].length, index + match[0].length + 32)
    if (
      /^\s*(?:(?:(?:should|must|may|shall)\s+)?not\s+(?:be\s+)?(?:called|used|invoked|run)\b|(?:are|is)\s+not\s+to\s+be\s+(?:called|used|invoked|run)\b)/iu.test(suffix)
      || /^\s*(?:호출|사용|실행)(?:은|는|을|를)?\s*(?:하지\s*(?:마|말|않)|금지|제외)/iu.test(suffix)
    ) {
      continue
    }
    return true
  }
  return false
}

/**
 * Resolve a closed one-tool workflow whose cardinality phrase is separated
 * from the registered tool name by concrete arguments or sentence structure.
 * This is deliberately narrower than the ordinary per-name matcher: exactly
 * one registered capability must be named, every other tool must be excluded,
 * and the remaining affirmative text must describe one tool invocation rather
 * than several semantic actions that happen to share a transport tool.
 */
function singleClosedToolWithExactCallBoundary(
  input: string,
  identities: readonly ToolCardinalityIdentity[],
): string | null {
  if (
    !(
      CLOSED_OTHER_TOOLS_ENGLISH_PATTERN.test(input)
      || CLOSED_OTHER_TOOLS_KOREAN_PATTERN.test(input)
    )
  ) {
    return null
  }

  const positive = inputPositiveCapabilityScope(input)
  if (
    !SINGLE_TOOL_EXACT_CALL_PATTERN.test(positive)
    || REPEATED_SEMANTIC_ACTION_EXACT_ONCE_PATTERN.test(positive)
  ) {
    return null
  }

  const registered = new Set<string>()
  for (const identity of identities) {
    if (
      [identity.tool, ...identity.aliases]
        .some((name) => identityOccurrences(positive, name).length > 0)
    ) {
      registered.add(identity.tool.toLowerCase())
    }
  }
  if (registered.size === 1) return [...registered][0] ?? null
  if (registered.size > 1 || identities.length > 0) return null

  const structural = new Set(
    [...positive.matchAll(TOOL_NAME_PATTERN)].map((match) => match[0].toLowerCase()),
  )
  return structural.size === 1 ? [...structural][0] ?? null : null
}

/**
 * Preserve the ordered occurrences of registered tool identities in a global
 * named-call boundary. Unlike a name-only execution sequence, this list may
 * contain duplicates: two different semantic calls can legitimately use the
 * same tool with different arguments.
 */
export function namedToolCallSequence(
  input: string,
  identities: readonly ToolCardinalityIdentity[] = [],
): string[] {
  if (!hasAffirmativeGlobalNamedToolsExactOnceBoundary(input)) return []
  const positive = inputPositiveCapabilityScope(input)
  const occurrences = new Map<string, { tool: string; index: number }>()
  for (const match of positive.matchAll(TOOL_NAME_PATTERN)) {
    const index = match.index
    if (index === undefined) continue
    const tool = match[0].toLowerCase()
    occurrences.set(`${tool}:${index}`, { tool, index })
  }
  for (const identity of identities) {
    const tool = identity.tool.toLowerCase()
    for (const index of [...new Set(
      [identity.tool, ...identity.aliases]
        .flatMap((alias) => identityOccurrences(positive, alias)),
    )]) {
      occurrences.set(`${tool}:${index}`, { tool, index })
    }
  }
  return [...occurrences.values()]
    .sort((left, right) => left.index - right.index || left.tool.localeCompare(right.tool))
    .map(({ tool }) => tool)
}

/**
 * Extract explicit per-turn tool cardinality constraints from affirmative
 * request scope. Tool names are structural identifiers, and the exact-once
 * marker must immediately follow that identifier (apart from connective
 * grammar). Negative clauses are removed first, so “do not call process.read”
 * can never become a positive execution budget.
 */
export function exactOnceToolNames(
  input: string,
  identities: readonly ToolCardinalityIdentity[] = [],
): Set<string> {
  const positive = inputPositiveCapabilityScope(input)
  const limits = new Set<string>()
  const structuralToolNames = [...new Set(
    [...positive.matchAll(TOOL_NAME_PATTERN)].map((match) => match[0]),
  )]
  const allIdentities = [...new Set([
    ...structuralToolNames,
    ...identities.flatMap((identity) => [identity.tool, ...identity.aliases]),
  ])]
  for (const toolName of structuralToolNames) {
    if (toolNameCarriesExactOnceMarker(positive, toolName, allIdentities)) {
      limits.add(toolName.toLowerCase())
    }
  }
  for (const identity of identities) {
    // Canonical dotted tool names already passed through the stricter
    // structural grammar above. Running them through the looser endpoint
    // alias window can turn "terminal.run with command A once, then command B
    // once" into a one-call budget for terminal.run itself. Aliases describe
    // the capability identity; the canonical tool name is not its own alias.
    if (
      canonicalIdentityCarriesExactOnceMarker(
        positive,
        identity.tool,
        allIdentities,
      )
      || identity.aliases.some((alias) =>
        aliasCarriesExactOnceMarker(positive, alias, allIdentities)
      )
    ) {
      limits.add(identity.tool.toLowerCase())
    }
  }
  // A user may put one cardinality boundary in front of a bounded tool list
  // instead of repeating it after every identifier: "use only the following
  // tools, exactly once each" / "다음 등록 도구를 각각 정확히 한 번". This is
  // still structural authority, but only when the wording is explicitly about
  // named tools. Do not treat "run each action/step once" as a name-level
  // budget because several semantic actions can legitimately share one tool.
  if (hasAffirmativeGlobalNamedToolsExactOnceBoundary(input)) {
    const callSequence = namedToolCallSequence(input, identities)
    const mentionCounts = new Map<string, number>()
    for (const name of callSequence) {
      mentionCounts.set(name, (mentionCounts.get(name) ?? 0) + 1)
    }
    for (const name of callSequence) {
      if (mentionCounts.get(name) === 1) limits.add(name)
    }
  }
  const separatedSingleToolBoundary = singleClosedToolWithExactCallBoundary(
    input,
    identities,
  )
  if (separatedSingleToolBoundary) limits.add(separatedSingleToolBoundary)
  return limits
}

/**
 * A closed named-call list can still constrain the enabled tool surface when
 * repeated occurrences make a name-only exact-once workflow unrepresentable.
 */
export function inputClosesNamedToolCallSet(
  input: string,
  identities: readonly ToolCardinalityIdentity[] = [],
): boolean {
  return namedToolCallSequence(input, identities).length > 0
    && (
      CLOSED_OTHER_TOOLS_ENGLISH_PATTERN.test(input)
      || CLOSED_OTHER_TOOLS_KOREAN_PATTERN.test(input)
    )
}

/**
 * A closed exact-once set is stronger than a per-tool cardinality limit: the
 * user also explicitly excluded every unnamed tool. Keep this structural and
 * conservative so ordinary “call X once” requests can still use unrelated
 * evidence or recovery tools when the user did not close the tool surface.
 */
export function inputClosesExactOnceToolSet(
  input: string,
  identities: readonly ToolCardinalityIdentity[] = [],
): boolean {
  const namedSequence = namedToolCallSequence(input, identities)
  if (namedSequence.length > new Set(namedSequence).size) return false
  return exactOnceToolNames(input, identities).size > 0
    && (
      CLOSED_OTHER_TOOLS_ENGLISH_PATTERN.test(input)
      || CLOSED_OTHER_TOOLS_KOREAN_PATTERN.test(input)
    )
}

/**
 * Resolve the registered tool names owned by a structural closed-call
 * boundary. This is the shared projection used both before prompt assembly
 * and when the durable run contract is recovered later in mode routing. A
 * null result means the request did not establish a closed boundary; an
 * unregistered or purely negative name can never enter the returned set.
 */
export function resolveClosedExactToolNames(
  input: string,
  tools: readonly CardinalityAwareTool[],
): string[] | null {
  const identities = resolveToolCardinalityIdentities(tools)
  if (
    !inputClosesExactOnceToolSet(input, identities)
    && !inputClosesNamedToolCallSet(input, identities)
  ) return null

  const registered = new Map(
    tools.map((tool) => [tool.name.toLowerCase(), tool.name]),
  )
  const namedSequence = namedToolCallSequence(input, identities)
  const boundaryNames = namedSequence.length > 0
    ? namedSequence
    : [...exactOnceToolNames(input, identities)]
  const allowed = [...new Set(boundaryNames
    .map((name) => registered.get(name.toLowerCase()))
    .filter((name): name is string => Boolean(name)))]
  return allowed.length > 0 ? allowed : null
}

function cardinalityTargetStrings(value: unknown): string[] {
  if (!value || typeof value !== 'object') return []
  if (Array.isArray(value)) {
    return value.flatMap((entry) => cardinalityTargetStrings(entry))
  }
  const targets: string[] = []
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (CARDINALITY_TARGET_ARGUMENT_PATTERN.test(key)) {
      if (typeof entry === 'string') targets.push(entry)
      continue
    }
    if (entry && typeof entry === 'object') {
      targets.push(...cardinalityTargetStrings(entry))
    }
  }
  return targets
}

/**
 * Map an action to the registered capability identity whose exact cardinality
 * it consumes. Besides the canonical tool name, URL/endpoint-shaped arguments
 * can target a registered runtime alias through another transport tool. This
 * prevents an exhausted provider budget from being bypassed with webfetch or
 * browser navigation while leaving prose/query arguments and unrelated hosts
 * alone.
 */
function cardinalityToolForAction(
  tool: string,
  input: Record<string, unknown> | undefined,
  exactOnce: ReadonlySet<string>,
  identities: readonly ToolCardinalityIdentity[],
): string {
  const normalizedTool = tool.toLowerCase()
  if (exactOnce.has(normalizedTool) || !input) return normalizedTool
  const targets = cardinalityTargetStrings(input)
  if (targets.length === 0) return normalizedTool
  for (const identity of identities) {
    const canonical = identity.tool.toLowerCase()
    if (
      exactOnce.has(canonical)
      && identity.aliases.some((alias) =>
        targets.some((target) => identityOccurrences(target, alias).length > 0)
      )
    ) {
      return canonical
    }
  }
  return normalizedTool
}

export function closedExactOnceToolBudgetComplete(
  input: string,
  history: Array<{
    tool: string
    input?: Record<string, unknown>
    executionObserved?: boolean
  }> | undefined,
  identities: readonly ToolCardinalityIdentity[] = [],
): boolean {
  return inputClosesExactOnceToolSet(input, identities)
    && exactOnceToolBudgetComplete(input, history, identities)
}

export function exactOnceToolBudgetComplete(
  input: string,
  history: Array<{
    tool: string
    input?: Record<string, unknown>
    executionObserved?: boolean
  }> | undefined,
  identities: readonly ToolCardinalityIdentity[] = [],
): boolean {
  const required = exactOnceToolNames(input, identities)
  if (required.size === 0) return false
  const observed = new Set((history ?? [])
    .filter((entry) => entry.executionObserved !== false)
    .map((entry) => cardinalityToolForAction(
      entry.tool,
      entry.input,
      required,
      identities,
    )))
  return [...required].every((tool) => observed.has(tool))
}

export function remainingExactOnceToolNames(
  input: string,
  history: Array<{
    tool: string
    input?: Record<string, unknown>
    executionObserved?: boolean
  }> | undefined,
  identities: readonly ToolCardinalityIdentity[] = [],
): string[] {
  const exactOnce = exactOnceToolNames(input, identities)
  const observed = new Set((history ?? [])
    .filter((entry) => entry.executionObserved !== false)
    .map((entry) => cardinalityToolForAction(
      entry.tool,
      entry.input,
      exactOnce,
      identities,
    )))
  return [...exactOnce].filter((tool) => !observed.has(tool))
}

export function partitionToolCallsByExactOnceBudget(
  input: string,
  history: Array<{
    tool: string
    input?: Record<string, unknown>
    executionObserved?: boolean
  }> | undefined,
  calls: ToolCall[],
  identities: readonly ToolCardinalityIdentity[] = [],
): {
  executable: ToolCall[]
  exhausted: ToolCall[]
  nonCanonical: NonCanonicalExactOnceToolCall[]
} {
  const exactOnce = exactOnceToolNames(input, identities)
  if (exactOnce.size === 0 || calls.length === 0) {
    return { executable: calls, exhausted: [], nonCanonical: [] }
  }
  const observed = new Set(
    (history ?? [])
      .filter((entry) => entry.executionObserved !== false)
      .map((entry) => cardinalityToolForAction(
        entry.tool,
        entry.input,
        exactOnce,
        identities,
      ))
      .filter((tool) => exactOnce.has(tool)),
  )
  const executable: ToolCall[] = []
  const exhausted: ToolCall[] = []
  const nonCanonical: NonCanonicalExactOnceToolCall[] = []
  for (const call of calls) {
    const name = cardinalityToolForAction(
      call.name,
      call.arguments,
      exactOnce,
      identities,
    )
    if (exactOnce.has(name) && observed.has(name)) {
      exhausted.push(call)
      continue
    }
    if (exactOnce.has(name) && name !== call.name.toLowerCase()) {
      // The user bounded a registered capability, not an arbitrary transport
      // that happens to target the same endpoint. Reject the substitute before
      // execution so it cannot spend the one permitted attempt on a homepage,
      // browser navigation, or incompatible HTTP method. An explicitly named
      // transport remains canonical for itself because cardinalityToolForAction
      // returns that tool name before considering endpoint aliases.
      nonCanonical.push({ call, canonicalTool: name })
      continue
    }
    executable.push(call)
    if (exactOnce.has(name)) observed.add(name)
  }
  return { executable, exhausted, nonCanonical }
}

export function buildCanonicalExactOnceToolRepairMessage(
  rejected: readonly NonCanonicalExactOnceToolCall[],
): Message {
  const mappings = [...new Set(rejected.map(({ call, canonicalTool }) =>
    `${call.name} -> ${canonicalTool}`
  ))]
  const canonicalTools = [...new Set(rejected.map(({ canonicalTool }) => canonicalTool))]
  return {
    role: 'system',
    metadata: { reminderKind: 'canonical_exact_tool_capability' },
    content: [
      '[Canonical exact tool capability guard]',
      `Skipped endpoint-targeted substitute call(s): ${mappings.join(', ')}.`,
      `Call the registered canonical tool directly: ${canonicalTools.join(', ')}.`,
      'The rejected substitute was not executed and did not consume the user\'s exactly-once allowance.',
      'A source URL returned by the canonical tool may still be fetched later when the user did not forbid additional tools.',
    ].join(' '),
  }
}

export function partitionToolCallsBySequence(
  sequence: readonly string[] | undefined,
  history: Array<{
    tool: string
    status?: 'success' | 'error'
    executionObserved?: boolean
  }> | undefined,
  calls: ToolCall[],
  countRejectedAttempts = false,
): {
  executable: ToolCall[]
  outOfOrder: ToolCall[]
  nextTool?: string
  workflowFailed: boolean
} {
  if (!sequence || sequence.length === 0 || calls.length === 0) {
    return { executable: calls, outOfOrder: [], workflowFailed: false }
  }
  if (new Set(sequence).size !== sequence.length) {
    // A string sequence has no action identity beyond the tool name. Repeated
    // names therefore cannot distinguish a required semantic action from a
    // preparatory or follow-up call through the same tool. Ignore this
    // unrepresentable ordering contract rather than consuming false steps or
    // closing a no-retry workflow prematurely. Distinct-name sequences retain
    // the strict behavior below.
    return { executable: calls, outOfOrder: [], workflowFailed: false }
  }
  let sequenceIndex = 0
  let workflowFailed = false
  for (const entry of history ?? []) {
    if (!countRejectedAttempts && entry.executionObserved === false) continue
    if (entry.tool !== sequence[sequenceIndex]) {
      if (countRejectedAttempts) workflowFailed = true
      if (workflowFailed) break
      continue
    }
    sequenceIndex += 1
    if (countRejectedAttempts && entry.status === 'error') {
      workflowFailed = true
      break
    }
    if (sequenceIndex >= sequence.length) break
  }
  const executable: ToolCall[] = []
  const outOfOrder: ToolCall[] = []
  for (const call of calls) {
    if (workflowFailed || call.name !== sequence[sequenceIndex]) {
      outOfOrder.push(call)
      continue
    }
    executable.push(call)
    sequenceIndex += 1
  }
  return {
    executable,
    outOfOrder,
    ...(sequence[sequenceIndex] ? { nextTool: sequence[sequenceIndex] } : {}),
    workflowFailed,
  }
}
