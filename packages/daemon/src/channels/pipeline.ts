import type {
  ApprovalScope,
  ChannelAttachment,
  IChannel,
  IncomingMessage,
  ManualJobRunResult,
  SessionMeta,
  SkillMetadata,
} from '@sepilotd/core'
import {
  createTokenSpeedTracker,
  formatTokenSpeedStats,
  type TokenSpeedTracker,
} from '@sepilotd/api-client'
import { createHash, randomUUID } from 'node:crypto'
import { mkdir, realpath, stat } from 'node:fs/promises'
import { performance } from 'node:perf_hooks'
import { createLogger } from '../logger.js'
import type { ObservabilitySeverity } from '../observability/events.js'
import type { ChannelPipelineCapabilities } from '../server/runtime/capabilities.js'
import { ChannelAccessController } from './access.js'
import {
  ChannelAgentExecutor,
  isChannelAgentAbortedError,
  isChannelAgentInactivityError,
  type ChannelAgentInactivityError,
  type ChannelApprovalRequestEvent,
} from './agent-executor.js'
import {
  ChannelIntentClassifier,
  explicitParallelCommand,
  loadIntentClassifierConfigFromEnv,
  maxParallelRunsFromEnv,
  type ChannelFollowupResolution,
  type IntentClassifierActiveRunSnapshot,
} from './intent-classifier.js'
import {
  ChannelMessageNormalizer,
  type NormalizedIncomingChannelMessage,
} from './normalizer.js'
import { channelMessage, detectChannelLang, channelLangSignal, type ChannelLang } from './i18n.js'
import {
  ChannelPostProcessor,
  type ChannelAgentResponse,
  type ChannelHookEmitter,
} from './post-processor.js'
import { ChannelReplayGuard } from './replay-guard.js'
import { ChannelResponseDispatcher } from './response-dispatcher.js'
import {
  ChannelSessionResolver,
} from './session-resolver.js'
import { channelSafeAgentResponse } from './safe-response.js'
import type { ChannelPipelineStage } from '../server/runtime/channel-pipeline-monitor.js'
import type { PendingQuestion } from '../tools/question.js'
import {
  extractExplicitMemoryDirective,
  rememberUserMemory,
} from '../memory/user-memory.js'
import { resolveToolPath } from '../tools/path-utils.js'
import { deriveChannelScopeTags, isMemoryVisibleInScope } from '../memory/scope.js'
import type { ChannelSessionBinding } from '../server/runtime/channel-sessions.js'
import { formatWhen, parseWhen, SchedulerParseError } from '../scheduler/time-parser.js'
import {
  hasPendingScheduledRun,
  type JobRun,
  type JobStore,
  type ScheduledJob,
} from '../scheduler/job-store.js'
import { DAEMON_VERSION } from '../version.js'
import {
  applyDefaultModel,
  currentModelTarget,
  formatAvailableModels,
  pullOllamaModel,
  resolveModelTarget as resolveConfiguredModelTarget,
  selfTestProviderModel,
} from '../server/runtime/model-control.js'
import { readOutput } from '../media/image-gen/files.js'
import { MarketplaceSource } from '../skills/sources/marketplace.js'
import type { ChannelOriginRecord } from './channel-origin-store.js'

const log = createLogger('router')
const CHANNEL_APPROVAL_TTL_MS = 11 * 60_000
const APPROVAL_COMMAND_PATTERN = /^\/(approve|deny)(?:@\S+)?(?:\s+(.+))?$/i
const APPROVAL_SCOPES = new Set(['once', 'session', 'always', 'run', 'command', 'task', 'session-all', 'session_all'])
const FEEDBACK_COMMAND_PATTERN = /^\/(feedback|rate)(?:@\S+)?(?:\s+([\s\S]*))?$/i
const SESSION_COMMAND_PATTERN = /^\/([a-z][\w-]*)(?:@\S+)?(?:\s+([\s\S]*))?$/i
const SKILLS_COMMAND_PATTERN = /^\/skills(?:@\S+)?(?:\s+([\s\S]*))?$/i
const SKILL_SHORTCUT_PATTERN = /^\/([a-z][\w.-]*)(?:@\S+)?(?:\s+([\s\S]*))?$/i
const STOP_COMMAND_PATTERN = /^\/(stop|cancel|abort)(?:@\S+)?(?:\s*)$/i
const SKILL_SHORTCUT_RESERVED = new Set([
  'abort', 'about', 'approve', 'async', 'cancel', 'capabilities', 'capability',
  'cd', 'clear', 'commands', 'current', 'cwd', 'deny', 'diag', 'feedback',
  'fork', 'health', 'help', 'history', 'llm', 'load', 'memories', 'memory',
  'model', 'models', 'new', 'parallel', 'pwd', 'rate', 'reset', 'resume',
  'schedule', 'schedules', 'self', 'session', 'sessions', 'skills', 'status',
  'stop', 'tps', 'usage', 'use', 'workdir', 'workspace',
])
const WORKSPACE_COMMAND_PATTERN = /^\/(cd|cwd|pwd|workdir|workspace)(?:@\S+)?(?:\s+([\s\S]*))?$/i
const SCHEDULE_COMMAND_PATTERN = /^\/schedules?(?:@\S+)?(?:\s+([\s\S]*))?$/i
const SESSION_LIST_LIMIT = 8
const ACTIVE_CHANNEL_RUN_STALE_MS = 30 * 60_000
const CHANNEL_ACTIVITY_INTERVAL_MS = 4_000
const CHANNEL_RUN_ATTENTION_MS = 2 * 60_000
const STATUS_PREVIEW_CHARS = 120
const APPROVAL_INTENT_LLM_TIMEOUT_MS = 2_500
const IMAGE_GEN_FILE_ROUTE_PATTERN = /\/image-gen\/files\/([^\s)\]}>,]+)/g

function sharedGroupContextEnabled(
  config: ChannelPipelineCapabilities['config'],
): boolean {
  const pipelineConfig = (config as {
    channelPipeline?: { sharedGroupContext?: unknown }
  }).channelPipeline
  if (typeof pipelineConfig?.sharedGroupContext === 'boolean') {
    return pipelineConfig.sharedGroupContext
  }

  const legacyChannelsConfig = config.channels as unknown
  return Boolean(
    legacyChannelsConfig
      && typeof legacyChannelsConfig === 'object'
      && 'sharedGroupContext' in legacyChannelsConfig
      && legacyChannelsConfig.sharedGroupContext === true,
  )
}

function maxGlobalChannelRuns(
  config: ChannelPipelineCapabilities['config'],
): number {
  const value = (config as {
    channelPipeline?: { maxGlobalRuns?: unknown }
  }).channelPipeline?.maxGlobalRuns
  return typeof value === 'number' && Number.isInteger(value) && value > 0
    ? value
    : 8
}

type ChannelApprovalCommand = {
  action: 'approve' | 'deny' | 'feedback'
  requestId: string
  scope: ApprovalScope
  note?: string
}

type ChannelApprovalIntent = Omit<ChannelApprovalCommand, 'requestId'>

type ChannelFeedbackCommand =
  | {
      kind: 'feedback'
      rating: 'positive' | 'neutral' | 'negative'
      reason: 'channel-command' | 'channel-reaction'
      note?: string
      silent: boolean
    }
  | { kind: 'help' }

type ChannelApprovalOrigin = {
  channelType: string
  channelId: string
  senderId: string
  sequence: number
  toolName: string
  requestedAt: number
  // Conversation language captured from the original request, so the approval
  // confirmation mirrors the user even when the reply is a terse '/approve'
  // command that carries no language signal of its own.
  lang: ChannelLang
  cleanup: ReturnType<typeof setTimeout>
}

type ChannelApprovalOriginMatch = ChannelApprovalOrigin & {
  requestId: string
}

type ChannelQuestionOrigin = {
  sessionId: string
  channelType: string
  channelId: string
  senderId: string
  requestedAt: number
}

type ActiveChannelRun = {
  runId: string
  kind: 'main' | 'fork'
  parentRunId?: string
  parentSessionId?: string
  sessionId?: string
  controller: AbortController
  userPrompt: string
  triggerMessageId: string
  messageId: string
  senderId?: string
  startedAt: number
  phase: 'processing' | 'waiting_approval' | 'waiting_question'
  waitingSince?: number
  waitingForId?: string
  waitingForLabel?: string
  progress?: ChannelProgressReporter
}

type ChannelProgressReporter = {
  stop(): void
}

type ChannelEventMeta = {
  sessionId?: string
  taskId?: string
  provider?: string
  model?: string
}

type ChannelDeliveryPurpose =
  | 'agent_response'
  | 'approval_request'
  | 'approval_response'
  | 'command_response'
  | 'correction_ack'
  | 'error_response'
  | 'no_provider'
  | 'progress'
  | 'question_request'

function hashIdentifier(value?: string): string | undefined {
  if (!value) return undefined
  return createHash('sha256').update(value).digest('hex').slice(0, 32)
}

function errorName(error: unknown): string {
  return error instanceof Error ? error.name : typeof error
}

type ChannelFileMemory = {
  getPromptContext(): Promise<{
    longTermMemory?: string
    todayNote?: string
    yesterdayNote?: string
  }>
  readMemory?(): Promise<string | undefined>
  readDailyNote?(date?: Date): Promise<string | undefined>
}

type ChannelSessionCommand =
  | { action: 'help' }
  | { action: 'current' }
  | { action: 'new' }
  | { action: 'list'; query?: string }
  | { action: 'resume'; sessionRef?: string }
  | { action: 'memory' }
  | { action: 'self'; selfAction?: 'overview' | 'skills' | 'capabilities' | 'schedules' }
  | { action: 'model'; modelAction?: 'current' | 'list' | 'switch' | 'pull'; args?: string }
  | { action: 'tps'; tpsAction: 'current' | 'reset' | 'help' }

type ChannelWorkspaceCommand =
  | { action: 'show' }
  | { action: 'set'; path: string }

type ChannelSkillsCommand =
  | { action: 'list'; query?: string }
  | { action: 'search'; query?: string }
  | { action: 'enable'; id?: string }
  | { action: 'disable'; id?: string }
  | { action: 'install'; source?: string; expectedDigest?: string }
  | { action: 'help' }

type NaturalApprovalResolution =
  | { kind: 'command'; command: ChannelApprovalCommand }
  | { kind: 'ambiguous'; requestIds: string[] }

function normalizeApprovalScopeToken(raw: string | undefined): ApprovalScope | null {
  const normalized = raw?.trim().toLowerCase()
  if (!normalized || !APPROVAL_SCOPES.has(normalized)) return null
  if (normalized === 'command' || normalized === 'task') return 'run'
  if (normalized === 'session_all') return 'session-all'
  return normalized as ApprovalScope
}

function parseApprovalCommand(text: string): ChannelApprovalCommand | null {
  const match = text.trim().match(APPROVAL_COMMAND_PATTERN)
  if (!match) return null

  const action = match[1]?.toLowerCase() === 'approve' ? 'approve' : 'deny'
  const rest = match[2]?.trim() ?? ''
  if (!rest) return null

  const [requestId, ...tail] = rest.split(/\s+/)
  if (!requestId) return null

  if (action === 'deny') {
    return {
      action,
      requestId,
      scope: 'once',
      note: tail.join(' ').trim() || undefined,
    }
  }

  const scope = normalizeApprovalScopeToken(tail[0]) ?? 'once'
  const noteParts = scope === 'once' ? tail : tail.slice(1)
  return {
    action,
    requestId,
    scope,
    note: noteParts.join(' ').trim() || undefined,
  }
}

// Deterministic approval replies: exact short phrases advertised in the
// channel prompt must work without depending on the LLM classifier. Longer or
// ambiguous prose still falls through to classifyApprovalReplyWithLLM.
const APPROVE_SYMBOL_RE = /^(?:👍|👍🏻|👍🏼|👍🏽|👍🏾|👍🏿|👌|👌🏻|👌🏼|👌🏽|👌🏾|👌🏿|✅)+$/u
const DENY_SYMBOL_RE = /^(?:👎|👎🏻|👎🏼|👎🏽|👎🏾|👎🏿|❌|✖️|✖)+$/u
const APPROVE_ONCE_TEXTS = new Set([
  '네',
  '예',
  '응',
  'ㅇㅇ',
  '진행',
  '진행해',
  '해봐',
  '승인',
  '승인해',
  '허용',
  '허용해',
  '좋아',
  '좋아요',
  'ok',
  'okay',
  'yes',
  'y',
  'approve',
  'proceed',
])
const DENY_ONCE_TEXTS = new Set([
  '아니',
  '아니요',
  '아뇨',
  'ㄴㄴ',
  '안돼',
  '안되',
  '하지마',
  '거절',
  '거절해',
  'no',
  'n',
  'deny',
  'reject',
])
const APPROVE_ALWAYS_TEXTS = new Set([
  '항상승인',
  '항상승인해',
  '항상허용',
  '항상허용해',
  '앞으로항상승인',
  '앞으로항상승인해',
  '앞으로항상허용',
  '앞으로항상허용해',
  'always',
  'alwaysapprove',
])
const APPROVE_RUN_TEXTS = new Set([
  '이번작업은계속승인해',
  '이번작업계속승인해',
  '이번작업은모두승인해',
  '이번작업모두승인해',
  '이번요청은계속승인해',
  '이번요청계속승인해',
  '이번명령은계속승인해',
  '이번명령계속승인해',
])
const APPROVE_SESSION_ALL_TEXTS = new Set([
  '이번세션은모두승인해',
  '이번세션모두승인해',
  '이세션은모두승인해',
  '이세션모두승인해',
])

function parseApprovalSymbol(text: string): ChannelApprovalIntent | null {
  const compact = text.trim().replace(/\s+/g, '').toLowerCase()
  if (!compact) return null
  if (DENY_SYMBOL_RE.test(compact)) return { action: 'deny', scope: 'once' }
  if (APPROVE_SYMBOL_RE.test(compact)) return { action: 'approve', scope: 'once' }
  if (APPROVE_SESSION_ALL_TEXTS.has(compact)) return { action: 'approve', scope: 'session-all' }
  if (APPROVE_RUN_TEXTS.has(compact)) return { action: 'approve', scope: 'run' }
  if (APPROVE_ALWAYS_TEXTS.has(compact)) return { action: 'approve', scope: 'always' }
  if (DENY_ONCE_TEXTS.has(compact)) return { action: 'deny', scope: 'once' }
  if (APPROVE_ONCE_TEXTS.has(compact)) return { action: 'approve', scope: 'once' }
  return null
}

function approvalDecisionForAction(action: ChannelApprovalCommand['action']): 'approved' | 'denied' | 'feedback' {
  if (action === 'approve') return 'approved'
  if (action === 'feedback') return 'feedback'
  return 'denied'
}

function approvalApprovedForAction(action: ChannelApprovalCommand['action']): boolean {
  return action === 'approve'
}

function parseApprovalIntentClassifierResponse(content: unknown): ChannelApprovalIntent | null {
  const text = typeof content === 'string'
    ? content
    : Array.isArray(content)
      ? content
        .map((part) => part && typeof part === 'object' && 'text' in part ? String(part.text) : '')
        .join('\n')
      : ''
  const match = text.match(/\{[\s\S]*\}/)
  if (!match) return null
  try {
    const parsed = JSON.parse(match[0]!) as Record<string, unknown>
    const action = typeof parsed.action === 'string' ? parsed.action.toLowerCase() : ''
    if (action !== 'approve' && action !== 'deny' && action !== 'feedback') {
      return null
    }
    const rawScope = typeof parsed.scope === 'string' ? parsed.scope.toLowerCase() : ''
    const scope = normalizeApprovalScopeToken(rawScope) ?? 'once'
    const note = typeof parsed.note === 'string' && parsed.note.trim()
      ? parsed.note.trim()
      : undefined
    return { action, scope, note }
  } catch {
    return null
  }
}

function parseFeedbackRatingToken(
  token: string,
): 'positive' | 'neutral' | 'negative' | null {
  const normalized = token.trim().toLowerCase()
  if (!normalized) return null
  if (
    /^(good|great|positive|helpful|up|yes|y|like|liked|love|thanks|thank-you|ok|👍|👍🏻|👍🏼|👍🏽|👍🏾|👍🏿|✅|좋음|좋아|도움|도움됨|만족|고마워|감사)$/u
      .test(normalized)
  ) {
    return 'positive'
  }
  if (
    /^(bad|poor|negative|unhelpful|down|no|n|dislike|wrong|👎|👎🏻|👎🏼|👎🏽|👎🏾|👎🏿|❌|별로|나쁨|불만|틀림|틀렸어|도움안됨|도움\s*안\s*됨)$/u
      .test(normalized)
  ) {
    return 'negative'
  }
  if (/^(neutral|meh|mixed|normal|보통|중립|그저그래)$/u.test(normalized)) {
    return 'neutral'
  }
  return null
}

function parseFeedbackReaction(text: string): 'positive' | 'negative' | null {
  const compact = text.trim().replace(/\s+/g, '')
  if (!compact) return null
  if (/^(👍|👍🏻|👍🏼|👍🏽|👍🏾|👍🏿|✅|👏|👏🏻|👏🏼|👏🏽|👏🏾|👏🏿|❤️|❤)+$/u.test(compact)) {
    return 'positive'
  }
  if (/^(👎|👎🏻|👎🏼|👎🏽|👎🏾|👎🏿|❌|😞|😡)+$/u.test(compact)) {
    return 'negative'
  }
  return null
}

function parseFeedbackCommand(text: string): ChannelFeedbackCommand | null {
  const commandMatch = text.trim().match(FEEDBACK_COMMAND_PATTERN)
  if (commandMatch) {
    const rest = commandMatch[2]?.trim() ?? ''
    if (!rest) return { kind: 'help' }
    const [ratingToken, ...noteParts] = rest.split(/\s+/)
    const rating = parseFeedbackRatingToken(ratingToken ?? '')
    if (!rating) return { kind: 'help' }
    const note = noteParts.join(' ').trim()
    return {
      kind: 'feedback',
      rating,
      reason: 'channel-command',
      note: note || undefined,
      silent: false,
    }
  }

  const reaction = parseFeedbackReaction(text)
  if (!reaction) return null
  return {
    kind: 'feedback',
    rating: reaction,
    reason: 'channel-reaction',
    silent: true,
  }
}

function formatApprovalToolInput(input: Record<string, unknown>): string {
  const serialized = JSON.stringify(input, null, 2)
  return serialized.length > 1000
    ? `${serialized.slice(0, 997)}...`
    : serialized
}

function formatApprovalResolutionText(
  command: ChannelApprovalCommand,
  toolName?: string,
  lang: ChannelLang = 'ko',
): string {
  const tool = toolName ? ` ${toolName}` : ''
  if (command.action === 'feedback') {
    return channelMessage('approvalFeedback', lang, { tool })
  }
  if (command.action === 'deny') {
    return channelMessage('approvalDenied', lang, { tool })
  }

  switch (command.scope) {
    case 'run':
      return channelMessage('approvalRun', lang)
    case 'session-all':
      return channelMessage('approvalSessionAll', lang)
    case 'always':
      return channelMessage('approvalAlways', lang)
    case 'session':
      return channelMessage('approvalSession', lang)
    case 'once':
    default:
      return channelMessage('approvalOnce', lang, { tool })
  }
}

function parseSessionCommand(text: string): ChannelSessionCommand | null {
  const match = text.trim().match(SESSION_COMMAND_PATTERN)
  if (!match) return null

  const name = match[1]?.toLowerCase()
  const rest = match[2]?.trim() || undefined
  switch (name) {
    case 'help':
    case 'commands':
      return { action: 'help' }
    case 'session':
    case 'current':
    case 'status':
      return { action: 'current' }
    case 'new':
    case 'reset':
    case 'clear':
      return { action: 'new' }
    case 'sessions':
    case 'history':
      return { action: 'list', query: rest }
    case 'resume':
    case 'load':
    case 'use':
      return { action: 'resume', sessionRef: rest }
    case 'memory':
    case 'memories':
      return { action: 'memory' }
    case 'self':
    case 'about':
      return { action: 'self', selfAction: 'overview' }
    case 'skills':
      return { action: 'self', selfAction: 'skills' }
    case 'capabilities':
    case 'capability':
      return { action: 'self', selfAction: 'capabilities' }
    case 'schedules':
      return { action: 'self', selfAction: 'schedules' }
    case 'model':
    case 'models':
    case 'llm':
      return { action: 'model', ...parseModelCommandArgs(rest, name === 'models') }
    case 'tps':
    case 'usage':
      return { action: 'tps', tpsAction: parseTpsCommandArgs(rest) }
    default:
      return null
  }
}

function parseTpsCommandArgs(args: string | undefined): Extract<ChannelSessionCommand, { action: 'tps' }>['tpsAction'] {
  const [verb] = args?.trim().split(/\s+/).filter(Boolean) ?? []
  if (!verb) return 'current'
  const normalized = verb.toLowerCase()
  if (normalized === 'current' || normalized === 'show' || normalized === 'stats') return 'current'
  if (normalized === 'reset' || normalized === 'clear') return 'reset'
  if (normalized === 'help' || normalized === 'usage') return 'help'
  return 'help'
}

function parseSkillsCommand(text: string): ChannelSkillsCommand | null {
  const match = text.trim().match(SKILLS_COMMAND_PATTERN)
  if (!match) return null

  const rest = match[1]?.trim() ?? ''
  if (!rest) return { action: 'list' }

  const [verbRaw, ...tail] = rest.split(/\s+/)
  const verb = verbRaw?.toLowerCase()
  const args = tail.join(' ').trim()
  switch (verb) {
    case 'help':
    case 'usage':
      return { action: 'help' }
    case 'installed':
    case 'list':
    case 'ls':
    case 'all':
      return { action: 'list', query: args || undefined }
    case 'search':
    case 'find':
    case 'store':
    case 'marketplace':
      return { action: 'search', query: args || undefined }
    case 'enable':
    case 'on':
      return { action: 'enable', id: tail[0] }
    case 'disable':
    case 'off':
      return { action: 'disable', id: tail[0] }
    case 'install':
    case 'add':
      return { action: 'install', ...parseSkillInstallArgs(tail) }
    default:
      return { action: 'search', query: rest }
  }
}

function parseSkillInstallArgs(args: string[]): Pick<Extract<ChannelSkillsCommand, { action: 'install' }>, 'source' | 'expectedDigest'> {
  if (args.length === 0) return {}
  if (args[0]?.toLowerCase() === 'confirm') {
    return {
      source: args[1],
      expectedDigest: args[2],
    }
  }

  const confirmIndex = args.findIndex((arg) => arg === '--confirm' || arg === '--digest')
  if (confirmIndex >= 0) {
    return {
      source: args.slice(0, confirmIndex).join(' ').trim() || undefined,
      expectedDigest: args[confirmIndex + 1],
    }
  }

  return { source: args.join(' ').trim() || undefined }
}

function parseSkillShortcutCommand(text: string): { skillId: string; args: string } | null {
  const match = text.trim().match(SKILL_SHORTCUT_PATTERN)
  if (!match) return null
  const skillId = match[1]?.trim()
  if (!skillId) return null
  if (SKILL_SHORTCUT_RESERVED.has(skillId.toLowerCase())) return null
  return {
    skillId,
    args: match[2]?.trim() ?? '',
  }
}

function skillsHelpText(): string {
  return [
    '사용법:',
    '/skills 또는 /skills installed [검색어] - 설치된 스킬 조회',
    '/skills search <query> - 설치/마켓플레이스 스킬 검색',
    '/skills enable <id> - 스킬 활성화',
    '/skills disable <id> - 스킬 비활성화',
    '/skills install <source> - 설치 미리보기',
    '/skills install <source> --confirm <digest> - 미리보기한 exact digest 설치',
    '/<skill-id> <작업 내용> - 특정 스킬로 작업 실행',
  ].join('\n')
}

function formatSkillLine(skill: SkillMetadata): string {
  const state = skill.enabled === false ? '비활성' : '활성'
  const tools = skill.tools?.length ? ` · tools: ${skill.tools.slice(0, 4).join(', ')}` : ''
  return `- ${skill.id}@${skill.version} · ${state}: ${truncateForTelegram(skill.description, 90)}${tools}`
}

function filterSkills(skills: SkillMetadata[], query: string | undefined): SkillMetadata[] {
  const needle = query?.trim().toLowerCase()
  if (!needle) return skills
  return skills.filter((skill) =>
    skill.id.toLowerCase().includes(needle)
    || skill.name.toLowerCase().includes(needle)
    || skill.description.toLowerCase().includes(needle)
    || skill.tags?.some((tag) => tag.toLowerCase().includes(needle)),
  )
}

function formatSkillCommandError(error: unknown): string {
  if (error instanceof Error) {
    if (error.name === 'SkillValidationError' && 'result' in error) {
      try {
        const result = error.result as { errors?: string[]; warnings?: string[] }
        const details = [...(result.errors ?? []), ...(result.warnings ?? [])].filter(Boolean)
        return details.length ? details.join('; ') : error.message
      } catch {
        return error.message
      }
    }
    if (error.name === 'SkillDigestMismatchError') {
      return error.message
    }
    if (error.name === 'SkillAlreadyExistsError') {
      return `${error.message}. 이미 설치된 스킬은 /skills enable <id>로 활성화하거나 CLI/admin에서 update를 사용하세요.`
    }
    return error.message
  }
  return String(error)
}

function parseModelCommandArgs(
  args: string | undefined,
  listByDefault = false,
): Pick<Extract<ChannelSessionCommand, { action: 'model' }>, 'modelAction' | 'args'> {
  const trimmed = args?.trim()
  if (!trimmed) return { modelAction: listByDefault ? 'list' : 'current' }

  const [verb, ...rest] = trimmed.split(/\s+/)
  const normalized = verb?.toLowerCase()
  const tail = rest.join(' ').trim()
  if (normalized === 'current' || normalized === 'now' || normalized === 'status') {
    return { modelAction: 'current' }
  }
  if (normalized === 'list' || normalized === 'available' || normalized === 'ls') {
    return { modelAction: 'list' }
  }
  if (
    normalized === 'pull'
    || normalized === 'get'
    || normalized === 'fetch'
    || normalized === 'install'
    || normalized === '가져와'
    || normalized === '다운로드'
    || normalized === '설치'
  ) {
    return { modelAction: 'pull', args: tail || undefined }
  }
  if (
    normalized === 'switch'
    || normalized === 'use'
    || normalized === 'set'
    || normalized === 'change'
  ) {
    return { modelAction: 'switch', args: tail || undefined }
  }
  return { modelAction: 'switch', args: trimmed }
}

function cleanWorkspacePathCandidate(path: string): string {
  let candidate = path.trim()
  if (
    (candidate.startsWith('"') && candidate.endsWith('"'))
    || (candidate.startsWith("'") && candidate.endsWith("'"))
    || (candidate.startsWith('`') && candidate.endsWith('`'))
  ) {
    candidate = candidate.slice(1, -1).trim()
  }
  return candidate
    .replace(/[),.;!?]+$/u, '')
    .replace(/(?:으로|로|에서|에|를|을)$/u, '')
    .trim()
}

function parseWorkspaceCommand(text: string): ChannelWorkspaceCommand | null {
  const trimmed = text.trim()
  if (!trimmed) return null

  const slash = trimmed.match(WORKSPACE_COMMAND_PATTERN)
  if (slash) {
    const name = slash[1]?.toLowerCase()
    const rest = slash[2]?.trim()
    if (!rest || name === 'pwd') {
      return { action: 'show' }
    }
    return { action: 'set', path: cleanWorkspacePathCandidate(rest) }
  }

  return null
}

function formatSessionLine(session: SessionMeta): string {
  const id = session.id.slice(0, 8)
  const title = session.title.replace(/\s+/g, ' ').trim() || '(untitled)'
  const clippedTitle = title.length > 52 ? `${title.slice(0, 49)}...` : title
  return `- ${id} [${session.status}, ${session.messageCount} msgs] ${clippedTitle}`
}

function formatElapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  if (seconds < 60) return `${seconds}초`
  const minutes = Math.floor(seconds / 60)
  const rest = seconds % 60
  return rest > 0 ? `${minutes}분 ${rest}초` : `${minutes}분`
}

function compactStatusPreview(value: string | undefined): string {
  const normalized = (value ?? '').replace(/\s+/g, ' ').trim()
  if (!normalized) return ''
  return normalized.length > STATUS_PREVIEW_CHARS
    ? `${normalized.slice(0, STATUS_PREVIEW_CHARS - 3)}...`
    : normalized
}

function formatActiveRunLine(activeRun: ActiveChannelRun | undefined): string {
  if (!activeRun) return 'Current run: idle'

  const elapsed = formatElapsed(Date.now() - activeRun.startedAt)
  const waitingElapsed = activeRun.waitingSince
    ? formatElapsed(Date.now() - activeRun.waitingSince)
    : elapsed
  switch (activeRun.phase) {
    case 'waiting_approval':
      return [
        `Current run: waiting for approval for ${waitingElapsed}`,
        activeRun.waitingForLabel ? `Tool: ${activeRun.waitingForLabel}` : undefined,
        activeRun.waitingForId ? `Request: ${activeRun.waitingForId}` : undefined,
      ].filter(Boolean).join('\n')
    case 'waiting_question':
      return [
        `Current run: waiting for your answer for ${waitingElapsed}`,
        activeRun.waitingForLabel ? `Question: ${activeRun.waitingForLabel}` : undefined,
        activeRun.waitingForId ? `Question ID: ${activeRun.waitingForId}` : undefined,
      ].filter(Boolean).join('\n')
    case 'processing':
    default:
      return `Current run: processing for ${elapsed}`
  }
}

function formatBlockingRunBusyText(activeRun: ActiveChannelRun): string {
  const elapsed = formatElapsed(Date.now() - activeRun.startedAt)
  const waitingElapsed = activeRun.waitingSince
    ? formatElapsed(Date.now() - activeRun.waitingSince)
    : elapsed

  if (activeRun.phase === 'waiting_approval') {
    return [
      '이전 요청은 도구 승인 응답을 기다리는 중입니다.',
      `대기 시간: ${waitingElapsed}`,
      activeRun.waitingForLabel ? `도구: ${activeRun.waitingForLabel}` : undefined,
      activeRun.waitingForId
        ? `승인하려면 "네" 또는 /approve ${activeRun.waitingForId} 로 답장하세요.`
        : '승인 메시지에 답장하면 이어서 처리합니다.',
    ].filter(Boolean).join('\n')
  }

  if (activeRun.phase === 'waiting_question') {
    return [
      '이전 요청은 사용자 답변을 기다리는 중입니다.',
      `대기 시간: ${waitingElapsed}`,
      activeRun.waitingForLabel ? `질문: ${activeRun.waitingForLabel}` : undefined,
      '이 채팅에 답장하면 이어서 처리합니다.',
    ].filter(Boolean).join('\n')
  }

  const attention = formatRunAttentionLine(activeRun)
  return [
    `이전 요청을 아직 처리 중입니다. 경과 시간: ${elapsed}.`,
    attention,
  ].filter(Boolean).join('\n')
}

function formatActiveRunsStatus(activeRuns: ReadonlyArray<ActiveChannelRun>): string {
  if (activeRuns.length === 0) return 'Current run: idle'

  const sorted = [...activeRuns].sort((a, b) => a.startedAt - b.startedAt)
  let forkIndex = 0
  const lines = sorted.map((run) => {
    const label = run.kind === 'main' ? '[메인]' : `[병렬 #${++forkIndex}]`
    const elapsed = formatElapsed(Date.now() - run.startedAt)
    const waitingElapsed = run.waitingSince
      ? formatElapsed(Date.now() - run.waitingSince)
      : elapsed
    const promptPreview = compactStatusPreview(run.userPrompt)
    switch (run.phase) {
      case 'waiting_approval':
        return [
          `${label} 도구 승인 대기 ${waitingElapsed}`,
          run.waitingForLabel ? `  도구: ${run.waitingForLabel}` : undefined,
          promptPreview ? `  요청: ${promptPreview}` : undefined,
        ].filter(Boolean).join('\n')
      case 'waiting_question':
        return [
          `${label} 사용자 답변 대기 ${waitingElapsed}`,
          run.waitingForLabel ? `  질문: ${run.waitingForLabel}` : undefined,
          promptPreview ? `  요청: ${promptPreview}` : undefined,
        ].filter(Boolean).join('\n')
      case 'processing':
      default:
        return [
          `${label} 처리 중 ${elapsed}`,
          promptPreview ? `  요청: ${promptPreview}` : undefined,
        ].filter(Boolean).join('\n')
    }
  })
  return lines.join('\n')
}

function formatRunAttentionLine(activeRun: ActiveChannelRun | undefined): string | undefined {
  if (!activeRun || activeRun.phase !== 'processing') return undefined
  const elapsedMs = Date.now() - activeRun.startedAt
  if (elapsedMs < CHANNEL_RUN_ATTENTION_MS) return undefined
  return 'Health: still processing. If this stays here for several minutes with no tool approval or question, the provider/tool may be slow; use /status again, /new for a fresh session, or retry after the final response.'
}

function imageGenAttachmentsFromText(text: string): ChannelAttachment[] {
  const attachments: ChannelAttachment[] = []
  const seen = new Set<string>()
  IMAGE_GEN_FILE_ROUTE_PATTERN.lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = IMAGE_GEN_FILE_ROUTE_PATTERN.exec(text)) !== null) {
    const encodedId = match[1]
    if (!encodedId) continue
    let fileId: string
    try {
      fileId = decodeURIComponent(encodedId)
    } catch {
      fileId = encodedId
    }
    if (seen.has(fileId)) continue
    seen.add(fileId)
    const output = readOutput(fileId)
    if (!output) continue
    attachments.push({
      type: 'image',
      name: `${fileId.replace(/[^a-z0-9._-]+/gi, '-')}.png`,
      mimeType: output.mime,
      data: output.bytes.toString('base64'),
      dataType: 'base64',
    })
  }
  return attachments
}

// No recorded origin means the question was enqueued by a code path
// that did not go through notifyQuestionRequest (e.g. a web UI run
// or an in-memory state wipe). In that case we cannot attribute the
// question to this channel sender, so we refuse the match rather
// than letting any sender on the same bound session answer it.
// Channel-originated questions always record an origin in
// notifyQuestionRequest before delivery.
export function matchesQuestionOrigin(
  origin: ChannelQuestionOrigin | undefined,
  normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
): boolean {
  if (!origin) return false
  return origin.channelType === normalized.message.channelType
    && origin.channelId === normalized.message.channelId
    && origin.senderId === normalized.message.sender.id
}

export class ChannelMessagePipeline {
  private readonly runtime: ChannelPipelineCapabilities
  private readonly emitChannelHook: ChannelHookEmitter
  private readonly accessController: ChannelAccessController
  private readonly agentExecutor: ChannelAgentExecutor
  private readonly normalizer: ChannelMessageNormalizer
  private readonly postProcessor: ChannelPostProcessor
  private readonly replayGuard: ChannelReplayGuard
  private readonly responseDispatcher: ChannelResponseDispatcher
  private readonly sessionResolver: ChannelSessionResolver
  private readonly approvalOrigins = new Map<string, ChannelApprovalOrigin>()
  private readonly questionOrigins = new Map<string, ChannelQuestionOrigin>()
  private readonly activeChannelRuns = new Map<string, Map<string, ActiveChannelRun>>()
  private readonly tokenSpeedTrackers = new Map<string, TokenSpeedTracker>()
  private readonly maxParallelRuns: number
  private readonly intentClassifier: ChannelIntentClassifier
  private originLoadPromise?: Promise<void>
  private approvalOriginSequence = 0

  constructor(runtime: ChannelPipelineCapabilities, emitChannelHook: ChannelHookEmitter) {
    this.runtime = runtime
    this.emitChannelHook = emitChannelHook
    this.accessController = new ChannelAccessController(runtime)
    this.agentExecutor = new ChannelAgentExecutor(runtime)
    this.normalizer = new ChannelMessageNormalizer({
      sharedGroupContext: () => sharedGroupContextEnabled(this.runtime.config),
    })
    this.postProcessor = new ChannelPostProcessor(runtime, emitChannelHook)
    this.replayGuard = new ChannelReplayGuard(runtime)
    this.responseDispatcher = new ChannelResponseDispatcher()
    this.sessionResolver = new ChannelSessionResolver(runtime)
    this.maxParallelRuns = maxParallelRunsFromEnv()
    this.intentClassifier = new ChannelIntentClassifier(
      () => {
        const provider = this.runtime.providerRegistry.getDefault()
        return provider ? { models: provider.models, chat: provider.chat.bind(provider) } : undefined
      },
      loadIntentClassifierConfigFromEnv(),
    )
  }

  private tokenSpeedKeyFor(normalized: NormalizedIncomingChannelMessage): string {
    return normalized.sessionKey
      ?? `${normalized.message.channelType}:${normalized.message.channelId}:${normalized.message.sender?.id ?? 'unknown'}`
  }

  private tokenSpeedTrackerFor(normalized: NormalizedIncomingChannelMessage): TokenSpeedTracker {
    const key = this.tokenSpeedKeyFor(normalized)
    let tracker = this.tokenSpeedTrackers.get(key)
    if (!tracker) {
      tracker = createTokenSpeedTracker()
      this.tokenSpeedTrackers.set(key, tracker)
    }
    return tracker
  }

  private listActiveRuns(chatKey: string | undefined): ActiveChannelRun[] {
    if (!chatKey) return []
    const pool = this.activeChannelRuns.get(chatKey)
    if (!pool) return []
    const now = Date.now()
    const result: ActiveChannelRun[] = []
    for (const [runId, run] of pool.entries()) {
      if (now - run.startedAt >= ACTIVE_CHANNEL_RUN_STALE_MS) {
        run.progress?.stop()
        pool.delete(runId)
        continue
      }
      result.push(run)
    }
    if (pool.size === 0) {
      this.activeChannelRuns.delete(chatKey)
    }
    return result.sort((a, b) => a.startedAt - b.startedAt)
  }

  private addActiveRun(chatKey: string, run: ActiveChannelRun): void {
    let pool = this.activeChannelRuns.get(chatKey)
    if (!pool) {
      pool = new Map()
      this.activeChannelRuns.set(chatKey, pool)
    }
    pool.set(run.runId, run)
  }

  private removeActiveRun(chatKey: string | undefined, runId: string): void {
    if (!chatKey) return
    const pool = this.activeChannelRuns.get(chatKey)
    if (!pool) return
    const run = pool.get(runId)
    run?.progress?.stop()
    pool.delete(runId)
    if (pool.size === 0) {
      this.activeChannelRuns.delete(chatKey)
    }
  }

  private countForkRuns(chatKey: string | undefined): number {
    return this.listActiveRuns(chatKey).filter((run) => run.kind === 'fork').length
  }

  private countGlobalActiveRuns(): number {
    let total = 0
    for (const chatKey of [...this.activeChannelRuns.keys()]) {
      total += this.listActiveRuns(chatKey).length
    }
    return total
  }

  private canStartGlobalRun(): boolean {
    return this.countGlobalActiveRuns() < maxGlobalChannelRuns(this.runtime.config)
  }

  private activeRunsForSender(
    normalized: NormalizedIncomingChannelMessage,
    activeRuns: ActiveChannelRun[],
  ): ActiveChannelRun[] {
    const senderId = normalized.message.sender.id
    return activeRuns.filter((run) => !run.senderId || run.senderId === senderId)
  }

  private recordChannelEvent(
    normalizedOrMessage: NormalizedIncomingChannelMessage | IncomingMessage,
    eventType: string,
    attributes: Record<string, unknown> = {},
    severity: ObservabilitySeverity = 'info',
    meta: ChannelEventMeta = {},
  ): void {
    const observability = this.runtime.observability
    if (!observability) return

    let normalized: NormalizedIncomingChannelMessage | undefined
    let message: IncomingMessage
    if ('message' in normalizedOrMessage) {
      normalized = normalizedOrMessage
      message = normalizedOrMessage.message
    } else {
      message = normalizedOrMessage
    }

    try {
      observability.recordEvents([{
        source: 'channel',
        surface: message.channelType,
        eventType,
        severity,
        privacy: 'operational',
        sessionId: meta.sessionId,
        messageId: message.messageId,
        taskId: meta.taskId,
        channelIdHash: hashIdentifier(message.channelId),
        userIdHash: hashIdentifier(message.sender.id),
        provider: meta.provider,
        model: meta.model,
        attributes: {
          channelType: message.channelType,
          receiveOnly: normalized?.receiveOnly,
          hasReplyTarget: Boolean(normalized?.replyTarget),
          hasReplyToken: Boolean(normalized?.replyToken),
          ...attributes,
        },
      }])
    } catch {
      // Observability is best-effort and must never affect channel traffic.
    }
  }

  private async sendObserved(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    text: string,
    purpose: ChannelDeliveryPurpose,
    meta: ChannelEventMeta = {},
    attributes: Record<string, unknown> = {},
    options: { replyToMessageId?: string; attachments?: ChannelAttachment[] } = {},
  ): Promise<boolean> {
    try {
      const delivered = await this.responseDispatcher.send(channel, normalized, text, options)
      this.recordChannelEvent(
        normalized,
        delivered ? 'channel.delivery_succeeded' : 'channel.delivery_skipped',
        {
          purpose,
          textLength: text.length,
          ...attributes,
        },
        delivered ? 'info' : 'debug',
        meta,
      )
      return delivered
    } catch (error) {
      this.recordChannelEvent(
        normalized,
        'channel.delivery_failed',
        {
          purpose,
          textLength: text.length,
          errorName: errorName(error),
          ...attributes,
        },
        'error',
        meta,
      )
      throw error
    }
  }

  async handle(channel: IChannel, msg: IncomingMessage): Promise<void> {
    const run = this.runtime.channelPipelineMonitor?.start(msg.channelType)
    let dedupeKey: string | undefined
    try {
      const normalized = await this.measureStage('normalize', () =>
        this.normalizer.normalize(msg),
        msg.channelType,
      )
      this.recordChannelEvent(normalized, 'channel.message_received', {
        textLength: normalized.message.text.length,
      }, 'debug')
      const dedupe = await this.measureStage('replay_claim', () =>
        this.replayGuard.claim(normalized),
        msg.channelType,
      )
      dedupeKey = dedupe.key
      if (dedupe.duplicate) {
        this.runtime.channelPipelineMonitor?.record(msg.channelType, 'duplicate')
        log.info('Skipping duplicate inbound channel message', {
          channelType: msg.channelType,
          channelId: msg.channelId,
          messageId: msg.messageId,
          dedupeState: dedupe.state ?? 'processed',
        })
        await this.emitChannelHook(msg, {
          status: 'duplicate',
          dedupeState: dedupe.state ?? 'processed',
        })
        this.recordChannelEvent(normalized, 'channel.duplicate', {
          dedupeState: dedupe.state ?? 'processed',
        }, 'debug')
        return
      }

      const access = await this.measureStage('access_check', () =>
        this.accessController.evaluate(msg),
        msg.channelType,
      )
      if (!access.allowed) {
        this.runtime.channelPipelineMonitor?.record(msg.channelType, 'blocked')
        log.warn(`Blocked: ${msg.sender.id} on ${msg.channelType}`)
        await this.emitChannelHook(msg, {
          status: 'blocked',
          reason: access.reason,
        })
        this.recordChannelEvent(normalized, 'channel.blocked', {
          reason: access.reason,
        }, 'warning')
        await this.replayGuard.markProcessed(dedupeKey)
        return
      }

      await this.ensureChannelOriginsLoaded()

      const approvalCommand = parseApprovalCommand(normalized.message.text)
      if (approvalCommand) {
        await this.handleApprovalCommand(
          channel,
          normalized,
          approvalCommand,
          dedupeKey,
        )
        return
      }

      const naturalApproval = await this.resolveNaturalApproval(normalized)
      if (naturalApproval?.kind === 'command') {
        await this.handleApprovalCommand(
          channel,
          normalized,
          naturalApproval.command,
          dedupeKey,
        )
        return
      }
      if (naturalApproval?.kind === 'ambiguous') {
        await this.handleAmbiguousApprovalResponse(
          channel,
          normalized,
          naturalApproval.requestIds,
          dedupeKey,
        )
        return
      }

      const feedbackCommand = parseFeedbackCommand(normalized.message.text)
      if (feedbackCommand) {
        await this.handleFeedbackCommand(
          channel,
          normalized,
          feedbackCommand,
          dedupeKey,
        )
        return
      }

      const skillsCommand = parseSkillsCommand(normalized.message.text)
      if (skillsCommand) {
        await this.handleSkillsCommand(
          channel,
          normalized,
          skillsCommand,
          dedupeKey,
        )
        return
      }

      if (STOP_COMMAND_PATTERN.test(normalized.message.text.trim())) {
        await this.handleExplicitStopCommand(channel, normalized, dedupeKey)
        return
      }

      const sessionCommand = parseSessionCommand(normalized.message.text)
      if (sessionCommand) {
        await this.handleSessionCommand(
          channel,
          normalized,
          sessionCommand,
          dedupeKey,
        )
        return
      }

      const workspaceCommand = parseWorkspaceCommand(normalized.message.text)
      if (workspaceCommand) {
        await this.handleWorkspaceCommand(
          channel,
          normalized,
          workspaceCommand,
          dedupeKey,
        )
        return
      }

      const text = normalized.message.text.trim()
      const scheduleCommandText = resolveScheduleCommandText(text)
      if (scheduleCommandText) {
        await this.handleScheduleCommandMessage(channel, normalized, dedupeKey, scheduleCommandText)
        return
      }

      // Memory slash commands (/memory, /memory_search, /memory_remember,
      // /memory_forget, /memory_list, /memory_hot). Same scope guard as
      // the rest of the channel pipeline: only memories owned by the
      // calling user/channel are listed/searched/deletable.
      const lower = text.toLowerCase()
      if ((lower === '/memory' || lower.startsWith('/memory ') || lower.startsWith('/memory_'))
          && this.runtime.semanticIndex) {
        const scopeTags = deriveChannelScopeTags({
          channelType: normalized.message.channelType,
          channelId: normalized.message.channelId,
          senderId: normalized.message.sender?.id,
        })
        const reply = await handleMemoryCommand({
          semanticIndex: this.runtime.semanticIndex,
          text: normalized.message.text,
          scopeTags,
          actor: `channel:${normalized.message.channelType}:${normalized.message.sender?.id ?? 'unknown'}`,
        })
        await this.responseDispatcher.send(channel, normalized, reply)
        return
      }

      if (lower === '/health' || lower === '/diag') {
        await this.handleHealthCommand(channel, normalized, dedupeKey)
        return
      }

      const skillShortcut = parseSkillShortcutCommand(normalized.message.text)
      if (skillShortcut && await this.handleSkillShortcutCommand(
        channel,
        normalized,
        access.autonomy ?? this.runtime.autonomy,
        dedupeKey,
        skillShortcut,
      )) {
        return
      }

      if (await this.handlePendingQuestionAnswer(channel, normalized, dedupeKey)) {
        return
      }

      if (await this.handleExplicitMemoryDirective(channel, normalized, dedupeKey)) {
        return
      }

      await this.processChannelMessage(
        channel,
        normalized,
        access.autonomy ?? this.runtime.autonomy,
        dedupeKey,
      )
    } catch (error) {
      this.runtime.channelPipelineMonitor?.record(msg.channelType, 'error')
      await this.replayGuard.release(dedupeKey)
      log.error('Error processing message', { error: String(error) })
      const normalized = this.normalizer.normalize(msg)
      this.recordChannelEvent(normalized, 'channel.task_failed', {
        errorName: errorName(error),
      }, 'error', {
        taskId: msg.messageId,
      })
      try {
        await this.sendObserved(
          channel,
          normalized,
          '처리 중 오류가 발생했습니다. daemon은 요청을 받았지만 작업을 완료하지 못했습니다. 잠시 뒤 다시 시도하거나 /status로 현재 세션 상태를 확인해주세요.',
          'error_response',
        )
      } catch (sendError) {
        log.warn('Failed to send channel error response', { error: String(sendError) })
      }
      await this.emitChannelHook(msg, {
        status: 'error',
        error: String(error),
      })
    } finally {
      this.runtime.channelPipelineMonitor?.finish(run)
    }
  }

  private async dispatchAgentResponse(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    sessionId: string,
    responseText: string,
    provider: NonNullable<ReturnType<ChannelPipelineCapabilities['providerRegistry']['getDefault']>>,
    replyToMessageId?: string,
  ): Promise<ChannelAgentResponse> {
    log.info(`Agent response: ${responseText.length} chars`, {
      preview: responseText.slice(0, 100),
    })

    let responseDelivered = false
    if (responseText) {
      const attachments = imageGenAttachmentsFromText(responseText)
      responseDelivered = await this.measureStage(
        'response_dispatch',
        () => this.sendObserved(
          channel,
          normalized,
          responseText,
          'agent_response',
          {
            sessionId,
            taskId: normalized.message.messageId,
            provider: provider.id,
            model: this.runtime.config.agent?.defaultModel ?? provider.models[0]?.id ?? 'default',
          },
          {},
          {
            ...(replyToMessageId ? { replyToMessageId } : {}),
            ...(attachments.length > 0 ? { attachments } : {}),
          },
        ),
        normalized.message.channelType,
      )
      await this.runtime.sessions.appendEvent(sessionId, {
        type: 'assistant_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content: responseText,
      })
    }

    return { responseText, responseDelivered }
  }

  private async postProcessChannelMessage(
    msg: IncomingMessage,
    sessionId: string,
    provider: NonNullable<ReturnType<ChannelPipelineCapabilities['providerRegistry']['getDefault']>>,
    response: ChannelAgentResponse,
  ): Promise<void> {
    await this.measureStage('post_process', () =>
      this.postProcessor.complete(msg, sessionId, provider, response),
      msg.channelType,
    )
  }

  private async handleNoProviderMessage(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey?: string,
  ): Promise<void> {
    const delivered = await this.sendObserved(
      channel,
      normalized,
      'No LLM provider configured.',
      'no_provider',
      { taskId: normalized.message.messageId },
    )
    this.recordChannelEvent(normalized, 'channel.no_provider', {}, 'warning', {
      taskId: normalized.message.messageId,
    })
    await this.emitChannelHook(normalized.message, {
      status: 'no_provider',
      responseDelivered: delivered,
    })
    await this.replayGuard.markProcessed(dedupeKey)
    this.runtime.channelPipelineMonitor?.record(
      normalized.message.channelType,
      'no_provider',
    )
  }

  private async processChannelMessage(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey?: string,
  ): Promise<void> {
    const { message } = normalized
    const provider = this.runtime.providerRegistry.getDefault()
    if (!provider) {
      await this.handleNoProviderMessage(channel, normalized, dedupeKey)
      return
    }

    const chatKey = normalized.sessionKey
    const activeRuns = this.listActiveRuns(chatKey)

    if (chatKey && activeRuns.length > 0) {
      await this.routeFollowupForActiveRuns(channel, normalized, autonomy, dedupeKey, chatKey, activeRuns)
      return
    }

    await this.startMainRun(channel, normalized, autonomy, dedupeKey, message.text)
  }

  private async routeFollowupForActiveRuns(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey: string | undefined,
    chatKey: string,
    activeRuns: ActiveChannelRun[],
  ): Promise<void> {
    const { message } = normalized

    const blockingRun = activeRuns.find(
      (run) => run.phase === 'waiting_approval' || run.phase === 'waiting_question',
    )
    if (blockingRun) {
      const scheduleCommandText = resolveScheduleCommandText(message.text)
      if (scheduleCommandText) {
        await this.handleScheduleCommandMessage(
          channel,
          normalized,
          dedupeKey,
          scheduleCommandText,
          'schedule_command_during_wait',
        )
        return
      }

      const text = formatBlockingRunBusyText(blockingRun)
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        text,
        'channel_run_busy',
        { activeMessageId: blockingRun.messageId, activeRunPhase: blockingRun.phase },
      )
      return
    }

    // Explicit `/parallel ...` (and /new /async /fork) command — parsed, not
    // heuristically inferred. Natural-language follow-ups always go to the LLM
    // classifier below; there is no regex prefilter for prose.
    const parallelCommand = explicitParallelCommand(message.text)
    if (parallelCommand.matched) {
      this.recordChannelEvent(normalized, 'channel.intent_classified', {
        category: 'parallel_new',
        source: 'command',
        activeRunCount: activeRuns.length,
      }, 'info', { taskId: message.messageId })
      await this.handleParallelNewFollowup(
        channel, normalized, autonomy, dedupeKey, chatKey, activeRuns, parallelCommand.remainder,
      )
      return
    }

    const snapshots: IntentClassifierActiveRunSnapshot[] = activeRuns.map((run) => ({
      runId: run.runId,
      kind: run.kind,
      userPrompt: run.userPrompt,
      phase: run.phase,
      elapsedMs: Math.max(0, Date.now() - run.startedAt),
    }))

    let resolution: ChannelFollowupResolution
    try {
      resolution = await this.intentClassifier.classify({
        chatKey,
        newMessageText: message.text,
        activeRuns: snapshots,
      })
    } catch (error) {
      // Fail-safe: a thrown classifier must not spawn a mutating parallel run.
      // Fall back to status (acknowledge, keep the active run) instead of forking.
      log.warn('Intent classifier threw unexpectedly', { error: errorName(error) })
      resolution = { intent: { category: 'status' }, source: 'fallback' }
    }

    this.recordChannelEvent(normalized, 'channel.intent_classified', {
      category: resolution.intent.category,
      source: resolution.source,
      latencyMs: resolution.latencyMs,
      activeRunCount: activeRuns.length,
    }, 'info', { taskId: message.messageId })

    switch (resolution.intent.category) {
      case 'status':
        await this.handleStatusFollowup(channel, normalized, dedupeKey, activeRuns)
        return
      case 'cancel':
        await this.handleCancelFollowup(channel, normalized, dedupeKey, chatKey, activeRuns)
        return
      case 'duplicate':
        await this.handleDuplicateFollowup(channel, normalized, dedupeKey, activeRuns)
        return
      case 'correction':
        await this.handleCorrectionFollowup(
          channel,
          normalized,
          autonomy,
          dedupeKey,
          chatKey,
          activeRuns,
          resolution.intent.note,
        )
        return
      case 'parallel_new':
      default:
        await this.handleParallelNewFollowup(channel, normalized, autonomy, dedupeKey, chatKey, activeRuns)
        return
    }
  }

  private async handleScheduleCommandMessage(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey: string | undefined,
    scheduleCommandText: string,
    status = 'schedule_command',
  ): Promise<void> {
    if (!this.runtime.jobStore) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'Scheduler is not initialized.',
        'schedule_unavailable',
      )
      return
    }
    const schedulerRuntime = this.runtime as {
      schedulerDefaultTimezone?: string
      triggerSchedulerJob?: (id: string) => Promise<ManualJobRunResult>
    }
    const reply = await handleScheduleCommand({
      store: this.runtime.jobStore,
      text: scheduleCommandText,
      chatKey: normalized.sessionKey ?? '',
      channelType: normalized.message.channelType,
      replyToMessageId: normalized.message.messageId,
      defaultTimezone: schedulerRuntime.schedulerDefaultTimezone ?? this.runtime.config.scheduler?.timezone,
      triggerSchedulerJob: schedulerRuntime.triggerSchedulerJob,
    })
    await this.completeCommand(channel, normalized, dedupeKey, reply, status, { scheduleCommandText })
  }

  private async handleDuplicateFollowup(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey: string | undefined,
    activeRuns: ActiveChannelRun[],
  ): Promise<void> {
    // The classifier judged this message to be the same request an active
    // run is already handling (a verbatim re-send or a rephrase). Don't fork
    // a duplicate run — just acknowledge so the user knows it landed.
    const target = activeRuns[0]
    this.recordChannelEvent(normalized, 'channel.duplicate_resend_ignored', {
      activeMessageId: target?.messageId,
      activeRunPhase: target?.phase,
      activeRunCount: activeRuns.length,
    }, 'info', { taskId: normalized.message.messageId })
    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      '같은 요청이 이미 처리 중이에요 — 결과가 준비되면 보내드릴게요. ⏳',
      'channel_duplicate_resend',
      { activeMessageId: target?.messageId, activeRunPhase: target?.phase },
    )
  }

  private async handleStatusFollowup(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey: string | undefined,
    activeRuns: ActiveChannelRun[],
  ): Promise<void> {
    const text = formatActiveRunsStatus(activeRuns)
    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      text,
      'channel_run_status',
      { activeRunCount: activeRuns.length },
    )
  }

  private async handleCancelFollowup(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey: string | undefined,
    _chatKey: string,
    activeRuns: ActiveChannelRun[],
  ): Promise<void> {
    const targetRuns = this.activeRunsForSender(normalized, activeRuns)
    if (targetRuns.length === 0) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        '현재 이 채팅에서 요청자가 시작한 진행 중인 작업이 없습니다.',
        'channel_run_idle',
        { activeRunCount: activeRuns.length },
      )
      return
    }

    for (const run of targetRuns) {
      try {
        run.controller.abort(new Error('user_cancel'))
      } catch {
        // controller may already be aborted
      }
      this.recordChannelEvent(normalized, 'channel.run_aborted', {
        reason: 'user_cancel',
        runId: run.runId,
        kind: run.kind,
      }, 'info', { taskId: run.messageId })
    }
    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      `진행 중이던 ${targetRuns.length}개 작업을 모두 중단했습니다.`,
      'channel_run_cancelled',
      { activeRunCount: targetRuns.length },
    )
  }

  private async handleCorrectionFollowup(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey: string | undefined,
    _chatKey: string,
    activeRuns: ActiveChannelRun[],
    note: string | undefined,
  ): Promise<void> {
    const targetRuns = this.activeRunsForSender(normalized, activeRuns)
    const mainRun = targetRuns.find((run) => run.kind === 'main') ?? targetRuns[0]
    if (!mainRun) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        '현재 이 채팅에서 요청자가 정정할 수 있는 진행 중인 작업이 없습니다.',
        'channel_run_idle',
        { activeRunCount: activeRuns.length },
      )
      return
    }
    if (mainRun) {
      try {
        mainRun.controller.abort(new Error('correction'))
      } catch {
        // already aborted
      }
      this.recordChannelEvent(normalized, 'channel.run_aborted', {
        reason: 'correction',
        runId: mainRun.runId,
        kind: mainRun.kind,
      }, 'info', { taskId: mainRun.messageId })
    }
    const correctionText = note?.trim() || normalized.message.text.trim()
    const restartedPrompt = mainRun
      ? `${mainRun.userPrompt}\n\n[사용자 정정] ${correctionText}`
      : normalized.message.text
    await this.sendObserved(
      channel,
      normalized,
      '이전 진행을 중단하고 정정 사항을 반영해서 다시 시작합니다.',
      'correction_ack',
      { taskId: normalized.message.messageId },
    )
    await this.startMainRun(channel, normalized, autonomy, dedupeKey, restartedPrompt)
  }

  private async handleParallelNewFollowup(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey: string | undefined,
    chatKey: string,
    activeRuns: ActiveChannelRun[],
    promptOverride?: string,
  ): Promise<void> {
    if (this.countForkRuns(chatKey) >= this.maxParallelRuns) {
      const text = [
        `현재 ${activeRuns.length}개 작업이 진행 중입니다 (병렬 한도 ${this.maxParallelRuns}).`,
        '진행 중인 작업이 끝난 후에 다시 시도하거나, "취소"라고 보내서 모두 중단할 수 있습니다.',
        formatActiveRunsStatus(activeRuns),
      ].join('\n')
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        text,
        'channel_parallel_limit',
        { activeRunCount: activeRuns.length },
      )
      return
    }
    // Status / cancel / correction follow-ups all send an immediate ack.
    // parallel_new used to start a fork silently, which made the user
    // think their second message was dropped — the actual agent reply
    // can take minutes when the main run is busy waiting on approvals
    // or holding the LLM provider's request slot.
    try {
      await this.sendObserved(
        channel,
        normalized,
        '병렬로 별도 처리합니다 — 결과가 준비되면 보내드릴게요.',
        'agent_response',
        { taskId: normalized.message.messageId },
        { activeRunCount: activeRuns.length },
        { replyToMessageId: normalized.message.messageId },
      )
    } catch (error) {
      log.warn('Failed to send parallel_new ack', { error: errorName(error) })
    }
    this.recordChannelEvent(
      normalized,
      'channel.parallel_new_started',
      { activeRunCount: activeRuns.length },
      'info',
      { taskId: normalized.message.messageId },
    )
    const mainRun = activeRuns.find((run) => run.kind === 'main') ?? activeRuns[0]
    const parentSessionId = mainRun?.sessionId
    await this.startForkRun(
      channel,
      normalized,
      autonomy,
      dedupeKey,
      chatKey,
      mainRun?.runId,
      parentSessionId,
      promptOverride,
    )
  }

  private async startMainRun(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey: string | undefined,
    promptText: string,
  ): Promise<void> {
    const provider = this.runtime.providerRegistry.getDefault()
    if (!provider) {
      await this.handleNoProviderMessage(channel, normalized, dedupeKey)
      return
    }
    await this.executeChannelRun(channel, normalized, autonomy, dedupeKey, provider, {
      kind: 'main',
      promptText,
      resolveSession: () => this.sessionResolver.resolve(normalized, provider),
    })
  }

  private async startForkRun(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey: string | undefined,
    _chatKey: string,
    parentRunId: string | undefined,
    parentSessionId: string | undefined,
    promptOverride?: string,
  ): Promise<void> {
    const provider = this.runtime.providerRegistry.getDefault()
    if (!provider) {
      await this.handleNoProviderMessage(channel, normalized, dedupeKey)
      return
    }
    const started = await this.executeChannelRun(channel, normalized, autonomy, dedupeKey, provider, {
      kind: 'fork',
      promptText: promptOverride && promptOverride.length > 0 ? promptOverride : normalized.message.text,
      parentRunId,
      parentSessionId,
      resolveSession: () => this.sessionResolver.resolveFork(normalized, provider, parentSessionId),
      replyToMessageId: normalized.message.messageId,
    })
    if (!started) return
    this.recordChannelEvent(normalized, 'channel.fork_created', {
      parentSessionId,
      parentRunId,
    }, 'info', { taskId: normalized.message.messageId })
  }

  private async executeChannelRun(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey: string | undefined,
    provider: NonNullable<ReturnType<ChannelPipelineCapabilities['providerRegistry']['getDefault']>>,
    options: {
      kind: 'main' | 'fork'
      promptText: string
      parentRunId?: string
      parentSessionId?: string
      resolveSession: () => Promise<{
        sessionId: string
        session: SessionMeta
        previousMessages: Awaited<ReturnType<ChannelSessionResolver['resolve']>>['previousMessages']
      }>
      replyToMessageId?: string
    },
  ): Promise<boolean> {
    const { message } = normalized
    const chatKey = normalized.sessionKey
    const globalActiveRunCount = this.countGlobalActiveRuns()
    const maxGlobalRuns = maxGlobalChannelRuns(this.runtime.config)
    if (!this.canStartGlobalRun()) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        [
          `현재 전체 채널 작업이 ${globalActiveRunCount}개 진행 중입니다 (전역 한도 ${maxGlobalRuns}).`,
          '진행 중인 작업이 끝난 후 다시 시도하거나, 현재 채팅의 작업은 "취소"로 중단할 수 있습니다.',
        ].join('\n'),
        'channel_global_busy',
        {
          activeRunCount: globalActiveRunCount,
          maxGlobalRuns,
        },
      )
      return false
    }
    const runId = randomUUID()
    const controller = new AbortController()

    const activeRunState: ActiveChannelRun = {
      runId,
      kind: options.kind,
      parentRunId: options.parentRunId,
      parentSessionId: options.parentSessionId,
      controller,
      userPrompt: options.promptText,
      triggerMessageId: message.messageId,
      messageId: message.messageId,
      senderId: message.sender.id,
      startedAt: Date.now(),
      phase: 'processing',
    }

    if (chatKey) {
      this.addActiveRun(chatKey, activeRunState)
    }

    const taskStartedAt = performance.now()
    const eventMeta: ChannelEventMeta = {
      taskId: message.messageId,
      provider: provider.id,
      model: this.runtime.config.agent?.defaultModel ?? provider.models[0]?.id ?? 'default',
    }
    this.recordChannelEvent(normalized, 'channel.task_started', {
      autonomy,
      runId,
      runKind: options.kind,
    }, 'info', eventMeta)

    const progress = this.startChannelProgress(channel, normalized)
    activeRunState.progress = progress

    try {
      const { sessionId, session, previousMessages } = await this.measureStage(
        'session_resolve',
        () => options.resolveSession(),
        message.channelType,
      )
      activeRunState.sessionId = sessionId
      const sessionEventMeta: ChannelEventMeta = {
        ...eventMeta,
        sessionId,
      }
      const responseText = await this.measureStage(
        'agent_execute',
        async () => {
          try {
            const runMessage = {
              text: options.promptText,
              messageId: message.messageId,
              sender: message.sender
                ? { id: message.sender.id, name: message.sender.name }
                : undefined,
            }
            const scopeTags = deriveChannelScopeTags({
              channelType: message.channelType,
              channelId: message.channelId,
              senderId: message.sender?.id,
              sessionId,
            })
            return await this.agentExecutor.run(
              runMessage,
              provider,
              autonomy,
              sessionId,
              previousMessages,
              session.cwd,
              (event) => {
                activeRunState.phase = 'waiting_approval'
                activeRunState.waitingSince = Date.now()
                activeRunState.waitingForId = event.requestId
                activeRunState.waitingForLabel = event.toolCall.name
                progress.stop()
                return this.notifyApprovalRequest(channel, normalized, event)
              },
              (question) => {
                activeRunState.phase = 'waiting_question'
                activeRunState.waitingSince = Date.now()
                activeRunState.waitingForId = question.id
                activeRunState.waitingForLabel = compactStatusPreview(question.prompt)
                progress.stop()
                return this.notifyQuestionRequest(channel, normalized, question)
              },
              controller.signal,
              scopeTags,
              {
                channel: message.channelType,
                chatKey: normalized.sessionKey ?? message.channelId,
                triggerMessageId: message.messageId,
              },
              (usage) => {
                this.tokenSpeedTrackerFor(normalized).recordRun({
                  startedAt: taskStartedAt,
                  finishedAt: performance.now(),
                  usage,
                })
              },
            )
          } catch (error) {
            if (isChannelAgentInactivityError(error)) {
              const ctx = error.context ?? {}
              this.recordChannelEvent(normalized, 'channel.agent_inactivity', {
                inactivityMs: error.inactivityMs,
                sawModelOutput: ctx.sawModelOutput,
                lastEventType: ctx.lastEventType,
                provider: ctx.provider,
                model: ctx.model,
              }, 'warning', { ...sessionEventMeta, taskId: message.messageId })
              return this.formatAgentInactivityResponse(error)
            }
            if (isChannelAgentAbortedError(error)) {
              return ''
            }
            throw error
          }
        },
        message.channelType,
      )
      if (!responseText) {
        await this.replayGuard.markProcessed(dedupeKey)
        return true
      }
      const safeResponse = channelSafeAgentResponse(responseText)
      if (safeResponse.internalFallback) {
        this.recordChannelEvent(normalized, 'channel.agent_final_missing', {
          originalResponseTextLength: responseText.length,
          deliveredResponseTextLength: safeResponse.text.length,
          runId,
          runKind: options.kind,
        }, 'warning', sessionEventMeta)
      }
      const response = await this.dispatchAgentResponse(
        channel,
        normalized,
        sessionId,
        safeResponse.text,
        provider,
        options.replyToMessageId,
      )
      await this.postProcessChannelMessage(message, sessionId, provider, response)
      await this.replayGuard.markProcessed(dedupeKey)
      this.recordChannelEvent(normalized, 'channel.task_completed', {
        durationMs: Math.round(performance.now() - taskStartedAt),
        responseDelivered: response.responseDelivered,
        responseTextLength: response.responseText.length,
        runId,
        runKind: options.kind,
        agentFinalMissing: safeResponse.internalFallback,
      }, 'info', sessionEventMeta)
      this.runtime.channelPipelineMonitor?.record(
        message.channelType,
        safeResponse.internalFallback ? 'error' : 'processed',
      )
    } finally {
      progress.stop()
      this.removeActiveRun(chatKey, runId)
    }
    return true
  }

  private formatAgentInactivityResponse(error: ChannelAgentInactivityError): string {
    const ctx = error.context ?? {}
    const elapsed = formatElapsed(error.inactivityMs)
    const modelLabel = ctx.provider && ctx.model
      ? `${ctx.provider} / ${ctx.model}`
      : ctx.model ?? ctx.provider

    if (ctx.sawModelOutput === false) {
      // The run produced no tokens, no tool call — the LLM/provider request
      // itself never came back. Point the user at provider health, not at
      // their request being "too long" or a stuck tool.
      return [
        `모델이 ${elapsed} 동안 응답을 시작하지 않아 중단했습니다.`,
        modelLabel ? `대상: ${modelLabel}` : undefined,
        '',
        '이번 실행에서는 토큰도 도구 호출도 전혀 발생하지 않았습니다 — 모델/provider 쪽이 막혀 있을 가능성이 큽니다.',
        '확인해 보세요:',
        '- provider 엔드포인트(예: Ollama 서버)가 살아 있는지, 해당 모델이 로드/사용 가능한지',
        '- provider 쪽 요청 큐 적체나 rate limit',
        '- 필요하면 더 빠르고 안정적인 모델로 전환',
        '',
        '다시 시도하면 새 실행으로 처리합니다. /status로 세션 상태도 확인할 수 있습니다.',
      ].filter((line): line is string => line !== undefined).join('\n')
    }

    return [
      '작업이 응답 없이 오래 걸려서 중단했습니다.',
      `무응답 시간: ${elapsed}`,
      ctx.lastEventType ? `마지막 단계: ${ctx.lastEventType}` : undefined,
      modelLabel ? `모델: ${modelLabel}` : undefined,
      '',
      '가능한 원인:',
      '- 도구(파일/터미널/웹/MCP) 실행이 응답하지 않음',
      '- 모델 스트리밍이 중간에 끊김',
      '- 외부 서비스 응답 없음',
      '',
      '다시 시도하면 새 실행으로 처리합니다. 같은 현상이 반복되면 요청을 더 작게 나누거나 /status로 세션 상태를 확인해주세요.',
    ].filter((line): line is string => line !== undefined).join('\n')
  }


  private async handleExplicitMemoryDirective(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey?: string,
  ): Promise<boolean> {
    const directive = extractExplicitMemoryDirective(normalized.message.text)
    if (!directive) {
      return false
    }

    const result = await rememberUserMemory({
      content: directive.content,
      fileMemory: this.runtime.fileMemoryRegistry?.get(deriveChannelScopeTags({
        channelType: normalized.message.channelType,
        channelId: normalized.message.channelId,
        senderId: normalized.message.sender?.id,
      })) ?? this.runtime.fileMemory,
      semanticIndex: this.runtime.semanticIndex,
      tags: deriveChannelScopeTags({
        channelType: normalized.message.channelType,
        channelId: normalized.message.channelId,
        senderId: normalized.message.sender?.id,
      }),
    })

    let text: string
    let status = 'memory_remembered'
    if (result.status === 'saved') {
      text = [
        '기억했습니다. 이 사용자/채널 범위의 새 세션에서도 참고할게요.',
        '',
        `- ${result.content}`,
      ].join('\n')
    } else if (result.status === 'sensitive') {
      status = 'memory_sensitive_rejected'
      text = '토큰, 비밀번호, API 키처럼 보이는 값은 일반 메모리에 저장하지 않았습니다. 이런 값은 secret/config 저장소에 보관해야 합니다.'
    } else {
      status = 'memory_unavailable'
      text = '메모리 저장소가 준비되지 않아 기억하지 못했습니다. daemon 상태를 확인한 뒤 다시 시도해주세요.'
    }

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      text,
      status,
      {
        memoryStatus: result.status,
        fileMemoryAdded: result.fileMemoryAdded ?? 0,
        semanticMemorySaved: result.semanticMemorySaved ?? false,
      },
    )
    return true
  }

  private startChannelProgress(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
  ): ChannelProgressReporter {
    if (
      normalized.receiveOnly
      || !normalized.replyTarget
      || typeof channel.sendActivity !== 'function'
    ) {
      return { stop() {} }
    }

    let stopped = false
    const sendTyping = () => {
      if (stopped || !normalized.replyTarget) return
      channel.sendActivity?.(normalized.replyTarget, { type: 'typing' }).catch((error) => {
        log.warn('Failed to send channel activity', { error: String(error) })
      })
    }

    sendTyping()
    const activity = setInterval(sendTyping, CHANNEL_ACTIVITY_INTERVAL_MS)

    return {
      stop() {
        if (stopped) return
        stopped = true
        clearInterval(activity)
      },
    }
  }

  private async ensureChannelOriginsLoaded(): Promise<void> {
    if (!this.runtime.channelOriginStore) return
    if (!this.originLoadPromise) {
      this.originLoadPromise = this.loadChannelOrigins().catch((error) => {
        log.warn('Failed to load persisted channel origins', {
          error: errorName(error),
        })
      })
    }
    await this.originLoadPromise
  }

  private async loadChannelOrigins(): Promise<void> {
    const store = this.runtime.channelOriginStore
    if (!store) return
    const records = await store.list()
    const now = Date.now()
    for (const record of records) {
      if (record.kind === 'approval') {
        const remainingMs = CHANNEL_APPROVAL_TTL_MS - Math.max(0, now - record.requestedAt)
        if (remainingMs <= 0 || !record.toolName) {
          await store.delete('approval', record.id)
          continue
        }
        this.restoreApprovalOrigin(record, remainingMs)
        continue
      }
      if (record.kind === 'question' && record.sessionId) {
        this.questionOrigins.set(record.id, {
          sessionId: record.sessionId,
          channelType: record.channelType,
          channelId: record.channelId,
          senderId: record.senderId,
          requestedAt: record.requestedAt,
        })
      }
    }
  }

  private restoreApprovalOrigin(record: ChannelOriginRecord, remainingMs: number): void {
    this.forgetApprovalOrigin(record.id, { deletePersisted: false })
    const sequence = typeof record.sequence === 'number'
      ? record.sequence
      : ++this.approvalOriginSequence
    this.approvalOriginSequence = Math.max(this.approvalOriginSequence, sequence)
    const cleanup = setTimeout(() => {
      this.approvalOrigins.delete(record.id)
      this.deletePersistedChannelOrigin('approval', record.id)
    }, remainingMs)
    this.approvalOrigins.set(record.id, {
      channelType: record.channelType,
      channelId: record.channelId,
      senderId: record.senderId,
      sequence,
      toolName: record.toolName ?? 'tool',
      requestedAt: record.requestedAt,
      // Persisted origins predate the language field; the resolution path falls
      // back to detecting from the reply when the stored lang is unavailable.
      lang: detectChannelLang(undefined),
      cleanup,
    })
  }

  private persistChannelOrigin(record: ChannelOriginRecord): void {
    void this.runtime.channelOriginStore?.upsert(record).catch((error) => {
      log.warn('Failed to persist channel origin', {
        kind: record.kind,
        id: record.id,
        error: errorName(error),
      })
    })
  }

  private deletePersistedChannelOrigin(
    kind: ChannelOriginRecord['kind'],
    id: string,
  ): void {
    void this.runtime.channelOriginStore?.delete(kind, id).catch((error) => {
      log.warn('Failed to delete persisted channel origin', {
        kind,
        id,
        error: errorName(error),
      })
    })
  }

  private rememberApprovalOrigin(
    event: ChannelApprovalRequestEvent,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): void {
    const requestId = event.requestId
    this.forgetApprovalOrigin(requestId)
    const cleanup = setTimeout(() => {
      this.approvalOrigins.delete(requestId)
      this.deletePersistedChannelOrigin('approval', requestId)
    }, CHANNEL_APPROVAL_TTL_MS)
    this.approvalOrigins.set(requestId, {
      channelType: normalized.message.channelType,
      channelId: normalized.message.channelId,
      senderId: normalized.message.sender.id,
      sequence: ++this.approvalOriginSequence,
      toolName: event.toolCall.name,
      requestedAt: Date.now(),
      lang: detectChannelLang(normalized.message.text),
      cleanup,
    })
    const origin = this.approvalOrigins.get(requestId)
    if (origin) {
      this.persistChannelOrigin({
        kind: 'approval',
        id: requestId,
        channelType: origin.channelType,
        channelId: origin.channelId,
        senderId: origin.senderId,
        sequence: origin.sequence,
        toolName: origin.toolName,
        requestedAt: origin.requestedAt,
      })
    }
  }

  private forgetApprovalOrigin(
    requestId: string,
    options: { deletePersisted?: boolean } = {},
  ): void {
    const existing = this.approvalOrigins.get(requestId)
    if (existing) clearTimeout(existing.cleanup)
    this.approvalOrigins.delete(requestId)
    if (options.deletePersisted !== false) {
      this.deletePersistedChannelOrigin('approval', requestId)
    }
  }

  private matchingApprovalRequests(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): ChannelApprovalOriginMatch[] {
    return Array.from(this.approvalOrigins.entries())
      .filter(([, origin]) => (
        origin.channelType === normalized.message.channelType
        && origin.channelId === normalized.message.channelId
        && origin.senderId === normalized.message.sender.id
      ))
      .map(([requestId, origin]) => ({ requestId, ...origin }))
      .sort((left, right) => left.sequence - right.sequence)
  }

  private resolveApprovalToLatestRequest(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    intent: ChannelApprovalIntent,
    matches: ChannelApprovalOriginMatch[],
  ): NaturalApprovalResolution | null {
    const latest = matches[matches.length - 1]
    if (!latest) return null
    if (matches.length > 1) {
      log.info('Resolved natural approval reply to latest pending channel request', {
        channelType: normalized.message.channelType,
        channelId: normalized.message.channelId,
        requestId: latest.requestId,
        otherRequestIds: matches.slice(0, -1).map((match) => match.requestId),
      })
    }

    return {
      kind: 'command',
      command: {
        ...intent,
        requestId: latest.requestId,
      },
    }
  }

  private async classifyApprovalReplyWithLLM(
    text: string,
    toolName: string,
  ): Promise<ChannelApprovalIntent | null> {
    const provider = this.runtime.providerRegistry.getDefault()
    const configuredModel = this.runtime.config.agent?.defaultModel
    const model = configuredModel && provider?.models.some((candidate) => candidate.id === configuredModel)
      ? configuredModel
      : provider?.models?.[0]?.id
    if (!provider || !model) return null

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), APPROVAL_INTENT_LLM_TIMEOUT_MS)
    try {
      const response = await provider.chat({
        model,
        temperature: 0,
        maxTokens: 80,
        timeoutMs: APPROVAL_INTENT_LLM_TIMEOUT_MS,
        messages: [
          {
            role: 'system',
            content: [
              'Classify a short user reply to a pending tool approval request.',
              'Return JSON only: {"action":"approve|deny|feedback|unknown","scope":"once|session|always|run|session-all","note":"optional"}',
              'approve means clear consent to run the pending tool.',
              'deny means clear refusal to run it.',
              'feedback means the user wants the tool call changed before it runs.',
              'run means approve every tool request only for the current user command/task.',
              'session-all means approve every tool request in the current chat session.',
              'unknown means unrelated or unclear.',
              'The reply may be in ANY language. Recognize consent and refusal regardless of language — e.g. "yes/ok/sure", "네/예/좋아", "sí/vale", "oui/d\'accord", "はい/いいえ", "ja/nein", "да/нет" — and terse replies, typos, or keyboard-layout mistakes such as "goqhk" meaning "해봐".',
              'Be conservative: do not approve unless the reply is direct consent.',
            ].join(' '),
          },
          {
            role: 'user',
            content: `Pending tool: ${toolName}\nUser reply: ${text}`,
          },
        ],
      }, { signal: controller.signal })
      return parseApprovalIntentClassifierResponse(response.message.content)
    } catch (error) {
      log.debug('LLM approval reply classification failed', {
        error: errorName(error),
      })
      return null
    } finally {
      clearTimeout(timeout)
    }
  }

  private async resolveNaturalApproval(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): Promise<NaturalApprovalResolution | null> {
    const matches = this.matchingApprovalRequests(normalized)
    if (matches.length === 0) return null

    // 👍 / ✅ / 👎 / ❌ are unambiguous symbols — resolve them directly, the
    // same way a /approve command resolves. Any natural-language reply goes to
    // the LLM classifier; there is no regex that infers approve/deny from prose.
    const symbolIntent = parseApprovalSymbol(normalized.message.text)
    if (symbolIntent) {
      return this.resolveApprovalToLatestRequest(normalized, symbolIntent, matches)
    }

    const latest = matches[matches.length - 1]
    if (!latest) return null
    const classifiedIntent = await this.classifyApprovalReplyWithLLM(
      normalized.message.text,
      latest.toolName,
    )
    if (!classifiedIntent) return null

    return this.resolveApprovalToLatestRequest(normalized, {
      ...classifiedIntent,
      note: classifiedIntent.note ?? normalized.message.text.trim(),
    }, matches)
  }

  private async notifyApprovalRequest(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    event: ChannelApprovalRequestEvent,
  ): Promise<void> {
    this.rememberApprovalOrigin(event, normalized)
    const text = [
      `${event.toolCall.name} 도구를 실행해도 될까요?`,
      '',
      `요청 ID: ${event.requestId}`,
      '',
      '실행 입력:',
      '```json',
      formatApprovalToolInput(event.toolCall.arguments),
      '```',
      '',
      '그냥 답장하면 방금 보낸 요청에 적용됩니다.',
      '승인하려면 "네", "진행해", "해봐", "승인"처럼 답장하세요.',
      '거절하려면 "아니요", "하지마", "거절"처럼 답장하세요.',
      `명령으로는 /approve ${event.requestId} 또는 /deny ${event.requestId}도 사용할 수 있습니다.`,
      `"앞으로 항상 허용해" 또는 /approve ${event.requestId} always 로 기억시킬 수 있습니다.`,
      `"이번 작업은 계속 승인해" 또는 /approve ${event.requestId} run 으로 현재 명령 동안 모두 승인할 수 있습니다.`,
      `"이번 세션은 모두 승인해" 또는 /approve ${event.requestId} session-all 으로 현재 세션 동안 모두 승인할 수 있습니다.`,
    ].join('\n')

    await this.sendObserved(
      channel,
      normalized,
      text,
      'approval_request',
      { taskId: normalized.message.messageId },
      {
        requestId: event.requestId,
        toolName: event.toolCall.name,
      },
    )
    this.recordChannelEvent(normalized, 'channel.approval_requested', {
      requestId: event.requestId,
      toolName: event.toolCall.name,
    }, 'info', {
      taskId: normalized.message.messageId,
    })
  }

  private rememberQuestionOrigin(
    question: PendingQuestion,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): void {
    this.questionOrigins.set(question.id, {
      sessionId: question.sessionId,
      channelType: normalized.message.channelType,
      channelId: normalized.message.channelId,
      senderId: normalized.message.sender.id,
      requestedAt: Date.now(),
    })
    const origin = this.questionOrigins.get(question.id)
    if (origin) {
      this.persistChannelOrigin({
        kind: 'question',
        id: question.id,
        sessionId: origin.sessionId,
        channelType: origin.channelType,
        channelId: origin.channelId,
        senderId: origin.senderId,
        requestedAt: origin.requestedAt,
      })
    }
  }

  private forgetAnsweredQuestionOrigins(
    sessionId: string,
    pendingQuestions: PendingQuestion[],
  ): void {
    const pendingIds = new Set(pendingQuestions.map((question) => question.id))
    for (const [questionId, origin] of this.questionOrigins.entries()) {
      if (origin.sessionId === sessionId && !pendingIds.has(questionId)) {
        this.questionOrigins.delete(questionId)
        this.deletePersistedChannelOrigin('question', questionId)
      }
    }
  }

  private questionMatchesOrigin(
    question: PendingQuestion,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): boolean {
    return matchesQuestionOrigin(
      this.questionOrigins.get(question.id),
      normalized,
    )
  }

  private async notifyQuestionRequest(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    question: PendingQuestion,
  ): Promise<void> {
    this.rememberQuestionOrigin(question, normalized)
    const choices = question.choices?.length
      ? [
          '',
          '선택지:',
          ...question.choices.map((choice) => `- ${choice}`),
        ]
      : []
    const text = [
      '확인이 필요합니다.',
      '',
      question.prompt,
      ...choices,
      '',
      `질문 ID: ${question.id}`,
      '그냥 이 채팅에 답장하면 이어서 처리합니다.',
    ].join('\n')
    const delivered = await this.sendObserved(
      channel,
      normalized,
      text,
      'question_request',
      {
        sessionId: question.sessionId,
        taskId: normalized.message.messageId,
      },
      {
        questionId: question.id,
        choiceCount: question.choices?.length ?? 0,
      },
    )
    this.recordChannelEvent(normalized, 'channel.question_requested', {
      questionId: question.id,
      choiceCount: question.choices?.length ?? 0,
    }, 'info', {
      sessionId: question.sessionId,
      taskId: normalized.message.messageId,
    })
    await this.emitChannelHook(normalized.message, {
      status: 'question_requested',
      sessionId: question.sessionId,
      questionId: question.id,
      responseDelivered: delivered,
    })
  }

  private async handlePendingQuestionAnswer(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey?: string,
  ): Promise<boolean> {
    const questions = this.runtime.questions
    if (!normalized.sessionKey || !this.runtime.channelSessionStore || !questions) {
      return false
    }

    const binding = await this.runtime.channelSessionStore.get(normalized.sessionKey)
    if (!binding) return false

    const session = await this.runtime.sessions.get(binding.sessionId)
    if (!session || session.status === 'completed') {
      await this.runtime.channelSessionStore.delete(binding.key)
      return false
    }

    const pendingQuestions = questions.list(binding.sessionId)
    this.forgetAnsweredQuestionOrigins(binding.sessionId, pendingQuestions)
    if (pendingQuestions.length === 0) return false

    const matchingQuestions = pendingQuestions.filter((question) =>
      this.questionMatchesOrigin(question, normalized),
    )
    if (matchingQuestions.length === 0) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        '대기 중인 질문은 요청한 사용자만 답할 수 있습니다. 요청한 사용자로 답장하거나 웹에서 처리해주세요.',
        'question_origin_mismatch',
        { sessionId: binding.sessionId },
      )
      return true
    }

    if (matchingQuestions.length > 1) {
      const text = [
        '대기 중인 질문이 여러 개라 어떤 질문의 답인지 알 수 없습니다.',
        ...matchingQuestions.map((question) => `- ${question.id}: ${question.prompt}`),
        '웹에서 질문을 선택해 답하거나 하나만 남긴 뒤 다시 답장해주세요.',
      ].join('\n')
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        text,
        'question_ambiguous',
        {
          sessionId: binding.sessionId,
          questionIds: matchingQuestions.map((question) => question.id),
        },
      )
      return true
    }

    const question = matchingQuestions[0]!
    const questionOrigin = this.questionOrigins.get(question.id)
    const answer = normalized.message.text.trim()
    await this.runtime.sessions.appendEvent(binding.sessionId, {
      type: 'user_message',
      id: normalized.message.messageId,
      timestamp: normalized.message.timestamp,
      content: normalized.message.text,
    })
    await this.runtime.channelSessionStore.bind({
      ...binding,
      updatedAt: normalized.message.timestamp,
    })
    const answered = questions.answer(question.id, answer)
    if (answered) {
      this.questionOrigins.delete(question.id)
      this.deletePersistedChannelOrigin('question', question.id)
    }
    this.recordChannelEvent(
      normalized,
      answered ? 'channel.question_answered' : 'channel.question_not_found',
      {
        questionId: question.id,
        answerLength: answer.length,
        latencyMs: questionOrigin
          ? Math.max(0, Date.now() - questionOrigin.requestedAt)
          : undefined,
      },
      answered ? 'info' : 'warning',
      {
        sessionId: binding.sessionId,
        taskId: normalized.message.messageId,
      },
    )

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      answered
        ? '답변을 전달했습니다. 이어서 처리할게요.'
        : `대기 중인 질문을 찾을 수 없습니다: ${question.id}`,
      answered ? 'question_answered' : 'question_not_found',
      {
        sessionId: binding.sessionId,
        questionId: question.id,
      },
    )
    if (answered) {
      this.resumeActiveRunProgress(channel, normalized)
    }
    return true
  }

  private async resolveFeedbackSessionId(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): Promise<string | undefined> {
    if (!normalized.sessionKey || !this.runtime.channelSessionStore) {
      return undefined
    }
    try {
      const binding = await this.runtime.channelSessionStore.get(normalized.sessionKey)
      return binding?.sessionId
    } catch {
      return undefined
    }
  }

  private async finishSilentChannelFeedback(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey: string | undefined,
    extra: Record<string, unknown>,
  ): Promise<void> {
    await this.emitChannelHook(normalized.message, {
      status: 'feedback_recorded',
      responseDelivered: false,
      ...extra,
    })
    await this.replayGuard.markProcessed(dedupeKey)
    this.runtime.channelPipelineMonitor?.record(
      normalized.message.channelType,
      'processed',
    )
  }

  private async handleFeedbackCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    command: ChannelFeedbackCommand,
    dedupeKey?: string,
  ): Promise<void> {
    if (command.kind === 'help') {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        [
          '피드백 사용법:',
          '- /feedback good [메모]',
          '- /feedback bad [메모]',
          '- /feedback neutral [메모]',
          '',
          '답변에 👍 또는 👎만 보내도 조용히 만족도 신호로 기록합니다.',
        ].join('\n'),
        'feedback_help',
      )
      return
    }

    const sessionId = await this.resolveFeedbackSessionId(normalized)
    const targetMessageId =
      normalized.message.replyTo ?? normalized.message.messageId
    let recorded = false

    try {
      this.runtime.observability?.recordFeedback({
        sessionId,
        messageId: targetMessageId,
        rating: command.rating,
        reason: command.reason,
        note: command.note,
        source: 'channel',
        surface: normalized.message.channelType,
      })
      recorded = Boolean(this.runtime.observability)
    } catch {
      recorded = false
    }

    this.recordChannelEvent(
      normalized,
      'channel.feedback_received',
      {
        rating: command.rating,
        reason: command.reason,
        hasNote: Boolean(command.note),
        targetMessageId,
        recorded,
      },
      recorded ? 'info' : 'warning',
      {
        sessionId,
        taskId: normalized.message.messageId,
      },
    )

    const extra = {
      sessionId,
      rating: command.rating,
      reason: command.reason,
      targetMessageId,
      recorded,
    }

    if (command.silent) {
      await this.finishSilentChannelFeedback(normalized, dedupeKey, extra)
      return
    }

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      recorded
        ? '피드백을 기록했습니다. 답변 품질 지표에 반영할게요.'
        : '관측 저장소가 준비되지 않아 피드백을 저장하지 못했습니다.',
      recorded ? 'feedback_recorded' : 'feedback_unavailable',
      extra,
    )
  }

  private async handleAmbiguousApprovalResponse(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    requestIds: string[],
    dedupeKey?: string,
  ): Promise<void> {
    const delivered = await this.sendObserved(
      channel,
      normalized,
      [
        '대기 중인 승인 요청이 여러 개라서 어떤 요청인지 알 수 없습니다.',
        `요청 ID: ${requestIds.join(', ')}`,
        '승인하려면 /approve <요청 ID>, 거절하려면 /deny <요청 ID>로 답장해주세요.',
      ].join('\n'),
      'approval_response',
      { taskId: normalized.message.messageId },
      { requestCount: requestIds.length },
    )
    this.recordChannelEvent(normalized, 'channel.approval_ambiguous', {
      requestCount: requestIds.length,
    }, 'warning', {
      taskId: normalized.message.messageId,
    })
    await this.emitChannelHook(normalized.message, {
      status: 'approval_ambiguous',
      requestIds,
      responseDelivered: delivered,
    })
    await this.replayGuard.markProcessed(dedupeKey)
    this.runtime.channelPipelineMonitor?.record(
      normalized.message.channelType,
      'processed',
    )
  }

  private async handleApprovalCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    command: ChannelApprovalCommand,
    dedupeKey?: string,
  ): Promise<void> {
    const origin = this.approvalOrigins.get(command.requestId)
    const sameOrigin = origin
      && origin.channelType === normalized.message.channelType
      && origin.channelId === normalized.message.channelId
      && origin.senderId === normalized.message.sender.id

    if (!sameOrigin) {
      const delivered = await this.sendObserved(
        channel,
        normalized,
        `No live approval request for ${command.requestId} in this chat.`,
        'approval_response',
        { taskId: normalized.message.messageId },
        { resolved: false },
      )
      this.recordChannelEvent(normalized, 'channel.approval_not_found', {
        requestId: command.requestId,
      }, 'warning', {
        taskId: normalized.message.messageId,
      })
      await this.emitChannelHook(normalized.message, {
        status: 'approval_not_found',
        requestId: command.requestId,
        responseDelivered: delivered,
      })
      await this.replayGuard.markProcessed(dedupeKey)
      this.runtime.channelPipelineMonitor?.record(
        normalized.message.channelType,
        'processed',
      )
      return
    }

    const toolName = origin.toolName
    const decision = approvalDecisionForAction(command.action)
    const approved = approvalApprovedForAction(command.action)
    const scope = approved ? command.scope : 'once'
    const result = await this.runtime.approvalRegistry.respond(
      command.requestId,
      {
        decision,
        approved,
        note: command.note,
      },
      {
        approvedBy: `channel:${normalized.message.channelType}:${normalized.message.sender.id}`,
        scope,
      },
    )

    if (result.resolved) {
      this.forgetApprovalOrigin(command.requestId)
    }

    // Mirror the conversation language: prefer this reply's own script when it
    // has one, otherwise the language captured when the approval was requested,
    // otherwise Korean for back-compat.
    const resolutionLang: ChannelLang =
      channelLangSignal(normalized.message.text) ?? origin?.lang ?? 'ko'

    const delivered = await this.sendObserved(
      channel,
      normalized,
      result.resolved
        ? formatApprovalResolutionText(command, toolName, resolutionLang)
        : channelMessage('approvalNoLiveRequest', resolutionLang, { id: command.requestId }),
      'approval_response',
      { taskId: normalized.message.messageId },
      {
        requestId: command.requestId,
        resolved: result.resolved,
        decision,
        scope,
      },
    )

    const approvalLatencyMs = Math.max(0, Date.now() - origin.requestedAt)
    this.recordChannelEvent(
      normalized,
      result.resolved ? 'approval.responded' : 'channel.approval_not_found',
      {
        requestId: command.requestId,
        decision,
        approved,
        scope,
        toolName,
        approvalLatencyMs,
      },
      result.resolved ? 'info' : 'warning',
      {
        taskId: normalized.message.messageId,
      },
    )

    await this.emitChannelHook(normalized.message, {
      status: result.resolved ? 'approval_resolved' : 'approval_not_found',
      requestId: command.requestId,
      decision,
      approved,
      scope,
      responseDelivered: delivered,
    })
    await this.replayGuard.markProcessed(dedupeKey)
    this.runtime.channelPipelineMonitor?.record(
      normalized.message.channelType,
      'processed',
    )
    if (result.resolved) {
      // Reset waiting state for both approve and deny — after a denial
      // the agent loop continues with the denial-as-tool-output, so the
      // run is no longer "waiting for approval". Without this reset the
      // busy reminder shown on subsequent messages would point at a
      // waitingForId that has already been resolved.
      this.resumeActiveRunProgress(channel, normalized)
    }
  }

  private resumeActiveRunProgress(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): void {
    const activeRun = this.activeRunFor(normalized)
    if (!activeRun) return
    activeRun.progress?.stop()
    activeRun.phase = 'processing'
    activeRun.waitingSince = undefined
    activeRun.waitingForId = undefined
    activeRun.waitingForLabel = undefined
    activeRun.progress = this.startChannelProgress(channel, normalized)
  }

  private async completeCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey: string | undefined,
    text: string,
    status: string,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    const delivered = await this.sendObserved(
      channel,
      normalized,
      text,
      'command_response',
      {
        taskId: normalized.message.messageId,
        sessionId: typeof extra.sessionId === 'string' ? extra.sessionId : undefined,
      },
      { status },
    )
    this.recordChannelEvent(normalized, 'channel.command_completed', {
      status,
      responseDelivered: delivered,
    }, 'info', {
      taskId: normalized.message.messageId,
      sessionId: typeof extra.sessionId === 'string' ? extra.sessionId : undefined,
    })
    await this.emitChannelHook(normalized.message, {
      status,
      responseDelivered: delivered,
      ...extra,
    })
    await this.replayGuard.markProcessed(dedupeKey)
    this.runtime.channelPipelineMonitor?.record(
      normalized.message.channelType,
      'processed',
    )
  }

  private async handleSessionCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    command: ChannelSessionCommand,
    dedupeKey?: string,
  ): Promise<void> {
    switch (command.action) {
      case 'help':
        await this.handleSessionHelpCommand(channel, normalized, dedupeKey)
        return
      case 'current':
        await this.handleCurrentSessionCommand(channel, normalized, dedupeKey)
        return
      case 'new':
        await this.handleNewSessionCommand(channel, normalized, dedupeKey)
        return
      case 'list':
        await this.handleListSessionsCommand(channel, normalized, command.query, dedupeKey)
        return
      case 'resume':
        await this.handleResumeSessionCommand(channel, normalized, command.sessionRef, dedupeKey)
        return
      case 'memory':
        await this.handleMemoryStatusCommand(channel, normalized, dedupeKey)
        return
      case 'self':
        await this.handleSelfCommand(channel, normalized, command.selfAction, dedupeKey)
        return
      case 'model':
        await this.handleModelCommand(channel, normalized, command.args, dedupeKey, command.modelAction)
        return
      case 'tps':
        await this.handleTpsCommand(channel, normalized, command.tpsAction, dedupeKey)
        return
    }
  }

  private async handleTpsCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    action: Extract<ChannelSessionCommand, { action: 'tps' }>['tpsAction'],
    dedupeKey?: string,
  ): Promise<void> {
    const tracker = this.tokenSpeedTrackerFor(normalized)
    if (action === 'reset') {
      tracker.reset()
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'TPS 통계를 초기화했습니다. 이후 이 채팅에서 완료되는 LLM 응답부터 다시 집계합니다.',
        'tps_reset',
      )
      return
    }

    if (action === 'help') {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        '사용법: /tps [current|reset]',
        'tps_help',
      )
      return
    }

    const snapshot = tracker.snapshot()
    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      formatTokenSpeedStats(snapshot, { locale: 'ko' }),
      'tps_current',
      { tpsSampleCount: snapshot.session.sampleCount },
    )
  }

  private async handleExplicitStopCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    dedupeKey?: string,
  ): Promise<void> {
    const activeRuns = this.listActiveRuns(normalized.sessionKey)
    if (activeRuns.length === 0) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        '현재 이 채팅에서 진행 중인 작업이 없습니다.',
        'channel_run_idle',
      )
      return
    }

    await this.handleCancelFollowup(
      channel,
      normalized,
      dedupeKey,
      normalized.sessionKey ?? '',
      activeRuns,
    )
  }

  private async handleSkillsCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    command: ChannelSkillsCommand,
    dedupeKey?: string,
  ): Promise<void> {
    switch (command.action) {
      case 'help':
        await this.completeCommand(channel, normalized, dedupeKey, skillsHelpText(), 'skills_help')
        return
      case 'list':
        await this.handleSkillsListCommand(channel, normalized, command.query, dedupeKey)
        return
      case 'search':
        await this.handleSkillsSearchCommand(channel, normalized, command.query, dedupeKey)
        return
      case 'enable':
      case 'disable':
        await this.handleSkillToggleCommand(channel, normalized, command.action, command.id, dedupeKey)
        return
      case 'install':
        await this.handleSkillInstallCommand(channel, normalized, command.source, command.expectedDigest, dedupeKey)
        return
    }
  }

  private async handleSkillsListCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    query: string | undefined,
    dedupeKey?: string,
  ): Promise<void> {
    const skills = filterSkills(await this.runtime.skillRegistry.listAll(), query)
      .sort((a, b) => a.id.localeCompare(b.id))
    const visible = skills.slice(0, 20)
    const text = [
      query ? `설치된 스킬 검색 결과 (${visible.length}/${skills.length}):` : `설치된 스킬 (${skills.length}):`,
      visible.length ? visible.map(formatSkillLine).join('\n') : '(none)',
      skills.length > visible.length ? `... ${skills.length - visible.length}개 더 있음. /skills installed <검색어>로 좁혀보세요.` : '',
      '',
      '관리: /skills search <query> | /skills enable <id> | /skills disable <id> | /skills install <source>',
      '사용: /<skill-id> <작업 내용>',
    ].filter(Boolean).join('\n')
    await this.completeCommand(channel, normalized, dedupeKey, text, 'skills_list', {
      query,
      count: visible.length,
      totalCount: skills.length,
    })
  }

  private async handleSkillsSearchCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    query: string | undefined,
    dedupeKey?: string,
  ): Promise<void> {
    const trimmed = query?.trim()
    if (!trimmed) {
      await this.completeCommand(channel, normalized, dedupeKey, '사용법: /skills search <query>', 'skills_search_usage')
      return
    }

    const installed = await this.runtime.skillRegistry.search(trimmed)
    const lines = [
      `스킬 검색: ${trimmed}`,
      '',
      '설치됨:',
      installed.length ? installed.slice(0, 8).map(formatSkillLine).join('\n') : '(none)',
    ]

    try {
      if (this.runtime.marketplaceCatalog) {
        const source = new MarketplaceSource({
          catalog: this.runtime.marketplaceCatalog,
          urlPolicy: this.runtime.skillSourceUrlPolicy,
        })
        const marketplace = await source.search(trimmed, { limit: 8 })
        lines.push('', '마켓플레이스:')
        lines.push(
          marketplace.length
            ? marketplace.map((item) =>
                `- ${item.metadata.id}@${item.metadata.version} (${item.marketplace})${item.metadataOnly ? ' · metadata-only' : ''}: ${truncateForTelegram(item.metadata.description, 90)}`,
              ).join('\n')
            : '(none)',
        )
        if (marketplace.length > 0) {
          lines.push('', '설치 미리보기: /skills install <marketplace>/<skill-id>')
        }
      }
    } catch (error) {
      lines.push('', `마켓플레이스 검색 실패: ${formatSkillCommandError(error)}`)
    }

    await this.completeCommand(channel, normalized, dedupeKey, lines.join('\n'), 'skills_search', {
      query: trimmed,
      installedCount: installed.length,
    })
  }

  private async handleSkillToggleCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    action: 'enable' | 'disable',
    id: string | undefined,
    dedupeKey?: string,
  ): Promise<void> {
    if (!id) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `사용법: /skills ${action} <id>`,
        'skills_toggle_usage',
      )
      return
    }
    const existing = await this.runtime.skillRegistry.get(id)
    if (!existing) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `스킬을 찾을 수 없습니다: ${id}\n/skills installed 로 id를 확인하세요.`,
        'skills_toggle_not_found',
      )
      return
    }

    const enabled = action === 'enable'
    await this.runtime.skillRegistry.setEnabled(id, enabled)
    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      enabled
        ? `스킬을 활성화했습니다: ${id}`
        : `스킬을 비활성화했습니다: ${id}`,
      enabled ? 'skills_enabled' : 'skills_disabled',
      { skillId: id, enabled },
    )
  }

  private async handleSkillInstallCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    source: string | undefined,
    expectedDigest: string | undefined,
    dedupeKey?: string,
  ): Promise<void> {
    if (!source) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        [
          '사용법:',
          '/skills install <source>',
          '/skills install <source> --confirm <digest>',
          'source 예: marketplace/skill-id, https://.../SKILL.md, github repo/tree URL',
        ].join('\n'),
        'skills_install_usage',
      )
      return
    }

    try {
      if (!expectedDigest) {
        const preview = await this.runtime.installPipeline.preview({ source })
        const candidateLines = preview.fetched.map((item) =>
          `- ${item.metadata.id}@${item.metadata.version}: ${truncateForTelegram(item.metadata.description, 90)}`,
        )
        await this.completeCommand(
          channel,
          normalized,
          dedupeKey,
          [
            '스킬 설치 미리보기:',
            `Source: ${source}`,
            `Digest: ${preview.digest}`,
            '',
            candidateLines.length ? candidateLines.join('\n') : '(no candidates)',
            '',
            '설치하려면 같은 source와 digest로 명시 승인하세요:',
            `/skills install ${source} --confirm ${preview.digest}`,
          ].join('\n'),
          'skills_install_preview',
          { source, digest: preview.digest, count: preview.fetched.length },
        )
        return
      }

      const result = await this.runtime.installPipeline.install({ source, expectedDigest })
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        [
          `스킬 설치 완료 (${result.installed.length}):`,
          ...result.installed.map(formatSkillLine),
        ].join('\n'),
        'skills_installed',
        { source, digest: result.digest, count: result.installed.length },
      )
    } catch (error) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `스킬 설치를 완료하지 못했습니다: ${formatSkillCommandError(error)}`,
        'skills_install_failed',
        { source, errorName: errorName(error) },
      )
    }
  }

  private async handleSkillShortcutCommand(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    dedupeKey: string | undefined,
    command: { skillId: string; args: string },
  ): Promise<boolean> {
    if (typeof this.runtime.skillRegistry?.getForCwd !== 'function') return false
    const bound = await this.boundChannelSession(normalized).catch(() => null)
    const cwd = bound?.session.cwd ?? process.cwd()
    const entry = await this.runtime.skillRegistry.getForCwd(command.skillId, cwd).catch(() => null)
    if (!entry) return false

    if (entry.metadata.enabled === false) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `스킬이 비활성화되어 있습니다: ${command.skillId}\n활성화: /skills enable ${command.skillId}`,
        'skill_shortcut_disabled',
        { skillId: command.skillId },
      )
      return true
    }

    if (!command.args) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `사용법: /${command.skillId} <작업 내용>`,
        'skill_shortcut_usage',
        { skillId: command.skillId },
      )
      return true
    }

    const prompt = [
      `Use the installed skill "${command.skillId}" for this request.`,
      `First call the skill tool with skill="${command.skillId}" and args containing the user request, then follow that skill unless blocked by policy or missing prerequisites.`,
      '',
      `User request:\n${command.args}`,
    ].join('\n')

    const chatKey = normalized.sessionKey
    const activeRuns = this.listActiveRuns(chatKey)
    if (chatKey && activeRuns.length > 0) {
      await this.handleParallelNewFollowup(
        channel,
        normalized,
        autonomy,
        dedupeKey,
        chatKey,
        activeRuns,
        prompt,
      )
      return true
    }

    await this.startMainRun(channel, normalized, autonomy, dedupeKey, prompt)
    return true
  }

  private async handleHealthCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey?: string,
  ): Promise<void> {
    const lines: string[] = ['🩺 sepilotd health']

    // Daemon process.
    const uptimeMs = Math.floor(process.uptime() * 1000)
    const device = this.runtime.config?.device
    lines.push(
      `• daemon: running · uptime ${formatElapsed(uptimeMs)}`
      + (device ? ` · ${device.name} (${device.role})` : '')
      + (DAEMON_VERSION ? ` · v${DAEMON_VERSION}` : ''),
    )

    // LLM provider — without one the agent cannot reply at all.
    const provider = this.runtime.providerRegistry?.getDefault?.()
    lines.push(
      provider
        ? `• provider: ${provider.id} · model ${provider.models?.[0]?.id ?? 'default'}`
        : '• provider: ⚠️ none configured — the agent cannot respond until one is added',
    )

    // Scheduler.
    if (this.runtime.jobStore) {
      try {
        const jobs = this.runtime.jobStore.list()
        const pending = jobs.filter((j) => j.status === 'pending')
        const failed = jobs.filter((j) => j.status === 'failed' || (j.lastError != null && j.lastError !== ''))
        const nextPending = pending
          .filter((j) => typeof j.nextRunAt === 'number' && Number.isFinite(j.nextRunAt))
          .sort((a, b) => a.nextRunAt - b.nextRunAt)[0]
        let line = `• scheduler: ${pending.length} pending / ${jobs.length} total`
        if (failed.length > 0) line += ` · ⚠️ ${failed.length} failed`
        if (nextPending) line += ` · next ${formatWhen(nextPending)}`
        const chatKey = normalized.sessionKey
        if (chatKey) {
          const here = jobs.filter((j) => j.channelTarget === chatKey && j.status === 'pending').length
          if (here > 0) line += ` · ${here} for this chat`
        }
        lines.push(line)
      } catch (error) {
        lines.push(`• scheduler: error reading jobs (${errorName(error)})`)
      }
    } else {
      lines.push('• scheduler: not available')
    }

    // Memory subsystems.
    const memBits: string[] = []
    memBits.push(this.runtime.semanticIndex ? 'semantic ✓' : 'semantic ✗')
    memBits.push((this.runtime.fileMemory || this.runtime.fileMemoryRegistry) ? 'file ✓' : 'file ✗')
    if (this.runtime.remindersStore) {
      try {
        const reminders = await this.runtime.remindersStore.list()
        const pendingReminders = reminders.filter((r) => !r.firedAt && !r.cancelledAt).length
        memBits.push(`reminders ${pendingReminders} pending`)
      } catch {
        memBits.push('reminders ?')
      }
    }
    lines.push(`• memory: ${memBits.join(' · ')}`)

    // This chat: bound session + active runs.
    const chatKey = normalized.sessionKey
    const activeRuns = this.listActiveRuns(chatKey)
    const bound = await this.boundChannelSession(normalized).catch(() => null)
    lines.push(
      `• this chat: ${bound ? `session ${bound.session.id.slice(0, 8)}…` : 'no session bound'}`
      + (activeRuns.length > 0 ? ` · ${activeRuns.length} active run(s)` : '')
      + (chatKey ? '' : ' · (channel has no persistent session key)'),
    )

    // Channel pipeline counters.
    try {
      const stats = this.runtime.channelPipelineMonitor?.getStats?.()
      const here = stats?.byChannelType.find((c) => c.channelType === normalized.message.channelType)
      if (here) {
        lines.push(
          `• ${normalized.message.channelType} pipeline: ${here.processedEvents} processed`
          + ` · ${here.errorEvents} errors`
          + ` · ${here.inFlight} in-flight`
          + (here.blockedEvents > 0 ? ` · ${here.blockedEvents} blocked` : ''),
        )
      }
    } catch {
      // Monitoring is best-effort; omit the line if it throws.
    }

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      lines.join('\n'),
      'channel_health',
      {
        uptimeMs,
        providerConfigured: Boolean(provider),
        activeRunCount: activeRuns.length,
        sessionBound: Boolean(bound),
      },
    )
  }

  private async handleWorkspaceCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    command: ChannelWorkspaceCommand,
    dedupeKey?: string,
  ): Promise<void> {
    if (command.action === 'show') {
      await this.handleWorkspaceShowCommand(channel, normalized, dedupeKey)
      return
    }

    await this.handleWorkspaceSetCommand(channel, normalized, command.path, dedupeKey)
  }

  private async handleWorkspaceShowCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey?: string,
  ): Promise<void> {
    const current = await this.boundChannelSession(normalized)
    const cwd = current?.session.cwd ?? process.cwd()
    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      current
        ? [
            'Current workspace:',
            `Session: ${current.session.id}`,
            `Cwd: ${cwd}`,
          ].join('\n')
        : [
            'No active session is attached to this chat.',
            `Default cwd: ${cwd}`,
            'Set this chat workspace with /cwd <absolute-path>.',
          ].join('\n'),
      'workspace_cwd_current',
      current ? { sessionId: current.session.id, cwd } : { cwd },
    )
  }

  private async handleWorkspaceSetCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    rawPath: string,
    dedupeKey?: string,
  ): Promise<void> {
    if (!normalized.sessionKey || !this.runtime.channelSessionStore) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'This channel does not support persistent workspaces.',
        'workspace_cwd_unavailable',
      )
      return
    }

    const current = await this.boundChannelSession(normalized)
    let cwd: string
    try {
      cwd = await this.prepareWorkspaceCwd(rawPath, current?.session.cwd)
    } catch (error) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `작업 디렉토리를 설정할 수 없습니다: ${error instanceof Error ? error.message : String(error)}`,
        'workspace_cwd_invalid',
      )
      return
    }

    const session = current?.session ?? await this.createWorkspaceSession(normalized, cwd)
    let updated = session
    if (session.cwd !== cwd || session.status !== 'active') {
      const patch = session.status === 'active'
        ? { cwd }
        : { cwd, status: 'active' as const }
      const patched = await this.runtime.sessions.updateMeta?.(session.id, patch)
      if (patched) {
        updated = patched
      }
    }
    if (updated.cwd !== cwd) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'Session store does not support workspace updates.',
        'workspace_cwd_unavailable',
        { sessionId: session.id, cwd },
      )
      return
    }

    await this.runtime.channelSessionStore.bind({
      key: normalized.sessionKey,
      sessionId: updated.id,
      channelType: normalized.message.channelType,
      channelId: normalized.message.channelId,
      updatedAt: normalized.message.timestamp,
    })

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      [
        '작업 디렉토리를 설정했습니다.',
        `Cwd: ${updated.cwd ?? cwd}`,
        '이후 이 채팅의 파일/터미널/코드 작업은 이 경로를 기본 작업 위치로 사용합니다.',
      ].join('\n'),
      'workspace_cwd_updated',
      { sessionId: updated.id, cwd: updated.cwd ?? cwd },
    )
  }

  private async prepareWorkspaceCwd(rawPath: string, baseCwd?: string): Promise<string> {
    const cleaned = cleanWorkspacePathCandidate(rawPath)
    if (!cleaned) {
      throw new Error('path is empty')
    }

    const resolved = resolveToolPath(cleaned, baseCwd)
    await mkdir(resolved, { recursive: true })
    const real = await realpath(resolved)
    const info = await stat(real)
    if (!info.isDirectory()) {
      throw new Error(`path is not a directory: ${resolved}`)
    }
    return real
  }

  private async createWorkspaceSession(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    cwd: string,
  ): Promise<SessionMeta> {
    const provider = this.runtime.providerRegistry.getDefault()
    const now = new Date().toISOString()
    return this.runtime.sessions.create({
      id: randomUUID(),
      title: `[${normalized.message.channelType}] workspace ${cwd}`,
      createdAt: now,
      updatedAt: now,
      provider: provider?.id ?? 'unknown',
      model: this.runtime.config.agent?.defaultModel ?? provider?.models[0]?.id ?? 'default',
      device: this.runtime.config.device.name,
      status: 'active',
      cwd,
      tags: [`channel:${normalized.message.channelType}`],
    })
  }

  private async handleSessionHelpCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey?: string,
  ): Promise<void> {
    const text = [
      'Channel commands:',
      '/health - daemon, provider, scheduler, memory and pipeline status',
      '/session - show the active session for this chat',
      '/new - start a fresh session in this chat',
      '/sessions [query] - list or search recent sessions',
      '/resume <id> - attach this chat to a previous session',
      '/memory - show durable memory availability for this chat',
      '/skills [installed|search|install|enable|disable] - manage skills for this daemon',
      '/memory_stats | /memory_search <query> | /memory_list [N] | /memory_hot [N] | /memory_remember <text> | /memory_forget <id> | /memory_audit <id> | /memory_pin <id> | /memory_unpin <id> | /memory_pinned [N]',
      '/schedule - list scheduled tasks for this chat',
      '/schedule add <when> -- <instruction> - create a scheduled task',
      '/schedule edit <id> <when> -- <instruction> - update an existing scheduled task',
      '/schedule reschedule <id> <when> - update only the scheduled time',
      '/schedule show|pause|resume|run|runs|cancel <id> - manage a scheduled task',
      '/tps [current|reset] - show or reset token-per-second stats for this chat',
      '/usage [current|reset] - alias for /tps',
      '/stop - stop active work in this chat',
      '/remember <내용> - save durable user memory for future sessions',
      '/cwd [path] - show or set this chat workspace',
      '/approve <request> [once|run|session-all|session|always] - approve a live tool request',
      '/deny <request> [note] - deny a live tool request',
    ].join('\n')
    await this.completeCommand(channel, normalized, dedupeKey, text, 'session_help')
  }

  private async handleCurrentSessionCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey?: string,
  ): Promise<void> {
    const binding = await this.currentBinding(normalized)
    const activeRuns = this.activeRunsFor(normalized)
    const primaryActiveRun = activeRuns.find((run) => run.kind === 'main') ?? activeRuns[0]
    if (!binding) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        primaryActiveRun
          ? [
              '현재 이 채팅에서 요청을 처리 중이지만 세션 연결은 아직 준비 중입니다.',
              formatActiveRunsStatus(activeRuns),
              formatRunAttentionLine(primaryActiveRun),
            ].filter(Boolean).join('\n')
          : 'No active session is attached to this chat. Send a message to start one, or use /sessions to list previous sessions.',
        'session_not_found',
        primaryActiveRun ? { activeMessageId: primaryActiveRun.messageId } : {},
      )
      return
    }

    const session = await this.runtime.sessions.get(binding.sessionId)
    if (!session) {
      await this.runtime.channelSessionStore?.delete(binding.key)
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'The attached session no longer exists. I cleared this chat binding; send a message to start a new session.',
        'session_binding_cleared',
        { sessionId: binding.sessionId },
      )
      return
    }

    const text = [
      'Current session:',
      `ID: ${session.id}`,
      `Title: ${session.title}`,
      `Status: ${session.status}`,
      `Cwd: ${session.cwd ?? process.cwd()}`,
      `Messages: ${session.messageCount}`,
      `Updated: ${session.updatedAt}`,
      activeRuns.length > 0 ? formatActiveRunsStatus(activeRuns) : 'Current run: idle',
      formatRunAttentionLine(primaryActiveRun),
      ...this.formatPendingSessionWork(normalized, session.id),
    ].filter(Boolean).join('\n')
    await this.completeCommand(channel, normalized, dedupeKey, text, 'session_current', {
      sessionId: session.id,
      cwd: session.cwd ?? process.cwd(),
    })
  }

  private activeRunFor(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): ActiveChannelRun | undefined {
    const chatKey = normalized.sessionKey
    if (!chatKey) return undefined
    const runs = this.listActiveRuns(chatKey)
    return runs.find((run) => run.kind === 'main') ?? runs[0]
  }

  private activeRunsFor(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): ActiveChannelRun[] {
    return this.listActiveRuns(normalized.sessionKey)
  }

  private formatPendingSessionWork(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    sessionId: string,
  ): string[] {
    const lines: string[] = []
    const approvalRegistry = (this.runtime as {
      approvalRegistry?: {
        listForSession?: ChannelPipelineCapabilities['approvalRegistry']['listForSession']
      }
    }).approvalRegistry
    const pendingApprovals = (approvalRegistry?.listForSession?.(sessionId) ?? [])
      .filter((approval) => {
        const origin = this.approvalOrigins.get(approval.requestId)
        return !origin
          || (
            origin.channelType === normalized.message.channelType
            && origin.channelId === normalized.message.channelId
            && origin.senderId === normalized.message.sender.id
          )
      })
    for (const approval of pendingApprovals.slice(0, 3)) {
      const elapsed = formatElapsed(Date.now() - new Date(approval.requestedAt).getTime())
      lines.push([
        `Waiting approval: ${approval.tool} for ${elapsed}`,
        `Request: ${approval.requestId}`,
        `Reply "네" or /approve ${approval.requestId}.`,
      ].join('\n'))
    }
    if (pendingApprovals.length > 3) {
      lines.push(`Waiting approval: ${pendingApprovals.length - 3} more request(s).`)
    }

    const questions = (this.runtime as {
      questions?: {
        list?: ChannelPipelineCapabilities['questions']['list']
      }
    }).questions
    const pendingQuestions = (questions?.list?.(sessionId) ?? [])
      .filter((question) => this.questionMatchesOrigin(question, normalized))
    for (const question of pendingQuestions.slice(0, 3)) {
      const origin = this.questionOrigins.get(question.id)
      const elapsed = origin
        ? formatElapsed(Date.now() - origin.requestedAt)
        : 'unknown'
      lines.push([
        `Waiting question: ${compactStatusPreview(question.prompt)} for ${elapsed}`,
        `Question ID: ${question.id}`,
        'Reply in this chat to continue.',
      ].join('\n'))
    }
    if (pendingQuestions.length > 3) {
      lines.push(`Waiting question: ${pendingQuestions.length - 3} more question(s).`)
    }

    return lines
  }

  private async handleMemoryStatusCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey?: string,
  ): Promise<void> {
    const fileMemory = (this.runtime as { fileMemory?: ChannelFileMemory }).fileMemory
    let longTermChars = 0
    let todayChars = 0
    let yesterdayChars = 0
    let readError: string | undefined

    if (fileMemory) {
      try {
        const context = await fileMemory.getPromptContext()
        longTermChars = context.longTermMemory?.trim().length ?? 0
        todayChars = context.todayNote?.trim().length ?? 0
        yesterdayChars = context.yesterdayNote?.trim().length ?? 0
      } catch (error) {
        readError = error instanceof Error ? error.message : String(error)
      }
    }

    const text = [
      'Memory status:',
      `Long-term memory: ${longTermChars > 0 ? `available (${longTermChars} chars)` : 'empty or unavailable'}`,
      `Today note: ${todayChars > 0 ? `available (${todayChars} chars)` : 'empty or unavailable'}`,
      `Yesterday note: ${yesterdayChars > 0 ? `available (${yesterdayChars} chars)` : 'empty or unavailable'}`,
      `Semantic memory: ${this.runtime.semanticIndex ? 'enabled' : 'disabled'}`,
      readError ? `Read error: ${readError}` : undefined,
      '',
      'New sessions keep using long-term memory and recent daily notes.',
      'Save durable facts with /remember <내용>.',
    ].filter((line): line is string => line !== undefined).join('\n')

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      text,
      'memory_status',
      {
        longTermChars,
        todayChars,
        yesterdayChars,
        semanticEnabled: Boolean(this.runtime.semanticIndex),
        readError,
      },
    )
  }

  private async handleSelfCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    action: 'overview' | 'skills' | 'capabilities' | 'schedules' = 'overview',
    dedupeKey?: string,
  ): Promise<void> {
    const bound = await this.boundChannelSession(normalized)
    const cwd = bound?.session.cwd ?? process.cwd()
    const current = currentModelTarget(this.runtime)
    const modelLabel = current
      ? `${current.providerId} / ${current.modelId}`
      : 'not configured'
    const availableModels = this.describeAvailableModels()
    const skills = await this.runtime.skillRegistry.listForCwd(cwd).catch(() => [])
    const tools = this.runtime.toolRegistry.list().map((tool) => tool.name).sort()
    const chatKey = normalized.sessionKey
    const jobs = this.runtime.jobStore
      ? this.runtime.jobStore.list({
          status: ['pending'],
          ...(chatKey ? { channelTarget: chatKey } : {}),
        }).filter((job) => job.enabled)
      : []

    const skillLines = skills.slice(0, 12).map((skill) => (
      `- ${skill.id}@${skill.version}: ${truncateForTelegram(skill.description, 90)}`
    ))
    const toolLines = tools.slice(0, 24).map((tool) => `- ${tool}`)
    const scheduleLines = jobs
      .sort((a, b) => a.nextRunAt - b.nextRunAt)
      .slice(0, 8)
      .map((job) => `- ${job.id}: ${job.name} (${formatWhen(job)})`)

    const lines: string[] = []
    if (action === 'overview') {
      lines.push(
        'Agent snapshot:',
        `Device: ${this.runtime.config.device.name} (${this.runtime.config.device.role})`,
        `Autonomy: ${this.runtime.autonomy}`,
        `Data dir: ${this.runtime.dataDir}`,
        `Current model: ${modelLabel}`,
        `Workspace: ${cwd}`,
        `Installed skills: ${skills.length}`,
        `Registered tools: ${tools.length}`,
        `Scheduled tasks: ${jobs.length}`,
        '',
        'Limits:',
        '- I can only use registered tools and installed/enabled skills.',
        '- Skill installation requires explicit approval and validation.',
        '- Model switches are self-tested and rolled back on failure.',
      )
    } else if (action === 'skills') {
      lines.push(
        `Installed skills for ${cwd}: ${skills.length}`,
        skillLines.length ? skillLines.join('\n') : '(none)',
        skills.length > skillLines.length ? `... ${skills.length - skillLines.length} more` : '',
        '',
        'Find more with a natural request like "이 작업에 맞는 스킬 찾아줘"; install only after approval.',
      )
    } else if (action === 'capabilities') {
      lines.push(
        'Capabilities:',
        `Model: ${modelLabel}`,
        `Tools: ${tools.length}`,
        toolLines.length ? toolLines.join('\n') : '(none)',
        '',
        `Skills: ${skills.length}`,
        skillLines.length ? skillLines.join('\n') : '(none)',
        skills.length > skillLines.length ? `... ${skills.length - skillLines.length} more skills` : '',
      )
    } else {
      lines.push(
        chatKey ? 'Scheduled tasks for this chat:' : 'Scheduled tasks:',
        scheduleLines.length ? scheduleLines.join('\n') : '(none)',
      )
    }

    if (action === 'overview' && availableModels.length > 0) {
      lines.push('', 'Available models:', ...availableModels.slice(0, 12))
      if (availableModels.length > 12) lines.push(`... ${availableModels.length - 12} more`)
    }

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      lines.filter(Boolean).join('\n'),
      `self_${action}`,
      { cwd, model: current?.modelId, provider: current?.providerId },
    )
  }

  private describeAvailableModels(): string[] {
    return formatAvailableModels(this.runtime)
  }

  private async selfTestDefaultModel(): Promise<{ ok: boolean; latencyMs?: number; reason?: string }> {
    const target = currentModelTarget(this.runtime)
    if (!target) {
      const provider = this.runtime.providerRegistry?.getDefault()
      return {
        ok: false,
        reason: provider ? 'provider exposes no model' : 'no default provider after switch',
      }
    }
    const test = await selfTestProviderModel(this.runtime, target)
    return { ok: test.ok, latencyMs: test.latencyMs, reason: test.reason }
  }

  private async switchDefaultModel(providerId: string, modelId: string, reason: string): Promise<void> {
    await applyDefaultModel(this.runtime, { providerId, modelId }, reason)
  }

  private async handleModelCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    args: string | undefined,
    dedupeKey?: string,
    modelAction: 'current' | 'list' | 'switch' | 'pull' = args?.trim() ? 'switch' : 'current',
  ): Promise<void> {
    const channelType = normalized.message.channelType
    const senderId = normalized.message.sender?.id ?? ''
    const current = currentModelTarget(this.runtime)
    const currentProvider = current?.providerId ?? this.runtime.config?.agent?.defaultProvider ?? null
    const currentModel = current?.modelId ?? this.runtime.config?.agent?.defaultModel ?? null
    const currentLabel = `${currentProvider ?? '?'} / ${currentModel ?? '?'}`
    const available = this.describeAvailableModels()

    if (modelAction === 'current' || modelAction === 'list') {
      await this.completeCommand(
        channel, normalized, dedupeKey,
        [
          `Current model: ${currentLabel}`,
          available.length ? `\nAvailable:\n${available.join('\n')}` : '\n(no providers configured)',
          '\nSwitch with /model <model> or /model <provider> <model>. Pull Ollama models with /model pull <model>. Switches are self-tested; on failure the previous one is restored automatically.',
        ].join('\n'),
        modelAction === 'list' ? 'model_list' : 'model_current',
        { provider: currentProvider, model: currentModel },
      )
      return
    }

    const trust = this.runtime.channelAcl?.getTrustLevel(channelType, senderId)
    if (trust !== 'owner') {
      await this.completeCommand(
        channel, normalized, dedupeKey,
        'Switching the model is restricted to operators (channel owners). Ask an operator to run /model.',
        'model_denied',
        { trust: trust ?? 'unknown' },
      )
      return
    }

    if (modelAction === 'pull') {
      if (!args?.trim()) {
        await this.completeCommand(
          channel, normalized, dedupeKey,
          'Usage: /model pull <model> or /model pull <provider> <model>',
          'model_pull_usage',
          {},
        )
        return
      }
      try {
        const pulled = await pullOllamaModel(this.runtime, args.trim())
        await this.completeCommand(
          channel, normalized, dedupeKey,
          `${pulled.message}\nSwitch with /model ${pulled.providerId}/${pulled.model}.`,
          'model_pulled',
          {
            provider: pulled.providerId,
            model: pulled.model,
            latencyMs: pulled.latencyMs,
            alreadyAvailable: pulled.alreadyAvailable,
          },
        )
      } catch (err) {
        await this.completeCommand(
          channel, normalized, dedupeKey,
          `Failed to pull model: ${err instanceof Error ? err.message : String(err)}.`,
          'model_pull_error',
          {},
        )
      }
      return
    }

    if (!args?.trim()) {
      await this.completeCommand(
        channel, normalized, dedupeKey,
        'Usage: /model <model> or /model <provider> <model>',
        'model_switch_usage',
        {},
      )
      return
    }

    const resolution = resolveConfiguredModelTarget(this.runtime, args.trim())
    if (resolution.status !== 'matched') {
      const detail = resolution.status === 'ambiguous'
        ? `Ambiguous model "${resolution.query}". Use provider/model.\n${resolution.matches.map((match) => `- ${match.providerId}/${match.modelId}`).join('\n')}`
        : `Unknown model: "${resolution.query}".`
      await this.completeCommand(
        channel, normalized, dedupeKey,
        [
          detail,
          available.length ? `Available:\n${available.join('\n')}` : '(no providers configured)',
        ].join('\n'),
        resolution.status === 'ambiguous' ? 'model_ambiguous' : 'model_unknown',
        {},
      )
      return
    }
    const { providerId, modelId } = resolution.target
    if (providerId === currentProvider && modelId === currentModel) {
      await this.completeCommand(channel, normalized, dedupeKey, `Already using ${providerId} / ${modelId}.`, 'model_noop', {})
      return
    }

    try {
      await this.switchDefaultModel(providerId, modelId, 'channel.model.switch')
    } catch (err) {
      await this.completeCommand(
        channel, normalized, dedupeKey,
        `Failed to apply model switch: ${err instanceof Error ? err.message : String(err)} (still on ${currentLabel}).`,
        'model_switch_error',
        {},
      )
      return
    }

    const test = await this.selfTestDefaultModel()
    if (test.ok) {
      await this.completeCommand(
        channel, normalized, dedupeKey,
        `Switched to ${providerId} / ${modelId} (self-test ${test.latencyMs}ms).`,
        'model_switched',
        { provider: providerId, model: modelId, latencyMs: test.latencyMs },
      )
      return
    }

    if (!currentProvider || !currentModel) {
      await this.completeCommand(
        channel, normalized, dedupeKey,
        `Self-test failed for ${providerId} / ${modelId} (${test.reason}); no previous default model was available to restore.`,
        'model_self_test_failed_no_previous',
        { attempted: `${providerId}/${modelId}`, reason: test.reason },
      )
      return
    }

    try {
      await this.switchDefaultModel(currentProvider, currentModel, 'channel.model.rollback')
      await this.completeCommand(
        channel, normalized, dedupeKey,
        `Self-test failed for ${providerId} / ${modelId} (${test.reason}); reverted to ${currentLabel}.`,
        'model_rolled_back',
        { attempted: `${providerId}/${modelId}`, reason: test.reason, revertedTo: currentLabel },
      )
    } catch (rollbackErr) {
      await this.completeCommand(
        channel, normalized, dedupeKey,
        `Self-test failed AND rollback failed — the daemon is now on ${providerId} / ${modelId} despite the failure; fix it manually. (test: ${test.reason}; rollback: ${rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr)})`,
        'model_rollback_failed',
        {},
      )
    }
  }

  private async handleNewSessionCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    dedupeKey?: string,
  ): Promise<void> {
    if (!normalized.sessionKey || !this.runtime.channelSessionStore) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'This channel does not support session reset.',
        'session_command_unavailable',
      )
      return
    }

    const binding = await this.runtime.channelSessionStore.get(normalized.sessionKey)
    await this.runtime.channelSessionStore.delete(normalized.sessionKey)
    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      binding
        ? 'Started a fresh session for this chat. The previous session is still available in history. Long-term memory is unchanged and will be loaded into the next run.'
        : 'No active session was attached. Your next message will start a fresh session. Long-term memory is unchanged.',
      'session_reset',
      { previousSessionId: binding?.sessionId },
    )
  }

  private async handleListSessionsCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    query: string | undefined,
    dedupeKey?: string,
  ): Promise<void> {
    const result = await this.runtime.sessions.list({
      page: 1,
      perPage: SESSION_LIST_LIMIT,
      query,
    })
    if (result.items.length === 0) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        query
          ? `No sessions matched "${query}".`
          : 'No previous sessions found.',
        'session_list_empty',
        { query },
      )
      return
    }

    const suffix = result.totalCount > result.items.length
      ? `\n...and ${result.totalCount - result.items.length} more. Narrow with /sessions <query>.`
      : ''
    const text = [
      query
        ? `Sessions matching "${query}":`
        : 'Recent sessions:',
      ...result.items.map(formatSessionLine),
      '',
      'Use /resume <id> to attach this chat to a session.',
    ].join('\n') + suffix
    await this.completeCommand(channel, normalized, dedupeKey, text, 'session_list', {
      query,
      count: result.items.length,
      totalCount: result.totalCount,
    })
  }

  private async handleResumeSessionCommand(
    channel: IChannel,
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
    sessionRef: string | undefined,
    dedupeKey?: string,
  ): Promise<void> {
    if (!sessionRef) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'Usage: /resume <session-id-or-prefix>',
        'session_resume_usage',
      )
      return
    }
    if (!normalized.sessionKey || !this.runtime.channelSessionStore) {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        'This channel does not support session resume.',
        'session_command_unavailable',
      )
      return
    }

    const resolved = await this.resolveSessionRef(sessionRef)
    if (resolved.status === 'not_found') {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `No session matched "${sessionRef}". Use /sessions to list recent sessions.`,
        'session_resume_not_found',
        { sessionRef },
      )
      return
    }
    if (resolved.status === 'ambiguous') {
      const text = [
        `Multiple sessions matched "${sessionRef}":`,
        ...resolved.matches.map(formatSessionLine),
        '',
        'Use a longer session id prefix.',
      ].join('\n')
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        text,
        'session_resume_ambiguous',
        { sessionRef, count: resolved.matches.length },
      )
      return
    }

    let session = resolved.session
    if (session.status !== 'active' && this.runtime.sessions.updateMeta) {
      session = await this.runtime.sessions.updateMeta(session.id, { status: 'active' }) ?? session
    }
    if (session.status === 'completed') {
      await this.completeCommand(
        channel,
        normalized,
        dedupeKey,
        `Session ${session.id} is completed and this session store cannot reactivate it.`,
        'session_resume_completed',
        { sessionId: session.id },
      )
      return
    }

    await this.runtime.channelSessionStore.bind({
      key: normalized.sessionKey,
      sessionId: session.id,
      channelType: normalized.message.channelType,
      channelId: normalized.message.channelId,
      updatedAt: normalized.message.timestamp,
    })

    await this.completeCommand(
      channel,
      normalized,
      dedupeKey,
      [
        'Resumed session for this chat:',
        `ID: ${session.id}`,
        `Title: ${session.title}`,
        `Messages: ${session.messageCount}`,
      ].join('\n'),
      'session_resumed',
      { sessionId: session.id },
    )
  }

  private async currentBinding(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ) {
    if (!normalized.sessionKey || !this.runtime.channelSessionStore) {
      return null
    }
    return this.runtime.channelSessionStore.get(normalized.sessionKey)
  }

  private async boundChannelSession(
    normalized: ReturnType<ChannelMessageNormalizer['normalize']>,
  ): Promise<{ binding: ChannelSessionBinding; session: SessionMeta } | null> {
    const binding = await this.currentBinding(normalized)
    if (!binding) return null

    const session = await this.runtime.sessions.get(binding.sessionId)
    if (!session) {
      if (normalized.sessionKey && this.runtime.channelSessionStore) {
        await this.runtime.channelSessionStore.delete(normalized.sessionKey)
      }
      return null
    }

    return { binding, session }
  }

  private async resolveSessionRef(
    sessionRef: string,
  ): Promise<
    | { status: 'found'; session: SessionMeta }
    | { status: 'not_found' }
    | { status: 'ambiguous'; matches: SessionMeta[] }
  > {
    const ref = sessionRef.trim()
    const exact = await this.runtime.sessions.get(ref)
    if (exact) {
      return { status: 'found', session: exact }
    }

    const sessions = await this.runtime.sessions.list({ page: 1, perPage: 200 })
    const matches = sessions.items.filter((session) => session.id.startsWith(ref))
    if (matches.length === 0) {
      return { status: 'not_found' }
    }
    if (matches.length > 1) {
      return { status: 'ambiguous', matches: matches.slice(0, SESSION_LIST_LIMIT) }
    }
    return { status: 'found', session: matches[0]! }
  }

  private async measureStage<T>(
    stage: ChannelPipelineStage,
    action: () => T | Promise<T>,
    channelType?: string,
  ): Promise<T> {
    const startedAt = performance.now()
    try {
      return await action()
    } finally {
      this.runtime.channelPipelineMonitor?.recordStage(
        stage,
        performance.now() - startedAt,
        new Date(),
        channelType,
      )
    }
  }
}

/**
 * Handle /memory or /memory_<verb> slash commands sent over a channel
 * (Telegram, Slack, etc.). Returns the user-facing reply text. The
 * command is fully scope-aware: list / search / hot / forget only see
 * memories that belong to the caller's scope tags. remember writes
 * with the caller's scope so the memory shows up in subsequent reads.
 *
 * Verbs (Telegram convention `/memory_X` and CLI convention
 * `/memory X` both supported):
 *   help             — usage
 *   search <query>   — top-N hits with snippets
 *   list [N]         — most-recent visible memories
 *   hot [N]          — top-N by access counter
 *   remember <text>  — append a new memory; scope tags auto-applied
 *   forget <id>      — delete a memory the caller owns
 */
export async function handleMemoryCommand(args: {
  semanticIndex: {
    add(entry: { id?: string; content: string; source: 'user' | 'conversation' | 'document' | 'skill'; tags: string[] }): Promise<void>
    get(id: string): Promise<{ id: string; content: string; source: string; tags: string[] } | null>
    delete(id: string): Promise<void>
    search(query: string, options?: { limit?: number; minScore?: number; type?: 'semantic' | 'keyword' | 'hybrid' }): Promise<Array<{ id: string; content: string; source: string; tags: string[]; score?: number }>>
    listRecent(limit: number): Promise<Array<{ id: string; content: string; source: string; tags: string[] }>>
    listHotMemories(limit: number): Promise<Array<{ id: string; content: string; source: string; tags: string[]; accessCount: number; lastAccessedAt: string | null }>>
    listAudit?(options?: { memoryId?: string; limit?: number }): Promise<Array<{ id: string; memoryId: string; action: string; actor: string; reason?: string; createdAt: string }>>
    pin?(id: string): Promise<void>
    unpin?(id: string): Promise<void>
    isPinned?(id: string): Promise<boolean | null>
    listPinned?(limit: number): Promise<Array<{ id: string; content: string; source: string; tags: string[]; pinnedAt: string | null }>>
    recordAccess?(ids: string[]): Promise<void>
    recordAudit?(input: { memoryId: string; action: 'created' | 'updated' | 'deleted' | 'pruned' | 'maintenance'; actor: string; reason?: string; before?: { id: string; content: string; source: string; tags: string[] }; after?: { id: string; content: string; source: string; tags: string[] } }): Promise<unknown>
  }
  text: string
  scopeTags: string[]
  actor?: string
}): Promise<string> {
  const trimmed = args.text.trim()
  const HELP = '사용법: /memory help | /memory_stats | /memory_search <query> | /memory_list [N] | /memory_hot [N] | /memory_remember <text> | /memory_forget <id> | /memory_audit <id> | /memory_pin <id> | /memory_unpin <id> | /memory_pinned [N]'
  const parsed = parseMemoryCommand(trimmed)
  if (!parsed) return HELP
  const { verb, rest } = parsed
  const scopeTags = args.scopeTags ?? []

  if (verb === 'help' || !verb) return HELP

  if (verb === 'stats') {
    let all: Awaited<ReturnType<typeof args.semanticIndex.listRecent>>
    try {
      all = await args.semanticIndex.listRecent(5000)
    } catch (error) {
      return `통계 조회 실패: ${error instanceof Error ? error.message : String(error)}`
    }
    const visible = filterVisibleMemories(all, scopeTags)
    const bySource = new Map<string, number>()
    for (const entry of visible) {
      bySource.set(entry.source, (bySource.get(entry.source) ?? 0) + 1)
    }
    let pinnedCount = 0
    if (args.semanticIndex.listPinned) {
      try {
        const pinned = await args.semanticIndex.listPinned(500)
        pinnedCount = filterVisibleMemories(pinned, scopeTags).length
      } catch {
        // best-effort
      }
    }
    const sourceLine = Array.from(bySource.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([src, n]) => `${src}: ${n}`)
      .join(', ')
    const scopeLabel = scopeTags.length === 0
      ? '전체(글로벌)'
      : scopeTags.find((tag) => tag.startsWith('scope:user:'))?.replace('scope:user:', 'user:')
        ?? scopeTags[0]
    return [
      `📊 메모리 현황 (${scopeLabel})`,
      `· 총 ${visible.length}건${sourceLine ? ` (${sourceLine})` : ''}`,
      `· 핀: ${pinnedCount}건`,
    ].join('\n')
  }

  if (verb === 'search') {
    const query = rest.trim()
    if (!query) return '검색어가 필요합니다. 예: /memory_search react hooks'
    let raw: Awaited<ReturnType<typeof args.semanticIndex.search>>
    try {
      raw = await args.semanticIndex.search(query, { limit: 20, type: 'hybrid' })
    } catch (error) {
      return `검색 실패: ${error instanceof Error ? error.message : String(error)}`
    }
    const visible = filterVisibleMemories(raw, scopeTags).slice(0, 5)
    if (visible.length === 0) return '일치하는 기억이 없습니다.'
    if (args.semanticIndex.recordAccess && visible.length > 0) {
      void args.semanticIndex.recordAccess(visible.map((entry) => entry.id)).catch(() => {})
    }
    return `🔍 ${visible.length}건:\n` + visible.map((entry, idx) =>
      `${idx + 1}. ${truncateForTelegram(entry.content)}\n   id: ${entry.id}`,
    ).join('\n')
  }

  if (verb === 'list') {
    const n = clampInt(parseInt(rest, 10) || 10, 1, 50)
    const all = await args.semanticIndex.listRecent(Math.min(n * 4, 200))
    const visible = filterVisibleMemories(all, scopeTags).slice(0, n)
    if (visible.length === 0) return '저장된 기억이 없습니다.'
    return `📜 최근 ${visible.length}건:\n` + visible.map((entry, idx) =>
      `${idx + 1}. ${truncateForTelegram(entry.content)}\n   id: ${entry.id}`,
    ).join('\n')
  }

  if (verb === 'hot') {
    const n = clampInt(parseInt(rest, 10) || 10, 1, 50)
    const raw = await args.semanticIndex.listHotMemories(Math.min(n * 4, 200))
    const visible = filterVisibleMemories(raw, scopeTags).slice(0, n)
    if (visible.length === 0) return '아직 자주 쓰인 기억이 없습니다.'
    return `🔥 자주 사용된 ${visible.length}건:\n` + visible.map((entry, idx) =>
      `${idx + 1}. ${truncateForTelegram(entry.content)}\n   id: ${entry.id} · ${(entry as { accessCount: number }).accessCount}회`,
    ).join('\n')
  }

  if (verb === 'remember') {
    const content = rest.trim()
    if (!content) return '기억할 내용을 입력하세요. 예: /memory_remember 사용자가 한국어 응답을 선호함'
    const id = `tg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const tags = ['explicit-memory', 'source:channel-command', ...scopeTags]
    try {
      await args.semanticIndex.add({ id, content, source: 'user', tags })
    } catch (error) {
      return `저장 실패: ${error instanceof Error ? error.message : String(error)}`
    }
    if (args.semanticIndex.recordAudit) {
      await args.semanticIndex.recordAudit({
        memoryId: id,
        action: 'created',
        actor: args.actor ?? 'channel-slash',
        reason: '/memory_remember',
        after: { id, content, source: 'user', tags },
      }).catch(() => {})
    }
    return `✅ 저장되었습니다. id: ${id}`
  }

  if (verb === 'pin' || verb === 'unpin') {
    const id = rest.trim()
    if (!id) return `대상 id가 필요합니다. 예: /memory_${verb} tg-1234`
    const fn = verb === 'pin' ? args.semanticIndex.pin : args.semanticIndex.unpin
    if (!fn) return '핀 기능이 활성화되어 있지 않습니다.'
    const existing = await args.semanticIndex.get(id)
    if (!existing) return `기억을 찾을 수 없습니다 (id: ${id})`
    if (scopeTags.length > 0 && !isMemoryVisibleInScope(existing.tags, scopeTags)) {
      return `기억을 찾을 수 없습니다 (id: ${id})`
    }
    try {
      await fn.call(args.semanticIndex, id)
    } catch (error) {
      return `${verb} 실패: ${error instanceof Error ? error.message : String(error)}`
    }
    if (args.semanticIndex.recordAudit) {
      await args.semanticIndex.recordAudit({
        memoryId: id,
        action: 'updated',
        actor: args.actor ?? 'channel-slash',
        reason: verb === 'pin' ? '/memory_pin' : '/memory_unpin',
        before: existing,
        after: existing,
      }).catch(() => {})
    }
    return verb === 'pin'
      ? `📌 핀 설정 완료 (id: ${id}). prune 면제됩니다.`
      : `📍 핀 해제 (id: ${id}).`
  }

  if (verb === 'pinned') {
    if (!args.semanticIndex.listPinned) return '핀 기능이 활성화되어 있지 않습니다.'
    const n = clampInt(parseInt(rest, 10) || 20, 1, 100)
    let raw: Awaited<ReturnType<NonNullable<typeof args.semanticIndex.listPinned>>>
    try {
      raw = await args.semanticIndex.listPinned(Math.min(n * 4, 200))
    } catch (error) {
      return `pinned 목록 조회 실패: ${error instanceof Error ? error.message : String(error)}`
    }
    const visible = filterVisibleMemories(raw, scopeTags).slice(0, n)
    if (visible.length === 0) return '핀이 설정된 기억이 없습니다.'
    return `📌 핀된 ${visible.length}건:\n` + visible.map((entry, idx) =>
      `${idx + 1}. ${truncateForTelegram(entry.content)}\n   id: ${entry.id}`,
    ).join('\n')
  }

  if (verb === 'audit') {
    const id = rest.trim()
    if (!id) return '조회할 기억의 id가 필요합니다. 예: /memory_audit tg-1234'
    if (!args.semanticIndex.listAudit) return '감사 로그가 활성화되어 있지 않습니다.'
    const existing = await args.semanticIndex.get(id)
    if (!existing) return `기억을 찾을 수 없습니다 (id: ${id})`
    if (scopeTags.length > 0 && !isMemoryVisibleInScope(existing.tags, scopeTags)) {
      return `기억을 찾을 수 없습니다 (id: ${id})`
    }
    let entries: Awaited<ReturnType<NonNullable<typeof args.semanticIndex.listAudit>>>
    try {
      entries = await args.semanticIndex.listAudit({ memoryId: id, limit: 20 })
    } catch (error) {
      return `감사 로그 조회 실패: ${error instanceof Error ? error.message : String(error)}`
    }
    if (entries.length === 0) return `감사 기록이 없습니다 (id: ${id}).`
    const lines = entries.slice(0, 10).map((entry, idx) => {
      const reason = entry.reason ? ` · ${truncateForTelegram(entry.reason)}` : ''
      const ts = entry.createdAt.replace('T', ' ').replace(/\..+$/, '')
      return `${idx + 1}. ${entry.action} · ${ts} · ${entry.actor}${reason}`
    })
    return `📋 ${id} 감사 기록 (${entries.length}):\n${lines.join('\n')}`
  }

  if (verb === 'forget') {
    const id = rest.trim()
    if (!id) return '삭제할 기억의 id가 필요합니다. 예: /memory_forget tg-1234'
    const existing = await args.semanticIndex.get(id)
    if (!existing) return `기억을 찾을 수 없습니다 (id: ${id})`
    if (scopeTags.length > 0 && !isMemoryVisibleInScope(existing.tags, scopeTags)) {
      return `기억을 찾을 수 없습니다 (id: ${id})`
    }
    try {
      await args.semanticIndex.delete(id)
    } catch (error) {
      return `삭제 실패: ${error instanceof Error ? error.message : String(error)}`
    }
    if (args.semanticIndex.recordAudit) {
      await args.semanticIndex.recordAudit({
        memoryId: id,
        action: 'deleted',
        actor: args.actor ?? 'channel-slash',
        reason: '/memory_forget',
        before: existing,
      }).catch(() => {})
    }
    return `🗑️ 삭제했습니다 (id: ${id}).`
  }

  return HELP
}

function parseMemoryCommand(text: string): { verb: string; rest: string } | null {
  // Accepts both "/memory_<verb> <rest>" and "/memory <verb> <rest>".
  const lower = text.toLowerCase()
  if (!lower.startsWith('/memory')) return null

  // Telegram convention: /memory_search react hooks → verb='search'
  const underscore = lower.match(/^\/memory_([a-z]+)(\s|$)/)
  if (underscore) {
    const verb = underscore[1]
    const head = `/memory_${verb}`
    const rest = text.slice(head.length).trim()
    return { verb, rest }
  }

  // Bare /memory or /memory help → list-style help.
  if (lower === '/memory' || lower.startsWith('/memory ')) {
    const rest = text.slice('/memory'.length).trim()
    if (!rest) return { verb: 'help', rest: '' }
    const space = rest.indexOf(' ')
    const verb = (space === -1 ? rest : rest.slice(0, space)).toLowerCase()
    const tail = space === -1 ? '' : rest.slice(space + 1).trim()
    return { verb, rest: tail }
  }
  return null
}

function filterVisibleMemories<T extends { tags: string[] }>(
  entries: T[],
  scopeTags: string[],
): T[] {
  return entries.filter((entry) => {
    if (entry.tags.includes('archived')) return false
    if (entry.tags.some((tag) => tag.startsWith('superseded'))) return false
    if (scopeTags.length === 0) return true
    return isMemoryVisibleInScope(entry.tags, scopeTags)
  })
}

function truncateForTelegram(content: string, max = 120): string {
  const cleaned = content.replace(/\s+/g, ' ').trim()
  if (cleaned.length <= max) return cleaned
  return `${cleaned.slice(0, Math.max(0, max - 1))}…`
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.max(min, Math.min(max, Math.floor(value)))
}

function isScheduleCommand(text: string): boolean {
  return SCHEDULE_COMMAND_PATTERN.test(text.trim())
}

function compactScheduleAliasText(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[\s"'`*_~\-–—.,!?！？。…。·:;()[\]{}]/gu, '')
}

function resolveScheduleCommandText(text: string): string | null {
  const trimmed = text.trim()
  if (isScheduleCommand(trimmed)) return trimmed

  const compact = compactScheduleAliasText(trimmed)
  if (!compact) return null

  const mentionsSchedule = /(schedule|schedules|reminder|스케줄|스케쥴|스케듈|예약|리마인더)/u.test(compact)
  const asksForList = /(list|리스트|목록|조회|확인|보여|알려|뭐있|뭐가있)/u.test(compact)
  if (mentionsSchedule && asksForList) {
    const asksForAll = /(all|전체|모두|전부)/u.test(compact)
    return asksForAll ? '/schedule list all' : '/schedule list'
  }

  const asksDailyConfirmation = /매일/u.test(compact)
    && /(알려주는거지|알려주는거맞|알려주는게맞|알려주는거야|알림맞|반복되는거|반복이야|매일해주는거|매일되는거|매일맞|매일이야|매일예약)/u.test(compact)
  return asksDailyConfirmation ? '/schedule list' : null
}

function parseScheduleCommand(text: string): { subcommand: string; rest: string } | null {
  const match = text.trim().match(SCHEDULE_COMMAND_PATTERN)
  if (!match) return null
  const body = (match[1] ?? '').trim()
  if (!body) return { subcommand: 'list', rest: '' }
  const [head = '', ...tail] = body.split(/\s+/)
  return { subcommand: head.toLowerCase(), rest: tail.join(' ').trim() }
}

function scheduleHelpText(): string {
  return [
    '사용법:',
    '/schedule list [all]',
    '/schedule add <when> -- <instruction>',
    '/schedule edit <id> <when> -- <instruction>',
    '/schedule reschedule <id> <when>',
    '/schedule show <id>',
    '/schedule pause <id>',
    '/schedule resume <id>',
    '/schedule run <id>',
    '/schedule runs <id> [limit|run-id]',
    '/schedule cancel <id>',
    '/schedule cancel all confirm',
    '',
    'id는 /schedule list에 보이는 앞 8자 prefix도 사용할 수 있습니다.',
    '예: /schedule add 2분 후 -- 상태를 확인해서 알려줘',
  ].join('\n')
}

function splitScheduleCreateInput(rest: string): { when: string; instruction: string } | null {
  const delimiters = [' -- ', ' | ']
  const candidates = delimiters
    .map((delimiter) => ({ delimiter, index: rest.indexOf(delimiter) }))
    .filter((candidate) => candidate.index >= 0)
    .sort((a, b) => a.index - b.index)
  const first = candidates[0]
  if (!first) return null
  const when = rest.slice(0, first.index).trim()
  const instruction = rest.slice(first.index + first.delimiter.length).trim()
  if (!when || !instruction) return null
  return { when, instruction }
}

function splitScheduleEditInput(rest: string): { id: string; when?: string; instruction?: string } | null {
  const trimmed = rest.trim()
  const firstSpace = trimmed.search(/\s/u)
  const id = firstSpace === -1 ? trimmed : trimmed.slice(0, firstSpace)
  const body = firstSpace === -1 ? '' : trimmed.slice(firstSpace + 1).trim()
  if (!id || !body) return null

  const delimiters = [' -- ', ' | ']
  const candidates = delimiters
    .map((delimiter) => ({ delimiter, index: body.indexOf(delimiter) }))
    .filter((candidate) => candidate.index >= 0)
    .sort((a, b) => a.index - b.index)
  const first = candidates[0]
  if (!first) return { id, when: body }

  const when = body.slice(0, first.index).trim() || undefined
  const instruction = body.slice(first.index + first.delimiter.length).trim() || undefined
  if (!when && !instruction) return null
  return { id, when, instruction }
}

function scheduleStatusLabel(job: ScheduledJob): string {
  if (job.status !== 'pending') return job.status
  return job.enabled ? 'pending' : 'paused'
}

function scheduleShortId(job: ScheduledJob): string {
  return job.id.slice(0, 8)
}

function formatScheduleListItem(job: ScheduledJob, index: number): string {
  const type = job.kind === 'recurring' ? 'recurring' : 'one-shot'
  const when = formatWhen(job)
  return `${index + 1}. ${scheduleShortId(job)} · ${type} · ${scheduleStatusLabel(job)} · ${when}\n   ${truncateForTelegram(job.instruction)}`
}

function formatScheduleDetail(job: ScheduledJob): string {
  const lines = [
    `id: ${job.id}`,
    `상태: ${scheduleStatusLabel(job)}`,
    `종류: ${job.kind === 'recurring' ? 'recurring' : 'one-shot'}`,
    `일정: ${formatWhen(job)}`,
    `다음 실행: ${hasPendingScheduledRun(job) ? new Date(job.nextRunAt).toISOString() : '—'}`,
    job.lastRunAt ? `최근 실행: ${new Date(job.lastRunAt).toISOString()}` : '최근 실행: 없음',
    `재시도: ${job.maxAttempts}회, backoff ${job.retryBackoffMs}ms`,
  ]
  if (job.timezone) lines.push(`timezone: ${job.timezone}`)
  if (job.lastError) lines.push(`최근 오류: ${truncateForTelegram(job.lastError)}`)
  lines.push(`작업: ${truncateForTelegram(job.instruction)}`)
  return lines.join('\n')
}

function formatScheduleRun(run: JobRun, index: number): string {
  const finished = run.finishedAt ? new Date(run.finishedAt).toISOString() : 'running'
  const duration = run.durationMs == null ? '' : ` · ${run.durationMs}ms`
  const detail = run.error ?? run.outputExcerpt
  const status = run.taskOutcome === 'incomplete' ? 'incomplete' : run.status
  const integrity = run.statusIntegrity === 'legacy-incomplete-conflict'
    ? ' · legacy stored status: success'
    : ''
  return detail
    ? `${index + 1}. ${status}${integrity} · ${finished}${duration} · run: ${run.id}\n   ${truncateForTelegram(detail)}`
    : `${index + 1}. ${status}${integrity} · ${finished}${duration} · run: ${run.id}`
}

function parseScheduleRunLimit(rest: string): number {
  const token = rest.split(/\s+/)[1]
  const parsed = token ? Number.parseInt(token, 10) : 5
  return clampInt(Number.isFinite(parsed) ? parsed : 5, 1, 20)
}

function parseScheduleRunId(rest: string): string | null {
  const token = rest.split(/\s+/)[1]?.trim()
  if (!token || /^\d+$/u.test(token)) return null
  return token
}

function resolveChatScheduleJob(args: {
  store: JobStore
  chatKey: string
  rawId: string
}): { job: ScheduledJob } | { reply: string } {
  const rawId = args.rawId.trim()
  if (!rawId) return { reply: '예약 id를 입력하세요. /schedule list로 id를 확인할 수 있습니다.' }

  const exact = args.store.get(rawId)
  if (exact) {
    if (exact.channelTarget !== args.chatKey) {
      return { reply: `예약을 찾을 수 없습니다 (id: ${rawId})` }
    }
    return { job: exact }
  }

  if (rawId.length < 4) {
    return { reply: 'id prefix는 최소 4자 이상 입력하세요. /schedule list에 보이는 8자를 쓰면 됩니다.' }
  }

  const matches = args.store
    .list({ channelTarget: args.chatKey })
    .filter((job) => job.id.startsWith(rawId))

  if (matches.length === 0) return { reply: `예약을 찾을 수 없습니다 (id: ${rawId})` }
  if (matches.length > 1) {
    return {
      reply: `id prefix가 여러 예약과 일치합니다: ${matches.slice(0, 5).map(scheduleShortId).join(', ')}`,
    }
  }
  return { job: matches[0]! }
}

function updateChatScheduleJob(args: {
  store: JobStore
  job: ScheduledJob
  when?: string
  instruction?: string
  defaultTimezone?: string
}): { job: ScheduledJob } | { reply: string } {
  const instruction = args.instruction ?? args.job.instruction
  if (!instruction.trim()) return { reply: '작업 내용은 비워둘 수 없습니다.' }

  if (!args.when) {
    if (args.job.kind === 'recurring' && args.job.cron) {
      const updated = args.store.updateRecurringJob({
        id: args.job.id,
        name: instruction.slice(0, 60),
        cron: args.job.cron,
        nextRunAt: args.job.nextRunAt,
        timezone: args.job.timezone,
        instruction,
        enabled: args.job.enabled,
      })
      return updated ? { job: updated } : { reply: `예약을 수정하지 못했습니다 (id: ${scheduleShortId(args.job)}).` }
    }
    if (args.job.kind === 'oneshot' && args.job.runAt) {
      const updated = args.store.updateOneShotJob({
        id: args.job.id,
        name: instruction.slice(0, 60),
        runAt: args.job.runAt,
        nextRunAt: args.job.nextRunAt,
        instruction,
        enabled: args.job.enabled,
      })
      return updated ? { job: updated } : { reply: `예약을 수정하지 못했습니다 (id: ${scheduleShortId(args.job)}).` }
    }
    return { reply: '기존 일정 정보를 확인할 수 없어 수정하지 못했습니다.' }
  }

  let parsed: ReturnType<typeof parseWhen>
  const timezone = args.job.timezone ?? args.defaultTimezone
  try {
    parsed = parseWhen(args.when, { timezone })
  } catch (error) {
    const message = error instanceof SchedulerParseError
      ? error.message
      : error instanceof Error
        ? error.message
        : String(error)
    return { reply: `일정을 해석하지 못했습니다: ${message}` }
  }

  if (parsed.kind !== args.job.kind) {
    return {
      reply: `기존 예약은 ${args.job.kind}입니다. one-shot/recurring 종류 변경은 /schedule cancel 후 /schedule add로 새로 만드세요.`,
    }
  }

  if (parsed.kind === 'recurring') {
    const updated = args.store.updateRecurringJob({
      id: args.job.id,
      name: instruction.slice(0, 60),
      cron: parsed.cron,
      nextRunAt: parsed.nextRunAt,
      timezone: timezone ?? null,
      instruction,
      enabled: args.job.enabled,
    })
    return updated ? { job: updated } : { reply: `예약을 수정하지 못했습니다 (id: ${scheduleShortId(args.job)}).` }
  }

  const updated = args.store.updateOneShotJob({
    id: args.job.id,
    name: instruction.slice(0, 60),
    runAt: parsed.runAt,
    nextRunAt: parsed.runAt,
    instruction,
    enabled: args.job.enabled,
  })
  return updated ? { job: updated } : { reply: `예약을 수정하지 못했습니다 (id: ${scheduleShortId(args.job)}).` }
}

export async function handleScheduleCommand(args: {
  store: JobStore
  text: string
  chatKey: string
  channelType?: string
  replyToMessageId?: string
  defaultTimezone?: string
  triggerSchedulerJob?: (id: string) => Promise<ManualJobRunResult>
}): Promise<string> {
  if (!args.chatKey) {
    return '채팅 컨텍스트를 확인할 수 없어 /schedule 명령을 처리할 수 없습니다.'
  }

  const command = parseScheduleCommand(args.text)
  if (!command) return scheduleHelpText()

  const sub = command.subcommand
  const rest = command.rest
  if (sub === 'help' || sub === 'usage') return scheduleHelpText()

  if (sub === 'list' || sub === 'ls' || sub === 'all') {
    const all = sub === 'all' || rest.toLowerCase() === 'all'
    const jobs = args.store.list({
      status: all ? undefined : ['pending'],
      channelTarget: args.chatKey,
    })
    if (jobs.length === 0) return '예약된 작업이 없습니다.'
    const lines = jobs.slice(0, 20).map(formatScheduleListItem)
    const suffix = jobs.length > 20 ? `\n... ${jobs.length - 20}개 더 있음` : ''
    return `📅 예약된 작업 (${jobs.length}):\n${lines.join('\n')}${suffix}\n관리: /schedule show <id> | /schedule cancel <id>`
  }

  if (sub === 'add' || sub === 'create' || sub === 'new') {
    const input = splitScheduleCreateInput(rest)
    if (!input) {
      return '예약 생성 형식: /schedule add <when> -- <instruction>\n예: /schedule add 2분 후 -- 상태를 확인해서 알려줘'
    }
    let parsed: ReturnType<typeof parseWhen>
    try {
      parsed = parseWhen(input.when, { timezone: args.defaultTimezone })
    } catch (error) {
      const message = error instanceof SchedulerParseError
        ? error.message
        : error instanceof Error
          ? error.message
          : String(error)
      return `일정을 해석하지 못했습니다: ${message}`
    }
    const job = args.store.create({
      name: input.instruction.slice(0, 60),
      kind: parsed.kind,
      cron: parsed.kind === 'recurring' ? parsed.cron : null,
      runAt: parsed.kind === 'oneshot' ? parsed.runAt : null,
      nextRunAt: parsed.kind === 'oneshot' ? parsed.runAt : parsed.nextRunAt,
      timezone: parsed.kind === 'recurring' ? (args.defaultTimezone ?? null) : null,
      instruction: input.instruction,
      channelType: args.channelType ?? null,
      channelTarget: args.chatKey,
      replyToMessageId: args.replyToMessageId ?? null,
      parentSessionId: null,
      enabled: true,
      createdBy: 'channel-cmd',
    })
    return `예약했습니다.\nid: ${scheduleShortId(job)}\n일정: ${formatWhen(job)}\n작업: ${truncateForTelegram(job.instruction)}`
  }

  if (sub === 'cancel' || sub === 'delete' || sub === 'remove' || sub === 'rm') {
    const [target, confirm] = rest.split(/\s+/)
    if (target === 'all') {
      const jobs = args.store.list({ status: ['pending'], channelTarget: args.chatKey })
      if (confirm !== 'confirm') {
        return `${jobs.length}개의 pending 예약이 있습니다. 모두 취소하려면 /schedule cancel all confirm 을 보내세요.`
      }
      for (const j of jobs) args.store.cancel(j.id)
      return `${jobs.length}개의 예약을 취소했습니다.`
    }
    const resolved = resolveChatScheduleJob({ store: args.store, chatKey: args.chatKey, rawId: target ?? '' })
    if ('reply' in resolved) return resolved.reply
    args.store.cancel(resolved.job.id)
    return `예약을 취소했습니다 (id: ${scheduleShortId(resolved.job)}).`
  }

  if (sub === 'show' || sub === 'info' || sub === 'get') {
    const resolved = resolveChatScheduleJob({ store: args.store, chatKey: args.chatKey, rawId: rest })
    if ('reply' in resolved) return resolved.reply
    return formatScheduleDetail(resolved.job)
  }

  if (sub === 'edit' || sub === 'update' || sub === 'reschedule') {
    const input = splitScheduleEditInput(rest)
    if (!input) {
      return '예약 수정 형식: /schedule edit <id> <when> -- <instruction>\n일정만 변경: /schedule reschedule <id> <when>'
    }
    const resolved = resolveChatScheduleJob({ store: args.store, chatKey: args.chatKey, rawId: input.id })
    if ('reply' in resolved) return resolved.reply
    const result = updateChatScheduleJob({
      store: args.store,
      job: resolved.job,
      when: input.when,
      instruction: sub === 'reschedule' ? undefined : input.instruction,
      defaultTimezone: args.defaultTimezone,
    })
    if ('reply' in result) return result.reply
    return `예약을 수정했습니다.\nid: ${scheduleShortId(result.job)}\n일정: ${formatWhen(result.job)}\n작업: ${truncateForTelegram(result.job.instruction)}`
  }

  if (sub === 'pause' || sub === 'disable') {
    const resolved = resolveChatScheduleJob({ store: args.store, chatKey: args.chatKey, rawId: rest })
    if ('reply' in resolved) return resolved.reply
    if (resolved.job.status !== 'pending') {
      return `pending 예약만 일시정지할 수 있습니다 (현재 상태: ${resolved.job.status}).`
    }
    args.store.setEnabled(resolved.job.id, false)
    return `예약을 일시정지했습니다 (id: ${scheduleShortId(resolved.job)}).`
  }

  if (sub === 'resume' || sub === 'enable') {
    const resolved = resolveChatScheduleJob({ store: args.store, chatKey: args.chatKey, rawId: rest })
    if ('reply' in resolved) return resolved.reply
    if (resolved.job.status !== 'pending') {
      return `pending 예약만 재개할 수 있습니다 (현재 상태: ${resolved.job.status}).`
    }
    args.store.setEnabled(resolved.job.id, true)
    return `예약을 재개했습니다 (id: ${scheduleShortId(resolved.job)}).`
  }

  if (sub === 'run' || sub === 'now' || sub === 'run-now') {
    if (!args.triggerSchedulerJob) return '수동 실행 기능이 아직 초기화되지 않았습니다.'
    const resolved = resolveChatScheduleJob({ store: args.store, chatKey: args.chatKey, rawId: rest })
    if ('reply' in resolved) return resolved.reply
    try {
      const result = await args.triggerSchedulerJob(resolved.job.id)
      if (!result.started) {
        return `예약이 현재 실행 가능하지 않아 지금 실행하지 못했습니다 (id: ${result.jobId}, 활성: ${resolved.job.enabled}, 상태: ${resolved.job.status}).`
      }
      if (result.status === 'failed') {
        return `예약을 지금 실행했지만 실패했습니다 (id: ${result.jobId}, run: ${result.runId}).`
      }
      return `예약을 지금 실행했습니다 (id: ${result.jobId}, run: ${result.runId}).`
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return `수동 실행에 실패했습니다: ${message}`
    }
  }

  if (sub === 'runs' || sub === 'history' || sub === 'logs') {
    const id = rest.split(/\s+/)[0] ?? ''
    const resolved = resolveChatScheduleJob({ store: args.store, chatKey: args.chatKey, rawId: id })
    if ('reply' in resolved) return resolved.reply
    const runId = parseScheduleRunId(rest)
    if (runId) {
      const run = args.store.getRun(resolved.job.id, runId)
      if (!run) return `실행 기록을 찾을 수 없습니다 (id: ${resolved.job.id}, run: ${runId}).`
      return `실행 기록 (${resolved.job.id}, run: ${run.id}):\n${formatScheduleRun(run, 0)}`
    }
    const limit = parseScheduleRunLimit(rest)
    const runs = args.store.listRuns(resolved.job.id, limit)
    if (runs.length === 0) return `실행 기록이 없습니다 (id: ${resolved.job.id}).`
    return `실행 기록 (${resolved.job.id}):\n${runs.map(formatScheduleRun).join('\n')}`
  }

  return scheduleHelpText()
}

export const __testables = {
  approvalApprovedForAction,
  approvalDecisionForAction,
  cleanWorkspacePathCandidate,
  compactStatusPreview,
  formatActiveRunsStatus,
  formatActiveRunLine,
  formatApprovalResolutionText,
  formatApprovalToolInput,
  formatBlockingRunBusyText,
  formatElapsed,
  formatRunAttentionLine,
  channelSafeAgentResponse,
  imageGenAttachmentsFromText,
  formatSessionLine,
  clampInt,
  filterVisibleMemories,
  isScheduleCommand,
  resolveScheduleCommandText,
  parseApprovalIntentClassifierResponse,
  parseApprovalCommand,
  parseApprovalSymbol,
  parseFeedbackCommand,
  parseFeedbackRatingToken,
  parseFeedbackReaction,
  parseMemoryCommand,
  parseModelCommandArgs,
  parseSessionCommand,
  parseSkillsCommand,
  parseSkillShortcutCommand,
  parseTpsCommandArgs,
  parseWorkspaceCommand,
  formatScheduleDetail,
  formatScheduleListItem,
  formatScheduleRun,
  parseScheduleCommand,
  parseScheduleRunLimit,
  parseScheduleRunId,
  resolveChatScheduleJob,
  splitScheduleCreateInput,
  splitScheduleEditInput,
  truncateForTelegram,
  updateChatScheduleJob,
}
