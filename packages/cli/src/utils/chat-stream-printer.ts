import chalk from 'chalk'
import { isUnsuccessfulAgentResult } from './run-outcome.js'
import {
  createTerminalChatStreamFrameConsumer,
  describeStopReason,
  shouldRenderStopCard,
  type DaemonChatStreamPayload,
  type DaemonPendingDecision,
  type RunStopLocale,
  type RunStopReason,
  type TerminalChatStreamFrame,
} from '@sepilotd/api-client'
import { formatRunStopCard, type RunStopHintMode } from '../tui/utils/run-stop-card.js'

interface Writer {
  write(text: string): void
}

interface BaseCliChatStreamPrinterOptions {
  onSessionId?: (sessionId: string) => void
  stdout?: Writer
  stderr?: Writer
  /** Fires when the daemon emits an `approval_request` frame so the host
   * shell can update a pending-counter or visual cue. */
  onApprovalRequested?: (requestId: string, toolName: string, sessionId?: string) => void
  /** Fires when the daemon emits a `question_request` frame so the host
   * shell can update a pending-counter or visual cue. */
  onQuestionRequested?: (questionId: string, sessionId?: string) => void
}

export interface InteractiveCliChatStreamPrinterOptions extends BaseCliChatStreamPrinterOptions {
  showArtifacts?: boolean
  showStateChanges?: boolean
  /** Show router, planner, reasoning, and per-LLM diagnostics. Defaults to true for API compatibility. */
  showDiagnostics?: boolean
  /**
   * How to spell approval hints.
   * - 'shell' (default): interactive chat shell slash commands.
   * - 'cli': top-level one-shot/run commands where slash commands are unavailable.
   */
  approvalHintMode?: 'shell' | 'cli'
  /** How to spell pending-question answer hints. Defaults to chat-shell slash commands. */
  questionHintMode?: 'shell' | 'cli'
}

export interface AnswerOnlyCliChatStreamPrinterOptions extends BaseCliChatStreamPrinterOptions {
  quiet?: boolean
  suppressContent?: boolean
  /**
   * How to spell the resolution hint after an `[approval: ...]` banner.
   * - `'shell'` (default): the user is sitting at the chat shell, so
   *   slash commands work — `/approve <id>` / `/deny <id>`.
   * - `'cli'`: the surrounding context is a piped/non-interactive
   *   `sepilot ask` run, so point at the top-level cli verbs that
   *   resolve the same daemon endpoint — `sepilot approve <id>` /
   *   `sepilot deny <id>`. Without this an automation user reads a
   *   slash hint and has no shell to type it into.
   */
  approvalHintMode?: 'shell' | 'cli'
  /** How to spell pending-question answer hints. Defaults to top-level cli commands. */
  questionHintMode?: 'shell' | 'cli'
}

export interface CliChatStreamPrinter {
  handleEvent: (event: DaemonChatStreamPayload) => void
  getContent: () => string
  hadError: () => boolean
  errorMessage: () => string | undefined
}

function truncate(text: string, limit: number): string {
  return text.length > limit ? text.slice(0, limit) : text
}

const TERMINAL_SECRET_ASSIGNMENT_RE =
  /((?:["']?[A-Za-z0-9_.-]*(?:api[_-]?key|authorization|credential|password|passwd|private[_-]?key|secret|token)[A-Za-z0-9_.-]*["']?)\s*[:=]\s*["']?)([^"'\s,;&}]{4,})/giu
const TERMINAL_BEARER_RE = /\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/giu
const TERMINAL_AWS_KEY_RE = /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/gu
const TERMINAL_JWT_RE = /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/gu

/** Keep live terminal previews useful without printing obvious credentials. */
export function redactTerminalPreview(text: string): string {
  return text
    .replace(TERMINAL_BEARER_RE, 'Bearer [redacted]')
    .replace(TERMINAL_AWS_KEY_RE, '[redacted-aws-key]')
    .replace(TERMINAL_JWT_RE, '[redacted-jwt]')
    .replace(TERMINAL_SECRET_ASSIGNMENT_RE, '$1[redacted]')
}

function formatQuestionAnswerHint(
  mode: 'shell' | 'cli',
  sessionId: string | undefined,
  questionId: string,
): string {
  if (mode === 'shell') {
    return `/answer ${questionId} <reply>`
  }
  return `sepilot answer ${sessionId ?? '<session-id>'} ${questionId} <reply>`
}

function formatApprovalDecisionHint(
  mode: 'shell' | 'cli' | undefined,
  sessionId: string | undefined,
  requestId: string,
): string {
  if (mode !== 'cli') {
    return `/approve ${requestId} [--run|--session-all] or /deny ${requestId}`
  }
  const sessionArg = sessionId ? ` --session ${sessionId}` : ''
  return [
    `sepilot approve ${requestId}${sessionArg} --scope run`,
    `sepilot deny ${requestId}${sessionArg}`,
  ].join(' or ')
}

function formatAcceptanceCriterionCount(count: number): string {
  return `${count} ${count === 1 ? 'AC' : 'ACs'}`
}

function acceptanceCriteriaSignature(
  criteria: ReadonlyArray<{ id?: unknown; text: string }>,
): string {
  return JSON.stringify(criteria.map((criterion) => [criterion.id ?? null, criterion.text]))
}

const DEFAULT_PROGRESS_HEARTBEAT_MS = 30_000
const THINKING_BATCH_INTERVAL_MS = 750
const THINKING_PREVIEW_CHARS = 100
const SUPERVISOR_PREVIEW_CHARS = 700
const SUPERVISOR_PREFIX_RE = /^\[supervisor\]\s*/i
const COMPLETION_GATE_THINKING_RE = /\bCompletion gate blocked\b/i

/**
 * `SEPILOTD_PROGRESS_HEARTBEAT_MS` throttles the `state_board` progress
 * heartbeat line. Default 30000ms; `0` disables it entirely. Read fresh on
 * every frame (not cached) so tests can flip it between assertions and a
 * long-lived REPL process picks up an env change on the next agent run.
 */
function progressHeartbeatIntervalMs(): number {
  const raw = process.env.SEPILOTD_PROGRESS_HEARTBEAT_MS
  if (raw === undefined || raw === '') return DEFAULT_PROGRESS_HEARTBEAT_MS
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_PROGRESS_HEARTBEAT_MS
}

/**
 * `criteria <total>` only — `AgentAcceptanceCriterion` carries no
 * structured met/status field today, so we render a bare count rather than
 * infer completion from text. `plan`/`todos` segments are omitted entirely
 * when the board has none of that kind.
 */
function formatStateBoardHeartbeat(frame: {
  criteriaTotal: number
  planTotal: number
  planDone: number
  todosTotal: number
  todosDone: number
  todos?: Array<{ content: string; status: string }>
}): string {
  const segments: string[] = []
  if (frame.criteriaTotal > 0) segments.push(`criteria ${frame.criteriaTotal}`)
  if (frame.planTotal > 0) segments.push(`plan ${frame.planDone}/${frame.planTotal}`)
  if (frame.todosTotal > 0) segments.push(`todos ${frame.todosDone}/${frame.todosTotal}`)
  // Show what the agent is actually working on right now, not just counts —
  // the structured in_progress todo item from the state board.
  const current = frame.todos?.find((todo) => todo.status === 'in_progress')
  if (current) segments.push(`현재: ${truncate(current.content, 80)}`)
  return segments.length > 0 ? `[진행] ${segments.join(' · ')}` : ''
}

function formatCoworkPlanLines(plan: Array<{ role: string; instruction: string }>): string[] {
  if (plan.length === 0) return ['  [cowork plan] 0 tasks']
  const lines = [`  [cowork plan] ${plan.length} task${plan.length === 1 ? '' : 's'}`]
  for (const [index, step] of plan.slice(0, 6).entries()) {
    lines.push(`    ${index + 1}. ${step.role}: ${truncate(step.instruction, 160)}`)
  }
  if (plan.length > 6) {
    lines.push(`    ... ${plan.length - 6} more`)
  }
  return lines
}

/**
 * Instance-scoped throttle for the `state_board` heartbeat: at most one
 * line per `SEPILOTD_PROGRESS_HEARTBEAT_MS`. Returns the line to print, or
 * `undefined` when this frame should be suppressed (too soon, disabled, or
 * nothing structured to show).
 */
function createStateBoardHeartbeat() {
  let lastHeartbeatAt: number | undefined
  return (frame: {
    criteriaTotal: number
    planTotal: number
    planDone: number
    todosTotal: number
    todosDone: number
    todos?: Array<{ content: string; status: string }>
  }): string | undefined => {
    const intervalMs = progressHeartbeatIntervalMs()
    if (intervalMs <= 0) return undefined
    const now = Date.now()
    if (lastHeartbeatAt !== undefined && now - lastHeartbeatAt < intervalMs) {
      return undefined
    }
    const line = formatStateBoardHeartbeat(frame)
    if (!line) return undefined
    lastHeartbeatAt = now
    return line
  }
}

function formatThinkingFrame(content: string): {
  label: 'thinking' | 'supervisor'
  text: string
  critical: boolean
} {
  const trimmed = content.trim()
  const isSupervisor = SUPERVISOR_PREFIX_RE.test(trimmed)
  const body = isSupervisor ? trimmed.replace(SUPERVISOR_PREFIX_RE, '') : trimmed
  const critical = COMPLETION_GATE_THINKING_RE.test(body)
  const limit = critical
    ? Number.POSITIVE_INFINITY
    : isSupervisor
      ? SUPERVISOR_PREVIEW_CHARS
      : THINKING_PREVIEW_CHARS
  const truncated = body.length > limit ? `${body.slice(0, limit - 3).trimEnd()}...` : body
  return {
    label: isSupervisor || critical ? 'supervisor' : 'thinking',
    text: truncated,
    critical,
  }
}

function formatDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '0ms'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`
}

function formatLlmRequestFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'llm_request' }>): string {
  const tools = frame.toolNames.length
    ? ` · tools ${frame.toolNames.slice(0, 5).join(', ')}${frame.toolNames.length > 5 ? `, +${frame.toolNames.length - 5}` : ''}`
    : ''
  const trace = frame.traceRef ? ` · trace ${frame.traceRef}` : ''
  const target = frame.providerId ? `${frame.providerId}/${frame.model}` : frame.model
  const role = frame.auxiliary ? 'aux' : 'main'
  const source = frame.source ? ` · ${frame.source}` : ''
  const timeout = frame.timeoutMs ? ` · timeout ${formatDurationMs(frame.timeoutMs)}` : ''
  return `[model:${role}] ${target}${source} · iteration ${frame.iteration}${timeout}${tools}${trace}`
}

function formatNodeTraceFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'node_trace' }>): string {
  const next = frame.nextEdge ? ` -> ${frame.nextEdge}` : ''
  return `[node] ${frame.node} ${formatDurationMs(frame.durationMs)}${next}`
}

function formatPlannerWorkingMemoryFrame(
  frame: Extract<TerminalChatStreamFrame, { kind: 'planner_working_memory' }>,
): string {
  const risks = frame.risks.length ? ` · risks ${frame.risks.length}` : ''
  const assumptions = frame.openAssumptions ? ` · assumptions ${frame.openAssumptions}` : ''
  const abandoned = frame.abandonedAlternatives ? ` · abandoned ${frame.abandonedAlternatives}` : ''
  const current = frame.currentStepTitle ? ` · current ${truncate(frame.currentStepTitle, 100)}` : ''
  const rationale = frame.currentStepRationale ? ` · ${truncate(frame.currentStepRationale, 120)}` : ''
  return `[planner] plan ${frame.planDone}/${frame.planTotal} · decisions ${frame.decisions}${risks}${assumptions}${abandoned}${current}${rationale}`
}

function formatSubagentProgressFrame(
  frame: Extract<TerminalChatStreamFrame, { kind: 'subagent_progress' }>,
): string {
  const label = frame.label ? `:${frame.label}` : ''
  return `[subagent${label}] ${frame.subagentId} · ${truncate(frame.detail, 180)}`
}

function formatPanelOpenFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'panel_open' }>): string {
  const names = frame.personas.map((persona) => persona.name).join(' · ')
  return `[panel] ${frame.personas.length} panelist${frame.personas.length === 1 ? '' : 's'}${names ? ` · ${truncate(names, 160)}` : ''}`
}

function formatDebateRoundFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'debate_round' }>): string {
  const decision = frame.round.finalDecision ? `:${frame.round.finalDecision}` : ''
  const rationale = frame.round.rationale ? ` · ${truncate(frame.round.rationale, 160)}` : ''
  return `[debate${decision}] ${truncate(frame.round.topic, 120)}${rationale}`
}

function formatRouteFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'mode_route_decision' }>): string {
  const persona = frame.persona ? `/${frame.persona}` : ''
  const confidence = frame.confidence === undefined ? '' : ` · conf ${frame.confidence.toFixed(2)}`
  const fallback = frame.fallback ? ' · fallback' : ''
  const reason = frame.reason ? ` · ${truncate(frame.reason, 160)}` : ''
  return `[mode route] ${frame.chosen}${persona}${confidence}${fallback}${reason}`
}

function formatRouterFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'router_decision' }>): string {
  const skills = frame.skillIds.length ? ` · skills ${frame.skillIds.join(',')}` : ''
  const fallback = frame.fallback ? ' · fallback' : ''
  const reason = frame.reason ? ` · ${truncate(frame.reason, 160)}` : ''
  return `[router] ${frame.mode}/${frame.persona} · ${frame.confidence}${fallback}${skills}${reason}`
}

function formatQualityGateFrame(
  frame: Extract<TerminalChatStreamFrame, { kind: 'quality_gate_verdict' }>,
): string {
  const reason = frame.blockingReason ? ` · ${truncate(frame.blockingReason, 180)}` : ''
  return `[quality gate:${frame.decision}] ${frame.phase} · backtracks ${frame.backtrackCount}${reason}`
}

function formatBacktrackFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'backtrack' }>): string {
  return `[backtrack] ${frame.phase} attempt ${frame.attempt} · ${truncate(frame.reason, 180)}`
}

function formatRecoveryFrame(frame: Extract<TerminalChatStreamFrame, { kind: 'recovery' }>): string {
  const recoverable = frame.recoverable ? 'recoverable' : 'terminal'
  return `[recovery:${frame.scope}/${frame.recoveryKind}] ${frame.action} · ${recoverable} · ${truncate(frame.message, 180)}`
}

function formatCheckpointFrame(
  frame: Extract<TerminalChatStreamFrame, { kind: 'edit_checkpoint_opened' | 'edit_checkpoint_resolved' }>,
): string {
  const checkpoint = frame.checkpoint
  const action = frame.kind === 'edit_checkpoint_opened' ? 'opened' : checkpoint.status
  const files = checkpoint.files.map((file) => file.path)
  const visibleFiles = files.slice(0, 3).join(', ')
  const more = files.length > 3 ? `, +${files.length - 3} more` : ''
  const suffix = visibleFiles ? ` · ${visibleFiles}${more}` : ''
  return `[edit checkpoint] ${action} ${checkpoint.checkpointId} · ${checkpoint.files.length} file${checkpoint.files.length === 1 ? '' : 's'} · ${truncate(checkpoint.label, 120)}${suffix}`
}

/** The legacy bare inactivity message — older daemons sent only this. */
const BARE_INACTIVITY_MESSAGE =
  /^(?:Agent|Resume) produced no output for \d+ms — aborted to release the run slot\.?$/

/**
 * Map a daemon error code to a more useful one-line message for the cli.
 * The daemon's raw `error.message` is correct but sometimes operationally
 * framed; for those codes we substitute a hint the user can act on.
 */
export function explainErrorCode(
  code: string | undefined,
  fallback: string,
  context: {
    stopReason?: RunStopReason
    pendingDecision?: DaemonPendingDecision
    locale?: RunStopLocale
  } = {},
): string {
  switch (code) {
    case 'APPROVAL_TIMEOUT':
    case 'QUESTION_TIMEOUT': {
      // Structured fields win over the daemon's prose: the pending decision
      // tells the user exactly what to answer, the stopReason how to resume.
      const pending = context.pendingDecision
      const copy = context.stopReason
        ? describeStopReason(context.stopReason, context.locale ?? 'en')
        : undefined
      const what = pending
        ? pending.kind === 'approval'
          ? `Pending approval ${pending.id}${pending.label ? ` (${pending.label})` : ''} — \`sepilot approve ${pending.id}\` / \`sepilot deny ${pending.id}\`.`
          : `Pending question ${pending.id}${pending.label ? ` (${pending.label})` : ''} — \`sepilot answer ${pending.id} <text>\`.`
        : undefined
      const parts = [copy?.body, what].filter((part): part is string => Boolean(part))
      return parts.length > 0 ? parts.join(' ') : fallback
    }
    case 'AGENT_INACTIVITY':
      // Current daemons report *why* it stalled — no token/tool from the
      // provider (check provider health, switch model) vs. a stuck step —
      // so surface that verbatim. Only the legacy bare message, which says
      // nothing useful, falls back to the generic hint.
      return fallback && !BARE_INACTIVITY_MESSAGE.test(fallback.trim())
        ? fallback
        : 'The agent stopped making progress and was aborted. Try a smaller, more concrete question or break it into steps.'
    case 'AGENT_CAPACITY_EXHAUSTED':
      return 'The daemon has no free run slots right now — wait a moment and retry, or close another running session.'
    case 'SERVICE_UNAVAILABLE':
      return 'The daemon is starting up or no provider is configured yet. Check `sepilot status` and `sepilot providers`.'
    case 'PROVIDER_CIRCUIT_OPEN':
      return 'The current provider has been failing repeatedly and is paused. Switch with `/provider <name>` or wait for the cooldown.'
    case 'INTERNAL_ERROR':
      return fallback
    default:
      return fallback
  }
}

function formatErrorLine(
  message: string,
  code?: string,
  context?: { stopReason?: RunStopReason; pendingDecision?: DaemonPendingDecision },
): string {
  const friendly = explainErrorCode(code, message, context)
  return code ? `Error [${code}]: ${friendly}` : `Error: ${friendly}`
}

/**
 * Stop card lines for a terminal frame that ended a run with a
 * non-completed structured stop reason. Empty when there is nothing to show.
 */
export function formatStopCardLines(
  stopReason: RunStopReason | undefined,
  hintMode: RunStopHintMode,
): string[] {
  if (!shouldRenderStopCard(stopReason)) return []
  return formatRunStopCard(stopReason, { hintMode }).split('\n')
}

function createCliChatStreamPrinter(
  handleFrame: (frame: TerminalChatStreamFrame) => void,
): CliChatStreamPrinter {
  let lastError: string | undefined
  let stopReason: unknown
  const inner = createTerminalChatStreamFrameConsumer({
    onFrame: (frame) => {
      if (frame.kind === 'error') {
        lastError = frame.message
      }
      handleFrame(frame)
    },
  })
  return {
    handleEvent: (event) => {
      if ('type' in event && event.type === 'done') stopReason = event.stopReason
      inner.handleEvent(event)
    },
    getContent: inner.getContent,
    hadError: () => lastError !== undefined || isUnsuccessfulAgentResult({ content: inner.getContent(), stopReason }),
    errorMessage: () => lastError,
  }
}

export function createInteractiveCliChatStreamPrinter(
  options: InteractiveCliChatStreamPrinterOptions = {},
): CliChatStreamPrinter {
  const stdout = options.stdout ?? process.stdout
  const questionHintMode = options.questionHintMode ?? 'shell'
  const hintMode: RunStopHintMode = options.approvalHintMode ?? 'shell'
  const showDiagnostics = options.showDiagnostics ?? true
  // The daemon re-emits run_contract mid-run (steering / re-plan). Repeat
  // the summary line so the operator knows the contract was refreshed.
  // Reprint the full criteria list only when the actual criteria changed.
  let lastPrintedCriteriaSignature: string | undefined
  let currentSessionId: string | undefined
  const stateBoardHeartbeat = createStateBoardHeartbeat()
  let pendingThinking = ''
  let pendingThinkingStartedAt: number | undefined

  const flushThinking = () => {
    if (!showDiagnostics || !pendingThinking.trim()) {
      pendingThinking = ''
      pendingThinkingStartedAt = undefined
      return
    }
    const thinking = formatThinkingFrame(pendingThinking)
    stdout.write(chalk.gray(`  [${thinking.label}] ${thinking.text}\n`))
    pendingThinking = ''
    pendingThinkingStartedAt = undefined
  }

  return createCliChatStreamPrinter((frame) => {
    // Provider thinking often arrives one token per frame. Preserve the raw
    // chunk order, but render consecutive chunks as a bounded live paragraph
    // instead of one terminal line per token. This is transport-level
    // coalescing only; it does not classify or reinterpret the content.
    if (frame.kind !== 'thinking') flushThinking()
    switch (frame.kind) {
      case 'session':
        currentSessionId = frame.sessionId
        options.onSessionId?.(frame.sessionId)
        break
      case 'state_board': {
        const line = stateBoardHeartbeat(frame)
        if (line) stdout.write(chalk.gray(`  ${line}\n`))
        break
      }
      case 'artifacts':
        if (options.showArtifacts) {
          stdout.write(chalk.gray(`  [artifacts] ${frame.count}\n`))
        }
        break
      case 'inline_text_start':
      case 'inline_text_end':
        stdout.write('\n')
        break
      case 'text':
        stdout.write(frame.text)
        break
      case 'message':
        stdout.write(`\n${frame.content}\n`)
        break
      case 'llm_request':
        if (!showDiagnostics) break
        stdout.write(chalk.gray(`  ${formatLlmRequestFrame(frame)}\n`))
        break
      case 'node_trace':
        if (!showDiagnostics) break
        stdout.write(chalk.gray(`  ${formatNodeTraceFrame(frame)}\n`))
        break
      case 'planner_working_memory':
        if (!showDiagnostics) break
        stdout.write(chalk.gray(`  ${formatPlannerWorkingMemoryFrame(frame)}\n`))
        if (frame.taskSummary) {
          stdout.write(chalk.gray(`    ${truncate(frame.taskSummary, 160)}\n`))
        }
        break
      case 'run_contract': {
        stdout.write(
          chalk.gray(
            `  [run contract] ${formatAcceptanceCriterionCount(frame.contract.acceptanceCriteria.length)} · ${truncate(frame.contract.summary, 100)}\n`,
          ),
        )
        if (!showDiagnostics) break
        const criteriaSignature = acceptanceCriteriaSignature(frame.contract.acceptanceCriteria)
        if (criteriaSignature !== lastPrintedCriteriaSignature) {
          lastPrintedCriteriaSignature = criteriaSignature
          frame.contract.acceptanceCriteria.forEach((criterion, index) => {
            stdout.write(chalk.gray(`    ${index + 1}. ${criterion.text}\n`))
          })
        }
        break
      }
      case 'steering_ack':
        stdout.write(
          chalk.yellow(
            `  [steering 접수] ${frame.noteId}${frame.steeringMessage ? `: ${truncate(frame.steeringMessage, 80)}` : ''}\n`,
          ),
        )
        break
      case 'steering_consumed':
        stdout.write(chalk.gray(`  [steering 반영됨] ${frame.noteId}\n`))
        break
      case 'thinking': {
        if (!showDiagnostics) break
        pendingThinking += frame.content
        pendingThinkingStartedAt ??= Date.now()
        const thinking = formatThinkingFrame(pendingThinking)
        if (
          thinking.critical
          || Date.now() - pendingThinkingStartedAt >= THINKING_BATCH_INTERVAL_MS
        ) {
          flushThinking()
        }
        break
      }
      case 'reasoning_step':
        if (!showDiagnostics) break
        stdout.write(
          chalk.gray(
            `  [reasoning] ${frame.label}${frame.detail ? `: ${truncate(frame.detail, 180)}` : ''}\n`,
          ),
        )
        break
      case 'mode_route_decision':
        if (!showDiagnostics) break
        stdout.write(chalk.gray(`  ${formatRouteFrame(frame)}\n`))
        break
      case 'router_decision':
        if (!showDiagnostics) break
        stdout.write(chalk.gray(`  ${formatRouterFrame(frame)}\n`))
        break
      case 'quality_gate_verdict':
        if (!showDiagnostics && frame.decision === 'pass') break
        stdout.write(
          (frame.decision === 'pass' ? chalk.gray : chalk.yellow)(`  ${formatQualityGateFrame(frame)}\n`),
        )
        break
      case 'backtrack':
        stdout.write(chalk.yellow(`  ${formatBacktrackFrame(frame)}\n`))
        break
      case 'recovery':
        stdout.write(chalk.yellow(`  ${formatRecoveryFrame(frame)}\n`))
        break
      case 'panel_open':
        stdout.write(chalk.gray(`  ${formatPanelOpenFrame(frame)}\n`))
        break
      case 'panel_turn_start':
        stdout.write(chalk.gray(`  [panel:${frame.personaName}] responding\n`))
        break
      case 'panel_turn_complete':
        stdout.write(chalk.green(`  [panel:${frame.personaName}] ${truncate(frame.text, 220)}\n`))
        break
      case 'panel_turn_failed':
        stdout.write(chalk.red(`  [panel:${frame.personaName}] failed ${truncate(frame.error, 220)}\n`))
        break
      case 'panel_synthesizing':
        stdout.write(chalk.gray(`  [panel] synthesizing ${frame.panelists} replies\n`))
        break
      case 'debate_round':
        stdout.write(
          (frame.round.finalDecision === 'reject' ? chalk.yellow : chalk.gray)(`  ${formatDebateRoundFrame(frame)}\n`),
        )
        break
      case 'edit_checkpoint_opened':
      case 'edit_checkpoint_resolved':
        stdout.write(chalk.yellow(`  ${formatCheckpointFrame(frame)}\n`))
        break
      case 'cowork_plan':
        for (const line of formatCoworkPlanLines(frame.plan)) {
          stdout.write(chalk.gray(`${line}\n`))
        }
        break
      case 'cowork_task_start':
        stdout.write(chalk.gray(`  [cowork:${frame.role}] started ${truncate(frame.instruction, 180)}\n`))
        break
      case 'cowork_task_complete':
        stdout.write(chalk.green(`  [cowork:${frame.role}] completed ${truncate(frame.result, 220)}\n`))
        break
      case 'cowork_task_failed':
        stdout.write(chalk.red(`  [cowork:${frame.role}] failed ${truncate(frame.error, 220)}\n`))
        break
      case 'cowork_synthesizing':
        stdout.write(chalk.gray(`  [cowork] synthesizing${frame.summary ? `: ${truncate(frame.summary, 180)}` : ''}\n`))
        break
      case 'cowork_discuss_request': {
        const choices = frame.choices?.length ? ` choices: ${frame.choices.join(' · ')}` : ''
        stdout.write(chalk.yellow(`  [cowork question] ${truncate(frame.prompt, 220)}${choices}\n`))
        break
      }
      case 'cowork_discuss_response':
        stdout.write(chalk.gray(`  [cowork answer] ${truncate(frame.response, 220)}\n`))
        break
      case 'subagent_progress':
        stdout.write(
          (frame.failed ? chalk.red : chalk.gray)(`  ${formatSubagentProgressFrame(frame)}\n`),
        )
        break
      case 'question_request': {
        if (frame.sessionId && frame.sessionId !== currentSessionId) {
          currentSessionId = frame.sessionId
          options.onSessionId?.(frame.sessionId)
        }
        const choices = frame.choices?.length ? ` choices: ${frame.choices.join(' · ')}` : ''
        stdout.write(chalk.yellow(`  [question] ${truncate(frame.prompt, 220)}${choices}\n`))
        stdout.write(
          chalk.gray(
            `    answer with: ${formatQuestionAnswerHint(questionHintMode, currentSessionId, frame.questionId)}\n`,
          ),
        )
        options.onQuestionRequested?.(frame.questionId, currentSessionId)
        break
      }
      case 'tool_call':
        stdout.write(chalk.cyan(`  [tool] ${redactTerminalPreview(frame.preview)}\n`))
        break
      case 'approval_request': {
        if (!frame.subagentId && frame.sessionId && frame.sessionId !== currentSessionId) {
          currentSessionId = frame.sessionId
          options.onSessionId?.(frame.sessionId)
        }
        // Show the tool name + truncated arg preview so the user knows what
        // they are about to allow without having to switch panes. The hint
        // line spells out the slash commands because new users won't know
        // they exist.
        stdout.write(chalk.yellow(`  [approval] ${redactTerminalPreview(frame.preview)}\n`))
        const approvalHint = formatApprovalDecisionHint(
          options.approvalHintMode,
          frame.sessionId ?? currentSessionId,
          frame.requestId,
        )
        stdout.write(
          chalk.gray(
            `    decide with: ${approvalHint} (10m timeout)\n`,
          ),
        )
        options.onApprovalRequested?.(frame.requestId, frame.toolName, frame.sessionId ?? currentSessionId)
        break
      }
      case 'auto_approval':
        // Positive signal that a remembered rule short-circuited the
        // prompt. Without it operators see no consent ui at all and have
        // to grep `decisions list` to figure out why — the gray status
        // line spells out the matched scope/pattern so future rounds are
        // predictable.
        stdout.write(
          chalk.gray(
            `  [auto-${frame.decision}] ${frame.toolName} via ${frame.scope} rule '${frame.pattern}'\n`,
          ),
        )
        break
      case 'tool_result': {
        const status = frame.status === 'success' ? chalk.green('✓') : chalk.red('✗')
        const posture = frame.postureLabel ? `, ${frame.postureLabel}` : ''
        stdout.write(
          chalk.gray(
            `  [result ${status}${frame.recoveryLabel ? `, ${frame.recoveryLabel}` : ''}${posture}] ${truncate(redactTerminalPreview(frame.output), 200)}\n`,
          ),
        )
        break
      }
      case 'done':
        stdout.write(
          chalk.gray(`  [tokens: ${frame.usage.inputTokens}→${frame.usage.outputTokens}]\n`),
        )
        for (const line of formatStopCardLines(frame.stopReason, hintMode)) {
          stdout.write(chalk.yellow(`  ${line}\n`))
        }
        break
      case 'error':
        stdout.write(
          chalk.red(
            `  ${formatErrorLine(frame.message, frame.code, { stopReason: frame.stopReason, pendingDecision: frame.pendingDecision })}\n`,
          ),
        )
        for (const line of formatStopCardLines(frame.stopReason, hintMode)) {
          stdout.write(chalk.yellow(`  ${line}\n`))
        }
        break
      case 'state_change':
        if (options.showStateChanges) {
          stdout.write(chalk.gray(`  [state] ${frame.state}\n`))
        }
        break
      case 'context_compact':
        // Always show — without it the user sees totalTokens drop in
        // /usage and wonders if the agent forgot earlier turns. The
        // gray "[context compacted: N → M tokens]" line tells them
        // the daemon summarised earlier rounds and how much head-room
        // they got back.
        stdout.write(
          chalk.gray(
            `  [context compacted: ${frame.beforeTokens.toLocaleString()} → ${frame.afterTokens.toLocaleString()} tokens]` +
              (frame.summary ? ` ${truncate(frame.summary, 120)}` : '') +
              '\n',
          ),
        )
        break
      case 'phase_change': {
        const entered = frame.enteredPhase ?? 'finalize'
        const closed = frame.closedPhase
          ? ` (closed ${frame.closedPhase.phase}: ${frame.closedPhase.closedTokens.toLocaleString()} tokens)`
          : ''
        stdout.write(chalk.gray(`  [phase] ${entered}${closed}\n`))
        break
      }
      case 'post_edit_findings': {
        // Visual hierarchy: broken callers in red (downstream damage),
        // edited file diagnostics in yellow, mere likely-callers in
        // gray. Without this split the operator can't tell whether
        // the edit *broke* anything or just touched things.
        const editedSummary = frame.editedFiles.slice(0, 3).join(', ')
        if (editedSummary) {
          stdout.write(chalk.gray(`  [post-edit] edited: ${editedSummary}\n`))
        }
        if (frame.brokenCallers.length > 0) {
          stdout.write(chalk.red(`    broken callers (${frame.brokenCallers.length}):\n`))
          for (const c of frame.brokenCallers.slice(0, 3)) {
            stdout.write(
              chalk.red(
                `      - ${c.file}: ${truncate(c.summary.replace(/^\[caller\]\s*/, '').split('\n')[0], 200)}\n`,
              ),
            )
          }
        }
        if (frame.ownDiagnostics.length > 0) {
          stdout.write(chalk.yellow(`    diagnostics (${frame.ownDiagnostics.length}):\n`))
          for (const d of frame.ownDiagnostics.slice(0, 3)) {
            stdout.write(
              chalk.yellow(`      - ${d.file}: ${truncate(d.summary.split('\n')[0], 200)}\n`),
            )
          }
        }
        if (frame.reverseCallers.length > 0 && frame.brokenCallers.length === 0) {
          stdout.write(
            chalk.gray(`    likely callers: ${frame.reverseCallers.slice(0, 3).join(', ')}\n`),
          )
        }
        break
      }
    }
  })
}

/**
 * Strip the trailing ```planner ... ``` JSON block (used internally by the
 * planner working memory channel) from an answer the user is about to see.
 * The block is *only* meaningful to the cli's planner panel, never to a
 * piped consumer of `sepilot ask`.
 */
const PLANNER_BLOCK_REGEX = /\n?```(?:planner|json:planner)\s*\n[\s\S]*?\n?```\s*$/i
function stripPlannerBlock(text: string): string {
  return text.replace(PLANNER_BLOCK_REGEX, '').trimEnd()
}

export function createAnswerOnlyCliChatStreamPrinter(
  options: AnswerOnlyCliChatStreamPrinterOptions = {},
): CliChatStreamPrinter {
  const stdout = options.stdout ?? process.stdout
  const stderr = options.stderr ?? process.stderr
  const questionHintMode = options.questionHintMode ?? 'cli'
  const answerHintMode: RunStopHintMode = options.approvalHintMode ?? 'cli'
  let needsTrailingNewline = false
  let pendingChunk = ''
  // The daemon re-emits run_contract mid-run (steering / re-plan). Repeat
  // the summary line so the operator knows the contract was refreshed.
  // Reprint the full criteria list only when the actual criteria changed.
  let lastPrintedCriteriaSignature: string | undefined
  let currentSessionId: string | undefined
  const stateBoardHeartbeat = createStateBoardHeartbeat()

  const flushContentNewline = () => {
    if (options.suppressContent || !needsTrailingNewline) return
    stdout.write('\n')
    needsTrailingNewline = false
  }

  const writeContent = (text: string) => {
    if (options.suppressContent) return
    stdout.write(text)
    needsTrailingNewline = !text.endsWith('\n')
  }

  return createCliChatStreamPrinter((frame) => {
    switch (frame.kind) {
      case 'session':
        currentSessionId = frame.sessionId
        options.onSessionId?.(frame.sessionId)
        break
      case 'artifacts':
        break
      case 'inline_text_start':
        break
      case 'inline_text_end':
        flushContentNewline()
        break
      case 'text': {
        if (pendingChunk.length > 0) {
          // Already in a fence-suspected buffer; just append.
          pendingChunk += frame.text
          break
        }
        // Look for the start of a ```planner fence appearing across the
        // current chunk. If a partial fence opener is at the tail (e.g. the
        // chunk ends with a stray '```' or '\n```pla'), buffer just enough
        // to recognise it on the next chunk.
        const merged = frame.text
        const fenceFull = merged.search(/\n?```(?:planner|json:planner)\b/i)
        if (fenceFull !== -1) {
          writeContent(merged.slice(0, fenceFull))
          pendingChunk = merged.slice(fenceFull)
          break
        }
        // Detect a partial fence opener at the chunk tail to defer printing.
        const partial = merged.match(/\n?`{1,3}(?:p|pl|pla|plan|plann|planne)?$/i)
        if (partial) {
          const cut = merged.length - partial[0].length
          writeContent(merged.slice(0, cut))
          pendingChunk = merged.slice(cut)
          break
        }
        writeContent(merged)
        break
      }
      case 'message':
        // The shared presenter emits `message` only for a non-streamed
        // assistant reply. A previous LLM iteration may have streamed a
        // progress sentence before a tool call, but that must not suppress
        // the later final reply. De-duplication of a streamed reply and its
        // matching message belongs to the presenter, at the segment boundary.
        writeContent(stripPlannerBlock(frame.content))
        break
      case 'llm_request':
      case 'node_trace':
      case 'planner_working_memory':
        break
      case 'thinking': {
        const thinking = formatThinkingFrame(frame.content)
        if (thinking.critical) {
          stderr.write(chalk.gray(`\n[${thinking.label}] ${thinking.text}\n`))
        }
        break
      }
      case 'reasoning_step':
        if (!options.quiet) {
          stderr.write(
            chalk.gray(
              `\n[reasoning] ${frame.label}${frame.detail ? `: ${truncate(frame.detail, 180)}` : ''}\n`,
            ),
          )
        }
        break
      case 'mode_route_decision':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n${formatRouteFrame(frame)}\n`))
        }
        break
      case 'router_decision':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n${formatRouterFrame(frame)}\n`))
        }
        break
      case 'quality_gate_verdict':
        if (!options.quiet || frame.decision !== 'pass') {
          const color = frame.decision === 'pass' ? chalk.gray : chalk.yellow
          stderr.write(color(`\n${formatQualityGateFrame(frame)}\n`))
        }
        break
      case 'backtrack':
        stderr.write(chalk.yellow(`\n${formatBacktrackFrame(frame)}\n`))
        break
      case 'recovery':
        stderr.write(chalk.yellow(`\n${formatRecoveryFrame(frame)}\n`))
        break
      case 'panel_open':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n${formatPanelOpenFrame(frame)}\n`))
        }
        break
      case 'panel_turn_start':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n[panel:${frame.personaName}] responding\n`))
        }
        break
      case 'panel_turn_complete':
        if (!options.quiet) {
          stderr.write(chalk.green(`\n[panel:${frame.personaName}] ${truncate(frame.text, 220)}\n`))
        }
        break
      case 'panel_turn_failed':
        stderr.write(chalk.red(`\n[panel:${frame.personaName}] failed ${truncate(frame.error, 220)}\n`))
        break
      case 'panel_synthesizing':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n[panel] synthesizing ${frame.panelists} replies\n`))
        }
        break
      case 'debate_round':
        if (!options.quiet || frame.round.finalDecision === 'reject') {
          const color = frame.round.finalDecision === 'reject' ? chalk.yellow : chalk.gray
          stderr.write(color(`\n${formatDebateRoundFrame(frame)}\n`))
        }
        break
      case 'edit_checkpoint_opened':
      case 'edit_checkpoint_resolved':
        stderr.write(chalk.yellow(`\n${formatCheckpointFrame(frame)}\n`))
        break
      case 'cowork_plan':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n${formatCoworkPlanLines(frame.plan).join('\n')}\n`))
        }
        break
      case 'cowork_task_start':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n[cowork:${frame.role}] started ${truncate(frame.instruction, 180)}\n`))
        }
        break
      case 'cowork_task_complete':
        if (!options.quiet) {
          stderr.write(chalk.green(`\n[cowork:${frame.role}] completed ${truncate(frame.result, 220)}\n`))
        }
        break
      case 'cowork_task_failed':
        stderr.write(chalk.red(`\n[cowork:${frame.role}] failed ${truncate(frame.error, 220)}\n`))
        break
      case 'cowork_synthesizing':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n[cowork] synthesizing${frame.summary ? `: ${truncate(frame.summary, 180)}` : ''}\n`))
        }
        break
      case 'cowork_discuss_request': {
        const choices = frame.choices?.length ? ` choices: ${frame.choices.join(' · ')}` : ''
        stderr.write(chalk.yellow(`\n[cowork question] ${truncate(frame.prompt, 220)}${choices}\n`))
        break
      }
      case 'cowork_discuss_response':
        if (!options.quiet) {
          stderr.write(chalk.gray(`\n[cowork answer] ${truncate(frame.response, 220)}\n`))
        }
        break
      case 'subagent_progress':
        if (!options.quiet || frame.failed) {
          const color = frame.failed ? chalk.red : chalk.gray
          stderr.write(color(`\n${formatSubagentProgressFrame(frame)}\n`))
        }
        break
      case 'question_request': {
        if (frame.sessionId && frame.sessionId !== currentSessionId) {
          currentSessionId = frame.sessionId
          options.onSessionId?.(frame.sessionId)
        }
        const choices = frame.choices?.length ? ` choices: ${frame.choices.join(' · ')}` : ''
        stderr.write(chalk.yellow(`\n[question] ${truncate(frame.prompt, 220)}${choices}\n`))
        stderr.write(
          chalk.gray(
            `  answer with: ${formatQuestionAnswerHint(questionHintMode, currentSessionId, frame.questionId)}\n`,
          ),
        )
        options.onQuestionRequested?.(frame.questionId, currentSessionId)
        break
      }
      case 'tool_result':
      case 'state_change':
        break
      case 'state_board': {
        // Progress chrome, not consent/content — respect quiet like
        // tool_call, not always-on like steering/approval.
        if (options.quiet) break
        const line = stateBoardHeartbeat(frame)
        if (line) stderr.write(chalk.gray(`\n${line}\n`))
        break
      }
      case 'run_contract': {
        if (!options.quiet) {
          stderr.write(
            chalk.gray(
              `\n[run contract] ${formatAcceptanceCriterionCount(frame.contract.acceptanceCriteria.length)} · ${truncate(frame.contract.summary, 120)}\n`,
            ),
          )
          // Full criteria list so the operator can see exactly what "done"
          // means for this run without querying the daemon separately —
          // the summary line alone only gives a count.
          const criteriaSignature = acceptanceCriteriaSignature(frame.contract.acceptanceCriteria)
          if (criteriaSignature !== lastPrintedCriteriaSignature) {
            lastPrintedCriteriaSignature = criteriaSignature
            frame.contract.acceptanceCriteria.forEach((criterion, index) => {
              stderr.write(chalk.gray(`  ${index + 1}. ${criterion.text}\n`))
            })
          }
        }
        break
      }
      case 'steering_ack':
        // Consent/visibility class, same as approval_request: the user
        // just queued a mid-run steering note and needs confirmation it
        // was received, even in quiet/piped mode.
        stderr.write(
          chalk.yellow(
            `\n[steering 접수] ${frame.noteId}${frame.steeringMessage ? `: ${truncate(frame.steeringMessage, 80)}` : ''}\n`,
          ),
        )
        break
      case 'steering_consumed':
        stderr.write(chalk.gray(`\n[steering 반영됨] ${frame.noteId}\n`))
        break
      case 'tool_call':
        if (!options.quiet) {
          stderr.write(chalk.cyan(`\n[tool: ${frame.toolName}]\n`))
        }
        break
      case 'approval_request': {
        if (!frame.subagentId && frame.sessionId && frame.sessionId !== currentSessionId) {
          currentSessionId = frame.sessionId
          options.onSessionId?.(frame.sessionId)
        }
        // User-safety critical: always surface approval prompts on
        // stderr, even in quiet/piped mode. Suppressing them would let
        // an `ask | tee answer.txt` invocation silently swallow the one
        // notification the user *must* see before the agent runs a
        // destructive tool. Other noise (tool_call cyan banner, done
        // usage line) still respects `quiet` because they're progress
        // chrome, not consent prompts.
        stderr.write(chalk.yellow(`\n[approval: ${frame.preview}]\n`))
        const hint = `  decide with: ${
          options.approvalHintMode === 'cli'
            ? formatApprovalDecisionHint('cli', frame.sessionId ?? currentSessionId, frame.requestId)
                .split(' or ')
                .map((command) => `\`${command}\``)
                .join(' or ')
            : formatApprovalDecisionHint(options.approvalHintMode, frame.sessionId ?? currentSessionId, frame.requestId)
        }`
        stderr.write(chalk.gray(`${hint}\n`))
        options.onApprovalRequested?.(frame.requestId, frame.toolName, frame.sessionId ?? currentSessionId)
        break
      }
      case 'auto_approval':
        // Same stderr-always rule as approval_request: this is the
        // operator-facing signal that *something they previously
        // remembered* let the daemon skip a consent prompt. Suppressing
        // it in quiet mode would defeat the visibility purpose — the
        // whole gap this banner closes is "why did no prompt appear?".
        stderr.write(
          chalk.gray(
            `\n[auto-${frame.decision}] ${frame.toolName} via ${frame.scope} rule '${frame.pattern}'\n`,
          ),
        )
        break
      case 'done':
        // Flush any tail still in the buffer (after stripping a planner
        // block if the buffer contained one).
        if (pendingChunk.length > 0) {
          writeContent(stripPlannerBlock(pendingChunk))
          pendingChunk = ''
        }
        flushContentNewline()
        if (!options.quiet) {
          stderr.write(
            chalk.gray(`\n[${frame.usage.inputTokens}→${frame.usage.outputTokens} tokens]\n`),
          )
        }
        // Operational signal like the approval banner: shown even in quiet
        // mode, on stderr so a piped stdout consumer never sees it.
        for (const line of formatStopCardLines(frame.stopReason, answerHintMode)) {
          stderr.write(chalk.yellow(`${line}\n`))
        }
        break
      case 'error':
        if (pendingChunk.length > 0) {
          writeContent(stripPlannerBlock(pendingChunk))
          pendingChunk = ''
        }
        flushContentNewline()
        stderr.write(
          chalk.red(
            `\n${formatErrorLine(frame.message, frame.code, { stopReason: frame.stopReason, pendingDecision: frame.pendingDecision })}\n`,
          ),
        )
        for (const line of formatStopCardLines(frame.stopReason, answerHintMode)) {
          stderr.write(chalk.yellow(`${line}\n`))
        }
        break
      case 'context_compact':
        // Like the approval banner, this is operational signal the user
        // needs even in quiet mode — without it a piped `ask` user sees
        // running token totals shrink with no explanation. Goes to
        // stderr so it never pollutes a piped stdout consumer.
        stderr.write(
          chalk.gray(
            `\n[context compacted: ${frame.beforeTokens.toLocaleString()} → ${frame.afterTokens.toLocaleString()} tokens]` +
              (frame.summary ? ` ${truncate(frame.summary, 120)}` : '') +
              '\n',
          ),
        )
        break
      case 'phase_change': {
        // Same rationale as context_compact — operational signal, stderr
        // so piped stdout consumers stay clean.
        if (!options.quiet) {
          const entered = frame.enteredPhase ?? 'finalize'
          const closed = frame.closedPhase
            ? ` (closed ${frame.closedPhase.phase}: ${frame.closedPhase.closedTokens.toLocaleString()} tokens)`
            : ''
          stderr.write(chalk.gray(`\n[phase] ${entered}${closed}\n`))
        }
        break
      }
      case 'post_edit_findings': {
        // Broken-caller signal is a near-failure: surface it even in
        // quiet mode, the operator probably wants to know an `ask`
        // run touched downstream files that no longer compile. Other
        // findings (own diagnostics, mere likely-callers) only render
        // outside quiet mode since they're informational.
        const editedSummary = frame.editedFiles.slice(0, 3).join(', ')
        if (frame.brokenCallers.length > 0) {
          stderr.write(
            chalk.red(
              `\n[post-edit] ${frame.brokenCallers.length} caller(s) broken${editedSummary ? ` after editing ${editedSummary}` : ''}\n`,
            ),
          )
          for (const c of frame.brokenCallers.slice(0, 3)) {
            stderr.write(
              chalk.red(
                `  - ${c.file}: ${truncate(c.summary.replace(/^\[caller\]\s*/, '').split('\n')[0], 200)}\n`,
              ),
            )
          }
          break
        }
        if (!options.quiet) {
          if (editedSummary) {
            stderr.write(chalk.gray(`\n[post-edit] edited: ${editedSummary}\n`))
          }
          if (frame.ownDiagnostics.length > 0) {
            stderr.write(chalk.yellow(`  diagnostics (${frame.ownDiagnostics.length}):\n`))
            for (const d of frame.ownDiagnostics.slice(0, 3)) {
              stderr.write(
                chalk.yellow(`    - ${d.file}: ${truncate(d.summary.split('\n')[0], 200)}\n`),
              )
            }
          }
        }
        break
      }
    }
  })
}
