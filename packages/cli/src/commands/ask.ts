import chalk from 'chalk'
import { writeFile } from 'node:fs/promises'
import { extname } from 'node:path'
import {
  getPersonaPanelPreset,
  selectedChatSkillRefs,
  type DaemonChatBackgroundListResult,
  type DaemonChatBackgroundStartResult,
  type DaemonChatBackgroundStatusResult,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import { DEFAULT_CLI_AGENT_MODE, buildCliChatOptions, parsePositiveIntegerCliOption } from '../chat-options.js'
import { output, getOutputFormat } from '../output/formatter.js'
import { formatChatStreamFailure } from '../utils/chat-stream-error.js'
import {
  createCliPendingDecisionTracker,
  formatPendingDecisionAbortReason,
  formatPendingDecisionWaitNotice,
} from '../utils/pending-decision.js'
import {
  createAnswerOnlyCliChatStreamPrinter,
  createInteractiveCliChatStreamPrinter,
} from '../utils/chat-stream-printer.js'
import { printApiError, printStreamError } from '../utils/error-message.js'
import { composeHangul, stripBracketedPasteDelimiters } from '../tui/utils/hangul.js'
import { detectCliLocale } from '../utils/locale.js'
import {
  forwardDaemonStreamWithResumeRecovery,
  isTerminalCliDaemonChatEvent,
} from '../utils/stream-resume.js'
import { DEFAULT_CLI_STREAM_IDLE_MS, resolveCliStreamIdleMs } from '../utils/stream-idle.js'
import {
  openChatStreamWithConnectTimeout,
  resolveCliStreamConnectMs,
} from '../utils/stream-connect.js'
import { resolveCliSyncChatTimeoutMs } from '../utils/sync-chat-timeout.js'
import { isUnsuccessfulAgentResult } from '../utils/run-outcome.js'

export { resolveCliStreamIdleMs } from '../utils/stream-idle.js'

export type BackgroundCommandSurface = 'ask' | 'assistant'
type BackgroundJobStatus = DaemonChatBackgroundListResult['jobs'][number]['status']

const BACKGROUND_JOB_STATUSES = new Set<BackgroundJobStatus>([
  'running',
  'completed',
  'failed',
  'cancelled',
])

interface BackgroundCommandHints {
  status: string
  list: string
  save: string
  wait: string
  cancel: string
}

function backgroundCommandHints(
  jobId: string,
  surface: BackgroundCommandSurface,
): BackgroundCommandHints {
  if (surface === 'assistant') {
    return {
      status: `sepilot assistant job ${jobId}`,
      list: 'sepilot assistant jobs',
      save: `sepilot assistant job ${jobId} --output <file>`,
      wait: `sepilot assistant job ${jobId} --wait`,
      cancel: `sepilot assistant cancel ${jobId}`,
    }
  }
  return {
    status: `sepilot ask --background-status ${jobId}`,
    list: 'sepilot ask --background-list',
    save: `sepilot ask --background-status ${jobId} --output <file>`,
    wait: `sepilot ask --background-status ${jobId} --wait`,
    cancel: `sepilot ask --background-cancel ${jobId}`,
  }
}

function normalizeBackgroundJobStatus(value?: string): BackgroundJobStatus | undefined {
  const normalized = value?.trim().toLowerCase()
  if (!normalized) return undefined
  if (!BACKGROUND_JOB_STATUSES.has(normalized as BackgroundJobStatus)) {
    throw new Error(
      `--status must be one of running, completed, failed, cancelled (got: ${value})`,
    )
  }
  return normalized as BackgroundJobStatus
}

function filterBackgroundJobList(
  result: DaemonChatBackgroundListResult,
  options: {
    status?: BackgroundJobStatus
    needsAction?: boolean
    limit?: number
  },
): DaemonChatBackgroundListResult {
  let jobs = result.jobs
  if (options.status) {
    jobs = jobs.filter((job) => job.status === options.status)
  }
  if (options.needsAction) {
    jobs = jobs.filter((job) => job.status === 'running' && Boolean(job.progress?.action))
  }
  if (options.limit !== undefined) {
    jobs = jobs.slice(0, options.limit)
  }
  return { jobs }
}

const ASK_COPY = {
  en: {
    expectsCodeWarn: (path: string, ext: string, issue: string) =>
      `Warning: --output ${path} expects code (${ext}) but the response looks like ${issue}.\n`,
    agentRanToolNote: '  The agent may have run a tool instead of writing code. File saved anyway — review before running.\n',
    outputIsDir: (path: string) => `--output path is a directory: ${path}`,
    outputNotWritable: (path: string) => `--output not writable: ${path} (permission denied)`,
    outputParentMissing: (path: string) => `--output parent directory missing: ${path}`,
    usage: (surface: BackgroundCommandSurface) => surface === 'assistant'
      ? 'Usage: sepilot assistant run "request" or echo "request" | sepilot assistant run\n'
      : 'Usage: sepilot ask "question" or echo "question" | sepilot ask\n',
    resolveFailed: (id: string, msg: string) => `Failed to resolve /${id}: ${msg}\n`,
    outputWrittenTo: (path: string) => `Output written to ${path}`,
    outputWrittenToNewline: (path: string) => `Output written to ${path}\n`,
    interactiveIgnoredBackground: (surface: BackgroundCommandSurface) => surface === 'assistant'
      ? '--interactive ignored because assistant run detaches by default; use --foreground for rich progress.\n'
      : '--interactive ignored because --background runs detached.\n',
    interactiveIgnoredOutput: (path: string) =>
      `--interactive ignored because --output writes only the final answer to ${path}.\n`,
    interactiveIgnoredPipe: '--interactive ignored because stdout is not a terminal; using answer-only output.\n',
    interactiveIgnoredFormat: (format: string) =>
      `--interactive ignored because --output-format ${format} is machine-readable.\n`,
    errorPrefix: 'Error:',
    noResponse: 'No response',
    rerunHint: (session: string) => `Re-run with --session ${session} to retry.`,
    sessionHint: 'Pass --session <id> next time to make retries idempotent.',
    unknownPanelPreset: (id: string) => `Unknown panel preset: ${id}\n`,
    backgroundQueued: (jobId: string, sessionId: string, commands: BackgroundCommandHints) =>
      `Background chat queued\n  job: ${jobId}\n  session: ${sessionId}\n  check: ${commands.status}\n  list: ${commands.list}\n  save: ${commands.save}\n  cancel: ${commands.cancel}`,
    backgroundRunning: (jobId: string, sessionId: string) =>
      `Background chat running\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundWaitingApproval: (
      jobId: string,
      sessionId: string,
      toolName: string,
      preview: string | undefined,
      command: string,
      cancelCommand: string,
    ) =>
      `Background chat waiting for approval\n  job: ${jobId}\n  session: ${sessionId}\n  tool: ${toolName}${preview ? `\n  request: ${preview}` : ''}\n  run: ${command}\n  cancel: ${cancelCommand}`,
    backgroundWaitingQuestion: (
      jobId: string,
      sessionId: string,
      prompt: string | undefined,
      choices: string[] | undefined,
      command: string,
      cancelCommand: string,
    ) =>
      `Background chat waiting for an answer\n  job: ${jobId}\n  session: ${sessionId}${prompt ? `\n  question: ${prompt}` : ''}${choices?.length ? `\n  choices: ${choices.join(' · ')}` : ''}\n  run: ${command}\n  cancel: ${cancelCommand}`,
    backgroundFailed: (jobId: string, message: string) =>
      `Background chat failed (${jobId}): ${message}`,
    backgroundCancelled: (jobId: string, sessionId: string) =>
      `Background chat cancelled\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundCancelAlreadyDone: (jobId: string, sessionId: string, status: string) =>
      `Background chat already ${status}\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundCompletedNoContent: (jobId: string, sessionId: string) =>
      `Background chat completed\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundListEmpty: 'No background chat jobs.',
    backgroundListHeader: 'Background chat jobs',
    backgroundListShow: (command: string) => `  show: ${command}`,
    backgroundListSave: (command: string) => `  save: ${command}`,
    backgroundListDecide: (approve: string, deny: string) => `  decide: ${approve}  OR  ${deny}`,
    backgroundListAnswer: (sessionId: string, questionId: string) =>
      `  answer: sepilot answer ${sessionId} ${questionId} <reply>`,
    backgroundListWait: (command: string) => `  wait: ${command}`,
    backgroundListCancel: (command: string) => `  cancel: ${command}`,
  },
  ko: {
    expectsCodeWarn: (path: string, ext: string, issue: string) =>
      `경고: --output ${path}은(는) 코드(${ext})를 기대하지만 응답이 ${issue}로 보입니다.\n`,
    agentRanToolNote: '  에이전트가 코드를 작성하는 대신 도구를 실행했을 수 있습니다. 파일은 그대로 저장됨 — 실행 전 검토하세요.\n',
    outputIsDir: (path: string) => `--output 경로가 디렉토리입니다: ${path}`,
    outputNotWritable: (path: string) => `--output 쓰기 불가: ${path} (권한 거부됨)`,
    outputParentMissing: (path: string) => `--output 상위 디렉토리 없음: ${path}`,
    usage: (surface: BackgroundCommandSurface) => surface === 'assistant'
      ? '사용법: sepilot assistant run "요청" 또는 echo "요청" | sepilot assistant run\n'
      : '사용법: sepilot ask "질문" 또는 echo "질문" | sepilot ask\n',
    resolveFailed: (id: string, msg: string) => `/${id} 해결 실패: ${msg}\n`,
    outputWrittenTo: (path: string) => `출력이 ${path}에 작성되었습니다`,
    outputWrittenToNewline: (path: string) => `출력이 ${path}에 작성되었습니다\n`,
    interactiveIgnoredBackground: (surface: BackgroundCommandSurface) => surface === 'assistant'
      ? '--interactive는 assistant run이 기본적으로 detached 실행되므로 무시됩니다. 진행 표시를 보려면 --foreground를 사용하세요.\n'
      : '--interactive는 --background가 detached로 실행되므로 무시됩니다.\n',
    interactiveIgnoredOutput: (path: string) =>
      `--interactive는 --output이 최종 답변만 ${path}에 쓰므로 무시됩니다.\n`,
    interactiveIgnoredPipe: '--interactive는 stdout이 터미널이 아니므로 무시됩니다. answer-only 출력으로 진행합니다.\n',
    interactiveIgnoredFormat: (format: string) =>
      `--interactive는 --output-format ${format}이 기계 판독 출력이므로 무시됩니다.\n`,
    errorPrefix: '오류:',
    noResponse: '응답 없음',
    rerunHint: (session: string) => `재시도하려면 --session ${session}로 다시 실행하세요.`,
    sessionHint: '재시도를 멱등하게 만들려면 다음에 --session <id>를 전달하세요.',
    unknownPanelPreset: (id: string) => `알 수 없는 패널 프리셋입니다: ${id}\n`,
    backgroundQueued: (jobId: string, sessionId: string, commands: BackgroundCommandHints) =>
      `백그라운드 채팅 작업 등록됨\n  job: ${jobId}\n  session: ${sessionId}\n  확인: ${commands.status}\n  목록: ${commands.list}\n  저장: ${commands.save}\n  취소: ${commands.cancel}`,
    backgroundRunning: (jobId: string, sessionId: string) =>
      `백그라운드 채팅 실행 중\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundWaitingApproval: (
      jobId: string,
      sessionId: string,
      toolName: string,
      preview: string | undefined,
      command: string,
      cancelCommand: string,
    ) =>
      `백그라운드 채팅이 승인을 기다리는 중\n  job: ${jobId}\n  session: ${sessionId}\n  tool: ${toolName}${preview ? `\n  요청: ${preview}` : ''}\n  실행: ${command}\n  취소: ${cancelCommand}`,
    backgroundWaitingQuestion: (
      jobId: string,
      sessionId: string,
      prompt: string | undefined,
      choices: string[] | undefined,
      command: string,
      cancelCommand: string,
    ) =>
      `백그라운드 채팅이 답변을 기다리는 중\n  job: ${jobId}\n  session: ${sessionId}${prompt ? `\n  질문: ${prompt}` : ''}${choices?.length ? `\n  선택지: ${choices.join(' · ')}` : ''}\n  실행: ${command}\n  취소: ${cancelCommand}`,
    backgroundFailed: (jobId: string, message: string) =>
      `백그라운드 채팅 실패 (${jobId}): ${message}`,
    backgroundCancelled: (jobId: string, sessionId: string) =>
      `백그라운드 채팅 취소됨\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundCancelAlreadyDone: (jobId: string, sessionId: string, status: string) =>
      `백그라운드 채팅이 이미 ${status} 상태입니다\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundCompletedNoContent: (jobId: string, sessionId: string) =>
      `백그라운드 채팅 완료\n  job: ${jobId}\n  session: ${sessionId}`,
    backgroundListEmpty: '백그라운드 채팅 작업이 없습니다.',
    backgroundListHeader: '백그라운드 채팅 작업',
    backgroundListShow: (command: string) => `  보기: ${command}`,
    backgroundListSave: (command: string) => `  저장: ${command}`,
    backgroundListDecide: (approve: string, deny: string) => `  결정: ${approve}  OR  ${deny}`,
    backgroundListAnswer: (sessionId: string, questionId: string) =>
      `  답변: sepilot answer ${sessionId} ${questionId} <reply>`,
    backgroundListWait: (command: string) => `  대기: ${command}`,
    backgroundListCancel: (command: string) => `  취소: ${command}`,
  },
} as const

const CODE_EXTENSIONS = new Set([
  '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.kt',
  '.rb', '.php', '.c', '.cpp', '.h', '.hpp', '.cs', '.swift', '.scala',
  '.sh', '.bash', '.zsh', '.fish', '.ps1',
  '.lua', '.pl', '.r', '.jl', '.dart', '.ex', '.exs',
])

/**
 * Heuristic: did the agent produce a markdown report (table / heading
 * with no fenced code block) instead of source code? Catches the
 * common failure where `ask --output app.py` saves a system-info
 * table to app.py because the agent ran a tool instead of writing
 * code. Returns null when the payload looks like real code.
 */
function detectMarkdownNotCode(payload: string): string | null {
  const hasFence = /```/.test(payload)
  if (hasFence) return null
  const hasTableRow = /^\s*\|.*\|\s*$/m.test(payload)
  const hasTableSeparator = /^\s*\|?\s*[:\-]+\s*\|/m.test(payload)
  const hasHeading = /^#{1,6}\s/m.test(payload)
  if (hasTableRow && hasTableSeparator) return 'markdown table'
  if (hasHeading && !/^[a-z_]\w*\s*[=(:]/im.test(payload)) {
    return 'markdown heading without code'
  }
  return null
}

async function writeAskOutput(path: string, payload: string): Promise<void> {
  const copy = ASK_COPY[detectCliLocale()] ?? ASK_COPY.en
  // Sanity check: if --output points at a code file but the payload
  // is clearly a markdown report (no fence + table/heading), warn the
  // user before clobbering the file. This catches the agent-ran-tool
  // failure mode where a system-info table replaces real code.
  const ext = extname(path).toLowerCase()
  if (CODE_EXTENSIONS.has(ext)) {
    const issue = detectMarkdownNotCode(payload)
    if (issue) {
      process.stderr.write(chalk.yellow(copy.expectsCodeWarn(path, ext, issue)))
      process.stderr.write(chalk.gray(copy.agentRanToolNote))
    }
  }
  try {
    await writeFile(path, payload, 'utf-8')
  } catch (err) {
    const code = (err as { code?: string }).code
    if (code === 'EISDIR') {
      console.error(chalk.red(copy.outputIsDir(path)))
      process.exit(1)
    }
    if (code === 'EACCES') {
      console.error(chalk.red(copy.outputNotWritable(path)))
      process.exit(1)
    }
    if (code === 'ENOENT') {
      console.error(chalk.red(copy.outputParentMissing(path)))
      process.exit(1)
    }
    throw err
  }
}

const ASK_SLASH_REGEX = /^\/([a-z0-9][a-z0-9-_]*)\b\s*([\s\S]*)$/i
const MODEL_STREAM_WAITING_THINKING = 'Still waiting for the model stream...'
const AGENT_PROGRESS_ACTION_PATTERN =
  /\b(?:analy[sz]e|audit|build|code|create|debug|design|develop|draft|fix|generate|implement|improve|inspect|investigate|migrate|plan|polish|refactor|repair|review|ship|test|trace|update|validate|write)\b|(?:분석|검사|검증|검토|계획|고쳐|구현|개발|개선|디버그|디자인|리뷰|만들|문서화|생성|설계|수정|작성|점검|추적|코딩|테스트)/iu
const AGENT_PROGRESS_OBJECT_PATTERN =
  /\b(?:agent|api|app|architecture|artifact|backend|bug|build|cli|client|codebase|component|database|docs?|file|files|front[- ]?end|games?|implementation|issue|layout|migration|output|package|page|plan|project|readme|repo|repository|result|screen|server|site|tests?|tool|ui|ux|web\s*(?:app|game|page|site|ui)?|website|workflow|workspace)\b|(?:agent|api|cli|ui|ux|결과|결과물|게임|계획|구현|도구|레포|레이아웃|마이그레이션|문서|백엔드|버그|빌드|사이트|산출물|서버|아키텍처|앱|에이전트|워크스페이스|웹|이슈|저장소|컴포넌트|코드베이스|클라이언트|테스트|파일|패키지|페이지|프로젝트|프론트|화면)/iu
const AGENT_PROGRESS_VISUAL_FEEDBACK_PATTERN =
  /\b(?:ui|ux|design|layout|screen|page|app|website|web\s*(?:app|page|site)?|mobile|desktop|responsive|button|text|visual)\b.{0,80}\b(?:bad|broken|ugly|off|wrong|messy|cramped|cluttered|unfinished|unpolished|overlap(?:ping)?|clipped|cut\s*off|overflow(?:ing)?|low\s*contrast|hard\s+to\s+read)\b|\b(?:bad|broken|ugly|off|wrong|messy|cramped|cluttered|unfinished|unpolished|overlap(?:ping)?|clipped|cut\s*off|overflow(?:ing)?|low\s*contrast|hard\s+to\s+read)\b.{0,80}\b(?:ui|ux|design|layout|screen|page|app|website|web\s*(?:app|page|site)?|mobile|desktop|responsive|button|text|visual)\b|(?:ui|ux|디자인|레이아웃|화면|페이지|앱|웹|모바일|데스크톱|반응형|버튼|텍스트|시각).{0,40}(?:엉망|구려|별로|깨져|깨짐|겹쳐|겹침|잘려|삐져|넘쳐|이상|안\s*맞|망가|대비가?\s*(?:낮|나쁘)|저대비|읽기\s*어려|미완성|어색)|(?:엉망|구려|별로|깨져|깨짐|겹쳐|겹침|잘려|삐져|넘쳐|이상|안\s*맞|망가|저대비|읽기\s*어려|미완성|어색).{0,40}(?:ui|ux|디자인|레이아웃|화면|페이지|앱|웹|모바일|데스크톱|반응형|버튼|텍스트|시각)/iu

export function parseAskSlashCommand(
  input: string,
): { commandId: string; args: string } | null {
  const match = input.match(ASK_SLASH_REGEX)
  if (!match) return null
  return { commandId: match[1], args: match[2].trim() }
}

export function shouldAutoUseInteractiveAskProgress(
  input: string,
  requestedMode: string | undefined,
): boolean {
  const mode = requestedMode?.trim().toLowerCase()
  if (mode && mode !== 'auto' && mode !== 'react' && mode !== 'instant') {
    return true
  }
  if (AGENT_PROGRESS_VISUAL_FEEDBACK_PATTERN.test(input)) {
    return true
  }
  return AGENT_PROGRESS_ACTION_PATTERN.test(input)
    && AGENT_PROGRESS_OBJECT_PATTERN.test(input)
}

const FENCED_BLOCK_REGEX = /```(?:[a-zA-Z0-9_-]+)?\s*\n([\s\S]*?)\n?```/
export const DEFAULT_BACKGROUND_POLL_MS = 1500

export interface BackgroundChatStatusClient {
  backgroundChatStatus(jobId: string): Promise<DaemonChatBackgroundStatusResult>
}

interface AskCommandOptions {
  url?: string
  model?: string
  provider?: string
  persona?: string
  mode?: string
  autonomy?: string
  thinkingLevel?: string
  skill?: string[]
  maxTokens?: string | number
  panelPreset?: string
  output?: string
  session?: string
  stripFences?: boolean
  background?: boolean
  wait?: boolean
  backgroundList?: boolean
  backgroundListStatus?: string
  backgroundListNeedsAction?: boolean
  backgroundListLimit?: string | number
  backgroundStatus?: string
  backgroundCancel?: string
  pollMs?: string | number
  interactive?: boolean
  backgroundCommandSurface?: BackgroundCommandSurface
}

/**
 * Strip the first markdown fenced code block out of the response so a piped
 * `ask --output file.py` invocation produces a directly executable file.
 * If the model returned no fence, fall back to the original text untouched.
 */
export function stripCodeFences(content: string): string {
  const match = content.match(FENCED_BLOCK_REGEX)
  if (!match) return content
  return match[1].trim() + '\n'
}

export async function openChatStreamWithIdleTimeout(
  responsePromise: Promise<Response>,
  aborter: AbortController,
  timeoutMs: number,
): Promise<Response> {
  const effectiveTimeoutMs = Number.isFinite(timeoutMs) && timeoutMs > 0
    ? timeoutMs
    : DEFAULT_CLI_STREAM_IDLE_MS
  let timer: ReturnType<typeof setTimeout> | null = null
  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error('stream-idle-timeout')
      aborter.abort(error)
      reject(error)
    }, effectiveTimeoutMs)
    timer.unref?.()
  })

  try {
    return await Promise.race([responsePromise, timeoutPromise])
  } finally {
    if (timer) {
      clearTimeout(timer)
    }
  }
}

export function isSubstantiveAskStreamEvent(event: unknown): boolean {
  if (!event || typeof event !== 'object') {
    return true
  }
  const candidate = event as { type?: unknown; content?: unknown; text?: unknown; state?: unknown; sessionId?: unknown }
  if (
    candidate.type === 'thinking'
    && (candidate.content === MODEL_STREAM_WAITING_THINKING || candidate.text === MODEL_STREAM_WAITING_THINKING)
  ) {
    return false
  }
  if (candidate.type === 'state_change') {
    return false
  }
  if (
    candidate.type === undefined
    && typeof candidate.sessionId === 'string'
    && Object.keys(candidate).every((key) => key === 'sessionId')
  ) {
    return false
  }
  return true
}

export function bindStreamTerminationSignals(
  aborter: AbortController,
  cancelRemoteRun?: () => void | Promise<void>,
): {
  interrupted: () => boolean
  exitCode: () => number | undefined
  waitForRemoteCancel: () => Promise<void>
  dispose: () => void
} {
  let receivedSignal: 'SIGINT' | 'SIGTERM' | undefined
  let remoteCancel = Promise.resolve()
  const onSignal = (signal: 'SIGINT' | 'SIGTERM') => {
    if (receivedSignal) return
    receivedSignal = signal
    aborter.abort(new Error(`cli-${signal.toLowerCase()}`))
    remoteCancel = Promise.resolve(cancelRemoteRun?.()).then(() => undefined, () => undefined)
  }
  const onSigint = () => onSignal('SIGINT')
  const onSigterm = () => onSignal('SIGTERM')
  process.once('SIGINT', onSigint)
  process.once('SIGTERM', onSigterm)
  return {
    interrupted: () => receivedSignal !== undefined,
    exitCode: () => receivedSignal === 'SIGINT' ? 130 : receivedSignal === 'SIGTERM' ? 143 : undefined,
    waitForRemoteCancel: () => remoteCancel,
    dispose: () => {
      process.off('SIGINT', onSigint)
      process.off('SIGTERM', onSigterm)
    },
  }
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms))
}

export async function waitForBackgroundChat(
  client: BackgroundChatStatusClient,
  jobId: string,
  pollMs: number,
  onStatus?: (status: DaemonChatBackgroundStatusResult) => void,
): Promise<DaemonChatBackgroundStatusResult> {
  while (true) {
    const status = await client.backgroundChatStatus(jobId)
    onStatus?.(status)
    if (status.status !== 'running') return status
    await sleep(pollMs)
  }
}

export function isBackgroundUnsuccessfulStatus(
  status: DaemonChatBackgroundStatusResult,
): boolean {
  return status.status === 'failed' || status.status === 'cancelled'
}

export function emitStreamJsonRecord(record: unknown): void {
  process.stdout.write(`${JSON.stringify(record)}\n`)
}

export function isIncompleteAgentResult(content: string): boolean {
  return content.split(/\r?\n/).some((line) => line.trimStart().startsWith('INCOMPLETE:'))
}

export async function flushStreamJsonRecord(record: unknown): Promise<void> {
  await new Promise<void>((resolve) => {
    const wrote = process.stdout.write(`${JSON.stringify(record)}\n`)
    if (wrote) resolve()
    else process.stdout.once('drain', resolve)
  })
}

export function emitBackgroundQueuedStreamJson(
  started: DaemonChatBackgroundStartResult,
): void {
  emitStreamJsonRecord({
    type: 'background_queued',
    jobId: started.jobId,
    sessionId: started.sessionId,
    status: started.status,
  })
}

export function emitBackgroundStatusStreamJson(
  status: DaemonChatBackgroundStatusResult,
): void {
  emitStreamJsonRecord({
    type: 'background_status',
    ...status,
  })
}

export async function emitBackgroundStreamJsonResult(
  result: DaemonChatBackgroundStartResult | DaemonChatBackgroundStatusResult,
  options: { startedAt?: number; failOnTerminalFailure?: boolean } = {},
): Promise<void> {
  const failed = options.failOnTerminalFailure
    && (result.status === 'failed' || result.status === 'cancelled')
  await flushStreamJsonRecord({
    type: 'result',
    subtype: failed ? 'error' : 'success',
    background: true,
    jobId: result.jobId,
    sessionId: result.sessionId,
    status: result.status,
    durationMs: typeof options.startedAt === 'number'
      ? Date.now() - options.startedAt
      : undefined,
    messageId: 'messageId' in result ? result.messageId : undefined,
    content: 'content' in result ? result.content : undefined,
    progress: 'progress' in result ? result.progress : undefined,
    error: 'error' in result ? result.error : undefined,
    createdAt: 'createdAt' in result ? result.createdAt : undefined,
    updatedAt: 'updatedAt' in result ? result.updatedAt : undefined,
  })
}

async function emitBackgroundJobListStreamJson(
  result: DaemonChatBackgroundListResult,
): Promise<void> {
  emitStreamJsonRecord({
    type: 'background_list',
    jobs: result.jobs,
  })
  await flushStreamJsonRecord({
    type: 'result',
    subtype: 'success',
    background: true,
    jobs: result.jobs,
  })
}

function truncateBackgroundDetail(text: string | undefined, limit = 180): string | undefined {
  if (!text) return undefined
  return text.length > limit ? `${text.slice(0, limit - 3).trimEnd()}...` : text
}

function truncateBackgroundChoices(choices: string[] | undefined): string[] | undefined {
  if (!choices?.length) return undefined
  return choices.slice(0, 8).map((choice) => truncateBackgroundDetail(choice, 80) ?? '')
}

function formatBackgroundActionSummary(
  status: Pick<DaemonChatBackgroundStatusResult, 'progress'>,
): string {
  const action = status.progress?.action
  if (!action) return ''
  if (action.type === 'approval') return ` action=approval:${action.requestId}`
  return ` action=question:${action.questionId}`
}

function renderBackgroundActionRequired(
  status: DaemonChatBackgroundStatusResult,
  surface: BackgroundCommandSurface,
): string | null {
  const copy = ASK_COPY[detectCliLocale()] ?? ASK_COPY.en
  const action = status.progress?.action
  if (!action) return null
  const cancel = backgroundCommandHints(status.jobId, surface).cancel
  if (action.type === 'approval') {
    const approve = `sepilot approve ${action.requestId} --session ${status.sessionId} --scope run`
    const deny = `sepilot deny ${action.requestId} --session ${status.sessionId}`
    const preview = truncateBackgroundDetail(action.preview ?? status.progress?.detail)
    return copy.backgroundWaitingApproval(
      status.jobId,
      status.sessionId,
      action.toolName,
      preview,
      `${approve}  OR  ${deny}`,
      cancel,
    )
  }
  const answer = `sepilot answer ${status.sessionId} ${action.questionId} <reply>`
  return copy.backgroundWaitingQuestion(
    status.jobId,
    status.sessionId,
    truncateBackgroundDetail(status.progress?.detail),
    truncateBackgroundChoices(action.choices),
    answer,
    cancel,
  )
}

export function createBackgroundActionNotifier(
  surface: BackgroundCommandSurface = 'ask',
): (status: DaemonChatBackgroundStatusResult) => void {
  let lastActionKey: string | undefined
  let lastProgressLabel: string | undefined
  return (status) => {
    const action = status.progress?.action
    if (!action) {
      const label = status.progress?.label?.replace(/[\r\n\x00-\x1f\x7f]/g, ' ').trim().slice(0, 160)
      if (status.status === 'running' && label && label !== lastProgressLabel) {
        lastProgressLabel = label
        process.stderr.write(chalk.gray(`  ${label}\n`))
      }
      return
    }
    const key = action.type === 'approval'
      ? `approval:${action.requestId}`
      : `question:${action.questionId}`
    if (key === lastActionKey) return
    lastActionKey = key
    const message = renderBackgroundActionRequired(status, surface)
    if (message) {
      process.stderr.write(chalk.yellow(`${message}\n`))
    }
  }
}

function renderBackgroundStatus(
  status: DaemonChatBackgroundStatusResult,
  surface: BackgroundCommandSurface,
): string {
  const copy = ASK_COPY[detectCliLocale()] ?? ASK_COPY.en
  if (status.status === 'failed') {
    return copy.backgroundFailed(status.jobId, status.error?.message ?? 'unknown error')
  }
  if (status.status === 'cancelled') {
    return copy.backgroundCancelled(status.jobId, status.sessionId)
  }
  if (status.status === 'completed') {
    return status.content?.trim()
      || copy.backgroundCompletedNoContent(status.jobId, status.sessionId)
  }
  const action = renderBackgroundActionRequired(status, surface)
  if (action) return action
  return copy.backgroundRunning(status.jobId, status.sessionId)
}

function renderBackgroundCancelStatus(
  status: DaemonChatBackgroundStatusResult,
): string {
  const copy = ASK_COPY[detectCliLocale()] ?? ASK_COPY.en
  if (status.status === 'cancelled') {
    return copy.backgroundCancelled(status.jobId, status.sessionId)
  }
  if (status.status === 'running') {
    return copy.backgroundRunning(status.jobId, status.sessionId)
  }
  return copy.backgroundCancelAlreadyDone(status.jobId, status.sessionId, status.status)
}

function renderBackgroundListFollowUp(
  job: DaemonChatBackgroundListResult['jobs'][number],
  copy: (typeof ASK_COPY)[keyof typeof ASK_COPY],
  surface: BackgroundCommandSurface,
): string[] {
  const commands = backgroundCommandHints(job.jobId, surface)
  if (job.status !== 'running') {
    return job.status === 'completed'
      ? [
          copy.backgroundListShow(commands.status),
          copy.backgroundListSave(commands.save),
        ]
      : [
          copy.backgroundListShow(commands.status),
        ]
  }
  const action = job.progress?.action
  if (action?.type === 'approval') {
    const approve = `sepilot approve ${action.requestId} --session ${job.sessionId} --scope run`
    const deny = `sepilot deny ${action.requestId} --session ${job.sessionId}`
    return [
      copy.backgroundListDecide(approve, deny),
      copy.backgroundListCancel(commands.cancel),
    ]
  }
  if (action?.type === 'question') {
    return [
      copy.backgroundListAnswer(job.sessionId, action.questionId),
      copy.backgroundListCancel(commands.cancel),
    ]
  }
  return [
    copy.backgroundListWait(commands.wait),
    copy.backgroundListCancel(commands.cancel),
  ]
}

function renderBackgroundJobList(
  result: DaemonChatBackgroundListResult,
  surface: BackgroundCommandSurface,
): string {
  const copy = ASK_COPY[detectCliLocale()] ?? ASK_COPY.en
  if (result.jobs.length === 0) return copy.backgroundListEmpty
  return [
    copy.backgroundListHeader,
    ...result.jobs.flatMap((job) => {
      const error = job.error?.message ? ` error=${truncateBackgroundDetail(job.error.message, 120)}` : ''
      const action = formatBackgroundActionSummary(job)
      return [
        `- ${job.jobId} ${job.status} session=${job.sessionId} updated=${job.updatedAt}${action}${error}`,
        ...renderBackgroundListFollowUp(job, copy, surface),
      ]
    }),
  ].join('\n')
}

async function emitBackgroundStatus(
  status: DaemonChatBackgroundStatusResult,
  options: Pick<AskCommandOptions, 'output' | 'stripFences' | 'backgroundCommandSurface'>,
): Promise<void> {
  const copy = ASK_COPY[detectCliLocale()] ?? ASK_COPY.en
  if (options.output && status.content) {
    await writeAskOutput(
      options.output,
      options.stripFences ? stripCodeFences(status.content) : status.content,
    )
    process.stderr.write(chalk.green(copy.outputWrittenToNewline(options.output)))
    return
  }
  console.log(renderBackgroundStatus(status, options.backgroundCommandSurface ?? 'ask'))
}

async function runAskStreamJson(
  client: DaemonClient,
  input: string,
  options: AskCommandOptions,
  chatOptions: ReturnType<typeof buildCliChatOptions>,
): Promise<void> {
  const startedAt = Date.now()
  let sessionId = options.session
  let content = ''
  let usage: unknown
  let stopReason: unknown
  let hadError = false

  const emit = (event: unknown): void => {
    emitStreamJsonRecord(event)
  }

  const STREAM_IDLE_MS = resolveCliStreamIdleMs()
  const STREAM_CONNECT_MS = resolveCliStreamConnectMs()
  const aborter = new AbortController()
  const termination = bindStreamTerminationSignals(aborter, async () => {
    if (sessionId) await client.cancelActiveRun(sessionId)
  })
  let lastTick = Date.now()
  const pendingDecisions = createCliPendingDecisionTracker()
  const watchdog = setInterval(() => {
    // A run blocked on a pending approval/question is deliberately silent.
    // Aborting it as an idle stream would report a human wait as a hang.
    if (pendingDecisions.pending()) return
    if (Date.now() - lastTick > STREAM_IDLE_MS) {
      aborter.abort(new Error('stream-idle-timeout'))
    }
  }, Math.min(5000, STREAM_IDLE_MS / 4))

  const tick = (event: unknown): void => {
    pendingDecisions.note(event)
    if (isSubstantiveAskStreamEvent(event)) {
      lastTick = Date.now()
    }
    if (event && typeof event === 'object') {
      const record = event as Record<string, unknown>
      if (typeof record.sessionId === 'string') sessionId = record.sessionId
      if (record.type === 'text_delta' && typeof record.text === 'string') {
        content += record.text
      }
      if (record.type === 'content' && typeof record.content === 'string') {
        content += record.content
      }
      if (record.type === 'message' && typeof record.content === 'string') {
        content = record.content
      }
      if (record.type === 'done') {
        usage = record.usage
        stopReason = record.stopReason
      }
      if (record.type === 'error') hadError = true
    }
    emit(event)
  }

  try {
    const res = await openChatStreamWithConnectTimeout(
      client.chatStream(input, options.session, chatOptions, { signal: aborter.signal }),
      {
        timeoutMs: STREAM_CONNECT_MS,
        abort: (error) => aborter.abort(error),
      },
    )
    if (!res.ok || !res.body) {
      hadError = true
      emit({ type: 'error', error: await formatChatStreamFailure(res) })
    } else {
      await forwardDaemonStreamWithResumeRecovery(res, {
        aborter,
        getSessionId: () => sessionId,
        onEvent: tick,
        openResumeStream: (resumeSessionId) => client.resumeSessionStream(
          resumeSessionId,
          undefined,
          { signal: aborter.signal },
        ),
        isTerminalEvent: isTerminalCliDaemonChatEvent,
        quiet: true,
        streamIdleMs: STREAM_IDLE_MS,
      })
    }
  } catch (err) {
    hadError = true
    if (!termination.interrupted()) {
      emit({ type: 'error', error: err instanceof Error ? err.message : String(err) })
    }
  } finally {
    clearInterval(watchdog)
    await termination.waitForRemoteCancel()
    termination.dispose()
  }

  // Flush the final result line before exiting. process.exit() drops
  // buffered stdout when piped, which could truncate this terminating
  // NDJSON line; await the drain, then signal failure via exitCode so
  // Node flushes naturally instead of a hard exit mid-write.
  const incomplete = termination.interrupted() || isUnsuccessfulAgentResult({ content, stopReason })
  await flushStreamJsonRecord({
    type: 'result',
    subtype: hadError || incomplete ? 'error' : 'success',
    sessionId,
    durationMs: Date.now() - startedAt,
    usage,
    stopReason,
    content,
  })
  if (termination.interrupted()) process.exitCode = termination.exitCode()
  else if (hadError || incomplete) process.exitCode = 1
}

export async function askCommand(question: string | undefined, options: AskCommandOptions) {
  const copy = ASK_COPY[detectCliLocale()] ?? ASK_COPY.en
  const backgroundCommandSurface = options.backgroundCommandSurface ?? 'ask'
  const quiet = !process.stdout.isTTY  // Output is piped — suppress non-content output
  const outputFormat = getOutputFormat()
  const warnIgnoredInteractive = (message: string) => {
    if (options.interactive) {
      process.stderr.write(chalk.yellow(message))
    }
  }
  let input = question
  const client = new DaemonClient(options.url)

  try {
    await ensureDaemon(client, { url: options.url })
  } catch (err) {
    process.stderr.write(chalk.red(`${err instanceof Error ? err.message : err}\n`))
    process.exit(1)
  }

  if (options.backgroundStatus) {
    try {
      const pollMs =
        parsePositiveIntegerCliOption(options.pollMs, '--poll-ms')
        ?? DEFAULT_BACKGROUND_POLL_MS
      const textActionNotifier = outputFormat === 'json' || outputFormat === 'stream-json'
        ? undefined
        : createBackgroundActionNotifier(backgroundCommandSurface)
      const status = options.wait
        ? await waitForBackgroundChat(
            client,
            options.backgroundStatus,
            pollMs,
            outputFormat === 'stream-json' ? emitBackgroundStatusStreamJson : textActionNotifier,
          )
        : await client.backgroundChatStatus(options.backgroundStatus)
      if (outputFormat === 'json') {
        output(status)
      } else if (outputFormat === 'stream-json') {
        if (!options.wait) emitBackgroundStatusStreamJson(status)
        await emitBackgroundStreamJsonResult(status, {
          failOnTerminalFailure: options.wait
            ? isBackgroundUnsuccessfulStatus(status)
            : status.status === 'failed',
        })
      } else {
        await emitBackgroundStatus(status, options)
      }
      if (status.status === 'failed' || (options.wait && status.status === 'cancelled')) process.exit(1)
    } catch (err) {
      if (printApiError(err, { hint: `Check the job id: ${options.backgroundStatus}` })) {
        process.exit(1)
      }
      console.error(chalk.red(`${copy.errorPrefix} ${err instanceof Error ? err.message : String(err)}`))
      process.exit(1)
    }
    return
  }

  if (options.backgroundCancel) {
    try {
      const status = await client.cancelBackgroundChat(options.backgroundCancel)
      if (outputFormat === 'json') {
        output(status)
      } else if (outputFormat === 'stream-json') {
        emitBackgroundStatusStreamJson(status)
        await emitBackgroundStreamJsonResult(status)
      } else {
        console.log(renderBackgroundCancelStatus(status))
      }
    } catch (err) {
      if (printApiError(err, { hint: `Check the job id: ${options.backgroundCancel}` })) {
        process.exit(1)
      }
      console.error(chalk.red(`${copy.errorPrefix} ${err instanceof Error ? err.message : String(err)}`))
      process.exit(1)
    }
    return
  }

  if (options.backgroundList) {
    let status: BackgroundJobStatus | undefined
    let limit: number | undefined
    try {
      status = normalizeBackgroundJobStatus(options.backgroundListStatus)
      limit = parsePositiveIntegerCliOption(options.backgroundListLimit, '--limit')
    } catch (err) {
      process.stderr.write(chalk.red(`${err instanceof Error ? err.message : err}\n`))
      process.exit(1)
    }
    try {
      const result = filterBackgroundJobList(await client.backgroundChatJobs(), {
        status,
        needsAction: options.backgroundListNeedsAction,
        limit,
      })
      if (outputFormat === 'json') {
        output(result)
      } else if (outputFormat === 'stream-json') {
        await emitBackgroundJobListStreamJson(result)
      } else {
        console.log(renderBackgroundJobList(result, backgroundCommandSurface))
      }
    } catch (err) {
      if (printApiError(err, { hint: 'Check daemon availability and retry.' })) {
        process.exit(1)
      }
      console.error(chalk.red(`${copy.errorPrefix} ${err instanceof Error ? err.message : String(err)}`))
      process.exit(1)
    }
    return
  }

  // Read from stdin if piped
  if (!input && !process.stdin.isTTY) {
    input = await new Promise<string>((resolve) => {
      let data = ''
      process.stdin.setEncoding('utf-8')
      process.stdin.on('data', (chunk) => { data += chunk })
      process.stdin.on('end', () => resolve(data.trim()))
    })
  }

  if (!input) {
    process.stderr.write(chalk.red(copy.usage(backgroundCommandSurface)))
    process.exit(1)
  }
  input = composeHangul(stripBracketedPasteDelimiters(input))

  let resolvedAgent: string | undefined
  // Slash command shortcut: '/cmd args ...' is resolved through the daemon
  // user-command store before the question is dispatched. Mirrors what the
  // TUI useChat hook already does so the cli/automation path stays consistent.
  const slash = parseAskSlashCommand(input)
  if (slash) {
    try {
      const resolved = await client.resolveUserCommand(slash.commandId, slash.args)
      input = resolved.prompt
      resolvedAgent = resolved.agent
    } catch (err) {
      process.stderr.write(chalk.red(
        copy.resolveFailed(slash.commandId, err instanceof Error ? err.message : String(err)),
      ))
      process.exit(1)
    }
  }

  let maxTokens: number | undefined
  try {
    maxTokens = parsePositiveIntegerCliOption(options.maxTokens, '--max-tokens')
  } catch (err) {
    process.stderr.write(chalk.red(`${err instanceof Error ? err.message : err}\n`))
    process.exit(1)
  }

  const panelPreset = options.panelPreset
    ? getPersonaPanelPreset(options.panelPreset)
    : undefined
  if (options.panelPreset && !panelPreset) {
    process.stderr.write(chalk.red(copy.unknownPanelPreset(options.panelPreset)))
    process.exit(1)
  }

  const requestedMode = options.mode ?? (panelPreset ? 'persona-panel' : resolvedAgent ?? DEFAULT_CLI_AGENT_MODE)
  let chatOptions: ReturnType<typeof buildCliChatOptions>
  try {
    chatOptions = buildCliChatOptions({
      model: options.model,
      provider: options.provider,
      persona: options.persona,
      // Let the daemon's LLM router pick the right built-in graph for one-shot
      // asks. Simple requests still fall back to react, while broad coding,
      // research, or review work can select the specialist graph without making
      // the user know the mode name first.
      mode: requestedMode,
      personaIds: panelPreset ? [...panelPreset.personaIds] : undefined,
      skillRefs: options.skill?.length
        ? selectedChatSkillRefs(options.skill)
        : undefined,
      thinkingLevel: options.thinkingLevel,
      maxTokens,
      autonomy: options.autonomy,
    })
  } catch (err) {
    process.stderr.write(chalk.red(`${err instanceof Error ? err.message : err}\n`))
    process.exit(1)
  }
  if (requestedMode && chatOptions && !chatOptions.mode) {
    chatOptions.mode = requestedMode as typeof chatOptions.mode
  }

  if (options.background) {
    warnIgnoredInteractive(copy.interactiveIgnoredBackground(backgroundCommandSurface))
    try {
      const startedAt = Date.now()
      const started = await client.startBackgroundChat(input, options.session, chatOptions)
      if (outputFormat === 'json' && !options.wait) {
        output(started)
        return
      }
      if (outputFormat === 'stream-json') {
        emitBackgroundQueuedStreamJson(started)
      }
      if (!options.wait) {
        if (outputFormat === 'stream-json') {
          await emitBackgroundStreamJsonResult(started, { startedAt })
          return
        }
        console.log(copy.backgroundQueued(
          started.jobId,
          started.sessionId,
          backgroundCommandHints(started.jobId, backgroundCommandSurface),
        ))
        return
      }
      if (outputFormat !== 'json' && outputFormat !== 'stream-json') {
        process.stderr.write(chalk.gray(`${copy.backgroundQueued(
          started.jobId,
          started.sessionId,
          backgroundCommandHints(started.jobId, backgroundCommandSurface),
        )}\n`))
      }
      const pollMs =
        parsePositiveIntegerCliOption(options.pollMs, '--poll-ms')
        ?? DEFAULT_BACKGROUND_POLL_MS
      const textActionNotifier = outputFormat === 'json' || outputFormat === 'stream-json'
        ? undefined
        : createBackgroundActionNotifier(backgroundCommandSurface)
      const status = await waitForBackgroundChat(
        client,
        started.jobId,
        pollMs,
        outputFormat === 'stream-json' ? emitBackgroundStatusStreamJson : textActionNotifier,
      )
      if (outputFormat === 'json') {
        output(status)
      } else if (outputFormat === 'stream-json') {
        await emitBackgroundStreamJsonResult(status, {
          startedAt,
          failOnTerminalFailure: true,
        })
      } else {
        await emitBackgroundStatus(status, options)
      }
      if (isBackgroundUnsuccessfulStatus(status)) process.exit(1)
    } catch (err) {
      const errorOptions = options.session
        ? { hint: copy.rerunHint(options.session) }
        : undefined
      if (printApiError(err, errorOptions)) {
        process.exit(1)
      }
      console.error(chalk.red(`${copy.errorPrefix} ${err instanceof Error ? err.message : String(err)}`))
      process.exit(1)
    }
    return
  }

  // stream-json mode: forward the SSE agent event stream as NDJSON on
  // stdout and synthesize a final `result` line — the automation contract
  // for headless callers (Chat Transport Policy: no /chat fallback, a
  // failed stream surfaces as an error event + non-zero exit).
  if (outputFormat === 'stream-json') {
    warnIgnoredInteractive(copy.interactiveIgnoredFormat('stream-json'))
    await runAskStreamJson(client, input, options, chatOptions)
    return
  }

  // JSON mode: use regular HTTP
  if (outputFormat === 'json') {
    warnIgnoredInteractive(copy.interactiveIgnoredFormat('json'))
    try {
      const data = {
        data: await client.chat(input, options.session, chatOptions, {
          timeoutMs: resolveCliSyncChatTimeoutMs(),
        }),
      }
      if (options.output && data.data?.content) {
        // Under --json + --output, write the full envelope to the file
        // so jq users can pipe `cat result.json | jq .content`. The
        // explicit --strip-fences flag still wins (it signals "I want
        // a directly executable file"), so it short-circuits to plain
        // content with the fence stripped.
        const payload = options.stripFences
          ? stripCodeFences(data.data.content)
          : JSON.stringify(data.data, null, 2)
        await writeAskOutput(options.output, payload)
        console.log(chalk.green(copy.outputWrittenTo(options.output)))
      } else {
        output(data.data)
      }
      if (isUnsuccessfulAgentResult(data.data)) process.exitCode = 1
    } catch (err) { console.error(chalk.red(`${copy.errorPrefix} ${err}`)); process.exit(1) }
    return
  }

  // Streaming mode: use SSE
  let activeSessionId = options.session
  try {
    const useInteractivePrinter = Boolean(
      (options.interactive || shouldAutoUseInteractiveAskProgress(input, requestedMode))
        && !quiet
        && !options.output,
    )
    if (options.interactive && !useInteractivePrinter) {
      warnIgnoredInteractive(
        options.output
          ? copy.interactiveIgnoredOutput(options.output)
          : copy.interactiveIgnoredPipe,
      )
    }
    const printer = useInteractivePrinter
      ? createInteractiveCliChatStreamPrinter({
          approvalHintMode: 'cli',
          questionHintMode: 'cli',
          showArtifacts: true,
          showStateChanges: true,
          // Auto-selected progress stays compact. The explicit flag is the
          // opt-in for router/planner/reasoning diagnostics.
          showDiagnostics: Boolean(options.interactive),
          onSessionId: (sessionId) => {
            activeSessionId = sessionId
          },
        })
      : createAnswerOnlyCliChatStreamPrinter({
          quiet,
          suppressContent: Boolean(options.output),
          // `ask` is invoked from a regular shell, not the chat REPL, so
          // tell the printer to spell the approval hint as `sepilot
          // approve <id>` rather than `/approve <id>` — slash commands
          // would mislead an automation user back to a shell that isn't
          // running.
          approvalHintMode: 'cli',
          questionHintMode: 'cli',
          onSessionId: (sessionId) => {
            activeSessionId = sessionId
          },
        })
    // Stream watchdog: if no chunk arrives for `STREAM_IDLE_MS`, abort
    // and surface a friendly hang message. Catches the case where the
    // daemon finishes (200 OK + close) but the SSE reader doesn't see
    // EOF — cli would otherwise wait forever.
    const STREAM_IDLE_MS = resolveCliStreamIdleMs()
    const STREAM_CONNECT_MS = resolveCliStreamConnectMs()
    const aborter = new AbortController()
    const termination = bindStreamTerminationSignals(aborter, async () => {
      if (activeSessionId) await client.cancelActiveRun(activeSessionId)
    })
    let lastTick = Date.now()
    // Pending-decision run state: while an approval/question is outstanding the
    // run is blocked on a human, not hung. The idle watchdog stays quiet and a
    // periodic notice replaces the dead air so the operator can see the run is
    // alive and exactly which command unblocks it.
    const pendingDecisions = createCliPendingDecisionTracker()
    let lastPendingNoticeAt = 0
    const PENDING_NOTICE_MS = 30_000
    const watchdog = setInterval(() => {
      const pending = pendingDecisions.pending()
      if (pending) {
        const now = Date.now()
        if (now - Math.max(lastPendingNoticeAt, pending.since) >= PENDING_NOTICE_MS) {
          lastPendingNoticeAt = now
          process.stderr.write(
            chalk.yellow(
              `\n${formatPendingDecisionWaitNotice(pending, now - pending.since, activeSessionId)}\n`,
            ),
          )
        }
        return
      }
      if (Date.now() - lastTick > STREAM_IDLE_MS) {
        aborter.abort(new Error('stream-idle-timeout'))
      }
    }, Math.min(5000, STREAM_IDLE_MS / 4))
    const tick = (event: unknown) => {
      pendingDecisions.note(event)
      if (isSubstantiveAskStreamEvent(event)) {
        lastTick = Date.now()
      }
      return printer.handleEvent(event as Parameters<typeof printer.handleEvent>[0])
    }
    try {
      const res = await openChatStreamWithConnectTimeout(
        client.chatStream(input, options.session, chatOptions, { signal: aborter.signal }),
        {
          timeoutMs: STREAM_CONNECT_MS,
          abort: (error) => aborter.abort(error),
        },
      )
      if (!res.ok || !res.body) {
        // User-visible chat must not hide stream/proxy failures behind a
        // synchronous /chat fallback. Surface the real transport failure so
        // operators can fix timeouts, auth, or provider capacity instead of
        // waiting on a request that may hang until a proxy 504.
        throw new Error(await formatChatStreamFailure(res))
      } else {
        await forwardDaemonStreamWithResumeRecovery(res, {
          aborter,
          getSessionId: () => activeSessionId,
          onEvent: tick,
          openResumeStream: (sessionId) => client.resumeSessionStream(
            sessionId,
            undefined,
            { signal: aborter.signal },
          ),
          isTerminalEvent: isTerminalCliDaemonChatEvent,
          quiet,
          streamIdleMs: STREAM_IDLE_MS,
        })
      }
    } catch (error) {
      if (termination.interrupted()) {
        process.exitCode = termination.exitCode()
        return
      }
      throw error
    } finally {
      clearInterval(watchdog)
      await termination.waitForRemoteCancel()
      termination.dispose()
    }

    const fullContent = printer.getContent()

    const unresolvedDecision = pendingDecisions.pending()
    if (unresolvedDecision) {
      // The run did not stall: it ended (or was cut) while still waiting for a
      // decision this non-interactive invocation cannot make on the user's
      // behalf. Say so instead of letting a generic timeout take the blame.
      process.stderr.write(
        chalk.yellow(`\n${formatPendingDecisionAbortReason(unresolvedDecision, activeSessionId)}\n`),
      )
      process.exit(1)
    }

    if (printer.hadError()) {
      // Daemon already printed the error via the stream printer; just exit
      // non-zero so callers can detect the failure.
      process.exit(1)
    }

    if (options.output && fullContent) {
      const payload = options.stripFences
        ? stripCodeFences(fullContent)
        : fullContent
      await writeAskOutput(options.output, payload)
      if (!quiet) process.stderr.write(chalk.green(copy.outputWrittenToNewline(options.output)))
      return
    }
    if (!options.output && !fullContent) {
      console.error(chalk.red(copy.noResponse))
      process.exit(1)
    }
  } catch (err) {
    // Stream-failure copy lives in printStreamError (idle watchdog +
    // half-open SSE drop). Status-aware http copy lives in printApiError.
    // Both helpers print the friendly message themselves and return true
    // so we can drop straight into process.exit(1).
    if (printStreamError(err, { sessionId: activeSessionId })) {
      process.exit(1)
    }
    const errorOptions = activeSessionId
      ? { hint: copy.rerunHint(activeSessionId) }
      : undefined
    if (printApiError(err, errorOptions)) {
      process.exit(1)
    }
    console.error(chalk.red(`${copy.errorPrefix} ${err instanceof Error ? err.message : String(err)}`))
    process.exit(1)
  }
}

export const __testables = {
  bindStreamTerminationSignals,
}
