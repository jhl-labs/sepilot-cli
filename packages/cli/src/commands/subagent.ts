import { readFile } from 'node:fs/promises'
import chalk from 'chalk'
import type {
  SubagentBackgroundDispatchResult,
  SubagentDelegationCategoriesResult,
  SubagentDispatchInput,
  SubagentDispatchResult,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const SUBAGENT_COPY = {
  en: {
    maxIterationsPositive: (raw: string) =>
      `--max-iterations must be a positive integer (got ${raw})`,
    truncated: ' (truncated)',
    dispatchFailed: (msg: string) => `subagent dispatch failed: ${msg}\n`,
    backgroundQueued: 'Background subagent queued',
    backgroundJob: (jobId: string) => `  job: ${jobId}`,
    backgroundStatus: (jobId: string) => `  status: sepilot jobs status ${jobId}`,
    backgroundWait: (jobId: string) => `  wait: sepilot jobs resume ${jobId}`,
    backgroundCancel: (jobId: string) => `  cancel: sepilot jobs cancel ${jobId}`,
    categoriesFailed: (msg: string) => `subagent categories failed: ${msg}\n`,
  },
  ko: {
    maxIterationsPositive: (raw: string) =>
      `--max-iterations는 양의 정수여야 합니다 (받은 값: ${raw})`,
    truncated: ' (잘림)',
    dispatchFailed: (msg: string) => `서브에이전트 디스패치 실패: ${msg}\n`,
    backgroundQueued: '백그라운드 서브에이전트 작업 등록됨',
    backgroundJob: (jobId: string) => `  job: ${jobId}`,
    backgroundStatus: (jobId: string) => `  확인: sepilot jobs status ${jobId}`,
    backgroundWait: (jobId: string) => `  대기: sepilot jobs resume ${jobId}`,
    backgroundCancel: (jobId: string) => `  취소: sepilot jobs cancel ${jobId}`,
    categoriesFailed: (msg: string) => `서브에이전트 카테고리 조회 실패: ${msg}\n`,
  },
} as const

function subagentCopy() {
  return SUBAGENT_COPY[detectCliLocale()] ?? SUBAGENT_COPY.en
}

export interface SubagentDispatchOptions {
  url?: string
  system?: string
  systemFile?: string
  category?: string
  agent?: string
  maxIterations?: string
  tools?: string
  model?: string
  parentSession?: string
  background?: boolean
}

/**
 * Minimal client surface used by `subagentDispatchCommand`. Real CLI runs
 * pass `DaemonClient` (which already implements this); unit tests can
 * inject a stub without standing up the full http transport.
 */
export interface SubagentDaemonClientLike {
  dispatchSubagent(input: SubagentDispatchInput): Promise<SubagentDispatchResult>
  dispatchSubagentBackground?(
    input: SubagentDispatchInput,
  ): Promise<SubagentBackgroundDispatchResult>
  listSubagentCategories?(): Promise<SubagentDelegationCategoriesResult>
}

export interface RunSubagentDispatchInput {
  prompt: string
  options: SubagentDispatchOptions
  client: SubagentDaemonClientLike
  /** Read system prompt from disk. Tests inject a fake reader to avoid I/O. */
  readSystemFile?: (path: string) => Promise<string>
  /** Output sink — defaults to the global `output()` formatter. Tests can
   * capture rendered text without mucking with stdout. */
  emit?: (result: SubagentDispatchResult | SubagentBackgroundDispatchResult) => void
}

function parseToolList(raw?: string): string[] | undefined {
  if (!raw) return undefined
  const tools = raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
  return tools.length > 0 ? tools : undefined
}

function parseMaxIterations(raw?: string): number | undefined {
  if (!raw) return undefined
  const n = Number.parseInt(raw, 10)
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(subagentCopy().maxIterationsPositive(JSON.stringify(raw)))
  }
  return n
}

function defaultEmit(result: SubagentDispatchResult): void {
  const copy = subagentCopy()
  output(result, (r) => {
    const head = `${chalk.cyan(`[subagent ${r.sessionId}]`)} status=${r.status} iterations=${r.iterations} tokens=${r.usage.inputTokens}+${r.usage.outputTokens}${r.truncated ? copy.truncated : ''}`
    return r.output ? `${head}\n\n${r.output}` : head
  })
}

function isBackgroundResult(
  result: SubagentDispatchResult | SubagentBackgroundDispatchResult,
): result is SubagentBackgroundDispatchResult {
  return 'jobId' in result
}

function defaultDispatchEmit(
  result: SubagentDispatchResult | SubagentBackgroundDispatchResult,
): void {
  if (isBackgroundResult(result)) {
    const copy = subagentCopy()
    output(result, (r) => [
      `${chalk.cyan(`[job ${r.jobId}]`)} ${copy.backgroundQueued}`,
      copy.backgroundJob(r.jobId),
      copy.backgroundStatus(r.jobId),
      copy.backgroundWait(r.jobId),
      copy.backgroundCancel(r.jobId),
    ].join('\n'))
    return
  }
  defaultEmit(result)
}

/**
 * Pure async core of the `subagent dispatch` cli command. Splitting it
 * out keeps it unit-testable without spawning a daemon — the integration
 * smoke is covered by the daemon-side route test.
 */
export async function runSubagentDispatch(
  input: RunSubagentDispatchInput,
): Promise<SubagentDispatchResult | SubagentBackgroundDispatchResult> {
  const { prompt, options, client } = input
  const reader = input.readSystemFile ?? ((path) => readFile(path, 'utf-8'))
  const emit = input.emit ?? defaultDispatchEmit

  let system = options.system
  if (options.systemFile) {
    system = await reader(options.systemFile)
  }

  const request: SubagentDispatchInput = {
    prompt,
    system,
    category: options.category,
    agentId: options.agent,
    maxIterations: parseMaxIterations(options.maxIterations),
    tools: parseToolList(options.tools),
    model: options.model,
    parentSessionId: options.parentSession,
  }

  const result = options.background
    ? await dispatchBackground(client, request)
    : await client.dispatchSubagent(request)

  emit(result)
  return result
}

async function dispatchBackground(
  client: SubagentDaemonClientLike,
  request: SubagentDispatchInput,
): Promise<SubagentBackgroundDispatchResult> {
  if (!client.dispatchSubagentBackground) {
    throw new Error('dispatchSubagentBackground is unavailable')
  }
  return client.dispatchSubagentBackground(request)
}

export async function subagentDispatchCommand(
  prompt: string,
  options: SubagentDispatchOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const result = await runSubagentDispatch({ prompt, options, client })
    if (!isBackgroundResult(result) && result.status === 'failed') {
      // Surface failures with a non-zero exit so automation pipelines
      // can detect them (the human-friendly banner already went to
      // stdout via `emit`).
      process.exit(1)
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(chalk.red(subagentCopy().dispatchFailed(message)))
    process.exit(1)
  }
}

function defaultCategoriesEmit(result: SubagentDelegationCategoriesResult): void {
  output(result, (r) => {
    return r.categories
      .map((category) => {
        const tools =
          category.toolHints === null ? 'parent allowed tools' : category.toolHints.join(', ')
        return `${chalk.cyan(category.id)}\n  ${category.description}\n  maxIterations=${category.defaultMaxIterations}\n  tools=${tools}`
      })
      .join('\n\n')
  })
}

export async function runSubagentCategories(input: {
  client: SubagentDaemonClientLike
  emit?: (result: SubagentDelegationCategoriesResult) => void
}): Promise<SubagentDelegationCategoriesResult> {
  if (!input.client.listSubagentCategories) {
    throw new Error('listSubagentCategories is unavailable')
  }
  const result = await input.client.listSubagentCategories()
  ;(input.emit ?? defaultCategoriesEmit)(result)
  return result
}

export async function subagentCategoriesCommand(options: { url?: string }): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    await runSubagentCategories({ client })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(chalk.red(subagentCopy().categoriesFailed(message)))
    process.exit(1)
  }
}
