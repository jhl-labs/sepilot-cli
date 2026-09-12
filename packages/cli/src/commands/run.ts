import type {
  DaemonChatBackgroundStartResult,
  DaemonChatBackgroundStatusResult,
  DaemonChatResult,
  DaemonChatRequestControlOptions,
  DaemonSkill,
  DaemonSkillDetail,
} from '@sepilotd/api-client'
import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import { buildCliChatOptions, parsePositiveIntegerCliOption } from '../chat-options.js'
import { output, getOutputFormat } from '../output/formatter.js'
import { formatChatStreamFailure } from '../utils/chat-stream-error.js'
import { createInteractiveCliChatStreamPrinter } from '../utils/chat-stream-printer.js'
import {
  forwardDaemonStreamWithResumeRecovery,
  isTerminalCliDaemonChatEvent,
} from '../utils/stream-resume.js'
import { resolveCliStreamIdleMs } from '../utils/stream-idle.js'
import {
  openChatStreamWithConnectTimeout,
  resolveCliStreamConnectMs,
} from '../utils/stream-connect.js'
import { isUnsuccessfulAgentResult } from '../utils/run-outcome.js'
import { createCliPendingDecisionTracker } from '../utils/pending-decision.js'
import { resolveCliSyncChatTimeoutMs } from '../utils/sync-chat-timeout.js'
import {
  bindStreamTerminationSignals,
  DEFAULT_BACKGROUND_POLL_MS,
  createBackgroundActionNotifier,
  emitBackgroundQueuedStreamJson,
  emitBackgroundStatusStreamJson,
  emitBackgroundStreamJsonResult,
  isBackgroundUnsuccessfulStatus,
  isSubstantiveAskStreamEvent,
  waitForBackgroundChat,
} from './ask.js'

/**
 * Minimal subset of {@link DaemonClient} used by {@link runCommandImpl}.
 *
 * Only the methods the run command actually invokes are exposed so that
 * tests can supply lightweight fakes without reconstructing the full HTTP
 * client. The real {@link DaemonClient} satisfies this shape via
 * structural typing.
 */
interface RunCommandClient {
  health(): Promise<unknown>
  cancelActiveRun(sessionId: string): Promise<unknown>
  skill(name: string, options?: { cwd?: string; workspaceRoot?: string }): Promise<DaemonSkillDetail>
  searchSkills(
    query: string,
    options?: { cwd?: string; workspaceRoot?: string },
  ): Promise<DaemonSkill[]>
  chat(
    message: string,
    sessionId: string | undefined,
    options: Record<string, unknown> | undefined,
    request?: DaemonChatRequestControlOptions,
  ): Promise<DaemonChatResult | null>
  chatStream(
    message: string,
    sessionId: string | undefined,
    options: Record<string, unknown> | undefined,
    request?: { signal?: AbortSignal },
  ): Promise<Response>
  resumeSessionStream(
    sessionId: string,
    options?: { force?: boolean },
    request?: { signal?: AbortSignal },
  ): Promise<Response>
  startBackgroundChat(
    message: string,
    sessionId: string | undefined,
    options: Record<string, unknown> | undefined,
  ): Promise<DaemonChatBackgroundStartResult>
  backgroundChatStatus(jobId: string): Promise<DaemonChatBackgroundStatusResult>
}

export interface RunCommandImplDeps {
  skillName: string
  input?: string
  model?: string
  maxTokens?: number
  maxIterations?: number
  background?: boolean
  wait?: boolean
  pollMs?: number
  client: RunCommandClient
}

async function runSkillStreamJson(
  client: RunCommandClient,
  message: string,
  chatOptions: Record<string, unknown>,
): Promise<void> {
  const startedAt = Date.now()
  let sessionId: string | undefined
  let content = ''
  let usage: unknown
  let stopReason: unknown
  let hadError = false

  const emit = (event: unknown): void => {
    process.stdout.write(`${JSON.stringify(event)}\n`)
  }

  const STREAM_IDLE_MS = resolveCliStreamIdleMs()
  const STREAM_CONNECT_MS = resolveCliStreamConnectMs()
  const aborter = new AbortController()
  const termination = bindStreamTerminationSignals(aborter, async () => {
    if (sessionId) await client.cancelActiveRun(sessionId)
  })
  const pendingDecisions = createCliPendingDecisionTracker()
  let lastTick = Date.now()
  const watchdog = setInterval(
    () => {
      if (pendingDecisions.pending()) return
      if (Date.now() - lastTick > STREAM_IDLE_MS) {
        aborter.abort(new Error('stream-idle-timeout'))
      }
    },
    Math.min(5000, STREAM_IDLE_MS / 4),
  )

  const tick = (event: unknown): void => {
    pendingDecisions.note(event)
    if (isSubstantiveAskStreamEvent(event)) lastTick = Date.now()
    if (event && typeof event === 'object') {
      const record = event as Record<string, unknown>
      if (typeof record.sessionId === 'string') sessionId = record.sessionId
      if (record.type === 'text_delta' && typeof record.text === 'string') content += record.text
      if (record.type === 'content' && typeof record.content === 'string') content += record.content
      if (record.type === 'message' && typeof record.content === 'string') content = record.content
      if (record.type === 'done') { usage = record.usage; stopReason = record.stopReason }
      if (record.type === 'error') hadError = true
    }
    emit(event)
  }

  try {
    const res = await openChatStreamWithConnectTimeout(
      client.chatStream(message, undefined, chatOptions, { signal: aborter.signal }),
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
    if (!termination.interrupted()) emit({ type: 'error', error: err instanceof Error ? err.message : String(err) })
  } finally {
    clearInterval(watchdog)
    await termination.waitForRemoteCancel()
    termination.dispose()
  }

  hadError ||= termination.interrupted() || isUnsuccessfulAgentResult({ content, stopReason })
  await new Promise<void>((resolve) => {
    const wrote = process.stdout.write(
      `${JSON.stringify({
        type: 'result',
        subtype: hadError ? 'error' : 'success',
        sessionId,
        durationMs: Date.now() - startedAt,
        usage,
        stopReason,
        content,
      })}\n`,
    )
    if (wrote) resolve()
    else process.stdout.once('drain', resolve)
  })
  if (termination.interrupted()) process.exitCode = termination.exitCode()
  else if (hadError) process.exitCode = 1
}

function formatUnsuccessfulBackgroundSkillRun(
  status: DaemonChatBackgroundStatusResult,
): string {
  const lines = status.status === 'cancelled'
    ? ['Background skill run cancelled']
    : [
        'Background skill run failed',
        `  error: ${status.error?.message ?? 'unknown error'}`,
      ]
  return [
    ...lines,
    `  job: ${status.jobId}`,
    `  session: ${status.sessionId}`,
    `  check: sepilot ask --background-status ${status.jobId}`,
    '  list: sepilot ask --background-list',
  ].join('\n')
}

function formatQueuedBackgroundSkillRun(started: DaemonChatBackgroundStartResult): string[] {
  return [
    `session: ${started.sessionId}`,
    `check: sepilot ask --background-status ${started.jobId}`,
    'list: sepilot ask --background-list',
    `save: sepilot ask --background-status ${started.jobId} --output <file>`,
    `cancel: sepilot ask --background-cancel ${started.jobId}`,
  ]
}

/**
 * Pure, dependency-injected implementation of `sepilotd run <skill>`.
 *
 * The daemon resolves skill content server-side from `skillRefs`; the CLI
 * just sends `skillRefs: [{ name }]` alongside the user request and the
 * daemon injects the skill into the system prompt before the agent loop
 * runs.
 *
 * Streaming behaviour mirrors `sepilot ask`: SSE is the default so the
 * user sees tool calls and partial messages as they happen — important
 * for skills whose agent loop runs for minutes (e.g. \`software-architect\`
 * reverse-engineering a codebase). Non-streaming \`POST /chat\` is used
 * only under \`--json\` for callers that explicitly need one final envelope.
 */
export async function runCommandImpl(deps: RunCommandImplDeps): Promise<void> {
  const cwd = process.cwd()
  const skillContext = { cwd, workspaceRoot: cwd }
  let skill: DaemonSkillDetail
  try {
    skill = await deps.client.skill(deps.skillName, skillContext)
  } catch {
    console.error(chalk.red(`Skill not found: ${deps.skillName}`))
    try {
      const searchData = await deps.client.searchSkills(deps.skillName, skillContext)
      if (searchData?.length) {
        console.log(chalk.gray('\nDid you mean:'))
        for (const candidate of searchData) {
          console.log(chalk.gray(`  ${candidate.name} — ${candidate.description}`))
        }
      }
    } catch {
      // Suggestion lookup is best-effort.
    }
    process.exit(1)
  }

  const userInput = deps.input ?? `Execute the "${deps.skillName}" skill`
  const baseOptions = buildCliChatOptions({ model: deps.model, maxTokens: deps.maxTokens }) ?? {}
  const chatOptions: Record<string, unknown> = {
    ...baseOptions,
    skillRefs: [{ name: deps.skillName }],
  }
  if (typeof deps.maxIterations === 'number' && deps.maxIterations >= 1) {
    chatOptions.maxIterations = deps.maxIterations
  }

  const outputFormat = getOutputFormat()
  const canPrintHumanHeader = outputFormat === 'text'
  if (canPrintHumanHeader) {
    console.log(chalk.cyan(`Running skill: ${skill.metadata.name} v${skill.metadata.version}`))
    console.log(chalk.gray(`${skill.metadata.description}\n`))
  }

  if (deps.background) {
    try {
      const startedAt = Date.now()
      const started = await deps.client.startBackgroundChat(userInput, undefined, chatOptions)
      if (outputFormat === 'json' && !deps.wait) {
        output(started)
        return
      }
      if (outputFormat === 'stream-json') {
        emitBackgroundQueuedStreamJson(started)
      }
      if (!deps.wait) {
        if (outputFormat === 'stream-json') {
          await emitBackgroundStreamJsonResult(started, { startedAt })
          return
        }
        console.log(chalk.cyan(`Background skill run queued: ${started.jobId}`))
        for (const line of formatQueuedBackgroundSkillRun(started)) {
          console.log(chalk.gray(line))
        }
        return
      }
      if (outputFormat !== 'json' && outputFormat !== 'stream-json') {
        process.stderr.write(chalk.gray(
          [
            `Background skill run queued: ${started.jobId}`,
            ...formatQueuedBackgroundSkillRun(started),
          ].join('\n') + '\n',
        ))
      }
      const status = await waitForBackgroundChat(
        deps.client,
        started.jobId,
        deps.pollMs ?? DEFAULT_BACKGROUND_POLL_MS,
        outputFormat === 'stream-json'
          ? emitBackgroundStatusStreamJson
          : outputFormat === 'json'
            ? undefined
            : createBackgroundActionNotifier(),
      )
      if (outputFormat === 'json') {
        output(status)
      } else if (outputFormat === 'stream-json') {
        await emitBackgroundStreamJsonResult(status, {
          startedAt,
          failOnTerminalFailure: true,
        })
      } else if (status.status === 'completed') {
        console.log(status.content?.trim() || chalk.gray(`Background skill run completed: ${status.jobId}`))
      } else if (status.status === 'cancelled') {
        console.error(chalk.red(formatUnsuccessfulBackgroundSkillRun(status)))
        process.exit(1)
      } else {
        console.error(chalk.red(formatUnsuccessfulBackgroundSkillRun(status)))
        process.exit(1)
      }
      if (isBackgroundUnsuccessfulStatus(status)) process.exit(1)
    } catch (err) {
      console.error(chalk.red(`Failed: ${err instanceof Error ? err.message : err}`))
      process.exit(1)
    }
    return
  }

  // JSON output: keep the legacy non-streaming envelope so `--json` callers
  // get a single parseable object.
  if (outputFormat === 'json') {
    try {
      const data = await deps.client.chat(userInput, undefined, chatOptions, {
        timeoutMs: resolveCliSyncChatTimeoutMs(),
      })
      if (data) output(data)
      if (isUnsuccessfulAgentResult(data)) process.exitCode = 1
    } catch (err) {
      console.error(chalk.red(`Failed: ${err instanceof Error ? err.message : err}`))
      process.exit(1)
    }
    return
  }

  if (outputFormat === 'stream-json') {
    await runSkillStreamJson(deps.client, userInput, chatOptions)
    return
  }

  // Default: stream tool events + partial output to the terminal.
  try {
    // Use the interactive printer (tool calls, results, thinking previews,
    // phase changes, context-compaction notes) so the user sees what the
    // agent is actually doing during a multi-minute skill run. The
    // answer-only printer is too quiet for an end-to-end skill like
    // `software-architect`, where the agent's final answer arrives only
    // after dozens of tool calls.
    let activeSessionId: string | undefined
    const printer = createInteractiveCliChatStreamPrinter({
      approvalHintMode: 'cli',
      questionHintMode: 'cli',
      showArtifacts: true,
      showDiagnostics: false,
      onSessionId: (sessionId) => {
        activeSessionId = sessionId
      },
    })
    const STREAM_IDLE_MS = resolveCliStreamIdleMs()
    const STREAM_CONNECT_MS = resolveCliStreamConnectMs()
    const aborter = new AbortController()
    const termination = bindStreamTerminationSignals(aborter, async () => {
      if (activeSessionId) await deps.client.cancelActiveRun(activeSessionId)
    })
    const pendingDecisions = createCliPendingDecisionTracker()
    let lastTick = Date.now()
    const watchdog = setInterval(
      () => {
        if (pendingDecisions.pending()) return
        if (Date.now() - lastTick > STREAM_IDLE_MS) {
          aborter.abort(new Error('stream-idle-timeout'))
        }
      },
      Math.min(5000, STREAM_IDLE_MS / 4),
    )
    const tick = (event: unknown) => {
      pendingDecisions.note(event)
      if (isSubstantiveAskStreamEvent(event)) lastTick = Date.now()
      if (event && typeof event === 'object') {
        const sessionId = (event as { sessionId?: unknown }).sessionId
        if (typeof sessionId === 'string') activeSessionId = sessionId
      }
      return printer.handleEvent(event as Parameters<typeof printer.handleEvent>[0])
    }

    try {
      const res = await openChatStreamWithConnectTimeout(
        deps.client.chatStream(userInput, undefined, chatOptions, { signal: aborter.signal }),
        { timeoutMs: STREAM_CONNECT_MS, abort: (error) => aborter.abort(error) },
      )
      if (!res.ok || !res.body) throw new Error(await formatChatStreamFailure(res))
      await forwardDaemonStreamWithResumeRecovery(res, {
        aborter,
        getSessionId: () => activeSessionId,
        onEvent: tick,
        openResumeStream: (resumeSessionId) => deps.client.resumeSessionStream(
          resumeSessionId, undefined, { signal: aborter.signal },
        ),
        isTerminalEvent: isTerminalCliDaemonChatEvent,
        streamIdleMs: STREAM_IDLE_MS,
      })
    } catch (error) {
      if (!termination.interrupted()) throw error
    } finally {
      clearInterval(watchdog)
      await termination.waitForRemoteCancel()
      termination.dispose()
    }
    if (termination.interrupted()) {
      process.exitCode = termination.exitCode()
      return
    }

    if (printer.hadError()) process.exit(1)
  } catch (err) {
    console.error(chalk.red(`Failed: ${err instanceof Error ? err.message : err}`))
    process.exit(1)
  }
}

export async function runCommand(
  skillName: string,
  options: {
    url?: string
    input?: string
    model?: string
    maxTokens?: string
    maxIters?: string
    background?: boolean
    wait?: boolean
    pollMs?: string
  },
): Promise<void> {
  const client = new DaemonClient(options.url)

  try {
    await ensureDaemon(client, { url: options.url })
  } catch (err) {
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    process.exit(1)
  }

  const maxIterations = (() => {
    if (!options.maxIters) return undefined
    const value = Number.parseInt(options.maxIters, 10)
    if (!Number.isFinite(value) || value < 1) {
      console.error(chalk.red(`--max-iters must be an integer ≥ 1 (got: ${options.maxIters})`))
      process.exit(1)
    }
    return value
  })()
  const maxTokens = (() => {
    try {
      return parsePositiveIntegerCliOption(options.maxTokens, '--max-tokens')
    } catch (err) {
      console.error(chalk.red(err instanceof Error ? err.message : String(err)))
      process.exit(1)
    }
  })()
  const pollMs = (() => {
    try {
      return parsePositiveIntegerCliOption(options.pollMs, '--poll-ms')
    } catch (err) {
      console.error(chalk.red(err instanceof Error ? err.message : String(err)))
      process.exit(1)
    }
  })()

  await runCommandImpl({
    skillName,
    input: options.input,
    model: options.model,
    maxTokens,
    maxIterations,
    background: options.background,
    wait: options.wait,
    pollMs,
    client: client as unknown as RunCommandClient,
  })
}
