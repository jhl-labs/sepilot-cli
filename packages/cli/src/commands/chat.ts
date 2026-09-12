import * as readline from 'node:readline'
import chalk from 'chalk'
import {
  delegationHealthDetail,
  delegationHealthLabel,
  createTokenSpeedTracker,
  formatTokenSpeedStats,
  formatToolCall,
  resolveApprovalDecision,
  runDaemonWsClientChat,
  resumableRunCopy,
  shouldFallbackFromInteractiveWsError,
  streamChatWithFallback,
} from '@sepilotd/api-client'
import type {
  DaemonAgentMode,
  DaemonChatStreamPayload,
  DaemonResumableRun,
  TokenSpeedTracker,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { DaemonWsClient } from '../client/ws.js'
import { buildCliChatOptions, parsePositiveIntegerCliOption } from '../chat-options.js'
import { formatProviderModelBadges } from '../utils/provider-display.js'
import { formatChatStreamFailure } from '../utils/chat-stream-error.js'
import { createInteractiveCliChatStreamPrinter } from '../utils/chat-stream-printer.js'
import { friendlyErrorMessage, printApiError, printStreamError } from '../utils/error-message.js'
import { createHangulReadlineInput } from '../utils/hangul-readline-input.js'
import { composeHangul, stripBracketedPasteDelimiters } from '../tui/utils/hangul.js'
import {
  compareDecisions,
  formatPastTime,
  isStaleRule,
} from './decisions.js'
import { buildStats, formatStatsSnapshot } from './stats.js'
import { formatQueuedSteerNote, submitSteer } from '../steer-shared.js'
import {
  forwardDaemonStreamWithResumeRecovery,
  isTerminalCliDaemonChatEvent,
} from '../utils/stream-resume.js'
import { resolveCliStreamIdleMs } from '../utils/stream-idle.js'
import {
  openChatStreamWithConnectTimeout,
  resolveCliStreamConnectMs,
} from '../utils/stream-connect.js'

const MODEL_STREAM_WAITING_THINKING = 'Still waiting for the model stream...'

function isSubstantiveChatStreamEvent(event: unknown): boolean {
  if (!event || typeof event !== 'object') return true
  const candidate = event as { type?: unknown; content?: unknown; text?: unknown; sessionId?: unknown }
  if (
    candidate.type === 'thinking'
    && (candidate.content === MODEL_STREAM_WAITING_THINKING || candidate.text === MODEL_STREAM_WAITING_THINKING)
  ) {
    return false
  }
  if (candidate.type === 'state_change') return false
  if (
    candidate.type === undefined
    && typeof candidate.sessionId === 'string'
    && Object.keys(candidate).every((key) => key === 'sessionId')
  ) {
    return false
  }
  return true
}

function createChatStreamIdleWatchdog(onTimeout: (error: Error) => void): {
  streamIdleMs: number
  touch: (event: unknown) => void
  clear: () => void
  timeoutError: () => Error | undefined
} {
  const streamIdleMs = resolveCliStreamIdleMs()
  let lastTick = Date.now()
  let timeoutError: Error | undefined
  const timer = setInterval(() => {
    if (Date.now() - lastTick <= streamIdleMs || timeoutError) return
    timeoutError = new Error('stream-idle-timeout')
    onTimeout(timeoutError)
  }, Math.min(5000, Math.max(250, streamIdleMs / 4)))
  timer.unref?.()
  return {
    streamIdleMs,
    touch: (event) => {
      if (isSubstantiveChatStreamEvent(event)) {
        lastTick = Date.now()
      }
    },
    clear: () => clearInterval(timer),
    timeoutError: () => timeoutError,
  }
}

function abortSignalAsPromise(
  signal: AbortSignal,
  getError: () => Error | undefined,
): Promise<never> {
  return new Promise((_, reject) => {
    if (signal.aborted) {
      reject(getError() ?? new Error('stream-idle-timeout'))
      return
    }
    signal.addEventListener('abort', () => {
      reject(getError() ?? new Error('stream-idle-timeout'))
    }, { once: true })
  })
}

function openBoundedChatStream(
  responsePromise: Promise<Response>,
  aborter: AbortController,
): Promise<Response> {
  return openChatStreamWithConnectTimeout(responsePromise, {
    timeoutMs: resolveCliStreamConnectMs(),
    abort: (error) => aborter.abort(error),
  })
}

/**
 * Format the autonomy level so the user can tell at a glance how
 * supervised the agent is. `autonomous` is highlighted because it is the
 * level where the daemon will pick tools without asking — running the cli
 * there without realising it is the most common foot-gun.
 */
function formatAutonomyLabel(level: string | undefined): string {
  switch (level) {
    case 'autonomous':
      return chalk.red('autonomous (approval-required tools blocked)')
    case 'accept-edits':
      return chalk.yellow('accept-edits (auto-allows fs.write)')
    case 'workspace-write':
      return chalk.magenta('workspace-write (workspace edits auto-allowed)')
    case 'readonly':
      return chalk.cyan('readonly (no mutations)')
    case 'supervised':
      return chalk.green('supervised (every risky tool needs approval)')
    default:
      return chalk.gray(level ?? 'unknown')
  }
}

function relativeTimeLabel(deltaMs: number): string {
  if (deltaMs <= 0) return 'expired'
  const totalSeconds = Math.floor(deltaMs / 1000)
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remMin = minutes % 60
  return remMin === 0 ? `${hours}h` : `${hours}h${remMin}m`
}

function formatModelSnapshot(snapshot: Awaited<ReturnType<DaemonClient['model']>>): string {
  const current = snapshot.current
    ? `${snapshot.current.providerId} / ${snapshot.current.modelId}`
    : 'not configured'
  const lines = [`Current model: ${current}`]
  if (snapshot.lines.length > 0) {
    lines.push('Available:', ...snapshot.lines.map((line) => `  ${line}`))
  } else {
    lines.push('Available: (no providers configured)')
  }
  lines.push('Switch with /model <model> or /model <provider>/<model>. Pull Ollama models with /model pull <model>.')
  return lines.join('\n')
}

function formatSkillRows(skills: Awaited<ReturnType<DaemonClient['skills']>>, limit = 20): string[] {
  if (skills.length === 0) return ['  (none)']
  const rows = skills.slice(0, limit).map((skill) => {
    const enabled = skill.enabled === false ? 'disabled' : 'enabled'
    const description = skill.description.replace(/\s+/g, ' ').trim()
    const clipped = description.length > 72 ? `${description.slice(0, 69)}...` : description
    return `  ${skill.id}@${skill.version} [${enabled}]${clipped ? ` - ${clipped}` : ''}`
  })
  if (skills.length > limit) rows.push(`  ... ${skills.length - limit} more`)
  return rows
}

function formatScheduledTaskRows(jobs: Awaited<ReturnType<DaemonClient['listScheduledTasks']>>, limit = 20): string[] {
  const pending = jobs
    .filter((job) => job.enabled !== false && job.status === 'pending')
    .sort((a, b) => a.nextRunAt - b.nextRunAt)
  if (pending.length === 0) return ['  (none)']
  const rows = pending.slice(0, limit).map((job) => {
    const when = Number.isFinite(job.nextRunAt)
      ? new Date(job.nextRunAt).toISOString()
      : 'unknown'
    return `  ${job.id} - ${job.name} (${when})`
  })
  if (pending.length > limit) rows.push(`  ... ${pending.length - limit} more`)
  return rows
}

async function printSelfSnapshot(
  client: DaemonClient,
  action: 'overview' | 'skills' | 'capabilities' | 'schedules',
): Promise<void> {
  const [model, config, skills, schedules] = await Promise.all([
    client.model().catch(() => null),
    client.config().catch(() => null),
    client.skills({ includeDisabled: true }).catch(() => []),
    client.listScheduledTasks({ all: true }).catch(() => []),
  ])

  if (action === 'skills') {
    console.log(chalk.yellow(`Installed skills: ${skills.length}`))
    console.log(chalk.gray(formatSkillRows(skills).join('\n')))
    return
  }

  if (action === 'schedules') {
    console.log(chalk.yellow('Scheduled tasks:'))
    console.log(chalk.gray(formatScheduledTaskRows(schedules).join('\n')))
    return
  }

  const current = model?.current
    ? `${model.current.providerId} / ${model.current.modelId}`
    : 'not configured'
  const autonomy = config?.agent?.autonomy ?? 'unknown'
  console.log(chalk.yellow(action === 'capabilities' ? 'Capabilities:' : 'Agent snapshot:'))
  console.log(chalk.gray(`  Model: ${current}`))
  console.log(chalk.gray(`  Autonomy: ${autonomy}`))
  if (config?.device?.name) {
    console.log(chalk.gray(`  Device: ${config.device.name} (${config.device.role ?? 'unknown'})`))
  }
  console.log(chalk.gray(`  Skills: ${skills.length}`))
  console.log(chalk.gray(`  Scheduled tasks: ${schedules.filter((job) => job.status === 'pending').length}`))
  if (model?.lines?.length) {
    console.log(chalk.gray('  Available models:'))
    for (const line of model.lines.slice(0, 12)) console.log(chalk.gray(`    ${line}`))
    if (model.lines.length > 12) console.log(chalk.gray(`    ... ${model.lines.length - 12} more`))
  }
  if (action === 'capabilities') {
    console.log(chalk.gray('  Skill list:'))
    console.log(chalk.gray(formatSkillRows(skills, 10).join('\n')))
  }
  console.log(chalk.gray('  Limits: registered tools and installed/enabled skills only; external skills require approval and validation.'))
}

function resumableRunActionHint(run: DaemonResumableRun): string {
  switch (run.mode) {
    case 'replay-risky':
      return '/resume --force'
    case 'replay-safe':
      return '/resume'
    default:
      return '/resume'
  }
}

export async function chatCommand(options: { url?: string; model?: string; provider?: string; maxTokens?: string; session?: string }) {
  const httpClient = new DaemonClient(options.url)
  let currentMaxTokens: number | undefined
  try {
    currentMaxTokens = parsePositiveIntegerCliOption(options.maxTokens, '--max-tokens')
  } catch (err) {
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    process.exit(1)
  }

  try {
    const health = await httpClient.health()
    console.log(chalk.gray(`Connected to sepilotd ${health.version}`))
  } catch {
    console.error(chalk.red('Cannot connect to sepilotd. Is the daemon running?'))
    process.exit(1)
  }

  let wsClient: DaemonWsClient | null = null
  try {
    wsClient = new DaemonWsClient(options.url)
    await wsClient.connect()
  } catch {
    wsClient = null
  }

  let sessionId: string | undefined = options.session
  if (sessionId) {
    try {
      const session = await httpClient.session(sessionId)
      console.log(chalk.gray(`Resuming session: ${session.title ?? sessionId}`))
    } catch {
      console.error(chalk.red(`Session ${sessionId} not found`))
      process.exit(1)
    }
  }
  let currentModel = options.model
  let currentProvider = options.provider
  let currentMode: DaemonAgentMode | undefined
  let currentThinkingLevel: string | undefined
  const tokenSpeedTracker = createTokenSpeedTracker()
  let rlRef: readline.Interface | null = null

  // Track pending approvals/questions so the readline prompt can show a counter.
  // The sets are fed live from stream frames and reconciled
  // after every send/decision via /api/v1/sessions/:id (which is the
  // authoritative list, in case the user has multiple cli windows open or
  // a previous decision timed out).
  const pendingApprovals = new Map<string, string | undefined>()
  const pendingQuestions = new Map<string, string | undefined>()
  const updatePromptIndicator = () => {
    if (!rlRef) return
    const parts = [
      pendingApprovals.size > 0
        ? `${pendingApprovals.size} approval${pendingApprovals.size === 1 ? '' : 's'}`
        : null,
      pendingQuestions.size > 0
        ? `${pendingQuestions.size} question${pendingQuestions.size === 1 ? '' : 's'}`
        : null,
    ].filter(Boolean)
    const prefix = parts.length > 0 ? chalk.yellow(`[${parts.join(' · ')} pending] `) : ''
    rlRef.setPrompt(prefix + chalk.blue('> '))
  }
  const syncPendingWaits = async (targetSessionId = sessionId) => {
    if (!targetSessionId) {
      pendingApprovals.clear()
      pendingQuestions.clear()
      updatePromptIndicator()
      return
    }
    try {
      const session = await httpClient.session(targetSessionId)
      for (const [requestId, sourceSessionId] of pendingApprovals) {
        if (!sourceSessionId || sourceSessionId === session.id) {
          pendingApprovals.delete(requestId)
        }
      }
      for (const item of session.pendingApprovals ?? []) {
        pendingApprovals.set(item.requestId, session.id)
      }
      for (const [questionId, sourceSessionId] of pendingQuestions) {
        if (!sourceSessionId || sourceSessionId === session.id) {
          pendingQuestions.delete(questionId)
        }
      }
      for (const item of session.pendingQuestions ?? []) {
        pendingQuestions.set(item.id, item.sessionId ?? session.id)
      }
      updatePromptIndicator()
    } catch { /* best-effort indicator */ }
  }

  const createPromptEventPrinter = (
    onStreamSessionId?: (nextSessionId: string) => void,
  ) => createInteractiveCliChatStreamPrinter({
    showDiagnostics: false,
    onSessionId: (nextSessionId) => {
      sessionId = nextSessionId
      onStreamSessionId?.(nextSessionId)
    },
    onApprovalRequested: (requestId, _toolName, sourceSessionId) => {
      pendingApprovals.set(requestId, sourceSessionId ?? sessionId)
      updatePromptIndicator()
    },
    onQuestionRequested: (questionId, sourceSessionId) => {
      pendingQuestions.set(questionId, sourceSessionId ?? sessionId)
      updatePromptIndicator()
    },
  })

  const createTrackedPromptEventHandler = (
    printer: ReturnType<typeof createPromptEventPrinter>,
  ) => {
    const startedAt = Date.now()
    return async (payload: DaemonChatStreamPayload) => {
      if ('type' in payload && payload.type === 'done') {
        tokenSpeedTracker.recordRun({
          startedAt,
          finishedAt: Date.now(),
          usage: payload.usage,
        })
      }
      await printer.handleEvent(payload)
    }
  }

  const sendPrompt = async (prompt: string, targetSessionId = sessionId) => {
    let promptSessionId = targetSessionId
    const chatOptions = buildCliChatOptions({
      model: currentModel,
      provider: currentProvider,
      mode: currentMode,
      thinkingLevel: currentThinkingLevel,
      maxTokens: currentMaxTokens,
    })
    const printer = createPromptEventPrinter((nextSessionId) => {
      promptSessionId = nextSessionId
    })
    const handleEvent = createTrackedPromptEventHandler(printer)

    const runSsePrompt = async () => {
      const aborter = new AbortController()
      const watchdog = createChatStreamIdleWatchdog((error) => aborter.abort(error))
      try {
        const response = await openBoundedChatStream(
          httpClient.chatStream(prompt, promptSessionId, chatOptions, {
            signal: aborter.signal,
          }),
          aborter,
        )
        if (!response.ok || !response.body) {
          throw new Error(await formatChatStreamFailure(response))
        }
        await forwardDaemonStreamWithResumeRecovery<DaemonChatStreamPayload>(response, {
          aborter,
          getSessionId: () => promptSessionId,
          onEvent: async (event) => {
            watchdog.touch(event)
            await handleEvent(event)
          },
          openResumeStream: (resumeSessionId) => httpClient.resumeSessionStream(
            resumeSessionId,
            undefined,
            { signal: aborter.signal },
          ),
          isTerminalEvent: isTerminalCliDaemonChatEvent,
          streamIdleMs: watchdog.streamIdleMs,
        })
      } finally {
        watchdog.clear()
      }
    }

    if (!wsClient) {
      await runSsePrompt()
      return
    }

    let wsSawEvent = false
    await streamChatWithFallback({
      runWebSocketChat: async () => {
        const aborter = new AbortController()
        const watchdog = createChatStreamIdleWatchdog((error) => {
          aborter.abort(error)
          wsClient?.close()
        })
        try {
          await Promise.race([
            runDaemonWsClientChat({
              wsClient,
              message: prompt,
              sessionId: promptSessionId,
              options: chatOptions,
              onEvent: async (event) => {
                wsSawEvent = true
                watchdog.touch(event)
                await handleEvent(event)
              },
            }),
            abortSignalAsPromise(aborter.signal, watchdog.timeoutError),
          ])
        } finally {
          watchdog.clear()
        }
      },
      fallbackStream: runSsePrompt,
      shouldFallback: (error) => shouldFallbackFromInteractiveWsError(error, {
        sawEvent: wsSawEvent,
      }),
    })
  }

  const resumeApproval = async (
    requestId: string,
    approved: boolean,
    targetSessionId = sessionId,
  ) => {
    let resumeSessionId = targetSessionId
    const printer = createPromptEventPrinter((nextSessionId) => {
      resumeSessionId = nextSessionId
    })
    const handleEvent = createTrackedPromptEventHandler(printer)
    const aborter = new AbortController()
    const watchdog = createChatStreamIdleWatchdog((error) => aborter.abort(error))
    try {
      const response = await openBoundedChatStream(
        httpClient.resumeApprovalStream(
          requestId,
          approved,
          resumeSessionId,
          undefined,
          { signal: aborter.signal },
        ),
        aborter,
      )
      if (!response.ok || !response.body) {
        throw new Error(await formatChatStreamFailure(response))
      }
      await forwardDaemonStreamWithResumeRecovery<DaemonChatStreamPayload>(response, {
        aborter,
        getSessionId: () => resumeSessionId,
        onEvent: async (event) => {
          watchdog.touch(event)
          await handleEvent(event)
        },
        openResumeStream: (resumeSessionId) => httpClient.resumeSessionStream(
          resumeSessionId,
          undefined,
          { signal: aborter.signal },
        ),
        isTerminalEvent: isTerminalCliDaemonChatEvent,
        streamIdleMs: watchdog.streamIdleMs,
      })
    } finally {
      watchdog.clear()
    }
  }

  console.log(chalk.gray(`Mode: ${wsClient ? 'streaming' : 'http'}${currentModel ? ` | Model: ${currentModel}` : ''}${currentProvider ? ` | Provider: ${currentProvider}` : ''}${currentMaxTokens ? ` | Max output: ${currentMaxTokens}` : ''}`))
  // One-off config fetch so the user sees their safety posture before they
  // start typing. A failure (daemon down, auth missing) is silent — the
  // session line above already tells them whether the daemon is reachable.
  try {
    const config = await httpClient.config()
    console.log(chalk.gray(`Autonomy: `) + formatAutonomyLabel(config.agent?.autonomy))
  } catch { /* best-effort header; falls through to /help line */ }
  console.log(chalk.gray('Type /help for commands, "exit" or "/exit" to quit\n'))

  const rl = readline.createInterface({
    input: createHangulReadlineInput(process.stdin),
    output: process.stdout,
    prompt: chalk.blue('> '),
    terminal: true,
  })
  rlRef = rl

  rl.prompt()

  rl.on('line', async (input) => {
    const normalizedInput = composeHangul(stripBracketedPasteDelimiters(input))
    const trimmed = normalizedInput.trim()
    if (!trimmed) { rl.prompt(); return }
    if (['exit', 'quit', '/exit', '/quit', '/q'].includes(trimmed.toLowerCase())) {
      wsClient?.close()
      rl.close()
      return
    }

    // Slash commands
    if (trimmed.startsWith('/')) {
      if (trimmed === '/approvals') {
        if (!sessionId) {
          console.log(chalk.gray('No active session'))
        } else {
          try {
            const session = await httpClient.session(sessionId)
            for (const [requestId, sourceSessionId] of pendingApprovals) {
              if (!sourceSessionId || sourceSessionId === session.id) {
                pendingApprovals.delete(requestId)
              }
            }
            for (const approval of session.pendingApprovals ?? []) {
              pendingApprovals.set(approval.requestId, session.id)
            }
            updatePromptIndicator()
            if (!session.pendingApprovals?.length) {
              console.log(chalk.gray('No pending approvals'))
            } else {
              const now = Date.now()
              console.log(chalk.yellow('\nPending approvals:'))
              for (const approval of session.pendingApprovals) {
                const askedAgo = relativeTimeLabel(now - new Date(approval.requestedAt).getTime())
                const expiresIn = relativeTimeLabel(new Date(approval.expiresAt).getTime() - now)
                // Same fix as cli sessions show (commit 60ea4f1):
                // relativeTimeLabel returns the bare string `expired`
                // when delta<=0, and stitching ` until timeout` after
                // it produced the ungrammatical "expired until
                // timeout". Render the past-deadline case as a bare
                // `expired` label so an operator scanning the
                // /approvals output can grep for it visually.
                const expiryLabel = expiresIn === 'expired'
                  ? 'expired'
                  : `${expiresIn} until timeout`
                const stateTags = `${approval.state}${approval.resumeAvailable ? ', resumable' : ''}`
                const preview = formatToolCall(
                  { name: approval.tool, arguments: approval.input },
                  200,
                )
                console.log()
                console.log(chalk.yellow(`  [${stateTags}] ${preview}`))
                console.log(
                  chalk.gray(
                    `    ${approval.requestId} · asked ${askedAgo} ago · ${expiryLabel}`,
                  ),
                )
                console.log(
                  chalk.gray(
                    `    decide: /approve ${approval.requestId} [--session|--always] · /deny ${approval.requestId} [why]`,
                  ),
                )
              }
            }
          } catch {
            console.log(chalk.red('Failed to fetch session approvals'))
          }
        }
        rl.prompt()
        return
      }

      if (trimmed === '/questions') {
        if (!sessionId) {
          console.log(chalk.gray('No active session'))
        } else {
          try {
            const session = await httpClient.session(sessionId)
            for (const [questionId, sourceSessionId] of pendingQuestions) {
              if (!sourceSessionId || sourceSessionId === session.id) {
                pendingQuestions.delete(questionId)
              }
            }
            for (const question of session.pendingQuestions ?? []) {
              pendingQuestions.set(question.id, question.sessionId ?? session.id)
            }
            updatePromptIndicator()
            if (!session.pendingQuestions?.length) {
              console.log(chalk.gray('No pending questions'))
            } else {
              console.log(chalk.yellow('\nPending questions:'))
              for (const question of session.pendingQuestions) {
                const choices = question.choices?.length
                  ? chalk.gray(` choices: ${question.choices.join(' · ')}`)
                  : ''
                console.log()
                console.log(chalk.yellow(`  [?] ${question.prompt}${choices}`))
                console.log(chalk.gray(`    answer: /answer ${question.id} <reply>`))
              }
            }
          } catch {
            console.log(chalk.red('Failed to fetch session questions'))
          }
        }
        rl.prompt()
        return
      }

      if (trimmed === '/answer' || trimmed.startsWith('/answer ')) {
        const tokens = trimmed.split(/\s+/).filter(Boolean)
        const questionId = tokens[1]
        const answer = tokens.slice(2).join(' ').trim()
        if (!questionId || !answer) {
          console.log(chalk.red('Usage: /answer <question-id> <reply>'))
          rl.prompt()
          return
        }
        const questionSessionId = pendingQuestions.get(questionId) ?? sessionId
        if (!questionSessionId) {
          console.log(chalk.gray('No active session'))
          rl.prompt()
          return
        }

        try {
          const result = await httpClient.answerSessionQuestion(
            questionSessionId,
            questionId,
            answer,
          )
          if (!result.answered) {
            console.log(chalk.red(`Question ${questionId} not found in session`))
          } else {
            pendingQuestions.delete(questionId)
            updatePromptIndicator()
            console.log(chalk.yellow(`Answered question ${questionId}.`))
          }
          await syncPendingWaits(questionSessionId)
        } catch (error) {
          console.log(chalk.red(`Failed to answer question: ${friendlyErrorMessage(error)}`))
        }
        rl.prompt()
        return
      }

      if (trimmed === '/resume' || trimmed.startsWith('/resume ')) {
        if (!sessionId) {
          console.log(chalk.gray('No active session'))
          rl.prompt()
          return
        }

        const resumeArgs = trimmed.split(/\s+/).slice(1)
        const forceReplay = resumeArgs.includes('--force')

        try {
          const session = await httpClient.session(sessionId)
          if (!session.resumableRun) {
            console.log(chalk.gray('No resumable run checkpoint in this session'))
            rl.prompt()
            return
          }

          if (session.resumableRun.forceRequired && !forceReplay) {
            console.log(chalk.yellow(resumableRunCopy(session.resumableRun)))
            console.log(
              chalk.gray(
                `Retry with ${resumableRunActionHint(session.resumableRun)} when you are ready to replay the tool.`,
              ),
            )
            rl.prompt()
            return
          }

          console.log(chalk.yellow(resumableRunCopy(session.resumableRun)))
          const printer = createPromptEventPrinter()
          const handleEvent = createTrackedPromptEventHandler(printer)
          const aborter = new AbortController()
          const watchdog = createChatStreamIdleWatchdog((error) => aborter.abort(error))
          try {
            const response = await openBoundedChatStream(
              httpClient.resumeSessionStream(
                sessionId,
                forceReplay ? { force: true } : undefined,
                { signal: aborter.signal },
              ),
              aborter,
            )
            if (!response.ok || !response.body) {
              throw new Error(await formatChatStreamFailure(response))
            }
            await forwardDaemonStreamWithResumeRecovery<DaemonChatStreamPayload>(response, {
              aborter,
              getSessionId: () => sessionId,
              onEvent: async (event) => {
                watchdog.touch(event)
                await handleEvent(event)
              },
              openResumeStream: (resumeSessionId) => httpClient.resumeSessionStream(
                resumeSessionId,
                undefined,
                { signal: aborter.signal },
              ),
              isTerminalEvent: isTerminalCliDaemonChatEvent,
              streamIdleMs: watchdog.streamIdleMs,
            })
          } finally {
            watchdog.clear()
          }
          console.log()
        } catch (error) {
          // Same rationale as the /approve catch above — route the
          // ApiHttpError envelope through friendlyErrorMessage so a
          // 4xx/5xx from the daemon (e.g. session gone, run already
          // completed) carries its status+code into the chat shell
          // instead of leaking only the bare envelope message.
          if (!printStreamError(error, { sessionId }) && !printApiError(error)) {
            console.log(chalk.red(`Failed to resume session: ${friendlyErrorMessage(error)}`))
          }
        }
        rl.prompt()
        return
      }

      if (
        trimmed === '/approve'
        || trimmed === '/deny'
        || trimmed.startsWith('/approve ')
        || trimmed.startsWith('/deny ')
      ) {
        const approved = trimmed === '/approve' || trimmed.startsWith('/approve ')
        const tokens = trimmed.split(' ').filter(Boolean)
        const requestId = tokens[1]
        // A fresh shell has no request namespace to resolve an id against.
        // Surface that state before generic syntax help so bare /approve and
        // /deny match the other session-bound slash command guards.
        if (!requestId && !sessionId) {
          console.log(chalk.gray('No active session'))
          rl.prompt()
          return
        }
        // --run: approve every tool request spawned by the current user
        // command. --session-all does the same for the current chat session.
        // --session: remember this answer for every subsequent call to the
        // same tool in the active session (handy when an agent legitimately
        // needs to run `git status` ten times in a row).
        // --always: persist across sessions until manually revoked.
        // Plain trailing tokens become a free-form reason that gets
        // forwarded to the agent as the approval `note`, so a denial isn't
        // a dead end — the agent can read why and pick a different path.
        let scope: 'once' | 'session' | 'always' | 'run' | 'session-all' = 'once'
        const noteTokens: string[] = []
        for (let i = 2; i < tokens.length; i++) {
          const tok = tokens[i]
          if (tok === '--session') scope = 'session'
          else if (tok === '--always') scope = 'always'
          else if (tok === '--run' || tok === '--command' || tok === '--task') scope = 'run'
          else if (tok === '--session-all' || tok === '--session_all') scope = 'session-all'
          else noteTokens.push(tok)
        }
        const note = noteTokens.join(' ').trim() || undefined
        if (!requestId) {
          console.log(chalk.red(
            approved
              ? 'Usage: /approve <request-id> [--run|--session-all|--session|--always] [reason...]'
              : 'Usage: /deny <request-id> [reason...]',
          ))
          rl.prompt()
          return
        }
        const approvalSessionId = pendingApprovals.get(requestId) ?? sessionId
        if (!approvalSessionId) {
          console.log(chalk.gray('No active session'))
          rl.prompt()
          return
        }
        if (!approved && scope !== 'once') {
          // Persisting a denial across calls is a bigger policy decision
          // than a one-off "no" — keep it explicit until we add a real
          // tool/path-scoped block command.
          console.log(chalk.red(
            'Scoped denials (--run / --session-all / --session / --always) are not supported on /deny. Use /approve --run or edit the policy file.',
          ))
          rl.prompt()
          return
        }

        try {
          const session = await httpClient.session(approvalSessionId)
          const approval = session.pendingApprovals?.find(
            (item) => item.requestId === requestId,
          )
          if (!approval) {
            console.log(chalk.red(`Approval ${requestId} not found in session`))
            rl.prompt()
            return
          }

          const resolution = await resolveApprovalDecision({
            requestId,
            approved,
            toolName: approval.tool,
            sessionId: approvalSessionId,
            approvalState: approval.state,
            resumeAvailable: approval.resumeAvailable,
            resumeStaleApproval: async () => {
              console.log(chalk.yellow(`Resuming ${approval.tool} from saved checkpoint...`))
              await resumeApproval(requestId, approved, approvalSessionId)
            },
            respondApproval: () => httpClient.respondApproval(
              requestId,
              approved,
              { sessionId: approvalSessionId, scope, note },
            ),
            recoverStaleApproval: async ({ prompt }) => {
              console.log(
                chalk.yellow(
                  `Recorded ${approved ? 'approval' : 'denial'} for stale request ${requestId}. Starting a fresh run...`,
                ),
              )
              await sendPrompt(prompt, approvalSessionId)
            },
          })

          if (resolution.outcome === 'resolved' || resolution.outcome === 'recorded') {
            const scopeLabel =
              scope === 'session' ? ' for the rest of this session'
                : scope === 'always' ? ' for every future session'
                  : ''
            console.log(
              chalk.yellow(
                `${approved ? 'Approved' : 'Denied'} request ${requestId}${scopeLabel}.`,
              ),
            )
            // Same surface contract as the cli verb: when scope is
            // session/always, surface the derived rule pattern so the
            // operator sees exactly which future invocations the daemon
            // will short-circuit on. Without it, an operator
            // approving `ls /tmp/X` --session would be surprised when
            // `ls /tmp/Y` re-prompts.
            if (resolution.rule?.pattern) {
              console.log(
                chalk.gray(`  pattern: ${resolution.rule.tool} → ${resolution.rule.pattern}`),
              )
            }
          }
          pendingApprovals.delete(requestId)

          if (resolution.outcome === 'resumed' || resolution.outcome === 'recovered') {
            console.log()
          }
          // Resync from the daemon after a decision: the daemon is the
          // source of truth for which requestIds are still pending (the
          // user might have approved on another surface, or this very
          // decision could have unblocked a chain that produced a fresh
          // request).
          await syncPendingWaits(approvalSessionId)
        } catch (error) {
          // ApiHttpError carries the daemon's structured envelope
          // (status + code + message). Without friendlyErrorMessage
          // the user sees the bare envelope.message — e.g. "approval
          // X expired at ..." with no clue that the daemon returned
          // 410 APPROVAL_EXPIRED. Surface the full status+code so
          // a chat-shell operator gets the same diagnostic the cli
          // verb already prints.
          console.log(chalk.red(`Failed to resolve approval: ${friendlyErrorMessage(error)}`))
        }
        rl.prompt()
        return
      }

      if (trimmed === '/decisions' || trimmed.startsWith('/decisions ')) {
        // Mirror `sepilot decisions list/clear` from the cli surface so
        // an operator inside the chat shell can audit and prune
        // remembered approval rules without dropping out to a separate
        // terminal. Default action is `list` to match the verb's most
        // common need (quick "what rules am I running with?").
        const tokens = trimmed.split(/\s+/).filter(Boolean)
        const action = tokens[1] ?? 'list'
        if (action === 'list') {
          // Mirror the cli verb's --stale / --scope / --tool filters
          // so chat-shell operators don't need to drop out for
          // audits.
          const tail = tokens.slice(2)
          const staleOnly = tail.includes('--stale')
          let scopeFilter: 'session' | 'always' | undefined
          let toolFilter: string | undefined
          for (let i = 0; i < tail.length; i++) {
            if (tail[i] === '--scope' && tail[i + 1]) {
              const candidate = tail[++i].toLowerCase()
              if (candidate === 'session' || candidate === 'always') {
                scopeFilter = candidate
              }
            } else if (tail[i] === '--tool' && tail[i + 1]) {
              toolFilter = tail[++i]
            }
          }
          try {
            const result = await httpClient.listRememberedApprovals()
            const allItems = result.decisions ?? []
            const filteredItems = allItems
              .filter((d) => !toolFilter || d.tool === toolFilter)
              .filter((d) => !scopeFilter || d.scope === scopeFilter)
              .filter((d) => !staleOnly || isStaleRule(d, Date.now()))
            const items = filteredItems
            const filterLabelParts = [
              staleOnly ? 'stale-only' : null,
              scopeFilter ? `scope=${scopeFilter}` : null,
              toolFilter ? `tool=${toolFilter}` : null,
            ].filter(Boolean).join(', ')
            if (items.length === 0) {
              if (filterLabelParts) {
                console.log(chalk.gray(
                  `No remembered approval decisions matched filter (${filterLabelParts}).`,
                ))
              } else {
                console.log(chalk.gray('No remembered approval decisions'))
              }
            } else {
              const headlinePrefix = staleOnly
                ? 'Stale remembered approval decisions'
                : 'Remembered approval decisions'
              const headlineFilter = filterLabelParts ? `, ${filterLabelParts}` : ''
              console.log(chalk.cyan(
                `${headlinePrefix} (${items.length} total${headlineFilter}, showing ${items.length}):`,
              ))
              const now = Date.now()
              const sortedItems = [...items].sort(compareDecisions)
              for (const decision of sortedItems) {
                const verb = decision.approved ? chalk.green('approved') : chalk.red('denied')
                const sessionLabel = decision.sessionId ? ` · session ${decision.sessionId}` : ''
                // Match `sepilot decisions list` so the chat shell
                // surfaces the same usage signal — a hits=0 rule is
                // a cleanup candidate; a hot rule warrants policy
                // re-examination.
                const hits = typeof decision.hitCount === 'number'
                  ? chalk.gray(` · hits: ${decision.hitCount}`)
                  : ''
                const lastLabel = formatPastTime(decision.lastHitAt, now)
                const last = lastLabel ? chalk.gray(` · last: ${lastLabel}`) : ''
                const stale = isStaleRule(decision, now)
                  ? chalk.yellow(' (stale?)')
                  : ''
                console.log(`  [${decision.scope}] ${decision.tool} → ${verb}${sessionLabel}${hits}${last}${stale}`)
              }
              const staleCount = sortedItems.filter((d) => isStaleRule(d, now)).length
              if (staleCount > 0 && !staleOnly) {
                // Hint suppressed when --stale already narrowed the
                // view; otherwise the operator just sees "N stale"
                // applied to the very rows they explicitly asked for.
                console.log('')
                console.log(chalk.gray(
                  `  ${staleCount} rule${staleCount === 1 ? '' : 's'} flagged stale (no hits, ≥7d old). List with /decisions list --stale, prune with /decisions clear --scope <session|always>.`,
                ))
              }
            }
          } catch (error) {
            console.log(chalk.red(
              `Failed to list decisions: ${error instanceof Error ? error.message : error}`,
            ))
          }
          rl.prompt()
          return
        }
        if (action === 'show') {
          // Verbose detail for a tool's rule(s). Mirrors `sepilot
          // decisions show <tool>` so chat-shell operators don't need
          // to drop out for the deep view.
          const targetTool = tokens[2]?.trim()
          if (!targetTool) {
            console.log(chalk.red('Usage: /decisions show <tool>'))
            rl.prompt()
            return
          }
          try {
            const result = await httpClient.listRememberedApprovals()
            const matches = (result.decisions ?? []).filter((d) => d.tool === targetTool)
            if (matches.length === 0) {
              console.log(chalk.gray(`No remembered approval decisions for tool '${targetTool}'.`))
            } else {
              const now = Date.now()
              const sortedMatches = [...matches].sort(compareDecisions)
              console.log(chalk.cyan(
                `Remembered approval decisions for '${targetTool}' (${matches.length}):`,
              ))
              for (const decision of sortedMatches) {
                const verb = decision.approved ? chalk.green('approved') : chalk.red('denied')
                const stale = isStaleRule(decision, now)
                console.log('')
                console.log(`  [${decision.scope}] ${decision.tool} → ${verb}${stale ? chalk.yellow(' (stale?)') : ''}`)
                console.log(chalk.gray(`    pattern:    ${decision.pattern}`))
                if (decision.sessionId) {
                  console.log(chalk.gray(`    sessionId:  ${decision.sessionId}`))
                }
                if (decision.createdAt) {
                  const createdAge = formatPastTime(decision.createdAt, now)
                  console.log(chalk.gray(
                    `    createdAt:  ${decision.createdAt}${createdAge ? ` (${createdAge})` : ''}`,
                  ))
                }
                if (typeof decision.hitCount === 'number') {
                  console.log(chalk.gray(`    hitCount:   ${decision.hitCount}`))
                }
                if (decision.lastHitAt) {
                  const lastAge = formatPastTime(decision.lastHitAt, now)
                  console.log(chalk.gray(
                    `    lastHitAt:  ${decision.lastHitAt}${lastAge ? ` (${lastAge})` : ''}`,
                  ))
                } else {
                  console.log(chalk.gray('    lastHitAt:  never matched'))
                }
              }
            }
          } catch (error) {
            console.log(chalk.red(
              `Failed to show decisions: ${error instanceof Error ? error.message : error}`,
            ))
          }
          rl.prompt()
          return
        }
        if (action === 'clear') {
          // Parse optional `--scope session|always` (omit to clear all
          // scopes). `--session <id>` is intentionally not exposed here:
          // the chat shell already has the active sessionId pinned, so
          // we apply scope filtering only — narrowing by other session
          // ids belongs in the cli verb where automation handles it.
          let scope: 'session' | 'always' | undefined
          let stale = false
          let toolFilter: string | undefined
          for (let i = 2; i < tokens.length; i++) {
            if (tokens[i] === '--scope' && tokens[i + 1]) {
              const candidate = tokens[++i].toLowerCase()
              if (candidate === 'session' || candidate === 'always') {
                scope = candidate
              } else {
                console.log(chalk.red(`Unknown scope: ${candidate}. Use session|always.`))
                rl.prompt()
                return
              }
            } else if (tokens[i] === '--stale') {
              stale = true
            } else if (tokens[i] === '--tool' && tokens[i + 1]) {
              toolFilter = tokens[++i]
            }
          }
          try {
            await httpClient.clearRememberedApprovals({ scope, stale, tool: toolFilter })
            const filterParts = [
              stale ? 'stale-only' : null,
              scope ? `scope=${scope}` : null,
              toolFilter ? `tool=${toolFilter}` : null,
            ].filter(Boolean).join(', ')
            console.log(chalk.green(
              `Cleared remembered decisions${filterParts ? ` (${filterParts})` : ''}.`,
            ))
          } catch (error) {
            console.log(chalk.red(
              `Failed to clear decisions: ${error instanceof Error ? error.message : error}`,
            ))
          }
          rl.prompt()
          return
        }
        console.log(chalk.red('Usage: /decisions [list [--stale] [--scope session|always] [--tool <name>] | show <tool> | clear [--scope session|always] [--tool <name>] [--stale]]'))
        rl.prompt()
        return
      }

      // /sessions list — REPL parity with `sepilot sessions list`.
      // Lets an operator pivot to a different session without
      // dropping out of the chat shell. `/session` (singular) still
      // shows the current session detail; `/sessions` (plural) maps
      // to the list verb so the muscle memory matches.
      if (trimmed === '/sessions' || trimmed.startsWith('/sessions ')) {
        const tokens = trimmed.split(/\s+/).filter(Boolean)
        const action = tokens[1] ?? 'list'
        if (action === 'list') {
          let statusFilter: 'active' | 'completed' | 'abandoned' | undefined
          let queryFilter: string | undefined
          let limit = 20
          for (let i = 2; i < tokens.length; i++) {
            if (tokens[i] === '--status' && tokens[i + 1]) {
              const candidate = tokens[++i].toLowerCase()
              if (candidate === 'active' || candidate === 'completed' || candidate === 'abandoned') {
                statusFilter = candidate
              } else {
                console.log(chalk.red(`Unknown status: ${candidate}. Use active|completed|abandoned.`))
                rl.prompt()
                return
              }
            } else if (tokens[i] === '--query' && tokens[i + 1]) {
              queryFilter = tokens[++i]
            } else if (tokens[i] === '--limit' && tokens[i + 1]) {
              const parsed = Number.parseInt(tokens[++i], 10)
              if (Number.isFinite(parsed) && parsed >= 0) limit = parsed
            }
          }
          try {
            const data = await httpClient.sessions(queryFilter, {
              metrics: true,
              status: statusFilter,
            })
            const items = data.items ?? []
            if (items.length === 0) {
              console.log(chalk.gray(
                statusFilter
                  ? `No ${statusFilter} sessions.`
                  : 'No sessions.',
              ))
            } else {
              const visible = limit === 0 ? items : items.slice(0, limit)
              const filterLabel = statusFilter ? `, status=${statusFilter}` : ''
              console.log(chalk.cyan(
                `Sessions (${data.totalCount} total${filterLabel}, showing ${visible.length}):`,
              ))
              for (const s of visible) {
                const counters = s.approvalCounters
                const hasApprovalActivity = counters
                  && (counters.approvalsRequested > 0 || counters.autoApprovalsApproved > 0)
                const approvalsLabel = hasApprovalActivity
                  ? `  ${chalk.gray(`appr: ${counters.approvalsRequested}/${counters.approvalsApproved}/${counters.approvalsDenied}`
                    + (counters.autoApprovalsApproved > 0 ? ` · auto: ${counters.autoApprovalsApproved}` : ''))}`
                  : ''
                // Mirror cli verb's fixed-width row so an operator
                // glancing across surfaces sees the same shape.
                const idCol = s.id.length > 18 ? `${s.id.slice(0, 17)}…` : s.id.padEnd(18)
                const titleCol = (s.title ?? '').length > 30
                  ? `${(s.title ?? '').slice(0, 29)}…`
                  : (s.title ?? '').padEnd(30)
                console.log(
                  `  ${idCol}  ${titleCol}  ${s.status.padEnd(10)}  ${String(s.messageCount).padStart(3)} msgs  ${s.provider}/${s.model}${approvalsLabel}`,
                )
              }
              if (visible.length < items.length) {
                console.log(chalk.gray(
                  `\n  …${items.length - visible.length} more. Use /sessions list --limit <n> or --limit 0.`,
                ))
              }
            }
          } catch (error) {
            console.log(chalk.red(
              `Failed to list sessions: ${error instanceof Error ? error.message : error}`,
            ))
          }
          rl.prompt()
          return
        }
        if (action === 'show') {
          // In-REPL deep view: pairs with /sessions list so an
          // operator who spotted an interesting row can pivot to its
          // detail inline. Mirrors `sepilot sessions show <id>` text
          // mode — Approvals block + Working memory keyDecisions +
          // pending count + small events tail. Skips the verbose
          // events tail (10 rows in cli verb) for a 5-row preview
          // since chat shell screen real estate is tighter.
          const targetId = tokens[2]?.trim()
          if (!targetId) {
            console.log(chalk.red('Usage: /sessions show <id>'))
            rl.prompt()
            return
          }
          try {
            const detail = await httpClient.session(targetId)
            console.log(chalk.cyan(`Session: ${detail.id}`))
            console.log(chalk.gray(`  title: ${detail.title || '(untitled)'}`))
            console.log(chalk.gray(`  status: ${detail.status}`))
            console.log(chalk.gray(`  provider: ${detail.provider}/${detail.model}`))
            console.log(chalk.gray(`  messages: ${detail.messageCount}`))
            const trace = detail.traceMetrics
            if (trace) {
              const requested = trace.approvalRequests ?? 0
              const approved = trace.approvalApproved ?? 0
              const denied = trace.approvalDenied ?? 0
              const autoApproved = trace.autoApprovalsApproved ?? 0
              if (requested > 0 || autoApproved > 0) {
                console.log(`  approvals: ${requested} requested · ${approved} approved · ${denied} denied`)
                if (autoApproved > 0) {
                  console.log(chalk.gray(`    auto (remembered rule): ${autoApproved} approved`))
                }
              }
            }
            const wm = detail.workingMemory
            const recentDecisions = (wm?.keyDecisions ?? []).slice(-5)
            if (recentDecisions.length > 0) {
              console.log(chalk.gray(`  decisions (last ${recentDecisions.length}):`))
              for (const decision of recentDecisions) {
                console.log(`    · ${decision.summary}`)
              }
            }
            const pending = detail.pendingApprovals ?? []
            if (pending.length > 0) {
              console.log(chalk.yellow(`  pending approvals: ${pending.length}`))
              for (const approval of pending.slice(0, 3)) {
                console.log(chalk.gray(`    · ${approval.tool} (${approval.requestId})`))
              }
            }
            const questions = detail.pendingQuestions ?? []
            if (questions.length > 0) {
              console.log(chalk.yellow(`  pending questions: ${questions.length}`))
              for (const question of questions.slice(0, 3)) {
                console.log(chalk.gray(`    · ${question.prompt} (${question.id})`))
              }
            }
            const events = detail.events ?? []
            const tailLimit = 5
            if (events.length > 0) {
              console.log(chalk.gray(`  events (${events.length}, last ${Math.min(events.length, tailLimit)}):`))
              for (const event of events.slice(-tailLimit)) {
                // Compact one-line per event — full render lives in
                // the cli verb. Use the type tag so an operator can
                // tell at a glance "approval_request, approval_response,
                // tool_call, etc." without parsing free-form copy.
                const tag = `[${event.type}]`
                const summary = (() => {
                  if (event.type === 'user_message' || event.type === 'assistant_message') {
                    const text = (event as { content?: string }).content ?? ''
                    return text.length > 60 ? `${text.slice(0, 57)}…` : text
                  }
                  if (event.type === 'tool_call' || event.type === 'approval_request') {
                    return (event as { tool?: string }).tool ?? ''
                  }
                  if (event.type === 'auto_approval') {
                    const e = event as { tool?: string; decision?: string; scope?: string }
                    return `${e.tool} ${e.decision} via ${e.scope} rule`
                  }
                  return ''
                })()
                console.log(chalk.gray(`    ${tag}${summary ? ` ${summary}` : ''}`))
              }
            }
          } catch (error) {
            console.log(chalk.red(
              `Failed to show session: ${error instanceof Error ? error.message : error}`,
            ))
          }
          rl.prompt()
          return
        }
        console.log(chalk.red('Usage: /sessions [list [--status …] [--query …] [--limit <n>] | show <id>]'))
        rl.prompt()
        return
      }

      // /stats — REPL parity with `sepilot stats`. Shares the same
      // buildStats + formatStatsSnapshot helpers as the cli verb so
      // the dashboard layout is byte-identical across surfaces.
      if (trimmed === '/stats') {
        try {
          const [sessions, decisionsResp] = await Promise.all([
            httpClient.sessions(undefined, { metrics: true }),
            httpClient.listRememberedApprovals(),
          ])
          const snapshot = buildStats(sessions, decisionsResp.decisions ?? [])
          console.log(formatStatsSnapshot(snapshot))
        } catch (error) {
          console.log(chalk.red(
            `Failed to load daemon snapshot: ${error instanceof Error ? error.message : error}`,
          ))
        }
        rl.prompt()
        return
      }

      handleSlashCommand(trimmed, {
        sessionId,
        currentModel,
        currentProvider,
        httpClient,
        tokenSpeedTracker,
        currentMaxTokens,
      })
        .then((result) => {
          if (result.exit) {
            wsClient?.close()
            rl.close()
            return
          }
          if (result.model) currentModel = result.model
          if (result.provider) currentProvider = result.provider
          if (result.mode) currentMode = result.mode
          if (result.thinking) currentThinkingLevel = result.thinking
          if ('maxTokens' in result) currentMaxTokens = result.maxTokens
          if (result.newSession) { sessionId = undefined; console.log(chalk.yellow('New session started')) }
        })
        .catch(() => {})
        .finally(() => rl.prompt())
      return
    }

    try {
      await sendPrompt(trimmed)
      console.log()
      // A run can leave new approvals pending or resolve old ones (e.g.
      // an approval on another surface, or a checkpoint replay). Reconcile
      // before drawing the next prompt so the indicator never lies.
      await syncPendingWaits()
    } catch (err) {
      // Surface the same friendly stream / http copy as `sepilot ask` so
      // the interactive shell doesn't drop the user back to the prompt
      // with an opaque `Error: terminated`. Falls through to the raw
      // message only if the error isn't a stream/http failure we know
      // how to classify.
      if (!printStreamError(err, { sessionId }) && !printApiError(err)) {
        console.log(chalk.red(`Error: ${err instanceof Error ? err.message : err}\n`))
      }
    }
    rl.prompt()
  })
}

async function handleSlashCommand(
  cmd: string,
  ctx: {
    sessionId?: string
    currentModel?: string
    currentProvider?: string
    currentMaxTokens?: number
    httpClient: DaemonClient
    tokenSpeedTracker?: TokenSpeedTracker
  },
): Promise<{
  model?: string
  provider?: string
  maxTokens?: number
  newSession?: boolean
  thinking?: string
  mode?: DaemonAgentMode
  exit?: boolean
}> {
  const [command, ...args] = cmd.split(' ')
  const tokenSpeedTracker = ctx.tokenSpeedTracker ?? createTokenSpeedTracker()

  switch (command) {
    case '/exit':
    case '/quit':
    case '/q':
      return { exit: true }

    case '/help':
      console.log(chalk.yellow(`
Commands:
  /model [name]     Show or switch daemon default LLM model (self-tested, rollback on failure)
  /model pull <name> Pull an Ollama model into the configured provider
  /provider <name>  Switch LLM provider
  /mode <mode>      Set agent mode or graph id
  /session          Show current session info
  /self             Show model, autonomy, skills, schedules, and runtime limits
  /skills           List installed skills
  /capabilities     Show available models and installed skill surface
  /schedules        List scheduled tasks
  /sessions list [--status active|completed|abandoned] [--limit <n>]  List sessions across the daemon
  /sessions show <id>          Show a session's detail inline (pairs with /sessions list)
  /stats                       Compact daemon dashboard (sessions, approvals, decisions)
  /approvals        List pending approvals in the current session
  /approve <id> [--session|--always] [why]  Approve a tool request (--session: remember for this session)
  /deny <id> [why]  Deny a pending tool request (optional reason gets forwarded to the agent)
  /questions        List pending agent questions in the current session
  /answer <id> <reply> Answer a pending agent question
  /steer <msg>      Send a mid-run steering note to the current session's active turn
  /resume [--force] Resume an interrupted run from the current session
  /new              Start new session
  /max-tokens <n|off|current> Set or clear the output token cap
  /thinking <level> Set thinking level (off|low|medium|high|max)
  /autonomy <level> Set autonomy (readonly|accept-edits|workspace-write|supervised|autonomous)
  /policy           Show the active tool policy (modes, deny lists)
  /decisions [list|show <tool>|clear [--scope session|always]]  List, inspect, or clear remembered approval rules
  /clear            Clear active context (past messages stay in terminal scrollback)
  /providers        List available providers
  /usage            Show token usage
  /tps              Show token-per-second speed for last response and current session
  /exit             Quit the interactive chat
  /help             Show this help
`))
      return {}

    case '/self':
      await printSelfSnapshot(ctx.httpClient, 'overview')
      return {}

    case '/skills':
      await printSelfSnapshot(ctx.httpClient, 'skills')
      return {}

    case '/capabilities':
      await printSelfSnapshot(ctx.httpClient, 'capabilities')
      return {}

    case '/schedules':
      await printSelfSnapshot(ctx.httpClient, 'schedules')
      return {}

    case '/model': {
      const action = args[0]?.toLowerCase()
      if (!action || action === 'current' || action === 'list' || action === 'available') {
        try {
          console.log(chalk.gray(formatModelSnapshot(await ctx.httpClient.model())))
        } catch (error) {
          console.log(chalk.red(`Failed to fetch model state: ${error instanceof Error ? error.message : error}`))
        }
        return {}
      }

      if (action === 'pull' || action === 'get' || action === 'fetch' || action === 'install') {
        const model = args.slice(1).join(' ').trim()
        if (!model) {
          console.log(chalk.red('Usage: /model pull <model>'))
          return {}
        }
        try {
          const result = await ctx.httpClient.pullModel({ model })
          console.log(chalk.yellow(result.pull.message))
          console.log(chalk.gray(`Switch with /model ${result.pull.providerId}/${result.pull.model}`))
        } catch (error) {
          console.log(chalk.red(`Failed to pull model: ${error instanceof Error ? error.message : error}`))
        }
        return {}
      }

      const target = args.join(' ').trim()
      try {
        const result = await ctx.httpClient.switchDefaultModel({ target })
        console.log((result.ok ? chalk.yellow : chalk.red)(result.message))
        if (result.ok) {
          return { provider: result.providerId, model: result.model }
        }
        return result.previousProviderId && result.previousModel
          ? { provider: result.previousProviderId, model: result.previousModel }
          : {}
      } catch (error) {
        console.log(chalk.red(`Failed to switch model: ${error instanceof Error ? error.message : error}`))
        return {}
      }
    }

    case '/steer': {
      const message = args.join(' ').trim()
      if (!message) {
        console.log(chalk.red('Usage: /steer <message>'))
        return {}
      }
      if (!ctx.sessionId) {
        console.log(chalk.red('No active session yet — send a message first.'))
        return {}
      }
      const result = await submitSteer(ctx.httpClient, ctx.sessionId, message)
      if (result.ok) {
        console.log(chalk.yellow(`[steer] ${formatQueuedSteerNote(result.noteId, result.pendingSteeringNoteCount)}`))
      } else if (result.noActiveRun) {
        console.log(chalk.red(result.guidance))
      } else {
        console.log(chalk.red(`Failed to steer: ${result.message}`))
      }
      return {}
    }

    case '/provider':
      if (!args[0]) { console.log(chalk.red('Usage: /provider <name>')); return {} }
      console.log(chalk.yellow(`Provider switched to ${args[0]}`))
      return { provider: args[0] }

    case '/mode': {
      const newMode = args.join(' ').trim()
      if (!newMode) {
        console.log(chalk.red('Usage: /mode <mode>'))
        return {}
      }
      console.log(chalk.yellow(`Agent mode switched to ${newMode}`))
      return { mode: newMode }
    }

    case '/session':
      if (ctx.sessionId) {
        try {
          const [session, config] = await Promise.all([
            ctx.httpClient.session(ctx.sessionId),
            ctx.httpClient.config().catch(() => null),
          ])
          console.log(chalk.gray(`Session: ${ctx.sessionId}`))
          console.log(chalk.gray(`Model: ${session.model || ctx.currentModel || 'default'}`))
          console.log(chalk.gray(`Provider: ${session.provider || ctx.currentProvider || 'default'}`))
          if (config?.agent?.autonomy) {
            console.log(chalk.gray('Autonomy: ') + formatAutonomyLabel(config.agent.autonomy))
          }
          if (session.delegation) {
            console.log(
              chalk.gray(
                `Delegation: ${delegationHealthLabel(session.delegation)} on ${session.delegation.targetDevice}`,
              ),
            )
            console.log(chalk.gray(`  ${delegationHealthDetail(session.delegation)}`))
          }
        } catch {
          console.log(chalk.gray(`Session: ${ctx.sessionId}`))
          console.log(chalk.gray(`Model: ${ctx.currentModel ?? 'default'}`))
          console.log(chalk.gray(`Provider: ${ctx.currentProvider ?? 'default'}`))
        }
      } else {
        console.log(chalk.gray('No active session'))
      }
      return {}

    case '/new':
      tokenSpeedTracker.reset()
      return { newSession: true }

    case '/max-tokens': {
      const value = args[0]
      if (!value || value === 'current') {
        console.log(chalk.gray(`Max output tokens: ${ctx.currentMaxTokens ?? 'provider/model default'}`))
        return {}
      }
      if (value === 'off' || value === 'default') {
        console.log(chalk.yellow('Max output tokens reset to provider/model default'))
        return { maxTokens: undefined }
      }
      try {
        const maxTokens = parsePositiveIntegerCliOption(value, '/max-tokens')
        console.log(chalk.yellow(`Max output tokens set to ${maxTokens}`))
        return { maxTokens }
      } catch (error) {
        console.log(chalk.red(error instanceof Error ? error.message : String(error)))
        return {}
      }
    }

    case '/thinking': {
      const level = args[0]
      if (!level || !['off', 'low', 'medium', 'high', 'max'].includes(level)) {
        console.log(chalk.red('Usage: /thinking <off|low|medium|high|max>'))
        return {}
      }
      console.log(chalk.yellow(`Thinking level set to ${level}`))
      return { thinking: level }
    }

    case '/policy': {
      try {
        const policy = await ctx.httpClient.policy()
        console.log(
          chalk.gray(
            `Defaults: mode=${policy.defaults.mode}, unmatched=${policy.defaults.unmatched_policy}, timeout=${policy.defaults.max_timeout_ms}ms`,
          ),
        )
        const tools = Object.entries(policy.tools).sort(([a], [b]) => a.localeCompare(b))
        if (tools.length === 0) {
          console.log(chalk.gray('No tool-specific rules.'))
        } else {
          console.log(chalk.yellow('\nTool rules:'))
          for (const [name, rule] of tools) {
            const counts: string[] = []
            if (rule.deny_patterns?.length) counts.push(`${rule.deny_patterns.length} deny_patterns`)
            if (rule.deny_paths?.length) counts.push(`${rule.deny_paths.length} deny_paths`)
            if (rule.deny_urls?.length) counts.push(`${rule.deny_urls.length} deny_urls`)
            if (rule.deny_executables?.length) counts.push(`${rule.deny_executables.length} deny_executables`)
            if (rule.allow_patterns?.length) counts.push(`${rule.allow_patterns.length} allow_patterns`)
            const modeColor =
              rule.mode === 'blocked' ? chalk.red
                : rule.mode === 'supervised' ? chalk.yellow
                  : chalk.green
            console.log(
              `  ${modeColor(rule.mode.padEnd(11))} ${name.padEnd(22)} ${chalk.gray(counts.join(', ') || 'no extra rules')}`,
            )
          }
        }
      } catch (error) {
        console.log(chalk.red(`Failed to fetch policy: ${error instanceof Error ? error.message : error}`))
      }
      return {}
    }

    case '/autonomy': {
      const level = args[0]
      const valid = ['readonly', 'accept-edits', 'workspace-write', 'supervised', 'autonomous']
      if (!level || !valid.includes(level)) {
        try {
          const config = await ctx.httpClient.config()
          console.log(chalk.gray('Autonomy: ') + formatAutonomyLabel(config.agent?.autonomy))
        } catch { /* fall through */ }
        console.log(chalk.red(`Usage: /autonomy <${valid.join('|')}>`))
        return {}
      }
      try {
        await ctx.httpClient.updateConfig({
          'agent.autonomy': level as 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous',
        })
        console.log(chalk.gray('Autonomy: ') + formatAutonomyLabel(level))
        if (level === 'autonomous') {
          // Loud warning so the user can't accidentally page-up the cli
          // history later and miss that they turned approvals off.
          console.log(
            chalk.red(
              '  Warning: tools will now run without approval prompts. Use `/autonomy supervised` to revert.',
            ),
          )
        }
      } catch (error) {
        console.log(
          chalk.red(`Failed to update autonomy: ${error instanceof Error ? error.message : error}`),
        )
      }
      return {}
    }

    case '/clear':
      console.clear()
      console.log(chalk.gray('Screen cleared. Conversation context is unchanged — use /new to start a fresh session.'))
      return {}

    case '/providers':
      try {
        const providers = await ctx.httpClient.providers()
        for (const p of providers) {
          console.log(chalk.cyan(`  ${p.name} (${p.id})`))
          const embedIds = new Set(p.embeddingModelIds)
          for (const m of p.models) {
            const role = embedIds.has(m.id) ? 'embed' : 'chat '
            console.log(chalk.gray(`    [${role}] ${m.id}${formatProviderModelBadges(m)}`))
          }
        }
      } catch {
        console.log(chalk.red('Failed to fetch providers'))
      }
      return {}

    case '/usage':
      try {
        const usage = await ctx.httpClient.usage()
        // Match `sepilot usage summary` formatting so the two paths
        // present the same numbers in the same shape.
        console.log(chalk.gray(`  Input tokens:  ${usage.inputTokens.toLocaleString()}`))
        console.log(chalk.gray(`  Output tokens: ${usage.outputTokens.toLocaleString()}`))
        console.log(chalk.gray(`  Cost:          $${usage.costUsd.toFixed(4)}`))
        console.log(chalk.gray(`  Requests:      ${usage.requestCount}`))
      } catch {
        console.log(chalk.red('Failed to fetch usage'))
      }
      return {}

    case '/tps': {
      const action = args[0]?.toLowerCase()
      if (action === 'reset') {
        tokenSpeedTracker.reset()
        console.log(chalk.gray('TPS stats reset.'))
        return {}
      }
      if (action && action !== 'current' && action !== 'info') {
        console.log(chalk.red('Usage: /tps [current|reset]'))
        return {}
      }
      console.log(chalk.gray(formatTokenSpeedStats(tokenSpeedTracker.snapshot())))
      return {}
    }

    default:
      console.log(chalk.red(`Unknown command: ${command}. Type /help for available commands.`))
      return {}
  }
}

export const __testables = {
  formatAutonomyLabel,
  handleSlashCommand,
  relativeTimeLabel,
  resumableRunActionHint,
}
