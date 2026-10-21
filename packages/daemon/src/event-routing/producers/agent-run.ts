import { createLogger } from '../../logger.js'
import type { EventRoutingConfig } from '../config-schema.js'
import type { RoutedEventInput } from '../types.js'
import { anyRuleMatchesSource, redactForEvent, EVENT_ROUTING_SURFACE } from './shared.js'

const logger = createLogger('event-routing')

const OUTPUT_MAX = 8_000
const ERROR_MAX = 1_000
const INPUT_EXCERPT_MAX = 500
const SUBJECT_MAX = 80

export interface HookRegistryLike {
  register(event: 'post:agent:run', handler: {
    id: string
    priority: number
    handle(payload: { event: 'post:agent:run'; data: Record<string, unknown> }): Promise<{ action: 'continue' }>
  }): { dispose(): void }
}

export interface RegisterAgentRunProducerDeps {
  hooks: HookRegistryLike
  ingest: (event: RoutedEventInput) => unknown
  getConfig: () => EventRoutingConfig
  isEventRoutingSession: (sessionId: string) => Promise<boolean>
}

const STATUS_TYPE: Record<string, string> = {
  success: 'completed',
  error: 'failed',
  aborted: 'cancelled',
}

function firstLineSubject(input: string | undefined): string | null {
  if (!input) return null
  const firstLine = input.split(/\r?\n/, 1)[0]?.trim() ?? ''
  if (!firstLine) return null
  return firstLine.length > SUBJECT_MAX ? firstLine.slice(0, SUBJECT_MAX) : firstLine
}

export function agentRunEventFromHook(data: Record<string, unknown>): RoutedEventInput | null {
  const mode = data.mode
  const runId = data.runId
  if (typeof mode !== 'string' || !mode) return null
  if (typeof runId !== 'string' || !runId) return null
  const status = data.status
  if (status === 'incomplete') return null
  if (data.surface === EVENT_ROUTING_SURFACE) return null

  const type = typeof status === 'string' ? STATUS_TYPE[status] ?? status : 'unknown'
  const sessionId = typeof data.sessionId === 'string' ? data.sessionId : undefined
  const provider = typeof data.provider === 'string' ? data.provider : undefined
  const model = typeof data.model === 'string' ? data.model : undefined
  const surface = typeof data.surface === 'string' ? data.surface : undefined
  const durationMs = typeof data.durationMs === 'number' ? data.durationMs : undefined
  const detached = typeof data.detached === 'boolean' ? data.detached : undefined
  const backgroundJobId = typeof data.backgroundJobId === 'string' ? data.backgroundJobId : undefined
  const roleBotId = typeof data.roleBotId === 'string' ? data.roleBotId : undefined
  const resumed = data.resumed === true ? true : undefined
  const usage = data.usage as { inputTokens?: unknown; outputTokens?: unknown } | undefined
  const normalizedUsage =
    usage && typeof usage.inputTokens === 'number' && typeof usage.outputTokens === 'number'
      ? { inputTokens: usage.inputTokens, outputTokens: usage.outputTokens }
      : undefined
  const input = typeof data.input === 'string' ? data.input : undefined
  const output = typeof data.output === 'string' ? data.output : undefined
  const error = typeof data.error === 'string' ? data.error : undefined
  const inputExcerpt = redactForEvent(input, INPUT_EXCERPT_MAX)
  const attempt = typeof data.attempt === 'number' && data.attempt >= 1 ? data.attempt : undefined
  const fallbackPending = typeof data.fallbackPending === 'boolean' ? data.fallbackPending : undefined

  return {
    source: 'agent-run',
    type,
    subject: firstLineSubject(inputExcerpt),
    dedupeKey: `agent-run:${runId}`,
    occurredAt: Date.now(),
    payload: {
      runId,
      sessionId,
      mode,
      surface,
      provider,
      model,
      status,
      durationMs,
      detached,
      backgroundJobId,
      roleBotId,
      resumed,
      usage: normalizedUsage,
      attempt,
      fallbackPending,
      inputExcerpt,
      output: redactForEvent(output, OUTPUT_MAX),
      error: redactForEvent(error, ERROR_MAX),
    },
  }
}

export function registerAgentRunProducer(deps: RegisterAgentRunProducerDeps): { dispose(): void } {
  return deps.hooks.register('post:agent:run', {
    id: 'event-routing:agent-run',
    priority: 1000,
    async handle(payload) {
      const data = payload.data
      if (anyRuleMatchesSource(deps.getConfig(), 'agent-run')) {
        void (async () => {
          try {
            const event = agentRunEventFromHook(data)
            if (!event) return
            const sessionId = typeof data.sessionId === 'string' ? data.sessionId : undefined
            const resumed = data.resumed === true
            const hasSurface = typeof data.surface === 'string' && data.surface.length > 0
            if (resumed && !hasSurface && sessionId && (await deps.isEventRoutingSession(sessionId))) {
              return
            }
            await deps.ingest(event)
          } catch (error) {
            logger.error('agent-run producer failed to ingest', { error: error instanceof Error ? error.message : String(error) })
          }
        })()
      }
      return { action: 'continue' }
    },
  })
}
