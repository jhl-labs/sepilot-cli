import {
  hasIncompleteAnswerStem,
  stripAnswerProtocolStem,
} from '../agent/interim-progress.js'
import type {
  JobRunStatus,
  JobRunStatusIntegrity,
  JobRunTaskOutcome,
  RunStopKind,
  RunStopReason,
} from '@sepilotd/core'
import type { JobExecutionResult } from './engine.js'

const MAX_INCOMPLETE_REASON_LENGTH = 500
// Text fallback for rows with no recorded stop kind: the pre-stop-reason
// wording, plus the structured wording emitted when the kind was observed but
// (older daemon, failed column write) not persisted alongside it.
const SCHEDULED_INCOMPLETE_ERROR =
  /^Scheduled agent (?:reported an incomplete result|stopped: (?:incomplete|blocked)\/)(?:[.:/]|$)?/iu

export function projectStoredJobRunOutcome(input: {
  status: JobRunStatus
  error: string | null
  outputExcerpt: string | null
  /**
   * Structured termination cause recorded by current daemons. Older rows have
   * none, so the text probes below remain the compatibility fallback.
   */
  stopKind?: RunStopKind | null
}): {
  taskOutcome: JobRunTaskOutcome
  statusIntegrity: JobRunStatusIntegrity
} {
  if (input.status === 'running') {
    return { taskOutcome: 'running', statusIntegrity: 'consistent' }
  }

  // Prefer the recorded stop kind: it states the cause outright instead of
  // inferring it from how the final answer happened to be worded.
  const incomplete = input.stopKind != null
    ? input.stopKind === 'incomplete' || input.stopKind === 'blocked'
    : SCHEDULED_INCOMPLETE_ERROR.test(input.error?.trim() ?? '')
      || hasIncompleteAnswerStem(input.outputExcerpt ?? '')
  if (incomplete) {
    return {
      taskOutcome: 'incomplete',
      statusIntegrity: input.status === 'success'
        ? 'legacy-incomplete-conflict'
        : 'consistent',
    }
  }
  return {
    taskOutcome: input.status === 'success' ? 'complete' : 'failed',
    statusIntegrity: 'consistent',
  }
}

function compactReason(value: string): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  if (normalized.length <= MAX_INCOMPLETE_REASON_LENGTH) return normalized
  return `${normalized.slice(0, MAX_INCOMPLETE_REASON_LENGTH - 1).trimEnd()}…`
}

export function scheduledAgentExecutionResult(input: {
  /** Raw agent protocol output, before a channel strips ANSWER:/INCOMPLETE:. */
  rawOutput: string
  /** Optional surface-safe output persisted in run history instead of raw protocol text. */
  output?: string
  usage?: JobExecutionResult['usage']
  /** Internal fallback responses are incomplete even when they have no literal stem. */
  forceIncomplete?: boolean
  /** Structured termination cause from the agent's done event, when observed. */
  stopReason?: RunStopReason
}): JobExecutionResult {
  const output = input.output ?? input.rawOutput
  const incomplete = input.forceIncomplete === true
    || hasIncompleteAnswerStem(input.rawOutput)
    || input.stopReason?.kind === 'incomplete'
    || input.stopReason?.kind === 'blocked'
  if (!incomplete) {
    return {
      ...(output ? { output } : {}),
      ...(input.usage ? { usage: input.usage } : {}),
      ...(input.stopReason ? { stopReason: input.stopReason } : {}),
    }
  }

  const blocker = compactReason(stripAnswerProtocolStem(input.rawOutput))
  return {
    ...(output ? { output } : {}),
    ...(input.usage ? { usage: input.usage } : {}),
    ...(input.stopReason ? { stopReason: input.stopReason } : {}),
    outcome: 'incomplete',
    failureReason: formatScheduledIncompleteReason(blocker, input.stopReason),
  }
}

/**
 * Lead with the machine-readable cause. The retained tool output that follows
 * can run to hundreds of characters, so a reader (or an operator scanning
 * last_error) previously had to guess whether the run crashed, was blocked, or
 * simply ran out of a deliberate budget with its state still resumable.
 */
export function formatScheduledIncompleteReason(
  blocker: string,
  stopReason?: RunStopReason,
): string {
  const head = stopReason
    ? `Scheduled agent stopped: ${stopReason.kind}/${stopReason.code}`
      + (stopReason.resumable ? ' (resumable' : ' (not resumable')
      + (stopReason.nextActions.length > 0
        ? `; next: ${stopReason.nextActions.join(', ')})`
        : ')')
    : 'Scheduled agent reported an incomplete result'
  const detail = stopReason?.summary?.trim() || blocker
  return detail ? `${head}: ${detail}` : `${head}.`
}
