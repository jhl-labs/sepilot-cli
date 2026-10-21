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
const SCHEDULED_INCOMPLETE_ERROR =
  /^Scheduled agent (?:reported an incomplete result|stopped: (?:incomplete|blocked)\/)(?:[.:/]|$)?/iu

export function projectStoredJobRunOutcome(input: {
  status: JobRunStatus
  error: string | null
  outputExcerpt: string | null
  stopKind?: RunStopKind | null
}): {
  taskOutcome: JobRunTaskOutcome
  statusIntegrity: JobRunStatusIntegrity
} {
  if (input.status === 'running') {
    return { taskOutcome: 'running', statusIntegrity: 'consistent' }
  }

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
  rawOutput: string
  output?: string
  usage?: JobExecutionResult['usage']
  forceIncomplete?: boolean
  stopReason?: RunStopReason
}): JobExecutionResult {
  const output = input.output ?? input.rawOutput
  const incomplete = input.forceIncomplete === true
    || hasIncompleteAnswerStem(input.rawOutput)
    || input.stopReason?.kind === 'incomplete'
    || input.stopReason?.kind === 'blocked'
    || input.stopReason?.kind === 'error'
    || input.stopReason?.kind === 'cancelled'
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
