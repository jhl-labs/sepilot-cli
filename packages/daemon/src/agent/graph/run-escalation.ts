import type { RunStopCode, RunStopReason } from '@sepilotd/core'
import { isIncompleteOutput } from '../stop-reason.js'
import { hasIncompleteAnswerStem } from '../interim-progress.js'
import type { GraphEscalationTarget } from './registry.js'
import type { AgentState } from './types.js'

export type EscalationRunState = Pick<AgentState, 'output' | 'completionDiagnostics'>

export function agentDeclaredIncomplete(state: EscalationRunState | undefined): boolean {
  if (!state) return false
  if (isIncompleteOutput(state.output)) return true
  const diagnostics = state.completionDiagnostics
  if (diagnostics?.presentation?.removedProtocolLabels?.includes('INCOMPLETE')) return true
  const raw = diagnostics?.rawReporterOutput
  return typeof raw === 'string' && hasIncompleteAnswerStem(raw)
}

const ESCALATABLE_STOP_CODES: ReadonlySet<RunStopCode> = new Set<RunStopCode>([
  'iteration_budget',
  'node_budget',
  'observation_budget',
  'no_progress',
  'stuck_repeat',
  'completion_gate',
  'wall_clock',
  'inactivity',
  'agent_incomplete',
])

export function stopReasonAgentIncomplete(): RunStopReason {
  return {
    kind: 'incomplete',
    code: 'agent_incomplete',
    summary: 'The agent reported that it could not complete the task.',
    resumable: false,
    nextActions: ['retry'],
  }
}

export function withEscalationOffer(
  stopReason: RunStopReason,
  state: EscalationRunState | undefined,
  target: GraphEscalationTarget | undefined,
): RunStopReason {
  if (!target) return stopReason
  const reason = stopReason.kind === 'completed' && agentDeclaredIncomplete(state)
    ? stopReasonAgentIncomplete()
    : stopReason
  if (!ESCALATABLE_STOP_CODES.has(reason.code)) return reason
  if (reason.nextActions.includes('escalate')) return reason
  return {
    ...reason,
    detail: {
      ...(reason.detail ?? {}),
      escalateMode: target.mode,
      ...(target.bot ? { escalateBot: target.bot } : {}),
    },
    nextActions: ['escalate', ...reason.nextActions],
  }
}
