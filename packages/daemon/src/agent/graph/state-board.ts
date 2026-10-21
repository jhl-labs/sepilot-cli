import { randomUUID } from 'node:crypto'
import { USER_INSTRUCTION_PRECEDENCE, type AgentSteeringNote } from '../user-steering.js'
export type { AgentSteeringNote } from '../user-steering.js'
import type { AgentStateBoardCompletion, TodoItem } from '@sepilotd/core'
import type {
  AgentAcceptanceCriterion,
  AgentFailedAttempt,
  AgentOpenQuestion,
  AgentSeedContract,
  AgentState,
  GraphExecutionContext,
} from './types.js'
import { formatRunContractForPrompt } from '../task-contract.js'
import { criterionEvidenceEpisode } from '../criterion-evidence-review.js'
import { formatEvidenceLedgerForPrompt } from './evidence-ledger.js'
import {
  formatPlannerWorkingMemory,
  type PlannerWorkingMemoryBoardSections,
} from './planner-working-memory.js'

export const STATE_BOARD_PREFIX = '[Agent state board]'

export function isStateBoardEnabled(): boolean {
  return process.env.SEPILOTD_STATE_BOARD !== '0'
}

export const MAX_BOARD_FAILED_ATTEMPTS = 10
const MAX_BOARD_PLAN_STEPS = 20
const MAX_BOARD_TODO_ITEMS = 20
const MAX_BOARD_OPEN_QUESTIONS = 10
const MAX_REASON_CHARS = 200

export interface AgentStateBoardPlanStep {
  id: string
  title: string
  status: string
  depth: number
}

const MAX_STEERING_NOTES = 20

export function createSteeringNote(
  note: { message: string; kind?: 'instruction' | 'question' },
): AgentSteeringNote {
  return {
    id: randomUUID(),
    message: note.message,
    kind: note.kind ?? 'instruction',
    createdAt: Date.now(),
  }
}

export function appendSteeringNote(
  state: { steeringNotes?: AgentSteeringNote[] },
  note: { message: string; kind?: 'instruction' | 'question' },
): AgentSteeringNote {
  const entry = createSteeringNote(note)
  state.steeringNotes = [...(state.steeringNotes ?? []), entry].slice(-MAX_STEERING_NOTES)
  return entry
}

export function takeUnconsumedSteeringNotes(state: {
  steeringNotes?: AgentSteeringNote[]
}): AgentSteeringNote[] {
  const fresh = (state.steeringNotes ?? []).filter(
    (note) => note.consumedAt === undefined && note.cancelledAt === undefined,
  )
  const now = Date.now()
  for (const n of fresh) n.consumedAt = now
  return fresh
}

export interface AgentStateBoard {
  goal: string
  completionCriteria: AgentAcceptanceCriterion[]
  contract?: AgentSeedContract
  plan: AgentStateBoardPlanStep[]
  todos: TodoItem[]
  decisions: string[]
  failedAttempts: AgentFailedAttempt[]
  openQuestions: AgentOpenQuestion[]
  evidenceSection: string | null
  completion?: AgentStateBoardCompletion
  steeringNotes: AgentSteeringNote[]
}

export type AgentStateBoardSource = Pick<
  AgentState,
  | 'seedContract'
  | 'evidenceLedger'
  | 'plannerWorkingMemory'
  | 'backtrackReasons'
  | 'failedAttempts'
  | 'openQuestions'
  | 'todoList'
  | 'completionDiagnostics'
> & {
  toolCallHistory?: AgentState['toolCallHistory']
  steeringNotes?: AgentSteeringNote[]
}

function projectCompletion(
  source: AgentStateBoardSource,
): AgentStateBoardCompletion | undefined {
  const diagnostics = source.completionDiagnostics
  if (!diagnostics) return undefined

  const contractCriteria = source.seedContract?.acceptanceCriteria ?? []
  const canonicalIds = new Map(
    contractCriteria.map((criterion) => [criterion.id.toUpperCase(), criterion.id]),
  )
  const latestById = new Map<string, AgentStateBoardCompletion['criterionVerdicts'][number]>()
  const snapshot = diagnostics.criterionVerdictSnapshot
  const currentEpisode = criterionEvidenceEpisode(source)
  const snapshotIsCurrent =
    snapshot?.toolResultCount === currentEpisode.toolResultCount
    && snapshot.toolResultFingerprint === currentEpisode.toolResultFingerprint
  const verdictCandidates = snapshotIsCurrent
    ? snapshot.criterionVerdicts
    : diagnostics.presentation?.removedCriterionVerdicts ?? []
  for (const candidate of verdictCandidates) {
    const id = canonicalIds.get(candidate.id.toUpperCase())
    if (!id) continue
    latestById.set(id, {
      id,
      verdict: candidate.verdict,
      ...(candidate.evidenceToolCallIds?.length
        ? { evidenceToolCallIds: [...candidate.evidenceToolCallIds] }
        : {}),
    })
  }
  const criterionVerdicts = contractCriteria
    .map((criterion) => latestById.get(criterion.id))
    .filter((verdict): verdict is NonNullable<typeof verdict> => verdict !== undefined)
  const gate = diagnostics.gate
    ? {
        decision: diagnostics.gate.decision,
        unmet: [...diagnostics.gate.unmet],
        ...(diagnostics.gate.reason ? { reason: diagnostics.gate.reason } : {}),
        ...(diagnostics.gate.budgetExhausted ? { budgetExhausted: true } : {}),
      }
    : undefined

  if (criterionVerdicts.length === 0 && !gate) return undefined
  return {
    criterionVerdicts,
    ...(gate ? { gate } : {}),
  }
}

function compressReason(reason: string): string {
  const normalized = reason.replace(/\s+/g, ' ').trim()
  return normalized.length > MAX_REASON_CHARS
    ? `${normalized.slice(0, MAX_REASON_CHARS - 1)}…`
    : normalized
}

function mergeFailedAttempts(
  source: AgentStateBoardSource,
  planner: PlannerWorkingMemoryBoardSections,
): AgentFailedAttempt[] {
  const merged: AgentFailedAttempt[] = [
    ...(source.failedAttempts ?? []).map((attempt) => ({
      ...attempt,
      reason: compressReason(attempt.reason),
    })),
  ]
  const seen = new Set(merged.map((attempt) => attempt.reason))
  const push = (attempt: AgentFailedAttempt) => {
    if (seen.has(attempt.reason)) return
    seen.add(attempt.reason)
    merged.push(attempt)
  }
  ;(source.backtrackReasons ?? []).forEach((reason, index) => {
    push({
      signature: `backtrack:${index}`,
      tool: 'quality-gate',
      reason: compressReason(reason),
      ts: 0,
    })
  })
  planner.abandoned.forEach((alt, index) => {
    push({
      signature: `abandoned:${index}`,
      tool: 'planner',
      reason: compressReason(`${alt.description} — ${alt.reason}`),
      ts: 0,
    })
  })
  return merged
}

function mergeOpenQuestions(
  source: AgentStateBoardSource,
  planner: PlannerWorkingMemoryBoardSections,
): AgentOpenQuestion[] {
  const merged: AgentOpenQuestion[] = [...(source.openQuestions ?? [])]
  const seen = new Set(merged.map((question) => question.text))
  for (const text of planner.openQuestions) {
    if (seen.has(text)) continue
    seen.add(text)
    merged.push({ id: `OQ${merged.length + 1}`, text, blocking: true })
  }
  return merged
}

export function buildStateBoard(
  source: AgentStateBoardSource,
  context?: GraphExecutionContext,
): AgentStateBoard {
  const planner = formatPlannerWorkingMemory(source.plannerWorkingMemory)
  return {
    goal: source.seedContract?.summary ?? '',
    completionCriteria: source.seedContract?.acceptanceCriteria ?? [],
    contract: source.seedContract,
    plan: planner.plan,
    todos: source.todoList ?? [],
    decisions: planner.decisions.map(compressReason),
    failedAttempts: mergeFailedAttempts(source, planner),
    openQuestions: mergeOpenQuestions(source, planner),
    evidenceSection: formatEvidenceLedgerForPrompt(source as AgentState, context),
    completion: projectCompletion(source),
    steeringNotes: source.steeringNotes ?? [],
  }
}

export function formatSeedContract(contract: AgentSeedContract | undefined): string | null {
  return formatRunContractForPrompt(contract)
    ?.replace('[Durable run contract]', '[Run contract]')
    ?? null
}

function boardIsEmpty(board: AgentStateBoard): boolean {
  return (
    !board.contract
    && board.plan.length === 0
    && board.todos.length === 0
    && board.decisions.length === 0
    && board.failedAttempts.length === 0
    && board.openQuestions.length === 0
    && board.evidenceSection === null
    && board.steeringNotes.every(
      (note) => note.cancelledAt !== undefined || (note.kind === 'question' && note.consumedAt !== undefined),
    )
  )
}

export function formatStateBoard(board: AgentStateBoard): string | null {
  if (boardIsEmpty(board)) return null
  const lines = [
    STATE_BOARD_PREFIX,
    'Rebuilt fresh from structured run state each turn; survives context compaction. Goal and acceptance criteria are verbatim — a history summary never replaces them.',
  ]
  const contractSection = formatSeedContract(board.contract)
  if (contractSection) {
    lines.push(contractSection.replace('[Run contract]', 'Run contract (verbatim):'))
  }
  if (board.plan.length > 0) {
    lines.push('Plan (planner working memory):')
    for (const step of board.plan.slice(0, MAX_BOARD_PLAN_STEPS)) {
      lines.push(`${'  '.repeat(step.depth)}- [${step.status}] ${step.id}: ${step.title}`)
    }
    if (board.plan.length > MAX_BOARD_PLAN_STEPS) {
      lines.push(`(+${board.plan.length - MAX_BOARD_PLAN_STEPS} more plan steps)`)
    }
  }
  if (board.todos.length > 0) {
    lines.push('Todo list:')
    for (const item of board.todos.slice(0, MAX_BOARD_TODO_ITEMS)) {
      lines.push(`- [${item.status}] ${item.content}`)
    }
    if (board.todos.length > MAX_BOARD_TODO_ITEMS) {
      lines.push(`(+${board.todos.length - MAX_BOARD_TODO_ITEMS} more todo items)`)
    }
  }
  if (board.decisions.length > 0) {
    lines.push('Decisions taken:')
    for (const decision of board.decisions.slice(-5)) lines.push(`- ${decision}`)
  }
  if (board.openQuestions.length > 0) {
    lines.push('Open questions (unresolved — verify or escalate, do not guess):')
    for (const question of board.openQuestions.slice(0, MAX_BOARD_OPEN_QUESTIONS)) {
      lines.push(`- ${question.id}: ${question.text}${question.blocking ? ' (blocking)' : ''}`)
    }
    if (board.openQuestions.length > MAX_BOARD_OPEN_QUESTIONS) {
      lines.push(`(+${board.openQuestions.length - MAX_BOARD_OPEN_QUESTIONS} more open questions)`)
    }
  }
  if (board.failedAttempts.length > 0) {
    lines.push('Failed attempts (do NOT repeat these approaches):')
    const recent = board.failedAttempts.slice(-MAX_BOARD_FAILED_ATTEMPTS)
    for (const attempt of recent) {
      lines.push(`- [${attempt.tool}] ${attempt.reason}`)
    }
    if (board.failedAttempts.length > recent.length) {
      lines.push(`(+${board.failedAttempts.length - recent.length} earlier failed attempts)`)
    }
  }
  const activeSteeringNotes = board.steeringNotes.filter(
    (note) => note.cancelledAt === undefined && (note.kind === 'instruction' || note.consumedAt === undefined),
  )
  if (activeSteeringNotes.length > 0) {
    lines.push('## User steering (act on these now)')
    lines.push(USER_INSTRUCTION_PRECEDENCE)
    for (const note of activeSteeringNotes) {
      lines.push(`- [${note.kind}] ${note.message}`)
    }
  }
  if (board.evidenceSection) {
    lines.push(board.evidenceSection)
  }
  return lines.join('\n')
}
