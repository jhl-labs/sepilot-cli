import type {
  AgentEvidenceLedger,
  AgentOpenQuestion,
  AgentState,
  GraphExecutionContext,
} from './types.js'

/**
 * Open-question escalation: when the agent has surfaced a blocking unresolved
 * question and progress has stalled (no new verified evidence), route the
 * question to the human via the injected `context.requestQuestion` HITL
 * channel instead of letting the agent guess. Bounded per run
 * (MAX_ESCALATIONS) and headless-safe: without a requestQuestion handler the
 * loop keeps moving and the question stays recorded for the final report.
 */
export const MAX_ESCALATIONS = 2

/**
 * How many loop iterations without any new verified evidence count as a
 * progress stall for escalation purposes.
 */
export const PROGRESS_STALL_ITERATIONS = 3

export function countVerifiedEvidence(ledger: AgentEvidenceLedger | undefined): number {
  if (!ledger) return 0
  return (
    ledger.validationRuns.filter((entry) => entry.verified).length
    + ledger.artifactReadBacks.filter((entry) => entry.verified).length
  )
}

/**
 * Promote the planner's unverified open assumptions into the run's open
 * questions. Structural promotion only: an assumption the planner still holds
 * as unverified is an unresolved blocking premise — no content interpretation
 * happens here. Already-promoted texts keep their entry (and askedAt).
 */
export function promoteOpenQuestionsFromPlannerMemory(
  state: Pick<AgentState, 'openQuestions' | 'plannerWorkingMemory'>,
): void {
  const assumptions = state.plannerWorkingMemory?.openAssumptions ?? []
  if (assumptions.length === 0) return
  const list = (state.openQuestions ??= [])
  for (const assumption of assumptions) {
    if (assumption.verified === true) continue
    const text = assumption.text.trim()
    if (!text || list.some((q) => q.text === text)) continue
    list.push({ id: `OQ${list.length + 1}`, text, blocking: true })
  }
}

export function shouldEscalateOpenQuestions(
  state: Pick<AgentState, 'openQuestions' | 'escalationCount'>,
  opts: { progressStalled: boolean },
): AgentOpenQuestion | undefined {
  if (!opts.progressStalled) return undefined
  if ((state.escalationCount ?? 0) >= MAX_ESCALATIONS) return undefined
  return (state.openQuestions ?? []).find(
    (q) => q.blocking && q.askedAt === undefined,
  )
}

/**
 * Ask the human the escalated question through the HITL channel, inject the
 * answer into the conversation, and mark the question asked so it is never
 * re-escalated. Returns the answer, or undefined in headless runs (no
 * requestQuestion handler) where this is a recorded no-op.
 */
export async function escalateOpenQuestion(
  state: Pick<AgentState, 'messages' | 'escalationCount'>,
  question: AgentOpenQuestion,
  context: Pick<GraphExecutionContext, 'requestQuestion'> & {
    sessionId: string
  },
  now: () => number = () => Date.now(),
): Promise<string | undefined> {
  if (!context.requestQuestion) return undefined
  const answer = await context.requestQuestion({
    sessionId: context.sessionId,
    prompt: [
      'The agent is blocked on an unresolved question and needs your input to continue without guessing.',
      `Question ${question.id}: ${question.text}`,
    ].join('\n'),
  })
  question.askedAt = now()
  state.escalationCount = (state.escalationCount ?? 0) + 1
  state.messages.push({
    role: 'system',
    content: [
      `[Open question resolved by user] ${question.id}: ${question.text}`,
      `Answer: ${answer}`,
      'Use this answer as authoritative; do not re-ask or guess around it.',
    ].join('\n'),
  })
  return answer
}
