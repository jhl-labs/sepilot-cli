import type {
  AgentEvidenceLedger,
  AgentOpenQuestion,
  AgentState,
  GraphExecutionContext,
} from './types.js'
import { raceWithAbort, throwIfAborted } from '../../abort.js'

export const MAX_ESCALATIONS = 2

export const PROGRESS_STALL_ITERATIONS = 3

export function countVerifiedEvidence(ledger: AgentEvidenceLedger | undefined): number {
  if (!ledger) return 0
  return (
    ledger.validationRuns.filter((entry) => entry.verified).length
    + ledger.artifactReadBacks.filter((entry) => entry.verified).length
  )
}

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

export async function escalateOpenQuestion(
  state: Pick<AgentState, 'messages' | 'escalationCount'>,
  question: AgentOpenQuestion,
  context: Pick<GraphExecutionContext, 'requestQuestion' | 'signal'> & {
    sessionId: string
  },
  now: () => number = () => Date.now(),
): Promise<string | undefined> {
  if (!context.requestQuestion) return undefined
  throwIfAborted(context.signal, 'Open question aborted')
  const answer = await raceWithAbort(context.requestQuestion({
    sessionId: context.sessionId,
    prompt: [
      'The agent is blocked on an unresolved question and needs your input to continue without guessing.',
      `Question ${question.id}: ${question.text}`,
    ].join('\n'),
  }), context.signal, 'Open question aborted')
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
