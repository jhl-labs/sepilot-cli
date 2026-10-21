import type {
  AgentEvidenceLedger,
  AgentFailedAttempt,
  AgentOpenQuestion,
  AgentState,
} from './graph/types.js'
import { cloneEvidenceLedger, emptyEvidenceLedger } from './graph/evidence-ledger.js'
import { signatureOf } from './stuck-tool-repeat.js'

export interface SubagentFindings {
  evidence: AgentEvidenceLedger
  failedAttempts: AgentFailedAttempt[]
  openQuestions: AgentOpenQuestion[]
  sessionId: string
  category: string
}

export function extractSubagentFindings(
  childFinalState: Partial<AgentState>,
  origin: { sessionId: string; category: string },
): SubagentFindings {
  const evidence = cloneEvidenceLedger(childFinalState.evidenceLedger) ?? emptyEvidenceLedger()

  const failedAttempts: AgentFailedAttempt[] = []
  const seenSignatures = new Set<string>()
  const pushFailedAttempt = (attempt: AgentFailedAttempt): void => {
    if (seenSignatures.has(attempt.signature)) return
    seenSignatures.add(attempt.signature)
    failedAttempts.push(attempt)
  }
  for (const attempt of childFinalState.failedAttempts ?? []) {
    pushFailedAttempt({ ...attempt })
  }
  for (const entry of childFinalState.evidenceLedger?.errors ?? []) {
    const signature = signatureOf({
      tool: entry.tool,
      input: entry.path ? { path: entry.path } : {},
    })
    pushFailedAttempt({
      signature,
      tool: entry.tool,
      reason: entry.summary ?? 'tool error',
      ts: entry.ts ?? Date.now(),
    })
  }

  const openQuestions: AgentOpenQuestion[] = []
  const seenQuestionIds = new Set<string>()
  const pushOpenQuestion = (question: AgentOpenQuestion): void => {
    if (seenQuestionIds.has(question.id)) return
    seenQuestionIds.add(question.id)
    openQuestions.push(question)
  }
  for (const question of childFinalState.openQuestions ?? []) {
    pushOpenQuestion({ ...question })
  }
  const assumptions = childFinalState.plannerWorkingMemory?.openAssumptions ?? []
  assumptions.forEach((assumption, index) => {
    if (assumption.verified || assumption.digest) return
    pushOpenQuestion({
      id: `${origin.sessionId}:assume-${index}`,
      text: assumption.text,
      blocking: false,
    })
  })

  return {
    evidence,
    failedAttempts,
    openQuestions,
    sessionId: origin.sessionId,
    category: origin.category,
  }
}
