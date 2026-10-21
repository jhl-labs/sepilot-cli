import type { AgentEvidenceLedger, AgentState } from './types.js'
import { emptyEvidenceLedger, nextEvidenceOrder, pushBoundedEvidence } from './evidence-ledger.js'
import type { SubagentFindings } from '../subagent-findings.js'

export function isSubagentBoardRollupEnabled(): boolean {
  const raw = process.env.SEPILOTD_SUBAGENT_BOARD_ROLLUP?.trim().toLowerCase()
  return raw !== 'off' && raw !== '0' && raw !== 'false'
}

const EVIDENCE_BUCKETS: Array<keyof AgentEvidenceLedger> = [
  'sourceReads',
  'sourceSearches',
  'artifactWrites',
  'artifactReadBacks',
  'validationRuns',
  'errors',
]

export function rollupSubagentFindings(parent: AgentState, findings: SubagentFindings): void {
  if (!isSubagentBoardRollupEnabled()) return
  const origin = { sessionId: findings.sessionId, category: findings.category }

  const ledger = (parent.evidenceLedger ??= emptyEvidenceLedger())
  for (const bucket of EVIDENCE_BUCKETS) {
    for (const entry of findings.evidence[bucket] ?? []) {
      pushBoundedEvidence(ledger[bucket], {
        ...entry,
        order: nextEvidenceOrder(ledger),
        origin: entry.origin ?? origin,
      })
    }
  }

  const failedAttempts = (parent.failedAttempts ??= [])
  const knownSignatures = new Set(failedAttempts.map((attempt) => attempt.signature))
  for (const attempt of findings.failedAttempts) {
    if (knownSignatures.has(attempt.signature)) continue
    knownSignatures.add(attempt.signature)
    failedAttempts.push({ ...attempt })
  }

  const openQuestions = (parent.openQuestions ??= [])
  const knownQuestionIds = new Set(openQuestions.map((question) => question.id))
  for (const question of findings.openQuestions) {
    if (knownQuestionIds.has(question.id)) continue
    knownQuestionIds.add(question.id)
    openQuestions.push({ ...question })
  }
}
