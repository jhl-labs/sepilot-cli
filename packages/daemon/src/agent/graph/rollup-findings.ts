import type { AgentEvidenceLedger, AgentState } from './types.js'
import { emptyEvidenceLedger, nextEvidenceOrder, pushBoundedEvidence } from './evidence-ledger.js'
import type { SubagentFindings } from '../subagent-findings.js'

/**
 * Roll-up toggle (default on). Set SEPILOTD_SUBAGENT_BOARD_ROLLUP=off to fall
 * back to the pure text-return behaviour (no structured merge into the parent
 * board). General option, not a model/dataset branch. PLAN_065 T6.
 */
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

/**
 * Merge an isolated subagent's structured findings into the parent board.
 *
 * Evidence entries flow through `pushBoundedEvidence` (PLAN_025, 80/bucket FIFO)
 * so a chatty subagent cannot blow the parent ledger; each rolled-up entry
 * carries `origin` provenance (subagent sessionId/category) so the parent knows
 * who found it. Failed-attempts dedup by structural signature (PLAN_024) and
 * open-questions dedup by id, so re-rolling the same findings is idempotent for
 * those buckets. This is a pure structured-field merge — the free-text subagent
 * summary is never parsed for findings.
 */
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
