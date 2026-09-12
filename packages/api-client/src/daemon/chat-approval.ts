import type { DaemonApprovalResponseResult } from './types.js'
import type { RecoverApprovalDecisionInput } from './chat-surface-types.js'
import { approvalRecoveryStartDetail } from './chat-surface-utils.js'

export function buildRecoveredApprovalPrompt(
  toolName: string,
  approved: boolean | import('@sepilotd/core').ApprovalDecisionStatus,
  note?: string,
): string {
  const decision = typeof approved === 'boolean'
    ? (approved ? 'approved' : 'denied')
    : approved

  if (decision === 'approved') {
    return `Continue the interrupted task. The pending approval for "${toolName}" was granted after reconnect, but the original run could not resume. Continue from the latest session state, and invoke that tool again yourself only if it is still needed.`
  }

  if (decision === 'feedback') {
    return `Continue the interrupted task. The pending approval for "${toolName}" came back with feedback after reconnect${note ? `: ${note}` : ''}. Continue from the latest session state, revise the approach, and only invoke that tool again if the feedback has been addressed.`
  }

  return `Continue the interrupted task. The pending approval for "${toolName}" was denied after reconnect, and the original run could not resume. Continue from the latest session state without using that tool unless the user later changes the decision.`
}

export interface ApprovalDecisionResult {
  outcome: 'resumed' | 'recovered' | 'recorded' | 'resolved'
  /**
   * Carried over from the daemon\'s respond envelope when scope
   * is session/always — the cli surfaces tool+pattern in the
   * /approve success line so operators can predict short-circuit
   * behaviour without grepping `decisions list`. Undefined for
   * once-scope (default), feedback, or stale resume paths
   * (those never hit the live respond envelope).
   */
  rule?: { tool: string; pattern: string }
}

export async function resolveApprovalDecision(params: {
  requestId: string
  approved: boolean | import('@sepilotd/core').ApprovalDecisionStatus
  toolName: string
  sessionId?: string
  approvalState?: 'live' | 'stale'
  resumeAvailable?: boolean
  note?: string
  resumeStaleApproval?: () => Promise<void>
  respondApproval: () => Promise<DaemonApprovalResponseResult>
  recoverStaleApproval?: (
    input: RecoverApprovalDecisionInput,
  ) => Promise<void>
}): Promise<ApprovalDecisionResult> {
  const requiresStaleRecovery = params.approvalState === 'stale'

  if (
    requiresStaleRecovery
    && params.sessionId
    && params.resumeAvailable
    && params.resumeStaleApproval
  ) {
    await params.resumeStaleApproval()
    return { outcome: 'resumed' }
  }

  const result = await params.respondApproval()
  if (result.state !== 'stale' && !requiresStaleRecovery) {
    return { outcome: 'resolved', rule: result.rule }
  }

  if (params.sessionId && params.recoverStaleApproval) {
    await params.recoverStaleApproval({
      sessionId: params.sessionId,
      prompt: buildRecoveredApprovalPrompt(
        params.toolName,
        params.approved,
        params.note,
      ),
      startLabel: 'Recovered run',
      startDetail: approvalRecoveryStartDetail(
        params.toolName,
        params.approved,
      ),
    })
    return { outcome: 'recovered', rule: result.rule }
  }

  return { outcome: 'recorded', rule: result.rule }
}
