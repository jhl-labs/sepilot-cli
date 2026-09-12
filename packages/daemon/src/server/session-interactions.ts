import type { SessionMeta } from '@sepilotd/core'
import type { RuntimeServices } from './runtime/types.js'

export type SessionInteractionRuntime = Pick<RuntimeServices, 'sessions'> & Partial<
  Pick<RuntimeServices, 'approvalRegistry' | 'questions'>
>

export interface SessionInteractionSummary {
  pendingApprovals: {
    total: number
    orphaned: number
  }
  pendingQuestions: {
    total: number
    orphaned: number
  }
}

export async function listAllSessionMetadata(
  runtime: Pick<RuntimeServices, 'sessions'>,
): Promise<SessionMeta[]> {
  const sessions: SessionMeta[] = []
  const perPage = 200
  for (let page = 1; ; page += 1) {
    const result = await runtime.sessions.list({ page, perPage })
    sessions.push(...result.items)
    if (!result.hasNextPage) {
      break
    }
  }
  return sessions
}

export function collectSessionInteractionState(
  runtime: SessionInteractionRuntime,
  sessions: readonly SessionMeta[],
) {
  const sessionIds = new Set(sessions.map((session) => session.id))
  const pendingApprovals = runtime.approvalRegistry?.listAll() ?? []
  const pendingQuestions = runtime.questions?.listAll() ?? []
  const orphanedPendingApprovals = pendingApprovals.filter(
    (approval) => !sessionIds.has(approval.sessionId),
  )
  const orphanedPendingQuestions = pendingQuestions.filter(
    (question) => !sessionIds.has(question.sessionId),
  )
  return {
    pendingApprovals,
    pendingQuestions,
    orphanedPendingApprovals,
    orphanedPendingQuestions,
  }
}

/**
 * Content-free interaction readiness for assistant/operator dashboards.
 * Deliberately reads only session metadata and in-memory registries; history
 * analysis remains owned by the full session-management endpoint.
 */
export async function buildSessionInteractionSummary(
  runtime: SessionInteractionRuntime,
): Promise<SessionInteractionSummary> {
  const state = collectSessionInteractionState(
    runtime,
    await listAllSessionMetadata(runtime),
  )
  return {
    pendingApprovals: {
      total: state.pendingApprovals.length,
      orphaned: state.orphanedPendingApprovals.length,
    },
    pendingQuestions: {
      total: state.pendingQuestions.length,
      orphaned: state.orphanedPendingQuestions.length,
    },
  }
}
