import type { SessionEvent, SessionMeta } from '@sepilotd/core'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import type { RuntimeServices } from '../runtime/types.js'
import {
  listUploadedFilesBySession,
  removeUploadedFilesByIds,
} from './file-registry.js'
import {
  buildSessionHistoryManagement,
  type SessionHistoryManagementRisk,
} from '../session-contracts.js'
import {
  collectSessionInteractionState,
  listAllSessionMetadata,
} from '../session-interactions.js'
import { zodRequestValidation } from './utils.js'

export type SessionManagementRuntime = Pick<RuntimeServices, 'sessions'> & Partial<
  Pick<
    RuntimeServices,
    | 'approvalCheckpoints'
    | 'approvalRegistry'
    | 'primaryAgents'
    | 'questions'
    | 'runCheckpoints'
    | 'sessionRuntimeSnapshots'
    | 'semanticIndex'
    | 'toolExecutions'
    | 'toolStatsStore'
    | 'usageTracker'
    | 'dreaming'
  >
>

export interface SessionRuntimeCleanupResult {
  pendingApprovals: number
  pendingQuestions: number
  approvalCheckpoints: number
  unavailableApprovalCheckpoints: number
  runCheckpoints: number
  toolExecutions: number
  usageRecords: number
  derivedMemories: number
  uploadedFiles: number
}

export interface SessionManagementSnapshot {
  totalSessions: number
  byStatus: Record<'active' | 'completed' | 'abandoned', number>
  pendingApprovals: {
    total: number
    orphaned: number
  }
  pendingQuestions: {
    total: number
    orphaned: number
  }
  runCheckpoints: {
    total: number
    unavailable: number
    locked: number
    orphaned: number
  }
  approvalCheckpoints: {
    total: number
    unavailable: number
    orphaned: number
  }
  toolExecutions: {
    total: number
    running: number
    completed: number
    unavailable: number
    orphaned: number
  }
  history: {
    totalSessions: number
    compactedSessions: number
    semanticRecallSessions: number
    documentRecallSessions: number
    memorySummarySessions: number
    attentionNeededSessions: number
    historyReadFailures: number
    totalCompactions: number
    totalTokensSaved: number
    totalSemanticContextEvents: number
    totalSemanticContextItems: number
    totalDocumentContextItems: number
    totalMemorySummaryEvents: number
    totalSemanticExtractions: number
    totalRagPromotions: number
    risksByCode: Partial<Record<SessionHistoryManagementRisk['code'], number>>
    attentionSessionIds: string[]
    failedHistorySessionIds: string[]
    semanticIndex?: ReturnType<RuntimeServices['semanticIndex']['getStatus']>
    dreaming?: ReturnType<RuntimeServices['dreaming']['getStatus']>
    memoryLifecycle?: Awaited<ReturnType<RuntimeServices['semanticIndex']['getLifecycleStatus']>>
  }
  orphaned: {
    pendingApprovalRequestIds: string[]
    pendingApprovalSessionIds: string[]
    pendingQuestionIds: string[]
    pendingQuestionSessionIds: string[]
    runCheckpointSessionIds: string[]
    approvalCheckpointRequestIds: string[]
    toolExecutionSessionIds: string[]
  }
}

const cleanupRequestSchema = z.preprocess(
  (value) => value ?? {},
  z.object({
    dryRun: z.boolean().default(true),
  }),
)

type CleanupRequestBody = z.output<typeof cleanupRequestSchema>

function uniqueSorted(values: Iterable<string>): string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right))
}

function collectAttachmentFileIds(events: readonly SessionEvent[]): Set<string> {
  const ids = new Set<string>()
  for (const event of events) {
    if (event.type !== 'user_message') continue
    for (const attachment of event.attachments ?? []) {
      ids.add(attachment.fileId)
    }
  }
  return ids
}

async function findUnreferencedSessionAttachmentIds(
  runtime: SessionManagementRuntime,
  sessionId: string,
  additionalCandidateIds: Iterable<string> = [],
): Promise<Set<string>> {
  if (
    typeof runtime.sessions.getEvents !== 'function' ||
    typeof runtime.sessions.list !== 'function'
  ) {
    return new Set()
  }

  const candidates = new Set(additionalCandidateIds)
  let sessions: SessionMeta[]
  try {
    for (const id of collectAttachmentFileIds(await runtime.sessions.getEvents(sessionId))) {
      candidates.add(id)
    }
    if (candidates.size === 0) return candidates
    sessions = await listAllSessionMetadata(runtime)
  } catch {
    return new Set()
  }

  try {
    await Promise.all(
      sessions
        .filter((session) => session.id !== sessionId)
        .map(async (session) => {
          const referenced = collectAttachmentFileIds(await runtime.sessions.getEvents(session.id))
          for (const id of referenced) candidates.delete(id)
        }),
    )
  } catch {
    // Fail closed: a history read failure must not delete a file that another
    // session or branch may still reference.
    return new Set()
  }

  return candidates
}

export async function buildHistoryManagementSnapshot(
  runtime: SessionManagementRuntime,
  sessions: SessionMeta[],
): Promise<SessionManagementSnapshot['history']> {
  let memoryLifecycle: Awaited<ReturnType<RuntimeServices['semanticIndex']['getLifecycleStatus']>> | undefined
  try {
    memoryLifecycle = runtime.semanticIndex?.getLifecycleStatus
      ? await runtime.semanticIndex.getLifecycleStatus()
      : undefined
  } catch {
    memoryLifecycle = undefined
  }
  const historyRuntime = {
    ...(runtime.semanticIndex?.getStatus
      ? { semanticIndex: runtime.semanticIndex.getStatus() }
      : {}),
    ...(runtime.dreaming?.getStatus
      ? { dreaming: runtime.dreaming.getStatus() }
      : {}),
    ...(memoryLifecycle ? { memoryLifecycle } : {}),
  }
  const history: SessionManagementSnapshot['history'] = {
    totalSessions: sessions.length,
    compactedSessions: 0,
    semanticRecallSessions: 0,
    documentRecallSessions: 0,
    memorySummarySessions: 0,
    attentionNeededSessions: 0,
    historyReadFailures: 0,
    totalCompactions: 0,
    totalTokensSaved: 0,
    totalSemanticContextEvents: 0,
    totalSemanticContextItems: 0,
    totalDocumentContextItems: 0,
    totalMemorySummaryEvents: 0,
    totalSemanticExtractions: 0,
    totalRagPromotions: 0,
    risksByCode: {},
    attentionSessionIds: [],
    failedHistorySessionIds: [],
    ...historyRuntime,
  }

  await Promise.all(sessions.map(async (session) => {
    try {
      const events = await runtime.sessions.getEvents(session.id)
      const status = buildSessionHistoryManagement(session, events, historyRuntime)
      if (status.compact.compactions > 0) {
        history.compactedSessions += 1
      }
      if (status.semanticRecall.contextEvents > 0) {
        history.semanticRecallSessions += 1
      }
      if (status.semanticRecall.documentItems > 0) {
        history.documentRecallSessions += 1
      }
      if (status.memorySummary.events > 0) {
        history.memorySummarySessions += 1
      }
      if (status.status === 'attention_needed') {
        history.attentionNeededSessions += 1
        history.attentionSessionIds.push(session.id)
      }
      for (const risk of status.risks) {
        history.risksByCode[risk.code] = (history.risksByCode[risk.code] ?? 0) + 1
      }
      history.totalCompactions += status.compact.compactions
      history.totalTokensSaved += status.compact.tokensSaved
      history.totalSemanticContextEvents += status.semanticRecall.contextEvents
      history.totalSemanticContextItems += status.semanticRecall.contextItems
      history.totalDocumentContextItems += status.semanticRecall.documentItems
      history.totalMemorySummaryEvents += status.memorySummary.events
      history.totalSemanticExtractions += status.memorySummary.semanticExtractions
      history.totalRagPromotions += status.memorySummary.ragPromotions
    } catch {
      history.historyReadFailures += 1
      history.failedHistorySessionIds.push(session.id)
    }
  }))

  history.attentionSessionIds = uniqueSorted(history.attentionSessionIds)
  history.failedHistorySessionIds = uniqueSorted(history.failedHistorySessionIds)
  return history
}

export async function cleanupSessionRuntimeState(
  runtime: SessionManagementRuntime,
  sessionId: string,
): Promise<SessionRuntimeCleanupResult> {
  // Session-owned ids join the same cross-session reference scan as journaled
  // attachments, so deleting the source session cannot break a branch.
  const sessionOwnedUploads = listUploadedFilesBySession(sessionId)
  const unreferencedAttachmentIds = await findUnreferencedSessionAttachmentIds(
    runtime,
    sessionId,
    sessionOwnedUploads.map((file) => file.id),
  )
  const runInspection = await runtime.runCheckpoints?.inspect(sessionId)
  const toolExecution = await runtime.toolExecutions?.get(sessionId)
  const approvalCheckpointResult = await runtime.approvalCheckpoints
    ?.deleteForSession(sessionId)
    ?? { deletedCheckpoints: 0, unavailableCheckpoints: 0 }

  const pendingApprovals = runtime.approvalRegistry?.cancelForSession(
    sessionId,
  ) ?? 0
  const pendingQuestions = runtime.questions?.cancelForSession(sessionId) ?? 0

  await Promise.all([
    runtime.primaryAgents?.clear(sessionId) ?? Promise.resolve(),
    runtime.runCheckpoints?.delete(sessionId) ?? Promise.resolve(),
    runtime.toolExecutions?.delete(sessionId) ?? Promise.resolve(),
  ])
  runtime.sessionRuntimeSnapshots?.delete(sessionId)
  // Per-session tool-stats bucket was never evicted — the Map grew for the
  // daemon's lifetime. Drop it here alongside the other per-session runtime state.
  runtime.toolStatsStore?.dispose(sessionId)

  // Complete-deletion (GDPR): purge the durable stores the delete path used to
  // leave orphaned — usage rows, session-derived memory entries + embeddings,
  // and uploaded files owned by the session.
  const usageRecords = runtime.usageTracker?.deleteBySession(sessionId) ?? 0
  const derivedMemories =
    (await runtime.semanticIndex?.deleteBySession(sessionId)) ?? 0
  const removedUploads = await removeUploadedFilesByIds(unreferencedAttachmentIds)

  return {
    pendingApprovals,
    pendingQuestions,
    approvalCheckpoints: approvalCheckpointResult.deletedCheckpoints,
    unavailableApprovalCheckpoints:
      approvalCheckpointResult.unavailableCheckpoints,
    runCheckpoints:
      runInspection && runInspection.status !== 'missing' ? 1 : 0,
    toolExecutions: toolExecution ? 1 : 0,
    usageRecords,
    derivedMemories,
    uploadedFiles: removedUploads.length,
  }
}

async function buildSessionManagementSnapshot(
  runtime: SessionManagementRuntime,
): Promise<SessionManagementSnapshot> {
  const sessions = await listAllSessionMetadata(runtime)
  const sessionIds = new Set(sessions.map((session) => session.id))
  const byStatus = {
    active: 0,
    completed: 0,
    abandoned: 0,
  }
  for (const session of sessions) {
    byStatus[session.status] += 1
  }

  const {
    pendingApprovals,
    pendingQuestions,
    orphanedPendingApprovals,
    orphanedPendingQuestions,
  } = collectSessionInteractionState(runtime, sessions)
  const runCheckpoints = await runtime.runCheckpoints?.list() ?? []
  const approvalCheckpoints = await runtime.approvalCheckpoints?.list() ?? []
  const toolExecutions = await runtime.toolExecutions?.list() ?? []
  const history = await buildHistoryManagementSnapshot(runtime, sessions)

  const orphanedRunCheckpoints = runCheckpoints.filter(
    (checkpoint) => !sessionIds.has(checkpoint.sessionId),
  )
  const orphanedApprovalCheckpoints = approvalCheckpoints.filter(
    (checkpoint) =>
      checkpoint.status === 'available'
      && !sessionIds.has(checkpoint.sessionId),
  )
  const orphanedToolExecutions = toolExecutions.filter(
    (execution) => !sessionIds.has(execution.sessionId),
  )

  return {
    totalSessions: sessions.length,
    byStatus,
    pendingApprovals: {
      total: pendingApprovals.length,
      orphaned: orphanedPendingApprovals.length,
    },
    pendingQuestions: {
      total: pendingQuestions.length,
      orphaned: orphanedPendingQuestions.length,
    },
    runCheckpoints: {
      total: runCheckpoints.length,
      unavailable: runCheckpoints.filter(
        (checkpoint) => checkpoint.status === 'unavailable',
      ).length,
      locked: runCheckpoints.filter((checkpoint) => checkpoint.locked).length,
      orphaned: orphanedRunCheckpoints.length,
    },
    approvalCheckpoints: {
      total: approvalCheckpoints.length,
      unavailable: approvalCheckpoints.filter(
        (checkpoint) => checkpoint.status === 'unavailable',
      ).length,
      orphaned: orphanedApprovalCheckpoints.length,
    },
    toolExecutions: {
      total: toolExecutions.length,
      running: toolExecutions.filter(
        (execution) => execution.status === 'running',
      ).length,
      completed: toolExecutions.filter(
        (execution) => execution.status === 'completed',
      ).length,
      unavailable: toolExecutions.filter(
        (execution) => execution.status === 'unavailable',
      ).length,
      orphaned: orphanedToolExecutions.length,
    },
    history,
    orphaned: {
      pendingApprovalRequestIds: uniqueSorted(
        orphanedPendingApprovals.map((approval) => approval.requestId),
      ),
      pendingApprovalSessionIds: uniqueSorted(
        orphanedPendingApprovals.map((approval) => approval.sessionId),
      ),
      pendingQuestionIds: uniqueSorted(
        orphanedPendingQuestions.map((question) => question.id),
      ),
      pendingQuestionSessionIds: uniqueSorted(
        orphanedPendingQuestions.map((question) => question.sessionId),
      ),
      runCheckpointSessionIds: uniqueSorted(
        orphanedRunCheckpoints.map((checkpoint) => checkpoint.sessionId),
      ),
      approvalCheckpointRequestIds: uniqueSorted(
        orphanedApprovalCheckpoints.map((checkpoint) => checkpoint.requestId),
      ),
      toolExecutionSessionIds: uniqueSorted(
        orphanedToolExecutions.map((execution) => execution.sessionId),
      ),
    },
  }
}

async function cleanupOrphanedSessionState(
  runtime: SessionManagementRuntime,
  snapshot: SessionManagementSnapshot,
): Promise<SessionRuntimeCleanupResult> {
  const result: SessionRuntimeCleanupResult = {
    pendingApprovals: 0,
    pendingQuestions: 0,
    approvalCheckpoints: 0,
    unavailableApprovalCheckpoints: 0,
    runCheckpoints: 0,
    toolExecutions: 0,
    // Orphan cleanup does not touch the durable per-session stores (those are
    // handled by the complete per-session delete path); report zero here.
    usageRecords: 0,
    derivedMemories: 0,
    uploadedFiles: 0,
  }

  for (const sessionId of snapshot.orphaned.pendingApprovalSessionIds) {
    result.pendingApprovals += runtime.approvalRegistry?.cancelForSession(
      sessionId,
    ) ?? 0
  }
  for (const sessionId of snapshot.orphaned.pendingQuestionSessionIds) {
    result.pendingQuestions += runtime.questions?.cancelForSession(sessionId) ?? 0
  }
  for (const requestId of snapshot.orphaned.approvalCheckpointRequestIds) {
    await runtime.approvalCheckpoints?.delete(requestId)
    result.approvalCheckpoints += 1
  }
  for (const sessionId of snapshot.orphaned.runCheckpointSessionIds) {
    await runtime.runCheckpoints?.delete(sessionId)
    result.runCheckpoints += 1
  }
  for (const sessionId of snapshot.orphaned.toolExecutionSessionIds) {
    await runtime.toolExecutions?.delete(sessionId)
    result.toolExecutions += 1
  }

  return result
}

export function registerSessionManagementRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/sessions/management', async (_request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }
    return { data: await buildSessionManagementSnapshot(runtime) }
  })

  app.post<{ Body: CleanupRequestBody }>('/sessions/management/cleanup', {
    preValidation: zodRequestValidation({
      body: {
        schema: cleanupRequestSchema,
        message: 'Invalid session management cleanup body',
      },
    }),
  }, async (request, reply) => {
    const { body } = request
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const before = await buildSessionManagementSnapshot(runtime)
    if (body.dryRun) {
      return {
        data: {
          dryRun: true,
          candidates: before.orphaned,
          deleted: {
            pendingApprovals: 0,
            pendingQuestions: 0,
            approvalCheckpoints: 0,
            unavailableApprovalCheckpoints: 0,
            runCheckpoints: 0,
            toolExecutions: 0,
          },
          before,
          after: before,
        },
      }
    }

    const deleted = await cleanupOrphanedSessionState(runtime, before)
    const after = await buildSessionManagementSnapshot(runtime)
    return {
      data: {
        dryRun: false,
        candidates: before.orphaned,
        deleted,
        before,
        after,
      },
    }
  })
}
