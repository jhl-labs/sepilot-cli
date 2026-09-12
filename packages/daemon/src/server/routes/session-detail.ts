import type { SessionEvent, SessionMeta } from '@sepilotd/core'
import { buildPendingApprovals } from '../approval-state.js'
import type { PendingApproval } from '../runtime/approvals.js'
import { assessRunResume } from '../runtime/resume.js'
import type { SessionDetailCapabilities } from '../runtime/capabilities.js'
import {
  buildSessionCompletionChecklist,
  buildSessionContractLedger,
  buildSessionContextEngine,
  buildSessionDebateRounds,
  buildSessionEvidenceManifest,
  buildSessionEvaluationGate,
  buildSessionHistoryManagement,
  buildSessionPlannerWorkingMemory,
  buildSessionRawEditCheckpoints,
  buildSessionRunContract,
  buildSessionTraceMetrics,
  buildSessionWorkingMemory,
} from '../session-contracts.js'

export function getDelegationId(tags: string[] = []): string | undefined {
  const tag = tags.find((value) => value.startsWith('delegation:'))
  return tag ? tag.slice('delegation:'.length) : undefined
}

interface DelegationSnapshot {
  delegationId: string
  targetDevice: string
  claimHealth: 'healthy' | 'degraded' | 'lost'
  startedAt: string
  updatedAt: string
  degradedSince?: string
  lastHeartbeatAt?: string
  lastError?: string
  leaseLossSource?: 'gateway' | 'comments' | 'transport'
}

export function getLatestDelegationSnapshot(
  events: SessionEvent[],
): DelegationSnapshot | undefined {
  const latest = [...events]
    .reverse()
    .find((event) => event.type === 'delegation_state')

  if (!latest || latest.type !== 'delegation_state') {
    return undefined
  }

  return {
    delegationId: latest.delegationId,
    targetDevice: latest.targetDevice,
    claimHealth: latest.claimHealth,
    startedAt: latest.startedAt,
    updatedAt: latest.timestamp,
    degradedSince: latest.degradedSince,
    lastHeartbeatAt: latest.lastHeartbeatAt,
    lastError: latest.lastError,
    leaseLossSource: latest.source,
  }
}

export function buildSessionMetaSnapshot(
  runtime: Pick<SessionDetailCapabilities, 'primaryAgents' | 'activeRuns'>,
  session: SessionMeta,
) {
  const primaryAgentId = runtime.primaryAgents?.get(session.id)
  return {
    ...session,
    isRunning: runtime.activeRuns?.get(session.id) != null,
    ...(primaryAgentId ? { primaryAgentId } : {}),
  }
}

export async function buildSessionPendingApprovals(
  runtime: Pick<SessionDetailCapabilities, 'approvalRegistry' | 'approvalCheckpoints'>,
  sessionId: string,
  events: SessionEvent[],
): Promise<PendingApproval[]> {
  return Promise.all(
    buildPendingApprovals(
      sessionId,
      events,
      runtime.approvalRegistry?.listForSession(sessionId) ?? [],
    ).map(async (approval) => ({
      ...approval,
      resumeAvailable: runtime.approvalCheckpoints
        ? await runtime.approvalCheckpoints.has(approval.requestId)
        : false,
    })),
  )
}

async function getMemoryLifecycleSnapshot(
  runtime: Pick<SessionDetailCapabilities, 'semanticIndex'>,
) {
  try {
    return await runtime.semanticIndex?.getLifecycleStatus()
  } catch {
    return undefined
  }
}

export async function buildSessionDetail(
  runtime: SessionDetailCapabilities,
  session: SessionMeta,
) {
  const events = await runtime.sessions.getEvents(session.id)
  const pendingQuestions = (runtime.questions?.list(session.id) ?? []).map(
    (question) => ({
      id: question.id,
      sessionId: question.sessionId,
      prompt: question.prompt,
      choices: question.choices,
    }),
  )
  const pendingApprovals = await buildSessionPendingApprovals(runtime, session.id, events)
  const resumableRunInspection = runtime.runCheckpoints
    ? await runtime.runCheckpoints.inspect(session.id)
    : { status: 'missing' as const }
  const resumableRun = resumableRunInspection.status === 'available'
    ? resumableRunInspection.checkpoint
    : null
  const resumableRunIssue = resumableRunInspection.status === 'unavailable'
    ? {
        status: resumableRunInspection.issue.status,
        message: resumableRunInspection.issue.message,
      }
    : undefined
  const delegationId = getDelegationId(session.tags)
  const resumableRunAssessment = resumableRun
    ? await assessRunResume(
        resumableRun,
        runtime.toolRegistry,
        runtime.toolExecutions,
      )
    : null
  const delegationState = delegationId
    ? runtime.delegationWorker.getSessionLeaseState(session.id, delegationId)
    : undefined
  const delegation = delegationState
    ? {
        delegationId: delegationState.delegationId,
        targetDevice: delegationState.targetDevice,
        claimHealth: delegationState.claimHealth,
        startedAt: delegationState.startedAt,
        updatedAt: delegationState.updatedAt,
        degradedSince: delegationState.degradedSince,
        lastHeartbeatAt: delegationState.lastHeartbeatAt,
        lastError: delegationState.lastError,
        leaseLossSource: delegationState.leaseLossSource,
      }
    : getLatestDelegationSnapshot(events)
  const sessionContracts = runtime.sessionRuntimeSnapshots
    ? await runtime.sessionRuntimeSnapshots.getOrRefresh(session, {
        events,
        pendingApprovals,
        pendingQuestions,
      })
    : (() => {
        const evidenceManifest = buildSessionEvidenceManifest(session, events)
        return {
          traceMetrics: buildSessionTraceMetrics(session, events),
          contextEngine: buildSessionContextEngine(
            session,
            events,
            pendingQuestions,
          ),
          completionChecklist: buildSessionCompletionChecklist(
            session,
            events,
            pendingApprovals,
            pendingQuestions,
          ),
          workingMemory: buildSessionWorkingMemory(
            session,
            events,
            pendingQuestions,
          ),
          runContract: buildSessionRunContract(events),
          contractLedger: buildSessionContractLedger(session, events, evidenceManifest),
          evidenceManifest,
          evaluationGate: buildSessionEvaluationGate(session, events, evidenceManifest),
        }
      })()
  const memoryLifecycle = await getMemoryLifecycleSnapshot(runtime)

  return {
    ...buildSessionMetaSnapshot(runtime, session),
    events,
    pendingQuestions,
    pendingApprovals,
    traceMetrics: sessionContracts.traceMetrics,
    contextEngine: sessionContracts.contextEngine,
    historyManagement: buildSessionHistoryManagement(session, events, {
      ...(runtime.semanticIndex?.getStatus
        ? { semanticIndex: runtime.semanticIndex.getStatus() }
        : {}),
      ...(runtime.dreaming?.getStatus
        ? { dreaming: runtime.dreaming.getStatus() }
        : {}),
      ...(memoryLifecycle
        ? { memoryLifecycle }
        : {}),
    }),
    completionChecklist: sessionContracts.completionChecklist,
    workingMemory: sessionContracts.workingMemory,
    runContract: sessionContracts.runContract,
    contractLedger: sessionContracts.contractLedger,
    evidenceManifest: sessionContracts.evidenceManifest,
    evaluationGate: sessionContracts.evaluationGate,
    // Raw (not the summarized SessionEditRollback[] shape used elsewhere in
    // sessionContracts) so these match exactly what the live SSE
    // onEditCheckpointResolved / onDebateRound / onPlannerWorkingMemoryUpdated
    // callbacks already send — the desktop/CLI clients can feed both through
    // the same store actions without an adapter.
    editRollbacks: buildSessionRawEditCheckpoints(events),
    debateRounds: buildSessionDebateRounds(events),
    plannerWorkingMemory: buildSessionPlannerWorkingMemory(events),
    resumableRun: resumableRun
      ? {
          stage: resumableRun.stage,
          checkpointedAt: resumableRun.checkpointedAt,
          mode: resumableRunAssessment?.mode ?? 'exact',
          forceRequired: resumableRunAssessment?.forceRequired ?? false,
          currentTool: resumableRunAssessment?.currentTool,
          currentToolCount: resumableRunAssessment?.currentToolCount,
          currentTools: resumableRunAssessment?.currentTools,
          journaledResultAvailable:
            resumableRunAssessment?.journaledResultAvailable ?? false,
          recoveryProbeAvailable:
            resumableRunAssessment?.recoveryProbeAvailable ?? false,
        }
      : undefined,
    resumableRunIssue,
    delegation,
  }
}
