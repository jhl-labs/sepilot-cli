import type { ISessionStore, SessionEvent, SessionMeta } from '@sepilotd/core'
import {
  buildSessionCompletionChecklist,
  buildSessionContractLedger,
  buildSessionContextEngine,
  buildSessionDebateRounds,
  buildSessionEditRollbacks,
  buildSessionEvidenceManifest,
  buildSessionEvaluationGate,
  buildSessionPlannerWorkingMemory,
  buildSessionRunContract,
  buildSessionTraceMetrics,
  buildSessionWorkingMemory,
  type PendingApprovalLike,
  type PendingQuestionLike,
  type SessionCompletionChecklist,
  type SessionContractLedger,
  type SessionContextEngine,
  type SessionDebateRounds,
  type SessionEditRollback,
  type SessionEvidenceManifest,
  type SessionEvaluationGate,
  type SessionPlannerWorkingMemory,
  type SessionRunContract,
  type SessionTraceMetrics,
  type SessionWorkingMemory,
} from '../session-contracts.js'
import type { SessionWatchChange } from './session-watch.js'

export interface SessionRuntimeSnapshot {
  sessionId: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  pendingApprovalSignature: string
  pendingQuestionSignature: string
  refreshedAt: string
  traceMetrics: SessionTraceMetrics
  contextEngine: SessionContextEngine
  completionChecklist: SessionCompletionChecklist
  workingMemory: SessionWorkingMemory
  editRollbacks: SessionEditRollback[]
  debateRounds: SessionDebateRounds
  plannerWorkingMemory: SessionPlannerWorkingMemory
  runContract: SessionRunContract
  contractLedger: SessionContractLedger
  evidenceManifest: SessionEvidenceManifest
  evaluationGate: SessionEvaluationGate
}

export interface SessionRuntimeSnapshotInput {
  events: SessionEvent[]
  pendingApprovals: PendingApprovalLike[]
  pendingQuestions: PendingQuestionLike[]
}

export interface SessionRuntimeSnapshotPendingState {
  pendingApprovals: PendingApprovalLike[]
  pendingQuestions: PendingQuestionLike[]
}

export interface SessionRuntimeSnapshotStoreDeps {
  sessions: ISessionStore
  getPendingState?: (
    session: SessionMeta,
    events: SessionEvent[],
  ) => Promise<SessionRuntimeSnapshotPendingState> | SessionRuntimeSnapshotPendingState
  now?: () => string
}

export interface SessionRuntimeSnapshotBroker {
  subscribeAll(listener: (change: SessionWatchChange) => void): () => void
}

function sessionIdFromChange(change: SessionWatchChange): string {
  return change.type === 'created'
    ? change.session.id
    : change.sessionId
}

function eventRevision(
  events: SessionEvent[],
  fallbackLastEventAt = '',
): Pick<
  SessionRuntimeSnapshot,
  'eventCount' | 'lastEventId' | 'lastEventAt'
> {
  const lastEvent = events.at(-1)
  return {
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt: lastEvent?.timestamp ?? fallbackLastEventAt,
  }
}

function pendingApprovalSignature(pendingApprovals: PendingApprovalLike[]): string {
  return [...pendingApprovals]
    .map((approval) => [
      approval.requestId,
      approval.state,
      approval.resumeAvailable ? 'resume' : 'no-resume',
    ].join(':'))
    .sort()
    .join('|')
}

function pendingQuestionSignature(pendingQuestions: PendingQuestionLike[]): string {
  return [...pendingQuestions]
    .map((question) => [
      question.id,
      question.prompt,
      question.choices?.join(',') ?? '',
    ].join(':'))
    .sort()
    .join('|')
}

function isSnapshotCurrent(
  snapshot: SessionRuntimeSnapshot,
  session: SessionMeta,
  input: SessionRuntimeSnapshotInput,
): boolean {
  const revision = eventRevision(input.events, session.updatedAt)
  return snapshot.eventCount === revision.eventCount
    && snapshot.lastEventId === revision.lastEventId
    && snapshot.lastEventAt === revision.lastEventAt
    && snapshot.pendingApprovalSignature === pendingApprovalSignature(input.pendingApprovals)
    && snapshot.pendingQuestionSignature === pendingQuestionSignature(input.pendingQuestions)
}

export class SessionRuntimeSnapshotStore {
  private readonly snapshots = new Map<string, SessionRuntimeSnapshot>()
  private readonly refreshes = new Map<string, Promise<SessionRuntimeSnapshot | null>>()
  private unsubscribe: (() => void) | null = null

  constructor(private readonly deps: SessionRuntimeSnapshotStoreDeps) {}

  attach(broker: SessionRuntimeSnapshotBroker): void {
    this.unsubscribe?.()
    this.unsubscribe = broker.subscribeAll((change) => {
      const sessionId = sessionIdFromChange(change)
      if (change.type === 'deleted') {
        this.snapshots.delete(sessionId)
        return
      }

      void this.refreshSession(sessionId).catch(() => {
        // Snapshot pre-warming must not affect session watch delivery.
      })
    })
  }

  detach(): void {
    this.unsubscribe?.()
    this.unsubscribe = null
  }

  get(sessionId: string): SessionRuntimeSnapshot | undefined {
    return this.snapshots.get(sessionId)
  }

  delete(sessionId: string): void {
    this.snapshots.delete(sessionId)
  }

  async waitForRefresh(sessionId: string): Promise<void> {
    await this.refreshes.get(sessionId)
  }

  async refreshSession(sessionId: string): Promise<SessionRuntimeSnapshot | null> {
    const active = this.refreshes.get(sessionId)
    if (active) {
      return active
    }

    const refresh = this.refreshSessionNow(sessionId)
      .finally(() => {
        if (this.refreshes.get(sessionId) === refresh) {
          this.refreshes.delete(sessionId)
        }
      })
    this.refreshes.set(sessionId, refresh)
    return refresh
  }

  async getOrRefresh(
    session: SessionMeta,
    input: SessionRuntimeSnapshotInput,
  ): Promise<SessionRuntimeSnapshot> {
    const existing = this.snapshots.get(session.id)
    if (existing && isSnapshotCurrent(existing, session, input)) {
      return existing
    }

    return this.buildAndStore(session, input)
  }

  private async refreshSessionNow(
    sessionId: string,
  ): Promise<SessionRuntimeSnapshot | null> {
    const session = await this.deps.sessions.get(sessionId)
    if (!session) {
      this.snapshots.delete(sessionId)
      return null
    }

    const events = await this.deps.sessions.getEvents(sessionId)
    const pending = await this.getPendingState(session, events)
    return this.buildAndStore(session, {
      events,
      pendingApprovals: pending.pendingApprovals,
      pendingQuestions: pending.pendingQuestions,
    })
  }

  private async getPendingState(
    session: SessionMeta,
    events: SessionEvent[],
  ): Promise<SessionRuntimeSnapshotPendingState> {
    return this.deps.getPendingState?.(session, events) ?? {
      pendingApprovals: [],
      pendingQuestions: [],
    }
  }

  private buildAndStore(
    session: SessionMeta,
    input: SessionRuntimeSnapshotInput,
  ): SessionRuntimeSnapshot {
    const revision = eventRevision(input.events, session.updatedAt)
    const evidenceManifest = buildSessionEvidenceManifest(session, input.events)
    const snapshot: SessionRuntimeSnapshot = {
      sessionId: session.id,
      eventCount: revision.eventCount,
      lastEventId: revision.lastEventId,
      lastEventAt: revision.lastEventAt,
      pendingApprovalSignature: pendingApprovalSignature(input.pendingApprovals),
      pendingQuestionSignature: pendingQuestionSignature(input.pendingQuestions),
      refreshedAt: this.deps.now?.() ?? new Date().toISOString(),
      traceMetrics: buildSessionTraceMetrics(session, input.events),
      contextEngine: buildSessionContextEngine(
        session,
        input.events,
        input.pendingQuestions,
      ),
      completionChecklist: buildSessionCompletionChecklist(
        session,
        input.events,
        input.pendingApprovals,
        input.pendingQuestions,
      ),
      workingMemory: buildSessionWorkingMemory(
        session,
        input.events,
        input.pendingQuestions,
      ),
      editRollbacks: buildSessionEditRollbacks(input.events),
      debateRounds: buildSessionDebateRounds(input.events),
      plannerWorkingMemory: buildSessionPlannerWorkingMemory(input.events),
      runContract: buildSessionRunContract(input.events),
      contractLedger: buildSessionContractLedger(session, input.events, evidenceManifest),
      evidenceManifest,
      evaluationGate: buildSessionEvaluationGate(session, input.events, evidenceManifest),
    }

    this.snapshots.set(session.id, snapshot)
    return snapshot
  }
}
