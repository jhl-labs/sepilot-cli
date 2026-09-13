import { SharedBoard } from './shared-board.js'

export type ActiveRunStatus = 'running'

export interface ActiveRunUpdate {
  sessionId: string
  graphId?: string
  currentNode?: string
  iteration: number
  maxIterations: number
  tokensInput?: number
  tokensOutput?: number
}

export interface ActiveRunSnapshot {
  sessionId: string
  graphId?: string
  currentNode?: string
  iteration: number
  maxIterations: number
  tokensInput: number
  tokensOutput: number
  pendingSteeringNoteCount: number
  oldestPendingSteeringAt?: string
  latestPendingSteeringAt?: string
  startedAt: string
  elapsedMs: number
  status: ActiveRunStatus
}

type DynamicActiveRunSnapshotFields =
  | 'pendingSteeringNoteCount'
  | 'oldestPendingSteeringAt'
  | 'latestPendingSteeringAt'

interface ActiveRunRecord extends Omit<ActiveRunSnapshot, DynamicActiveRunSnapshotFields> {
  startedAtMs: number
  updatedAtMs: number
}

const MAX_PENDING_STEERING_NOTES = 20

export interface LiveRunSteeringNote {
  id: string
  message: string
  kind: 'instruction' | 'question'
  createdAt: number
  consumedAt?: number
  cancelledAt?: number
}

export type SteeringCancellationTarget =
  | { selector: 'latest' | 'all' }
  | { noteId: string }

export type SteeringCancellationResult =
  | {
      status: 'cancelled'
      cancelledNoteIds: string[]
      pendingSteeringNoteCount: number
    }
  | {
      status: 'already_consumed' | 'already_cancelled' | 'not_found' | 'no_pending'
      cancelledNoteIds: []
      pendingSteeringNoteCount: number
    }

type ActiveRunCanceller = () => void | Promise<void>
type ActiveRunChangeListener = (sessionId: string, running: boolean) => void

/** Minimal shape the live registry needs — anything carrying a mutable
 *  `steeringNotes` array, per `agent/graph/state-board.ts` `AgentState`. Kept
 *  structural (not importing `AgentState` here) so this runtime module stays
 *  independent of the graph package. */
export interface LiveRunState {
  steeringNotes?: LiveRunSteeringNote[]
  [key: string]: unknown
}

export class ActiveRunRegistry {
  private evidenceReader?: (sessionId: string) => string
  setBackgroundEvidenceReader(reader: (sessionId: string) => string): void { this.evidenceReader = reader }
  backgroundEvidence(sessionId: string): string { return this.evidenceReader?.(sessionId) ?? '' }
  private readonly runs = new Map<string, ActiveRunRecord>()
  private readonly cancellers = new Map<string, ActiveRunCanceller>()

  /**
   * Live board shared by concurrent sibling subagents under the same parent run
   * (PLAN_065 T4). Juxtaposed here so it shares the run lifecycle: entries are
   * dropped in `finish()` when the parent run ends.
   */
  readonly sharedBoard = new SharedBoard()

  /**
   * Reference to the LIVE, in-flight `AgentState` object for a running graph
   * (not a checkpointed/cloned snapshot). The graph engine registers it
   * alongside its per-node `upsert()` progress update so routes such as
   * `POST /sessions/:id/steer` can mutate it directly (e.g. via
   * `appendSteeringNote`) and have the running loop observe the change on its
   * next turn. Cleared in `finish()`.
   */
  private readonly liveStates = new Map<string, LiveRunState>()

  private readonly pendingSteeringNotes = new Map<string, LiveRunSteeringNote[]>()

  constructor(private readonly onRunChange?: ActiveRunChangeListener) {}

  registerLiveState(sessionId: string, state: LiveRunState): void {
    const pending = this.pendingSteeringNotes.get(sessionId)
    if (pending && pending.length > 0) {
      state.steeringNotes = [...(state.steeringNotes ?? []), ...pending].slice(
        -MAX_PENDING_STEERING_NOTES,
      )
      this.pendingSteeringNotes.delete(sessionId)
    }
    this.liveStates.set(sessionId, state)
  }

  registerCanceller(sessionId: string, cancel: ActiveRunCanceller, options?: { ifAbsent?: boolean }): () => void {
    if (options?.ifAbsent && this.cancellers.has(sessionId)) return () => {}
    this.cancellers.set(sessionId, cancel)
    return () => {
      if (this.cancellers.get(sessionId) === cancel) {
        this.cancellers.delete(sessionId)
      }
    }
  }

  async cancel(sessionId: string): Promise<boolean> {
    const cancel = this.cancellers.get(sessionId)
    if (!cancel) return false
    await cancel()
    return true
  }

  getLiveState(sessionId: string): LiveRunState | null {
    return this.liveStates.get(sessionId) ?? null
  }

  queuePendingSteeringNote(sessionId: string, note: LiveRunSteeringNote): void {
    const list = this.pendingSteeringNotes.get(sessionId) ?? []
    list.push(note)
    if (list.length > MAX_PENDING_STEERING_NOTES) {
      list.splice(0, list.length - MAX_PENDING_STEERING_NOTES)
    }
    this.pendingSteeringNotes.set(sessionId, list)
  }

  /**
   * Cancel steering synchronously against the registry-owned live state.
   * Consumption and cancellation both run on the daemon event loop, so a
   * note can transition from pending to exactly one terminal state without a
   * read/modify/write gap between an HTTP route and the graph loop.
   */
  cancelPendingSteeringNotes(
    sessionId: string,
    target: SteeringCancellationTarget,
    now = Date.now(),
  ): SteeringCancellationResult {
    const notes = this.allSteeringNotesFor(sessionId)
    const pending = notes.filter(
      (note) => note.consumedAt === undefined && note.cancelledAt === undefined,
    )

    if ('noteId' in target) {
      const note = notes.find((candidate) => candidate.id === target.noteId)
      if (!note) return this.steeringCancellationMiss(sessionId, 'not_found')
      if (note.consumedAt !== undefined) {
        return this.steeringCancellationMiss(sessionId, 'already_consumed')
      }
      if (note.cancelledAt !== undefined) {
        return this.steeringCancellationMiss(sessionId, 'already_cancelled')
      }
      note.cancelledAt = now
      return {
        status: 'cancelled',
        cancelledNoteIds: [note.id],
        pendingSteeringNoteCount: this.pendingSteeringNotesFor(sessionId).length,
      }
    }

    if (pending.length === 0) {
      return this.steeringCancellationMiss(sessionId, 'no_pending')
    }
    const selected = target.selector === 'all'
      ? pending
      : [pending.reduce((latest, note) =>
          note.createdAt >= latest.createdAt ? note : latest)]
    for (const note of selected) note.cancelledAt = now
    return {
      status: 'cancelled',
      cancelledNoteIds: selected.map((note) => note.id),
      pendingSteeringNoteCount: this.pendingSteeringNotesFor(sessionId).length,
    }
  }

  upsert(update: ActiveRunUpdate, now = Date.now()): ActiveRunSnapshot {
    const existing = this.runs.get(update.sessionId)
    const startedAtMs = existing?.startedAtMs ?? now
    const record: ActiveRunRecord = {
      sessionId: update.sessionId,
      graphId: update.graphId ?? existing?.graphId,
      currentNode: update.currentNode ?? existing?.currentNode,
      iteration: update.iteration,
      maxIterations: update.maxIterations,
      tokensInput: update.tokensInput ?? existing?.tokensInput ?? 0,
      tokensOutput: update.tokensOutput ?? existing?.tokensOutput ?? 0,
      startedAt: existing?.startedAt ?? new Date(startedAtMs).toISOString(),
      elapsedMs: Math.max(0, now - startedAtMs),
      status: 'running',
      startedAtMs,
      updatedAtMs: now,
    }
    this.runs.set(update.sessionId, record)
    if (!existing) {
      this.onRunChange?.(update.sessionId, true)
    }
    return this.toSnapshot(record, now)
  }

  finish(sessionId: string): void {
    const wasRunning = this.runs.delete(sessionId)
    this.cancellers.delete(sessionId)
    this.sharedBoard.clear(sessionId)
    this.liveStates.delete(sessionId)
    this.pendingSteeringNotes.delete(sessionId)
    if (wasRunning) {
      this.onRunChange?.(sessionId, false)
    }
  }

  get(sessionId: string, now = Date.now()): ActiveRunSnapshot | null {
    const record = this.runs.get(sessionId)
    return record ? this.toSnapshot(record, now) : null
  }

  list(now = Date.now()): ActiveRunSnapshot[] {
    return [...this.runs.values()]
      .sort((a, b) => a.startedAtMs - b.startedAtMs || a.sessionId.localeCompare(b.sessionId))
      .map((record) => this.toSnapshot(record, now))
  }

  private toSnapshot(record: ActiveRunRecord, now: number): ActiveRunSnapshot {
    const pendingSteering = this.pendingSteeringNotesFor(record.sessionId)
    const createdTimes = pendingSteering
      .map((note) => note.createdAt)
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b)
    const oldestPendingSteeringAt = createdTimes[0]
    const latestPendingSteeringAt = createdTimes[createdTimes.length - 1]

    return {
      sessionId: record.sessionId,
      ...(record.graphId ? { graphId: record.graphId } : {}),
      ...(record.currentNode ? { currentNode: record.currentNode } : {}),
      iteration: record.iteration,
      maxIterations: record.maxIterations,
      tokensInput: record.tokensInput,
      tokensOutput: record.tokensOutput,
      pendingSteeringNoteCount: pendingSteering.length,
      ...(oldestPendingSteeringAt !== undefined
        ? { oldestPendingSteeringAt: new Date(oldestPendingSteeringAt).toISOString() }
        : {}),
      ...(latestPendingSteeringAt !== undefined
        ? { latestPendingSteeringAt: new Date(latestPendingSteeringAt).toISOString() }
        : {}),
      startedAt: record.startedAt,
      elapsedMs: Math.max(0, now - record.startedAtMs),
      status: record.status,
    }
  }

  private pendingSteeringNotesFor(sessionId: string): LiveRunSteeringNote[] {
    return this.allSteeringNotesFor(sessionId).filter(
      (note) => note.consumedAt === undefined && note.cancelledAt === undefined,
    )
  }

  private allSteeringNotesFor(sessionId: string): LiveRunSteeringNote[] {
    return [
      ...(this.liveStates.get(sessionId)?.steeringNotes ?? []),
      ...(this.pendingSteeringNotes.get(sessionId) ?? []),
    ]
  }

  private steeringCancellationMiss(
    sessionId: string,
    status: Exclude<SteeringCancellationResult['status'], 'cancelled'>,
  ): SteeringCancellationResult {
    return {
      status,
      cancelledNoteIds: [],
      pendingSteeringNoteCount: this.pendingSteeringNotesFor(sessionId).length,
    }
  }
}
