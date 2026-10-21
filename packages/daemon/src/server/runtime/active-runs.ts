import { SharedBoard } from './shared-board.js'
import type { AutonomyLevel } from '@sepilotd/core'

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

export interface LiveRunState {
  steeringNotes?: LiveRunSteeringNote[]
  [key: string]: unknown
}

export interface SessionRunContext {
  mode?: string
  cwd?: string
  workspaceRoot?: string
  provider?: string
  model?: string
  autonomy?: AutonomyLevel
  toolNames?: string[]
  bot?: string
  surface?: string
  recordedAt: number
}

type SessionRunAuthorityField = 'toolNames' | 'bot' | 'surface'

export type SessionRunContextUpdate =
  Omit<SessionRunContext, 'recordedAt' | SessionRunAuthorityField>
  & { [K in SessionRunAuthorityField]?: SessionRunContext[K] | null }

function mergeAuthorityField<T>(next: T | null | undefined, previous: T | undefined): T | undefined {
  if (next === null) return undefined
  return next ?? previous
}

export class ActiveRunRegistry {
  private evidenceReader?: (sessionId: string) => string
  private readonly runContexts = new Map<string, SessionRunContext>()

  recordRunContext(sessionId: string, update: SessionRunContextUpdate): void {
    if (!sessionId) return
    const previous = this.runContexts.get(sessionId)
    const { toolNames, bot, surface, ...rest } = update
    const context: SessionRunContext = { ...rest, recordedAt: Date.now() }
    const mergedToolNames = mergeAuthorityField(toolNames, previous?.toolNames)
    const mergedBot = mergeAuthorityField(bot, previous?.bot)
    const mergedSurface = mergeAuthorityField(surface, previous?.surface)
    if (mergedToolNames !== undefined) context.toolNames = mergedToolNames
    if (mergedBot !== undefined) context.bot = mergedBot
    if (mergedSurface !== undefined) context.surface = mergedSurface
    this.runContexts.delete(sessionId)
    this.runContexts.set(sessionId, context)
    while (this.runContexts.size > 500) {
      const oldest = this.runContexts.keys().next().value
      if (oldest === undefined) break
      this.runContexts.delete(oldest)
    }
  }

  lastRunContext(sessionId: string): SessionRunContext | undefined {
    return this.runContexts.get(sessionId)
  }

  updateRecordedAutonomy(sessionId: string, level: AutonomyLevel): void {
    const existing = this.runContexts.get(sessionId)
    if (!existing) return
    this.runContexts.set(sessionId, { ...existing, autonomy: level })
  }
  setBackgroundEvidenceReader(reader: (sessionId: string) => string): void { this.evidenceReader = reader }
  backgroundEvidence(sessionId: string): string { return this.evidenceReader?.(sessionId) ?? '' }
  private readonly runs = new Map<string, ActiveRunRecord>()
  private readonly cancellers = new Map<string, ActiveRunCanceller[]>()

  readonly sharedBoard = new SharedBoard()

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

  registerCanceller(sessionId: string, cancel: ActiveRunCanceller): () => void {
    const owners = this.cancellers.get(sessionId) ?? []
    owners.push(cancel)
    this.cancellers.set(sessionId, owners)
    return () => {
      const current = this.cancellers.get(sessionId)
      if (!current) return
      const index = current.indexOf(cancel)
      if (index >= 0) current.splice(index, 1)
      if (current.length === 0) this.cancellers.delete(sessionId)
    }
  }

  private readonly privilegeSetters = new Map<string, (level: AutonomyLevel) => void>()

  registerPrivilegeSetter(sessionId: string, setter: (level: AutonomyLevel) => void): () => void {
    this.privilegeSetters.set(sessionId, setter)
    return () => {
      if (this.privilegeSetters.get(sessionId) === setter) this.privilegeSetters.delete(sessionId)
    }
  }

  setPrivilege(sessionId: string, level: AutonomyLevel): boolean {
    this.updateRecordedAutonomy(sessionId, level)
    const setter = this.privilegeSetters.get(sessionId)
    if (!setter) return false
    setter(level)
    return true
  }

  async cancel(sessionId: string): Promise<boolean> {
    const owners = [...(this.cancellers.get(sessionId) ?? [])]
    if (owners.length === 0) return false
    for (const cancel of owners.reverse()) await cancel()
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
