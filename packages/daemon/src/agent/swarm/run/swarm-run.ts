import { EventEmitter } from 'node:events'
import type {
  SwarmAgentHandle,
  SwarmEvent,
  SwarmAgentRecoveryScenario,
  SwarmAgentRecoveryStatus,
  SwarmAgentRecoveryStep,
  SwarmAgentStartupEvidence,
  SwarmRun as SwarmRunSnapshot,
  SwarmRunStatus,
} from '@sepilotd/core'

export interface SwarmRunOpts {
  id: string
  goal: string
  worktree: { path: string; createdByDaemon: boolean; branch?: string; originRepo?: string }
}

export interface SwarmInteractiveLease {
  owner: string
  expiresAt: number
}

function interactiveLeaseTtlMs(): number {
  const raw = Number(process.env.SEPILOTD_SWARM_ATTACH_LEASE_MS ?? 120_000)
  return Number.isFinite(raw) && raw > 0 ? raw : 120_000
}

export class SwarmRun extends EventEmitter {
  readonly id: string
  readonly goal: string
  private worktree: { path: string; createdByDaemon: boolean; branch?: string; originRepo?: string }
  private status: SwarmRunStatus = 'pending'
  private agents = new Map<string, SwarmAgentHandle>()
  private activeHandle: string | undefined
  private createdAt = Date.now()
  private endedAt: number | undefined
  private recoveryAttempts = new Map<string, number>()
  private interactiveLeases = new Map<string, SwarmInteractiveLease>()

  constructor(opts: SwarmRunOpts) {
    super()
    this.setMaxListeners(0) // 0 = uncapped; SSE fan-out can exceed Node's default of 10
    this.id = opts.id
    this.goal = opts.goal
    this.worktree = opts.worktree
  }

  setWorktree(worktree: {
    path: string
    createdByDaemon: boolean
    branch?: string
    originRepo?: string
  }): void {
    this.worktree = worktree
  }

  private emitEvent(e: SwarmEvent): void {
    this.emit('event', e)
  }

  private normalizeLeaseOwner(owner: string): string {
    const trimmed = owner.trim()
    return trimmed || 'anonymous'
  }

  private pruneInteractiveLease(handleId: string, now = Date.now()): void {
    const lease = this.interactiveLeases.get(handleId)
    if (lease && lease.expiresAt <= now) {
      this.interactiveLeases.delete(handleId)
      this.emitEvent({
        type: 'supervisor.message',
        runId: this.id,
        text: `interactive_lease_expired ${handleId}`,
        ts: now,
      })
    }
  }

  start(): void {
    if (this.status !== 'pending') throw new Error('SwarmRun already started')
    this.status = 'running'
    this.emitEvent({ type: 'run.started', runId: this.id, goal: this.goal, ts: Date.now() })
  }

  finish(status: SwarmRunStatus): void {
    if (this.status !== 'running') return
    this.status = status
    this.endedAt = Date.now()
    this.emitEvent({ type: 'run.ended', runId: this.id, status, ts: Date.now() })
  }

  addAgent(handle: SwarmAgentHandle): void {
    this.agents.set(handle.handle, handle)
    this.emitEvent({ type: 'agent.spawned', runId: this.id, agent: handle, ts: Date.now() })
    if (handle.startupEvidence) {
      this.recordStartupEvidence(handle.handle, handle.startupEvidence)
    }
  }

  recordStartupEvidence(handleId: string, evidence: SwarmAgentStartupEvidence): void {
    const agent = this.agents.get(handleId)
    if (agent) {
      agent.startupEvidence = { ...evidence }
      agent.runtime = evidence.runtime
    }
    this.emitEvent({
      type: 'agent.startup',
      runId: this.id,
      handle: handleId,
      evidence: { ...evidence },
      ts: Date.now(),
    })
  }

  recoveryAttemptCount(
    handleId: string,
    scenario: SwarmAgentRecoveryScenario,
    action: SwarmAgentRecoveryStep,
  ): number {
    return this.recoveryAttempts.get(`${handleId}:${scenario}:${action}`) ?? 0
  }

  claimRecoveryAttempt(
    handleId: string,
    scenario: SwarmAgentRecoveryScenario,
    action: SwarmAgentRecoveryStep,
  ): number {
    const key = `${handleId}:${scenario}:${action}`
    const attempt = (this.recoveryAttempts.get(key) ?? 0) + 1
    this.recoveryAttempts.set(key, attempt)
    return attempt
  }

  recordRecoveryEvent(input: {
    handle: string
    scenario: SwarmAgentRecoveryScenario
    action: SwarmAgentRecoveryStep
    attempt: number
    maxAttempts: number
    status: SwarmAgentRecoveryStatus
    message: string
    replacementHandle?: string
    outputPreview?: string
  }): void {
    this.emitEvent({
      type: 'agent.recovery',
      runId: this.id,
      handle: input.handle,
      scenario: input.scenario,
      action: input.action,
      attempt: input.attempt,
      maxAttempts: input.maxAttempts,
      status: input.status,
      message: input.message,
      replacementHandle: input.replacementHandle,
      outputPreview: input.outputPreview,
      ts: Date.now(),
    })
  }

  acquireInteractiveLease(handleId: string, owner: string): boolean {
    if (!this.agents.has(handleId)) return false
    const normalizedOwner = this.normalizeLeaseOwner(owner)
    const now = Date.now()
    this.pruneInteractiveLease(handleId, now)
    const current = this.interactiveLeases.get(handleId)
    if (current && current.owner !== normalizedOwner) return false
    const lease = {
      owner: normalizedOwner,
      expiresAt: now + interactiveLeaseTtlMs(),
    }
    this.interactiveLeases.set(handleId, lease)
    this.emitEvent({
      type: 'supervisor.message',
      runId: this.id,
      text: `interactive_lease_acquired ${handleId} owner=${normalizedOwner}`,
      ts: now,
    })
    return true
  }

  renewInteractiveLease(handleId: string, owner: string): boolean {
    const normalizedOwner = this.normalizeLeaseOwner(owner)
    const now = Date.now()
    this.pruneInteractiveLease(handleId, now)
    const current = this.interactiveLeases.get(handleId)
    if (!current || current.owner !== normalizedOwner) return false
    current.expiresAt = now + interactiveLeaseTtlMs()
    this.emitEvent({
      type: 'supervisor.message',
      runId: this.id,
      text: `interactive_lease_renewed ${handleId} owner=${normalizedOwner}`,
      ts: now,
    })
    return true
  }

  releaseInteractiveLease(handleId: string, owner: string): boolean {
    const normalizedOwner = this.normalizeLeaseOwner(owner)
    const now = Date.now()
    this.pruneInteractiveLease(handleId, now)
    const current = this.interactiveLeases.get(handleId)
    if (!current) return true
    if (current.owner !== normalizedOwner) return false
    this.interactiveLeases.delete(handleId)
    this.emitEvent({
      type: 'supervisor.message',
      runId: this.id,
      text: `interactive_lease_released ${handleId} owner=${normalizedOwner}`,
      ts: now,
    })
    return true
  }

  isInteractiveHeld(handleId: string): boolean {
    this.pruneInteractiveLease(handleId)
    return this.interactiveLeases.has(handleId)
  }

  interactiveLeaseOwner(handleId: string): string | undefined {
    this.pruneInteractiveLease(handleId)
    return this.interactiveLeases.get(handleId)?.owner
  }

  getInteractiveLease(handleId: string): SwarmInteractiveLease | undefined {
    this.pruneInteractiveLease(handleId)
    const lease = this.interactiveLeases.get(handleId)
    return lease ? { ...lease } : undefined
  }

  /** Silent no-op when the handle is missing — supervisor + lifecycle paths can race during teardown. */
  removeAgent(handleId: string): void {
    if (!this.agents.delete(handleId)) return
    if (this.activeHandle === handleId) this.activeHandle = undefined
    this.interactiveLeases.delete(handleId)
    this.emitEvent({ type: 'agent.killed', runId: this.id, handle: handleId, ts: Date.now() })
  }

  /** Silent no-op when the handle is missing — same teardown-race rationale as removeAgent. */
  setStatus(handleId: string, status: SwarmAgentHandle['status']): void {
    const a = this.agents.get(handleId)
    if (!a) return
    a.status = status
    this.emitEvent({ type: 'agent.status', runId: this.id, handle: handleId, status, ts: Date.now() })
  }

  /** Throws on unknown handle — promoting a non-existent agent is a programmer error. */
  setActive(handleId: string): void {
    if (!this.agents.has(handleId)) throw new Error(`unknown handle: ${handleId}`)
    this.activeHandle = handleId
    this.emitEvent({ type: 'agent.active', runId: this.id, handle: handleId, ts: Date.now() })
  }

  getActive(): SwarmAgentHandle | undefined {
    return this.activeHandle ? this.agents.get(this.activeHandle) : undefined
  }

  getAgent(handleId: string): SwarmAgentHandle | undefined {
    return this.agents.get(handleId)
  }

  snapshot(): SwarmRunSnapshot {
    return {
      id: this.id,
      goal: this.goal,
      status: this.status,
      worktree: this.worktree,
      activeHandle: this.activeHandle,
      agents: [...this.agents.values()].map((a) => ({ ...a })),
      createdAt: this.createdAt,
      endedAt: this.endedAt,
    }
  }
}
