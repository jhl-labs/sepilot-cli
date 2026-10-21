export class SessionBusyRegistry {
  private readonly busy = new Set<string>()
  private readonly idleWaiters = new Map<string, Set<() => void>>()
  private readonly idleListeners = new Set<(sessionId: string) => void>()

  onIdle(listener: (sessionId: string) => void): () => void {
    this.idleListeners.add(listener)
    return () => { this.idleListeners.delete(listener) }
  }

  tryAcquire(sessionId: string): boolean {
    if (this.busy.has(sessionId)) return false
    this.busy.add(sessionId)
    return true
  }

  tryAcquireLease(sessionId: string): SessionBusyLease | undefined {
    if (!this.tryAcquire(sessionId)) return undefined
    return this.createLease(sessionId)
  }

  release(sessionId: string): void {
    if (!this.busy.delete(sessionId)) return
    const waiters = this.idleWaiters.get(sessionId)
    if (waiters) {
      this.idleWaiters.delete(sessionId)
      for (const resolve of waiters) resolve()
    }
    for (const listener of this.idleListeners) {
      try { listener(sessionId) } catch {}
    }
  }

  isBusy(sessionId: string): boolean {
    return this.busy.has(sessionId)
  }

  waitUntilIdle(sessionId: string): Promise<void> {
    if (!this.busy.has(sessionId)) return Promise.resolve()
    return new Promise((resolve) => {
      const waiters = this.idleWaiters.get(sessionId) ?? new Set<() => void>()
      waiters.add(resolve)
      this.idleWaiters.set(sessionId, waiters)
      if (!this.busy.has(sessionId)) {
        waiters.delete(resolve)
        if (waiters.size === 0) this.idleWaiters.delete(sessionId)
        resolve()
      }
    })
  }

  async acquireWithGrace(sessionId: string, graceMs = SESSION_BUSY_GRACE_MS): Promise<boolean> {
    if (this.tryAcquire(sessionId)) return true
    if (graceMs <= 0) return false
    const deadline = Date.now() + graceMs
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, SESSION_BUSY_POLL_MS))
      if (this.tryAcquire(sessionId)) return true
    }
    return false
  }

  async acquireLeaseWithGrace(
    sessionId: string,
    graceMs = SESSION_BUSY_GRACE_MS,
  ): Promise<SessionBusyLease | undefined> {
    if (this.tryAcquire(sessionId)) return this.createLease(sessionId)
    if (graceMs <= 0) return undefined
    const deadline = Date.now() + graceMs
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, SESSION_BUSY_POLL_MS))
      if (this.tryAcquire(sessionId)) return this.createLease(sessionId)
    }
    return undefined
  }

  private createLease(sessionId: string): SessionBusyLease {
    let owned = true
    return {
      release: () => {
        if (!owned) return
        owned = false
        this.release(sessionId)
      },
    }
  }
}

export interface SessionBusyLease {
  release(): void
}

export interface SessionBusyLeaseLifecycle {
  attachLease(lease: SessionBusyLease | undefined): boolean
  releasePreflight(): void
  transferToRun(): boolean
  releaseRun(): void
}

export function createSessionBusyLeaseLifecycle(): SessionBusyLeaseLifecycle {
  let phase: 'waiting' | 'preflight' | 'run' | 'released' = 'waiting'
  let lease: SessionBusyLease | undefined

  const releaseLease = () => {
    lease?.release()
    lease = undefined
  }

  return {
    attachLease: (nextLease) => {
      if (phase === 'released') {
        nextLease?.release()
        return false
      }
      if (phase !== 'waiting') return false
      lease = nextLease
      phase = 'preflight'
      return true
    },
    releasePreflight: () => {
      if (phase === 'run' || phase === 'released') return
      phase = 'released'
      releaseLease()
    },
    transferToRun: () => {
      if (phase !== 'preflight') return false
      phase = 'run'
      return true
    },
    releaseRun: () => {
      if (phase === 'released') return
      phase = 'released'
      releaseLease()
    },
  }
}

export const SESSION_BUSY_GRACE_MS = resolveSessionBusyGraceMs()
const SESSION_BUSY_POLL_MS = 50

function resolveSessionBusyGraceMs(): number {
  const raw = process.env.SEPILOTD_SESSION_BUSY_GRACE_MS
  if (raw !== undefined && raw !== '') {
    const parsed = Number(raw)
    if (Number.isFinite(parsed) && parsed >= 0) return Math.min(parsed, 60_000)
  }
  return 5_000
}
