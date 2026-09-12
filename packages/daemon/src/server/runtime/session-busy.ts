/**
 * Runtime-shared, in-process registry that guarantees at most one active turn
 * per session id across every entry point (WebSocket chat, HTTP/SSE chat
 * stream, stream resume). It exists because busy-tracking used to live in a
 * WebSocket-local `Set` with a check-then-add gap: the `has()` check and the
 * `add()` were separated by `await`s, so two turns for the same session could
 * both pass the check, and the HTTP/SSE path had no guard at all. Interleaved
 * turns write to the same session JSONL concurrently and can bind a tool_result
 * to the wrong tool_call.
 *
 * `tryAcquire` performs the has+add atomically (no `await` in between — the
 * event loop is single-threaded, so a synchronous check-and-set cannot be
 * interleaved), which closes the TOCTOU window. Callers must `release` in a
 * `finally`.
 */
export class SessionBusyRegistry {
  private readonly busy = new Set<string>()
  private readonly idleWaiters = new Map<string, Set<() => void>>()

  /**
   * Atomically claim the session. Returns `true` if the caller now owns the
   * turn, or `false` if another turn already holds it. Because there is no
   * `await` between the membership check and the insert, two concurrent turns
   * can never both observe the session as free.
   */
  tryAcquire(sessionId: string): boolean {
    if (this.busy.has(sessionId)) return false
    this.busy.add(sessionId)
    return true
  }

  /**
   * Atomically claim a session and return an idempotent release handle.
   *
   * Request handlers can have several cleanup paths (normal finally, socket
   * close, Fastify response finish). Returning a lease prevents an older
   * cleanup callback from releasing a newer turn that acquired the same
   * session after the original owner had already finished.
   */
  tryAcquireLease(sessionId: string): SessionBusyLease | undefined {
    if (!this.tryAcquire(sessionId)) return undefined
    return this.createLease(sessionId)
  }

  /** Release a previously acquired session so the next turn can proceed. */
  release(sessionId: string): void {
    if (!this.busy.delete(sessionId)) return
    const waiters = this.idleWaiters.get(sessionId)
    if (!waiters) return
    this.idleWaiters.delete(sessionId)
    for (const resolve of waiters) resolve()
  }

  /** True while a turn holds the session. */
  isBusy(sessionId: string): boolean {
    return this.busy.has(sessionId)
  }

  /**
   * Resolve once the current owner has completed all run cleanup and released
   * the session. This is an event-driven cancellation barrier: callers do not
   * need to guess a polling grace period before starting the next turn.
   */
  waitUntilIdle(sessionId: string): Promise<void> {
    if (!this.busy.has(sessionId)) return Promise.resolve()
    return new Promise((resolve) => {
      const waiters = this.idleWaiters.get(sessionId) ?? new Set<() => void>()
      waiters.add(resolve)
      this.idleWaiters.set(sessionId, waiters)
      // No await occurs between the first check and registration, but keep the
      // second check so future implementations that release synchronously from
      // registration hooks cannot strand a waiter.
      if (!this.busy.has(sessionId)) {
        waiters.delete(resolve)
        if (waiters.size === 0) this.idleWaiters.delete(sessionId)
        resolve()
      }
    })
  }

  /**
   * Claim the session, waiting up to `graceMs` for the current holder to
   * release it. Exists for the cancel→retype race: the client aborts a run
   * (Esc) and immediately sends the next turn, but the daemon-side agent
   * loop releases the busy slot asynchronously (the in-flight LLM call has
   * to abort first). Without the grace the very next turn gets a raw BUSY
   * error for a session that frees up milliseconds later.
   *
   * Each attempt is still the atomic `tryAcquire`; the grace only retries.
   * Returns `true` once the caller owns the turn, `false` if the session
   * stayed busy for the whole window (a genuinely concurrent turn).
   */
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

  /** Grace-period variant of {@link tryAcquireLease}. */
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

/**
 * Coordinates a session lease across request pre-flight and the main agent run.
 *
 * A client can disconnect while `acquireLeaseWithGrace()` is still waiting, so
 * request listeners must exist before the lease is available. Once the main
 * run starts, those same response `finish`/`close` callbacks must not release
 * the lease: the agent's run-level `finally` is then its sole owner.
 */
export interface SessionBusyLeaseLifecycle {
  /** Attach the result of lease acquisition. False means the request already ended. */
  attachLease(lease: SessionBusyLease | undefined): boolean
  /** Release a waiting/pre-flight lease, or remember an early disconnect. */
  releasePreflight(): void
  /** Atomically transfer cleanup responsibility to the run-level finally. */
  transferToRun(): boolean
  /** Release the run-owned lease. Idempotent. */
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
        // The socket closed while acquisition was pending. Never leave the
        // subsequently acquired lease orphaned or continue the cancelled run.
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

/**
 * How long a new turn waits for a cancelling run to release the session.
 * Override with SEPILOTD_SESSION_BUSY_GRACE_MS (0 disables the grace and
 * restores the old immediate-BUSY behaviour).
 */
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
