import type { IAuditLogger } from '@sepilotd/core'
import type { SepilotdConfig } from '../../config/schema.js'
import { randomUUID } from 'node:crypto'
import { createLogger } from '../../logger.js'

const log = createLogger('config-mutation-service')

export const CONFIG_TRANSACTION_COMMITTED_EVENT = 'config.transaction.committed'
export const CONFIG_TRANSACTION_ROLLED_BACK_EVENT = 'config.transaction.rolled-back'

export interface ConfigMutationServiceOptions {
  auditLogger?: IAuditLogger | null
  deviceName?: string
  /**
   * Returns the current in-memory configRevision so the service can
   * emit before/after markers in its audit event.
   */
  readRevision?: () => number | undefined
  /**
   * Called by apply() before the mutation runs. Must replace
   * runtime.config with a mutable structuredClone so that the
   * callback can mutate it freely. Returns the clone.
   * Used when runtime.config is frozen between mutations.
   */
  prepareMutableDraft?: () => SepilotdConfig
  /**
   * Called by apply() after the mutation succeeds. Receives the
   * draft and must deep-freeze it, then assign it back to
   * runtime.config.
   */
  commitDraft?: (draft: SepilotdConfig) => void
  /**
   * Takes a deep snapshot of the current runtime.config before a
   * mutation runs. Used for rollback on failure.
   */
  snapshotConfig?: () => SepilotdConfig
  /**
   * Restores runtime.config from a snapshot and reconfigures extensions
   * (providers, webhooks, MCP) to match. Called when a mutation fails
   * so the runtime is not left in a half-updated state.
   */
  restoreConfig?: (snapshot: SepilotdConfig) => Promise<void>
}

/**
 * Serializes config-mutation work so that concurrent requests do
 * not interleave read-modify-write steps against runtime.config or
 * config.yaml.
 *
 * Each `apply()` enqueues the callback after the previous one resolves
 * (or rejects), keeping the chain alive across failures so a bad
 * mutation does not wedge the service.
 *
 * If prepareMutableDraft/commitDraft are provided, the service
 * creates a mutable draft before the callback runs and freezes the
 * result after success — protecting frozen snapshots between
 * mutations.
 *
 * If snapshotConfig/restoreConfig are provided, the service takes a
 * snapshot before each mutation and restores it on failure — bringing
 * providers, webhooks, and MCP back in line with the pre-mutation
 * state.
 *
 * If constructed with an auditLogger + readRevision pair, each
 * committed (or rolled-back) transaction emits one audit event
 * carrying { description, transactionId, revisionBefore, revisionAfter }
 * so audit consumers can correlate per-domain events through a single
 * transaction marker.
 */
export class ConfigMutationService {
  private chain: Promise<void> = Promise.resolve()
  private activeMutations = 0
  private auditLogger: IAuditLogger | null
  private deviceName?: string
  private readRevision?: () => number | undefined
  private prepareMutableDraftFn?: () => SepilotdConfig
  private commitDraftFn?: (draft: SepilotdConfig) => void
  private snapshotConfigFn?: () => SepilotdConfig
  private restoreConfigFn?: (snapshot: SepilotdConfig) => Promise<void>

  constructor(options: ConfigMutationServiceOptions = {}) {
    this.auditLogger = options.auditLogger ?? null
    this.deviceName = options.deviceName
    this.readRevision = options.readRevision
    this.prepareMutableDraftFn = options.prepareMutableDraft
    this.commitDraftFn = options.commitDraft
    this.snapshotConfigFn = options.snapshotConfig
    this.restoreConfigFn = options.restoreConfig
  }

  configure(options: ConfigMutationServiceOptions): void {
    if (options.auditLogger !== undefined) {
      this.auditLogger = options.auditLogger
    }
    if (options.deviceName !== undefined) {
      this.deviceName = options.deviceName
    }
    if (options.readRevision !== undefined) {
      this.readRevision = options.readRevision
    }
    if (options.prepareMutableDraft !== undefined) {
      this.prepareMutableDraftFn = options.prepareMutableDraft
    }
    if (options.commitDraft !== undefined) {
      this.commitDraftFn = options.commitDraft
    }
    if (options.snapshotConfig !== undefined) {
      this.snapshotConfigFn = options.snapshotConfig
    }
    if (options.restoreConfig !== undefined) {
      this.restoreConfigFn = options.restoreConfig
    }
  }

  async apply<T>(description: string, fn: () => Promise<T>): Promise<T> {
    const transactionId = randomUUID()
    // NOTE: the rollback snapshot is captured inside runOne at execution time,
    // not here at enqueue time. Capturing at enqueue would let a later failing
    // mutation restore a stale pre-mutation state and silently revert an
    // earlier successful mutation that committed while this one was queued.
    const run = this.chain.then(() => this.runOne(description, transactionId, fn), () => this.runOne(description, transactionId, fn))
    this.chain = run.then(
      () => undefined,
      () => undefined,
    )
    return run
  }

  isApplying(): boolean {
    return this.activeMutations > 0
  }

  private async runOne<T>(description: string, transactionId: string, fn: () => Promise<T>): Promise<T> {
    this.activeMutations += 1
    try {
      const revisionBefore = this.readRevision?.()

      // Capture the rollback snapshot now, at execution start, after any
      // previously queued mutation has already committed. This reflects the
      // true pre-mutation state for THIS transaction.
      const snapshot = this.snapshotConfigFn?.()

      // If runtime.config is frozen between mutations, unfreeze it by
      // replacing with a mutable draft before the callback runs.
      const draft = this.prepareMutableDraftFn?.()

      try {
        const result = await fn()

        // Freeze the draft and commit it back to runtime.config.
        if (draft && this.commitDraftFn) {
          this.commitDraftFn(draft)
        }

        const revisionAfter = this.readRevision?.()
        await this.emitAudit(CONFIG_TRANSACTION_COMMITTED_EVENT, description, transactionId, revisionBefore, revisionAfter)
        return result
      } catch (error) {
        const revisionAfter = this.readRevision?.()
        log.warn('Config mutation failed', {
          description,
          transactionId,
          error: error instanceof Error ? error.message : String(error),
        })

        // Roll back runtime.config to the pre-mutation snapshot and
        // reconfigure extensions so the runtime is not left in a
        // half-updated state.
        if (snapshot && this.restoreConfigFn) {
          try {
            await this.restoreConfigFn(snapshot)
          } catch (rollbackErr) {
            log.warn('Rollback also failed', {
              description,
              transactionId,
              error: rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr),
            })
          }
        }

        await this.emitAudit(
          CONFIG_TRANSACTION_ROLLED_BACK_EVENT,
          description,
          transactionId,
          revisionBefore,
          revisionAfter,
          error instanceof Error ? error.message : String(error),
        )
        throw error
      }
    } finally {
      this.activeMutations -= 1
    }
  }

  private async emitAudit(
    event: string,
    description: string,
    transactionId: string,
    revisionBefore: number | undefined,
    revisionAfter: number | undefined,
    errorMessage?: string,
  ): Promise<void> {
    if (!this.auditLogger) return
    try {
      await this.auditLogger.log({
        timestamp: new Date().toISOString(),
        event,
        device: this.deviceName ?? 'unknown',
        description,
        transactionId,
        ...(revisionBefore !== undefined ? { revisionBefore } : {}),
        ...(revisionAfter !== undefined ? { revisionAfter } : {}),
        ...(errorMessage ? { error: errorMessage } : {}),
      })
    } catch (auditError) {
      log.warn('Failed to emit config transaction audit event', {
        description,
        transactionId,
        error:
          auditError instanceof Error ? auditError.message : String(auditError),
      })
    }
  }
}
