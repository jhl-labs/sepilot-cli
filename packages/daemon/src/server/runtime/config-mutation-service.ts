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
  readRevision?: () => number | undefined
  prepareMutableDraft?: () => SepilotdConfig
  commitDraft?: (draft: SepilotdConfig) => void
  snapshotConfig?: () => SepilotdConfig
  restoreConfig?: (snapshot: SepilotdConfig) => Promise<void>
}

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

      const snapshot = this.snapshotConfigFn?.()

      const draft = this.prepareMutableDraftFn?.()

      try {
        const result = await fn()

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
