export interface ModelCandidateIdentity {
  providerId: string
  model: string
}

export interface ModelFallbackStateOptions {
  failureThreshold?: number
  cooldownMs?: number
  clock?: () => number
}

export interface ModelFallbackRecord {
  providerId: string
  model: string
  consecutiveFailures: number
  openedUntil?: number
  lastFailure?: string
  lastFailureAt?: string
  lastSuccessAt?: string
}

export const DEFAULT_MODEL_FALLBACK_FAILURE_THRESHOLD = 1
export const DEFAULT_MODEL_FALLBACK_COOLDOWN_MS = 60_000

export class ModelFallbackState {
  private readonly failureThreshold: number
  private readonly cooldownMs: number
  private readonly clock: () => number
  private readonly records = new Map<string, ModelFallbackRecord>()

  constructor(options: ModelFallbackStateOptions = {}) {
    this.failureThreshold = Math.max(
      1,
      options.failureThreshold ?? DEFAULT_MODEL_FALLBACK_FAILURE_THRESHOLD,
    )
    this.cooldownMs = Math.max(0, options.cooldownMs ?? DEFAULT_MODEL_FALLBACK_COOLDOWN_MS)
    this.clock = options.clock ?? Date.now
  }

  isAvailable(identity: ModelCandidateIdentity): boolean {
    const record = this.records.get(this.key(identity))
    if (!record?.openedUntil) return true
    return record.openedUntil <= this.clock()
  }

  recordFailure(identity: ModelCandidateIdentity, error: unknown): ModelFallbackRecord {
    const now = this.clock()
    const record = this.records.get(this.key(identity)) ?? {
      providerId: identity.providerId,
      model: identity.model,
      consecutiveFailures: 0,
    }
    record.consecutiveFailures += 1
    record.lastFailure = this.errorMessage(error)
    record.lastFailureAt = new Date(now).toISOString()
    if (record.consecutiveFailures >= this.failureThreshold) {
      record.openedUntil = now + this.cooldownMs
    }
    this.records.set(this.key(identity), record)
    return { ...record }
  }

  recordSuccess(identity: ModelCandidateIdentity): ModelFallbackRecord {
    const now = this.clock()
    const record = this.records.get(this.key(identity)) ?? {
      providerId: identity.providerId,
      model: identity.model,
      consecutiveFailures: 0,
    }
    record.consecutiveFailures = 0
    record.openedUntil = undefined
    record.lastSuccessAt = new Date(now).toISOString()
    this.records.set(this.key(identity), record)
    return { ...record }
  }

  snapshot(): ModelFallbackRecord[] {
    return Array.from(this.records.values())
      .map((record) => ({ ...record }))
      .sort((a, b) => `${a.providerId}:${a.model}`.localeCompare(`${b.providerId}:${b.model}`))
  }

  private key(identity: ModelCandidateIdentity): string {
    return `${identity.providerId}:${identity.model}`
  }

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error)
  }
}

export function isProviderFallbackError(error: { code?: string } | null | undefined): boolean {
  return (
    error?.code === 'PROVIDER_ERROR'
    || error?.code === 'SERVICE_UNAVAILABLE'
    || error?.code === 'TIMEOUT'
    || error?.code === 'RATE_LIMITED'
  )
}
