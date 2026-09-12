export interface SchedulerMisfirePolicy {
  version: 1
  policy: 'skip' | 'run_once'
  maxLatenessMs: number
}

export function normalizeSchedulerMisfirePolicy(value: unknown): SchedulerMisfirePolicy {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('misfire must be an object')
  const raw = value as Record<string, unknown>
  if (raw.version !== 1 || (raw.policy !== 'skip' && raw.policy !== 'run_once')) {
    throw new Error('Unsupported scheduler misfire policy')
  }
  if (!Number.isSafeInteger(raw.maxLatenessMs) || Number(raw.maxLatenessMs) < 60_000
    || Number(raw.maxLatenessMs) > 30 * 86400_000) {
    throw new Error('maxLatenessMs must be between one minute and 30 days')
  }
  return { version: 1, policy: raw.policy, maxLatenessMs: Number(raw.maxLatenessMs) }
}

/** No new policy for legacy jobs; their engine-level grace remains unchanged. */
export function schedulerMisfirePolicyFromMetadata(
  metadata: Record<string, unknown> | null | undefined,
): SchedulerMisfirePolicy | null {
  return metadata?.schedulerMisfire === undefined ? null : normalizeSchedulerMisfirePolicy(metadata.schedulerMisfire)
}

export function schedulerMissedEvidence(
  job: { status: string; metadata?: Record<string, unknown> | null; lastError?: string | null },
): boolean {
  if (job.status !== 'cancelled') return false
  const evidence = job.metadata?.schedulerMissed
  if (evidence && typeof evidence === 'object' && !Array.isArray(evidence)) {
    const value = evidence as Record<string, unknown>
    if (value.version === 1 && value.reason === 'late' && Number.isFinite(value.scheduledAt)
      && Number.isFinite(value.detectedAt)) return true
  }
  // Exact legacy engine diagnostic, not natural-language intent routing.
  return /^missed by \d+ms$/.test(job.lastError ?? '')
}
