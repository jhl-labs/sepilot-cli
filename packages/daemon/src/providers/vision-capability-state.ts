const DEFAULT_IMAGE_INPUT_REJECTED_TTL_MS = 6 * 60 * 60 * 1000

interface ImageInputRejectedRecord {
  rejectedAt: number
  reason?: string
}

const imageInputRejected = new Map<string, ImageInputRejectedRecord>()

function keyFor(providerId: string, model: string): string {
  return `${providerId}\u0000${model}`
}

function imageInputRejectedTtlMs(): number {
  const raw = Number(process.env.SEPILOTD_IMAGE_INPUT_REJECTED_TTL_MS)
  if (Number.isFinite(raw) && raw > 0) {
    return Math.floor(raw)
  }
  return DEFAULT_IMAGE_INPUT_REJECTED_TTL_MS
}

export function markProviderModelImageInputRejected(
  providerId: string,
  model: string,
  reason?: string,
): void {
  if (!providerId || !model) return
  imageInputRejected.set(keyFor(providerId, model), {
    rejectedAt: Date.now(),
    reason,
  })
}

export function isProviderModelImageInputRejected(
  providerId: string,
  model: string,
): boolean {
  const key = keyFor(providerId, model)
  const record = imageInputRejected.get(key)
  if (!record) return false
  if (Date.now() - record.rejectedAt > imageInputRejectedTtlMs()) {
    imageInputRejected.delete(key)
    return false
  }
  return true
}

export function clearProviderModelImageInputRejected(
  providerId: string,
  model: string,
): void {
  imageInputRejected.delete(keyFor(providerId, model))
}

export function resetVisionCapabilityStateForTests(): void {
  imageInputRejected.clear()
}
