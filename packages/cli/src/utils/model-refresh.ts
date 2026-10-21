export const MODEL_REFRESH_TTL_MS = 5 * 60 * 1000

export interface ModelRefreshClient<T> {
  refreshProviderModels: (providerId: string) => Promise<unknown>
  providers: () => Promise<T>
}

export async function refreshProviderModelLists<T>(
  client: ModelRefreshClient<T>,
  providerIds: string[],
): Promise<T> {
  await Promise.allSettled(providerIds.map((id) => client.refreshProviderModels(id)))
  return client.providers()
}

export interface PickerItemIdentity {
  kind: 'model' | 'action'
  providerId?: string
  modelId?: string
  action?: string
  presetType?: string
}

export function pickerItemKey(item: PickerItemIdentity): string {
  return item.kind === 'model'
    ? `model:${item.providerId ?? ''}/${item.modelId ?? ''}`
    : `action:${item.action ?? ''}:${item.providerId ?? item.presetType ?? ''}`
}

export function remapPickerIndex(
  previousItems: readonly PickerItemIdentity[],
  nextItems: readonly PickerItemIdentity[],
  currentIndex: number,
): number {
  const previous = previousItems[currentIndex]
  if (previous) {
    const key = pickerItemKey(previous)
    const nextIndex = nextItems.findIndex((item) => pickerItemKey(item) === key)
    if (nextIndex >= 0) return nextIndex
  }
  return Math.max(0, Math.min(currentIndex, nextItems.length))
}

export interface ModelRefreshState {
  lastRefreshedAt: number | null
  inFlight: boolean
}

export function createModelRefreshState(): ModelRefreshState {
  return { lastRefreshedAt: null, inFlight: false }
}

export function shouldRefreshModels(
  state: ModelRefreshState,
  now: number = Date.now(),
  ttlMs: number = MODEL_REFRESH_TTL_MS,
): boolean {
  if (state.inFlight) return false
  if (state.lastRefreshedAt === null) return true
  return now - state.lastRefreshedAt >= ttlMs
}

export async function refreshModelsInBackground<T>(
  state: ModelRefreshState,
  fetchProviders: () => Promise<T>,
  onSuccess: (providers: T) => void,
  now: number = Date.now(),
  ttlMs: number = MODEL_REFRESH_TTL_MS,
): Promise<void> {
  if (!shouldRefreshModels(state, now, ttlMs)) return

  state.inFlight = true
  try {
    const providers = await fetchProviders()
    state.lastRefreshedAt = now
    onSuccess(providers)
  } catch {
  } finally {
    state.inFlight = false
  }
}
