// Background model-list rediscovery for the TUI model picker.
//
// Opening the picker used to trigger an unconditional `loadProviders()` call
// every single time, even if the picker had just been refreshed moments ago.
// This module extracts the gating logic (TTL cache + in-flight guard) into a
// pure, easily testable unit so the picker can refresh provider/model data in
// the background without spamming the daemon or flashing a loading state on
// every open. Failures are swallowed: a failed background refresh leaves the
// existing provider list untouched.

export const MODEL_REFRESH_TTL_MS = 5 * 60 * 1000

/**
 * Real per-provider rediscovery: ask the daemon to re-run model discovery
 * server-side (with its stored, unredacted credentials) for every configured
 * provider via POST /config/providers/:id/refresh-models, then reload the
 * provider list so the refreshed models show up. Individual provider refresh
 * failures are ignored — the subsequent `providers()` call simply returns
 * whatever the daemon currently knows.
 */
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

/**
 * Keep the picker selection stable when a background refresh replaces the
 * item list: re-locate the currently selected item by identity (provider or
 * provider/model key) in the new list, and clamp when it disappeared. The
 * picker has one virtual trailing "Add provider..." row, so the max valid
 * index is `nextItems.length` (not `length - 1`).
 */
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
  // Selected item vanished (or the virtual add-provider row was selected):
  // clamp to the new bounds, allowing the trailing virtual row.
  return Math.max(0, Math.min(currentIndex, nextItems.length))
}

export interface ModelRefreshState {
  lastRefreshedAt: number | null
  inFlight: boolean
}

export function createModelRefreshState(): ModelRefreshState {
  return { lastRefreshedAt: null, inFlight: false }
}

/**
 * Decide whether a background refresh should be dispatched right now.
 * Returns false while a refresh is already in flight, or while the last
 * successful refresh is still within the TTL window.
 */
export function shouldRefreshModels(
  state: ModelRefreshState,
  now: number = Date.now(),
  ttlMs: number = MODEL_REFRESH_TTL_MS,
): boolean {
  if (state.inFlight) return false
  if (state.lastRefreshedAt === null) return true
  return now - state.lastRefreshedAt >= ttlMs
}

/**
 * Run a background provider/model refresh, gated by `shouldRefreshModels`.
 * On success, updates `state.lastRefreshedAt` and calls `onSuccess` with the
 * fresh provider list. On failure, silently leaves `state` and the existing
 * list untouched (no error surfaced to the picker).
 */
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
    // Silent failure: keep the existing provider/model list as-is.
  } finally {
    state.inFlight = false
  }
}
