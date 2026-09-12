import { fetchSystemScope, type MemoryScope } from '@sepilotd/api-client'

// Cache the /system/scope round-trip per (baseUrl, token) — the cli
// constructs many short-lived DaemonClients in the same process and
// every outgoing request needs the userId header, so resolve it once.
const scopeCache = new Map<string, Promise<MemoryScope | null>>()
export function memoryScopeFor(baseUrl: string, token: string | null) {
  return () => {
    const key = baseUrl + '::' + (token ?? '')
    let pending = scopeCache.get(key)
    if (!pending) {
      pending = fetchSystemScope(baseUrl, token)
      scopeCache.set(key, pending)
    }
    return pending
  }
}

/** Test-only: drop the cached scope promise so the next DaemonClient
 *  rebuilds it. Production code should rely on the natural lifetime. */
export function resetCliDaemonScopeCache(): void {
  scopeCache.clear()
}

