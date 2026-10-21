import { fetchSystemScope, type MemoryScope } from '@sepilotd/api-client'

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

export function resetCliDaemonScopeCache(): void {
  scopeCache.clear()
}
