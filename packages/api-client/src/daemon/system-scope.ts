import { ApiHttpClient } from '../http.js'
import type { MemoryScope } from '../http.js'

/**
 * Boot-time helper that resolves the daemon's per-install memory scope id
 * over HTTP. Lives in api-client so surface packages (cli/desktop main+
 * preload/web) don't have to call `fetch` directly — that would trip the
 * `architecture/surface-no-raw-fetch` rule.
 */
export async function fetchSystemScope(
  baseUrl: string,
  token: string | null,
): Promise<MemoryScope | null> {
  // Scope discovery sits on the critical path for chat, but it must never
  // stall unrelated tray/system work for the normal 30-second HTTP timeout.
  const client = new ApiHttpClient({ baseUrl, token, timeoutMs: 2_000 })
  try {
    const body = await client.get<{ data?: { userId?: unknown } } | null>(
      '/api/v1/system/scope',
      { retry: { maxAttempts: 1 } },
    )
    const userId = body?.data?.userId
    if (typeof userId !== 'string' || userId.trim().length === 0) return null
    return { userId: userId.trim() }
  } catch {
    return null
  }
}
