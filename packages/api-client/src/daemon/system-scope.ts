import { ApiHttpClient } from '../http.js'
import type { MemoryScope } from '../http.js'

export async function fetchSystemScope(
  baseUrl: string,
  token: string | null,
): Promise<MemoryScope | null> {
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
