// DocClient — 글쓰기 모드 (canvas) doc session HTTP client.
// SSE event 구독은 browser EventSource로 직접 (renderer에서). 여기서는 REST만.

import type { DocChange, DocDiffPreview, DocSession } from '@sepilotd/core'
export type {
  DocChange,
  DocDiffPreview,
  DocOutlineEntry,
  DocSession,
} from '@sepilotd/core'

export interface DocClientOptions {
  baseUrl: string
  token: string | null
}

// Bare auth — used for GET/DELETE/POST calls that don't carry a JSON body.
// Fastify rejects requests with `content-type: application/json` and an empty
// body, so empty-body calls (close / undo / reload / accept / cancel / list)
// must NOT advertise a JSON content type.
function authHeaders(token: string | null): Record<string, string> {
  return token ? { authorization: `Bearer ${token}` } : {}
}

// JSON-body auth — for POST calls that do send a stringified body.
function jsonAuthHeaders(token: string | null): Record<string, string> {
  return { 'content-type': 'application/json', ...authHeaders(token) }
}

async function jsonOrThrow<T>(r: Response, label: string): Promise<T> {
  if (!r.ok) {
    let body: unknown = null
    try { body = await r.json() } catch { /* noop */ }
    throw new Error(`${label} failed (${r.status}): ${JSON.stringify(body)}`)
  }
  const raw = (await r.json()) as { data: T }
  return raw.data
}

export function createDocClient(opts: DocClientOptions) {
  const base = `${opts.baseUrl.replace(/\/$/, '')}/api/v1`

  return {
    async open(input: { path?: string | null; initialContent?: string }): Promise<DocSession> {
      const r = await fetch(`${base}/doc/open`, {
        method: 'POST',
        headers: jsonAuthHeaders(opts.token),
        body: JSON.stringify(input),
      })
      return jsonOrThrow<DocSession>(r, 'doc.open')
    },

    async get(id: string): Promise<DocSession> {
      const r = await fetch(`${base}/doc/${encodeURIComponent(id)}`, {
        headers: authHeaders(opts.token),
      })
      return jsonOrThrow<DocSession>(r, 'doc.get')
    },

    async userEdit(id: string, changes: DocChange[], expectedVersion?: number): Promise<{ version: number }> {
      const r = await fetch(`${base}/doc/${encodeURIComponent(id)}/user_edit`, {
        method: 'POST',
        headers: jsonAuthHeaders(opts.token),
        body: JSON.stringify({ changes, expectedVersion }),
      })
      return jsonOrThrow<{ version: number }>(r, 'doc.user_edit')
    },

    async save(id: string, path?: string): Promise<{ path: string; mtimeMs: number }> {
      const r = await fetch(`${base}/doc/${encodeURIComponent(id)}/save`, {
        method: 'POST',
        headers: jsonAuthHeaders(opts.token),
        body: JSON.stringify({ path }),
      })
      return jsonOrThrow<{ path: string; mtimeMs: number }>(r, 'doc.save')
    },

    async undo(id: string): Promise<{ version: number } | null> {
      const r = await fetch(`${base}/doc/${encodeURIComponent(id)}/undo`, {
        method: 'POST',
        headers: authHeaders(opts.token),
      })
      if (r.status === 404) return null
      return jsonOrThrow<{ version: number }>(r, 'doc.undo')
    },

    async reload(id: string): Promise<{ version: number }> {
      const r = await fetch(`${base}/doc/${encodeURIComponent(id)}/reload`, {
        method: 'POST',
        headers: authHeaders(opts.token),
      })
      return jsonOrThrow<{ version: number }>(r, 'doc.reload')
    },

    async close(id: string): Promise<{ ok: boolean }> {
      const r = await fetch(`${base}/doc/${encodeURIComponent(id)}`, {
        method: 'DELETE',
        headers: authHeaders(opts.token),
      })
      return jsonOrThrow<{ ok: boolean }>(r, 'doc.close')
    },

    async acceptPreview(id: string, previewId: string): Promise<{ version: number }> {
      const r = await fetch(
        `${base}/doc/${encodeURIComponent(id)}/diff/${encodeURIComponent(previewId)}/accept`,
        { method: 'POST', headers: authHeaders(opts.token) },
      )
      return jsonOrThrow<{ version: number }>(r, 'doc.diff.accept')
    },

    async cancelPreview(id: string, previewId: string): Promise<{ ok: boolean }> {
      const r = await fetch(
        `${base}/doc/${encodeURIComponent(id)}/diff/${encodeURIComponent(previewId)}/cancel`,
        { method: 'POST', headers: authHeaders(opts.token) },
      )
      return jsonOrThrow<{ ok: boolean }>(r, 'doc.diff.cancel')
    },

    async listPreviews(id: string): Promise<DocDiffPreview[]> {
      const r = await fetch(`${base}/doc/${encodeURIComponent(id)}/previews`, {
        headers: authHeaders(opts.token),
      })
      return jsonOrThrow<DocDiffPreview[]>(r, 'doc.previews')
    },

    /** SSE event stream URL — renderer가 EventSource로 직접 연결. token은 query param. */
    eventsUrl(): string {
      const params = new URLSearchParams()
      if (opts.token) params.set('token', opts.token)
      return `${base}/doc/events${params.size ? `?${params}` : ''}`
    },
  }
}

export type DocClient = ReturnType<typeof createDocClient>
