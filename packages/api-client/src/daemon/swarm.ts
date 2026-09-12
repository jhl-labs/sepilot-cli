import type {
  SwarmRun,
  SwarmRunStatus,
  SwarmEvent,
  SwarmAgentName,
  SwarmAgentRecoveryHint,
  SwarmAgentRecoveryStep,
  SwarmAgentStartupState,
} from '@sepilotd/core'

export type { SwarmAgentName, SwarmRun, SwarmRunStatus, SwarmEvent } from '@sepilotd/core'

export interface CreateRunInput {
  goal: string
  cwd?: string
  worktree?: string
  autonomy?: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
  autoApproveAgents?: boolean
  warmPool?: SwarmAgentName[]
  /**
   * Spawn the warm pool but do not start the supervisor loop. The run stays
   * alive until cancelled and can be driven directly through attach/sendKeys.
   */
  noSupervisor?: boolean
}

export interface DriveAgentInput {
  prompt: string
  followups?: string[]
  continue_prompt?: string
  max_turns?: number
  stop_patterns?: string[]
  timeout_sec?: number
  stable_sec?: number
  min_wait_ms?: number
  poll_ms?: number
  timeout_retries?: number
}

export type DriveAgentRequiredActionType =
  | 'trust_prompt'
  | 'tool_permission'

export interface DriveAgentRequiredAction {
  type: DriveAgentRequiredActionType
  lifecycleState: Extract<SwarmAgentStartupState, 'trust_required' | 'tool_permission_required'>
  action: Extract<SwarmAgentRecoveryStep, 'review_trust_prompt' | 'review_tool_permission'>
  reason: string
  suggested_input: string
  output_preview?: string
  snapshot: string
  recovery_hint?: SwarmAgentRecoveryHint
}

export interface DriveAgentTurn {
  turn: number
  prompt: string
  status: 'idle' | 'done' | 'blocked' | 'timeout' | 'dead'
  reason: string
  output: string
  snapshot: string
  durationMs: number
  required_action?: DriveAgentRequiredAction
}

export interface DriveAgentResult {
  runId: string
  handle: string
  status: 'done' | 'max_turns' | 'blocked' | 'timeout' | 'dead'
  turns: DriveAgentTurn[]
  lastOutput: string
  nextPrompt?: string
  required_action?: DriveAgentRequiredAction
}

export interface SwarmClient {
  createRun(input: CreateRunInput): Promise<{ runId: string }>
  list(status?: SwarmRunStatus): Promise<SwarmRun[]>
  /**
   * Recent runs (active and completed) reconstructed from the jsonl event
   * log. Newest first. `limit` defaults to 50, max 200.
   */
  history(limit?: number): Promise<SwarmRun[]>
  get(runId: string): Promise<SwarmRun>
  cancel(runId: string): Promise<{ status: 'cancelled' }>
  events(runId: string, since?: number): Promise<SwarmEvent[]>
  streamEvents(
    runId: string,
    opts?: { since?: number; signal?: AbortSignal },
  ): AsyncIterable<SwarmEvent>
  sendKeys(
    runId: string,
    handle: string,
    payload: {
      keys?: string
      keyName?: string
      enter?: boolean
      resize?: { cols: number; rows: number }
    },
  ): Promise<void>
  attachAgent(
    runId: string,
    handle: string,
    options?: { renew?: boolean },
  ): Promise<{ ok: true; lease?: { owner: string; expiresAt: number } }>
  detachAgent(
    runId: string,
    handle: string,
  ): Promise<void>
  captureAgent(
    runId: string,
    handle: string,
    options?: { lines?: number; raw?: boolean },
  ): Promise<{ text: string }>
  driveAgent(
    runId: string,
    handle: string,
    input: DriveAgentInput,
  ): Promise<DriveAgentResult>
}

export interface SwarmClientOpts {
  baseUrl: string
  token: string | null
  fetch?: typeof fetch
}

export function createSwarmClient(opts: SwarmClientOpts): SwarmClient {
  const f = opts.fetch ?? fetch
  const authHeaders = (): HeadersInit =>
    opts.token ? { authorization: `Bearer ${opts.token}` } : {}
  const jsonHeaders = (): HeadersInit => ({
    'content-type': 'application/json',
    ...authHeaders(),
  })
  const url = (p: string): string => `${opts.baseUrl.replace(/\/$/, '')}${p}`

  return {
    async createRun(input) {
      const res = await f(url('/api/v1/swarm/runs'), {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify(input),
      })
      if (!res.ok) throw new Error(`createRun failed: ${res.status} ${await res.text()}`)
      return res.json() as Promise<{ runId: string }>
    },
    async list(status) {
      const res = await f(
        url(`/api/v1/swarm/runs${status ? `?status=${encodeURIComponent(status)}` : ''}`),
        { headers: authHeaders() },
      )
      if (!res.ok) throw new Error(`list failed: ${res.status}`)
      return res.json() as Promise<SwarmRun[]>
    },
    async history(limit) {
      const q = limit ? `?limit=${limit}` : ''
      const res = await f(url(`/api/v1/swarm/runs/history${q}`), {
        headers: authHeaders(),
      })
      if (!res.ok) throw new Error(`history failed: ${res.status}`)
      return res.json() as Promise<SwarmRun[]>
    },
    async get(runId) {
      const res = await f(
        url(`/api/v1/swarm/runs/${encodeURIComponent(runId)}`),
        { headers: authHeaders() },
      )
      if (!res.ok) throw new Error(`get failed: ${res.status}`)
      return res.json() as Promise<SwarmRun>
    },
    async cancel(runId) {
      const res = await f(
        url(`/api/v1/swarm/runs/${encodeURIComponent(runId)}`),
        { method: 'DELETE', headers: authHeaders() },
      )
      if (!res.ok) throw new Error(`cancel failed: ${res.status}`)
      return res.json() as Promise<{ status: 'cancelled' }>
    },
    async events(runId, since) {
      const q = since ? `?since=${since}` : ''
      const res = await f(
        url(`/api/v1/swarm/runs/${encodeURIComponent(runId)}/events${q}`),
        { headers: authHeaders() },
      )
      if (!res.ok) throw new Error(`events failed: ${res.status}`)
      return res.json() as Promise<SwarmEvent[]>
    },
    async *streamEvents(runId, sopts) {
      const q = new URLSearchParams({ stream: '1', follow: '1' })
      if (sopts?.since) q.set('since', String(sopts.since))
      const res = await f(
        url(`/api/v1/swarm/runs/${encodeURIComponent(runId)}/events?${q}`),
        {
          headers: { ...authHeaders(), accept: 'text/event-stream' },
          signal: sopts?.signal,
        },
      )
      if (!res.ok || !res.body) throw new Error(`streamEvents failed: ${res.status}`)
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          const dataLine = frame.split('\n').find((l) => l.startsWith('data: '))
          if (!dataLine) continue
          try {
            yield JSON.parse(dataLine.slice(6)) as SwarmEvent
          } catch {
            /* ignore parse */
          }
        }
      }
    },
    async sendKeys(runId, handle, payload) {
      const res = await f(
        url(
          `/api/v1/swarm/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(handle)}/keys`,
        ),
        {
          method: 'POST',
          headers: jsonHeaders(),
          body: JSON.stringify(payload),
        },
      )
      if (!res.ok) throw new Error(`sendKeys failed: ${res.status}`)
    },
    async attachAgent(runId, handle, options) {
      const res = await f(
        url(
          `/api/v1/swarm/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(handle)}/attach`,
        ),
        {
          method: 'POST',
          headers: jsonHeaders(),
          body: JSON.stringify(options ?? {}),
        },
      )
      if (!res.ok) throw new Error(`attachAgent failed: ${res.status} ${await res.text()}`)
      return res.json() as Promise<{ ok: true; lease?: { owner: string; expiresAt: number } }>
    },
    async detachAgent(runId, handle) {
      const res = await f(
        url(
          `/api/v1/swarm/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(handle)}/detach`,
        ),
        {
          method: 'POST',
          headers: jsonHeaders(),
          body: JSON.stringify({}),
        },
      )
      if (!res.ok) throw new Error(`detachAgent failed: ${res.status} ${await res.text()}`)
    },
    async captureAgent(runId, handle, options) {
      const q = new URLSearchParams()
      if (typeof options?.lines === 'number') q.set('lines', String(options.lines))
      if (options?.raw) q.set('raw', '1')
      const query = q.toString()
      const suffix = query ? `?${query}` : ''
      const res = await f(
        url(
          `/api/v1/swarm/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(handle)}/capture${suffix}`,
        ),
        { headers: authHeaders() },
      )
      if (!res.ok) throw new Error(`captureAgent failed: ${res.status}`)
      return res.json() as Promise<{ text: string }>
    },
    async driveAgent(runId, handle, input) {
      const res = await f(
        url(
          `/api/v1/swarm/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(handle)}/drive`,
        ),
        {
          method: 'POST',
          headers: jsonHeaders(),
          body: JSON.stringify(input),
        },
      )
      if (!res.ok) throw new Error(`driveAgent failed: ${res.status} ${await res.text()}`)
      return res.json() as Promise<DriveAgentResult>
    },
  }
}
