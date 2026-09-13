// Only events with a real emit site belong here. Advertising an event that is
// never fired makes registering a hook for it a silent no-op — a false sense of
// a security/moderation gate. `pre:agent:think`/`post:agent:think` and the
// `session:start`/`session:end` lifecycle events were removed because nothing
// ever triggered them; re-add them here only together with an emit site.
export type HookEvent =
  | 'pre:agent:run' | 'post:agent:run'
  | 'pre:tool:execute' | 'post:tool:execute'
  | 'post:process:start' | 'post:process:exit'
  | 'pre:llm:call' | 'post:llm:call'
  | 'pre:channel:msg' | 'post:channel:msg'
  | 'post:file:edit'
  | 'pre:user:prompt'
  | 'pre:context:compact'
  | 'post:context:compact'
  | 'post:subagent:start' | 'post:subagent:stop'

export interface HookPayload {
  event: HookEvent
  data: Record<string, unknown>
}

export interface HookResult {
  action: 'continue' | 'skip' | 'abort'
  modifiedPayload?: HookPayload
  /** Human-readable reason, surfaced to the agent/user when a gating hook aborts. */
  reason?: string
}

export interface IHookHandler {
  id: string
  priority: number
  /** Cooperatively cancel external work when the registry's budget expires. */
  handle(payload: HookPayload, signal?: AbortSignal): Promise<HookResult>
}
