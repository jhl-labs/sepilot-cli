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
  reason?: string
}

export interface IHookHandler {
  id: string
  priority: number
  handle(payload: HookPayload, signal?: AbortSignal): Promise<HookResult>
}
