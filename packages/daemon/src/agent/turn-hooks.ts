import type { IHookRegistry } from '@sepilotd/core'

export interface PreUserPromptContext {
  hookRegistry?: IHookRegistry
  sessionId: string
  actor?: string
}

export interface PreUserPromptOutcome {
  aborted: boolean
  reason?: string
  prompt: string
}

export async function applyPreUserPromptHook(
  prompt: string,
  ctx: PreUserPromptContext,
): Promise<PreUserPromptOutcome> {
  if (!ctx.hookRegistry) return { aborted: false, prompt }
  const result = await ctx.hookRegistry.trigger({
    event: 'pre:user:prompt',
    data: { sessionId: ctx.sessionId, prompt, actor: ctx.actor },
  })
  if (result.action === 'abort') {
    return { aborted: true, reason: result.reason, prompt }
  }
  const rewritten = (result.modifiedPayload?.data as { prompt?: unknown } | undefined)?.prompt
  return { aborted: false, prompt: typeof rewritten === 'string' ? rewritten : prompt }
}

export interface AgentRunHookData {
  runId: string
  sessionId: string
  mode?: string
  provider?: string
  model?: string
  input?: string
  backgroundJobId?: string
  detached?: boolean
  surface?: string
  resumed?: true
  roleBotId?: string
  attempt?: number
  fallbackPending?: boolean
  status?: 'success' | 'error' | 'aborted' | 'incomplete'
  durationMs?: number
  usage?: { inputTokens: number; outputTokens: number }
  output?: string
  error?: string
}

export interface PreAgentRunOutcome {
  aborted: boolean
  reason?: string
}

export async function triggerPreAgentRun(
  hookRegistry: IHookRegistry | undefined,
  data: AgentRunHookData,
): Promise<PreAgentRunOutcome> {
  if (!hookRegistry) return { aborted: false }
  try {
    const result = await hookRegistry.trigger({
      event: 'pre:agent:run',
      data: { ...data },
    })
    if (result.action === 'abort') return { aborted: true, reason: result.reason }
    return { aborted: false }
  } catch {
    return { aborted: false }
  }
}

export async function triggerPostAgentRun(
  hookRegistry: IHookRegistry | undefined,
  data: AgentRunHookData,
): Promise<void> {
  if (!hookRegistry) return
  try {
    await hookRegistry.trigger({ event: 'post:agent:run', data: { ...data } })
  } catch {
  }
}
