import type { IHookRegistry } from '@sepilotd/core'

/**
 * Shared turn-lifecycle hook gates.
 *
 * `pre:user:prompt` and `pre:/post:agent:run` used to fire from a subset of the
 * turn entry points only (HTTP `/chat` and the ReAct engine), so simply
 * switching transport (WS) or mode (graph) silently bypassed moderation /
 * redaction / audit hooks. These helpers centralise the trigger + abort +
 * rewrite handling so every surface (HTTP, WS, cowork, resume) and every mode
 * (react, graph) gates identically.
 */

export interface PreUserPromptContext {
  hookRegistry?: IHookRegistry
  sessionId: string
  /** Actor label for audit/moderation handlers (e.g. resolved API actor). */
  actor?: string
}

export interface PreUserPromptOutcome {
  /** True when a handler aborted the turn — the caller must not proceed. */
  aborted: boolean
  /** Human-readable reason surfaced to the client on abort. */
  reason?: string
  /** The (possibly rewritten) prompt to feed the agent and journal. */
  prompt: string
}

/**
 * Fire `pre:user:prompt` before the prompt is committed to the session journal
 * or fed to the agent. A handler can abort the turn or rewrite the prompt via
 * `modifiedPayload.data.prompt`. With no registry or no handler this is a
 * no-op that returns the original prompt.
 */
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
  sessionId: string
  mode?: string
  provider?: string
  model?: string
  input?: string
  /** Present when the run belongs to POST /chat/background. */
  backgroundJobId?: string
  detached?: boolean
  surface?: string
  /** Completion evidence is attached to post:agent:run only. */
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

/**
 * Fire `pre:agent:run` at the run boundary (mode-router / resume) so graph and
 * resume turns gate identically to ReAct — previously only the ReAct engine
 * fired it, leaving most (graph) turns and every resume ungated.
 */
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
    // A crashing gate hook must not take down the turn; fail open like the
    // engine's own observation-hook isolation.
    return { aborted: false }
  }
}

/** Fire `post:agent:run`; observation-only, never throws. */
export async function triggerPostAgentRun(
  hookRegistry: IHookRegistry | undefined,
  data: AgentRunHookData,
): Promise<void> {
  if (!hookRegistry) return
  try {
    await hookRegistry.trigger({ event: 'post:agent:run', data: { ...data } })
  } catch {
    // Observation hooks must not fail the main agent flow.
  }
}
