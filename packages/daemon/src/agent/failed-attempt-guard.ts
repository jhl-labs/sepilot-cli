import { signatureOf } from './stuck-tool-repeat.js'
import type { AgentState } from './graph/types.js'

/**
 * Failed-attempt pre-execution guard: before re-running a tool call whose
 * structural signature (tool + stable-serialized args, shared with the
 * stuck-tool-repeat detector) was already recorded as failed this run, block
 * it and hand the model the recorded failure reason so it takes a different
 * approach instead of burning iterations on a known-bad action.
 *
 * Complementary to the stuck-repeat guard: that one catches "same call
 * repeating right now"; this one catches "retrying an approach that already
 * failed earlier". Blocking is bounded (MAX_FAILED_ATTEMPT_BLOCKS) — if the
 * model insists past the budget the call runs anyway with a warning, so the
 * guard can never deadlock a run.
 */
export const MAX_FAILED_ATTEMPT_BLOCKS = 2

function toSignatureInput(toolCall: { name: string; arguments?: unknown }): {
  tool: string
  input: Record<string, unknown>
} {
  return {
    tool: toolCall.name,
    input: (toolCall.arguments ?? {}) as Record<string, unknown>,
  }
}

export function recordFailedAttempt(
  state: Pick<AgentState, 'failedAttempts'>,
  toolCall: { name: string; arguments?: unknown },
  reason: string,
  now: () => number = () => Date.now(),
): void {
  const signature = signatureOf(toSignatureInput(toolCall))
  const list = (state.failedAttempts ??= [])
  const existing = list.find((f) => f.signature === signature)
  if (existing) {
    existing.reason = reason
    existing.ts = now()
    return
  }
  list.push({ signature, tool: toolCall.name, reason, ts: now() })
}

/**
 * Remove a stale failure after the structurally-identical action succeeds.
 *
 * Tool failures are observations about one execution, not permanent facts
 * about an invocation. A retry can legitimately recover after a source edit,
 * a restarted service, an approval, or a transient network failure. Keeping
 * the old row after a successful retry makes the guard contradict the newest
 * tool result and can drive the model back into the action it just completed.
 */
export function clearFailedAttempt(
  state: Pick<AgentState, 'failedAttempts'>,
  toolCall: { name: string; arguments?: unknown },
): boolean {
  const list = state.failedAttempts
  if (!list || list.length === 0) return false
  const signature = signatureOf(toSignatureInput(toolCall))
  const next = list.filter((attempt) => attempt.signature !== signature)
  if (next.length === list.length) return false
  state.failedAttempts = next
  return true
}

export function checkFailedAttempt(
  toolCall: { name: string; arguments?: unknown },
  failedAttempts: AgentState['failedAttempts'],
): { blocked: boolean; reason?: string } {
  const signature = signatureOf(toSignatureInput(toolCall))
  const hit = (failedAttempts ?? []).find((f) => f.signature === signature)
  return hit ? { blocked: true, reason: hit.reason } : { blocked: false }
}

const RECOVERY_MUTATION_TOOL_NAMES = new Set([
  'apply_patch',
  'fs.edit',
  'fs.write',
  'fs.append',
  'process.start',
  'process.stop',
  'service.start',
  'service.stop',
  'service.remove',
])

function isSuccessfulExternalTerminalAction(
  entry: NonNullable<AgentState['toolCallHistory']>[number],
): boolean {
  if (entry.tool !== 'terminal.run' || entry.status !== 'success') return false
  const network = entry.input.network
  if (network === 'external') return true
  return Boolean(
    network
    && typeof network === 'object'
    && !Array.isArray(network)
    && (network as Record<string, unknown>).mode === 'external',
  )
}

/**
 * Evaluate a recorded failure against the current run state. A compiler,
 * test, HTTP probe, or other exact action is no longer the same attempt after
 * a successful source/environment mutation. Permit that first post-mutation
 * retry; if it fails again, recordFailedAttempt refreshes the timestamp and
 * the ordinary bounded guard applies again.
 */
export function checkFailedAttemptAfterRecovery(
  toolCall: { name: string; arguments?: unknown },
  state: Pick<AgentState, 'failedAttempts' | 'toolCallHistory'>,
): { blocked: boolean; reason?: string } {
  const signature = signatureOf(toSignatureInput(toolCall))
  const hit = (state.failedAttempts ?? []).find((attempt) => attempt.signature === signature)
  if (!hit) return { blocked: false }

  const recovered = (state.toolCallHistory ?? []).some((entry) => (
    entry.status === 'success'
    && entry.ts >= hit.ts
    && (
      RECOVERY_MUTATION_TOOL_NAMES.has(entry.tool)
      // An external terminal action can repair the execution environment
      // without changing workspace files (for example by populating a package
      // cache). It invalidates an earlier terminal failure, but not a failed
      // file edit or another capability's structural failure.
      || (hit.tool === 'terminal.run' && isSuccessfulExternalTerminalAction(entry))
    )
  ))
  return recovered
    ? { blocked: false }
    : { blocked: true, reason: hit.reason }
}

/**
 * Structured tool-result text returned instead of executing a blocked call.
 */
export function buildFailedAttemptBlockOutput(reason: string): string {
  return [
    '[Failed-attempt guard] This exact tool call (same tool, same arguments) already failed earlier in this run and was not executed again.',
    `Recorded failure: ${reason}`,
    'Take a different approach: change the arguments, use another tool, or gather new evidence first. If this exact call is truly required, request it again and it will be allowed with a warning.',
  ].join('\n')
}

/**
 * Warning injected alongside the real result when the block budget is
 * exhausted and a previously-failed call is executed anyway.
 */
export function buildFailedAttemptWarning(reason: string): string {
  return `[Failed-attempt guard] Warning: this call was already recorded as failed earlier in this run (${reason}). Executing anyway because the block budget is exhausted.`
}
