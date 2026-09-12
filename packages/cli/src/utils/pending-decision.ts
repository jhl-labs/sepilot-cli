/**
 * A run blocked on a human decision (approval or question) is a first-class
 * run state, not a stalled run. The cli tracks it so the stream-idle watchdog
 * never reports "the stream hung" for a wait the operator themselves has to
 * end, and so the blocked state is visible instead of dead air.
 *
 * Mirrors the daemon-side state machine in
 * `packages/daemon/src/server/sse-response.ts`.
 */
export interface CliPendingDecision {
  kind: 'approval' | 'question'
  /** requestId / questionId — exactly what the operator resolves. */
  id: string
  /** Tool name for an approval, prompt text for a question. */
  label?: string
  since: number
}

export interface CliPendingDecisionTracker {
  note(event: unknown): void
  pending(): CliPendingDecision | null
}

const MODEL_STREAM_WAITING_THINKING = 'Still waiting for the model stream...'

/**
 * Keepalive-class frames prove nothing about the run's decision state: they
 * are emitted while the run is blocked as well as while it is working, so they
 * must neither set nor clear the pending state.
 */
function isDecisionNeutralFrame(record: Record<string, unknown>): boolean {
  if (record.type === 'thinking') {
    return record.content === MODEL_STREAM_WAITING_THINKING
      || record.text === MODEL_STREAM_WAITING_THINKING
  }
  if (record.type === 'state_change') return true
  if (record.type === undefined) return true
  return false
}

export function createCliPendingDecisionTracker(
  now: () => number = () => Date.now(),
): CliPendingDecisionTracker {
  let pending: CliPendingDecision | null = null
  return {
    note(event) {
      if (!event || typeof event !== 'object') return
      const record = event as Record<string, unknown>
      if (isDecisionNeutralFrame(record)) return
      if (record.type === 'approval_request' && typeof record.requestId === 'string') {
        pending = {
          kind: 'approval',
          id: record.requestId,
          label: typeof record.toolName === 'string' ? record.toolName : undefined,
          since: now(),
        }
        return
      }
      if (record.type === 'question_request' && typeof record.questionId === 'string') {
        pending = {
          kind: 'question',
          id: record.questionId,
          label: typeof record.prompt === 'string' ? record.prompt : undefined,
          since: now(),
        }
        return
      }
      // Any other frame means the run moved on — the decision was resolved
      // (approved, denied, answered, or auto-resolved) and the run is a normal
      // run again, watchdog included.
      pending = null
    },
    pending() {
      return pending
    },
  }
}

/**
 * Spelled exactly like the stream printer's hints so the operator can copy
 * either line and have it work.
 */
function resolveCommands(pending: CliPendingDecision, sessionId?: string): string {
  if (pending.kind === 'approval') {
    const sessionArg = sessionId ? ` --session ${sessionId}` : ''
    return `\`sepilot approve ${pending.id}${sessionArg} --scope run\` or \`sepilot deny ${pending.id}${sessionArg}\``
  }
  return `\`sepilot answer ${sessionId ?? '<session-id>'} ${pending.id} <reply>\``
}

/** Periodic "still blocked, here is how to unblock it" line. */
export function formatPendingDecisionWaitNotice(
  pending: CliPendingDecision,
  waitedMs: number,
  sessionId?: string,
): string {
  const what = pending.kind === 'approval'
    ? `approval ${pending.id}${pending.label ? ` (${pending.label})` : ''}`
    : `question ${pending.id}`
  return `[waiting ${Math.round(waitedMs / 1000)}s] run is paused on a pending ${what} — `
    + `it will not continue until you decide: ${resolveCommands(pending, sessionId)}`
}

/** Honest reason line when the run ends with a decision still outstanding. */
export function formatPendingDecisionAbortReason(
  pending: CliPendingDecision,
  sessionId?: string,
): string {
  const what = pending.kind === 'approval'
    ? `approval ${pending.id}${pending.label ? ` for ${pending.label}` : ''}`
    : `question ${pending.id}${pending.label ? `: ${pending.label}` : ''}`
  return `Run ended with a pending ${what} still unresolved — nothing hung, the run was waiting on your decision. `
    + `Resolve it with ${resolveCommands(pending, sessionId)}, or re-run interactively so the prompt can be answered in place.`
}
