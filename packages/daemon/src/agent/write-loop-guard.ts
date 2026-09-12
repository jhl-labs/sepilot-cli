import type { Message, ToolCall } from '@sepilotd/core'

/**
 * Guard against an agent that loops re-writing the same file forever.
 *
 * Observed in the vm-test real-usage session (CLI_BACKLOG.md B6): a stuck model
 * called `fs.write` against the same path 50+ times across a single run. The
 * inner engine loop is bounded by `effectiveMaxIterations`, but that bound is
 * high enough (~38) that a degenerate write loop still burns minutes and
 * rewrites the file dozens of times before the run ends.
 *
 * This is a deliberately *last-resort* cap: the default limit is generous
 * (10 writes to the same target) so legitimate iterative edits are never
 * affected. When the limit is reached the offending write turn is skipped and
 * the model is told to stop and finalize — the same shape as the existing
 * duplicate-tool-call repair, so no agent control flow changes.
 */
export const WRITE_LOOP_TOOL_NAMES: ReadonlySet<string> = new Set([
  'fs.write',
  'fs.edit',
  'fs.append',
  'apply_patch',
])

/** Default max writes to the same target before the guard trips. */
export const DEFAULT_WRITE_LOOP_LIMIT = 10

/**
 * Stable key identifying a write target. Path-based tools key on their path so
 * varying *content* to the same file is still counted. Pathless writes
 * (e.g. apply_patch) fall back to a serialization of the arguments, so only
 * byte-identical repeats are counted for those — accepted, since the observed
 * runaway is path-based fs.write.
 */
export function writeLoopTargetKey(
  name: string,
  args: Record<string, unknown> | undefined,
): string | null {
  if (!WRITE_LOOP_TOOL_NAMES.has(name)) return null
  const path = args?.path
  if (typeof path === 'string' && path.trim().length > 0) {
    return `${name}:path:${path.trim()}`
  }
  try {
    return `${name}:args:${JSON.stringify(args ?? {})}`
  } catch {
    return null
  }
}

export class WriteLoopTracker {
  private readonly counts = new Map<string, number>()

  constructor(private readonly limit: number = DEFAULT_WRITE_LOOP_LIMIT) {}

  /** Record one executed write tool call. */
  record(name: string, args: Record<string, unknown> | undefined): void {
    const key = writeLoopTargetKey(name, args)
    if (!key) return
    this.counts.set(key, (this.counts.get(key) ?? 0) + 1)
  }

  count(name: string, args: Record<string, unknown> | undefined): number {
    const key = writeLoopTargetKey(name, args)
    return key ? (this.counts.get(key) ?? 0) : 0
  }

  /**
   * Returns a human-readable target description when any of the upcoming tool
   * calls would write a target already written `limit` times. Null otherwise.
   */
  offendingTarget(toolCalls: ReadonlyArray<ToolCall>): string | null {
    for (const call of toolCalls) {
      const key = writeLoopTargetKey(call.name, call.arguments)
      if (!key) continue
      if ((this.counts.get(key) ?? 0) >= this.limit) {
        const path = call.arguments?.path
        return typeof path === 'string' && path.trim().length > 0
          ? path.trim()
          : call.name
      }
    }
    return null
  }
}

export function buildWriteLoopRepairMessage(target: string, limit: number): Message {
  return {
    role: 'system',
    content: [
      `You have already written "${target}" ${limit} times in this run and keep issuing more writes to it.`,
      'This is a write loop. Stop rewriting that file now.',
      'Either: (a) provide your final answer / summary to the user, (b) act on a DIFFERENT file or tool if other work genuinely remains, or (c) state the concrete blocker and stop.',
      'Do not write to that same target again.',
    ].join(' '),
  }
}
