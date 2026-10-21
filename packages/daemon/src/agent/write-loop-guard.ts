import type { Message, ToolCall } from '@sepilotd/core'

export const WRITE_LOOP_TOOL_NAMES: ReadonlySet<string> = new Set([
  'fs.write',
  'fs.edit',
  'fs.append',
  'apply_patch',
])

export const DEFAULT_WRITE_LOOP_LIMIT = 10

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

  record(name: string, args: Record<string, unknown> | undefined): void {
    const key = writeLoopTargetKey(name, args)
    if (!key) return
    this.counts.set(key, (this.counts.get(key) ?? 0) + 1)
  }

  count(name: string, args: Record<string, unknown> | undefined): number {
    const key = writeLoopTargetKey(name, args)
    return key ? (this.counts.get(key) ?? 0) : 0
  }

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
