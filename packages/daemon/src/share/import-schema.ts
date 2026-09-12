import type { SessionEvent } from '@sepilotd/core'
import { z } from 'zod'

const MAX_SHORT_TEXT = 1_024
const MAX_EVENT_TEXT = 512 * 1_024

const baseEventFields = {
  id: z.string().min(1).max(MAX_SHORT_TEXT),
  timestamp: z.string().min(1).max(100),
}

const transcriptEventSchemas = new Map<string, z.ZodTypeAny>([
  ['user_message', z.object({
    ...baseEventFields,
    type: z.literal('user_message'),
    content: z.string().max(MAX_EVENT_TEXT),
  })],
  ['assistant_message', z.object({
    ...baseEventFields,
    type: z.literal('assistant_message'),
    content: z.string().max(MAX_EVENT_TEXT),
  })],
])

export type ImportableSessionEventsParseResult =
  | { success: true; events: SessionEvent[] }
  | { success: false }

/**
 * Import only user/assistant transcript messages from an untrusted share.
 * Planner state, compacted context, tool history, approvals, memory context,
 * and delegation events can be promoted to system/tool authority when a
 * session resumes, so they must never cross this trust boundary.
 */
export function parseImportableSessionEvents(
  values: readonly unknown[],
): ImportableSessionEventsParseResult {
  const events: SessionEvent[] = []
  for (const value of values) {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      return { success: false }
    }
    const type = (value as { type?: unknown }).type
    if (typeof type !== 'string') return { success: false }
    const schema = transcriptEventSchemas.get(type)
    if (!schema) continue
    const parsed = schema.safeParse(value)
    if (!parsed.success) return { success: false }
    events.push(parsed.data as SessionEvent)
  }
  return { success: true, events }
}
