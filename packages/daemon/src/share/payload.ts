import type { SessionEvent, SessionMeta } from '@sepilotd/core'
import { homedir } from 'node:os'
import { sanitizeEvent } from '../sessions/sanitize.js'

export interface SharedSessionPayload {
  session: SessionMeta
  events: SessionEvent[]
}

export interface BuildSharedSessionPayloadOptions {
  /**
   * Include (still secret-redacted) tool_result output in the public payload.
   * Default false: tool stdout/stderr and file contents can carry credentials
   * or private data that do not match any secret pattern, so they are stripped
   * from public shares unless the operator opts in.
   */
  includeToolOutput?: boolean
}

const SHARE_SANITIZE_OPTIONS = { home: homedir() }

export function shareIncludesToolOutput(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  return env.SEPILOTD_SHARE_INCLUDE_TOOL_OUTPUT === '1'
}

/** Replace a tool_result output with a length-annotated placeholder. */
function strippedToolOutput(output: unknown): string {
  const length = typeof output === 'string' ? output.length : 0
  return `[tool output omitted from public share (${length} chars)]`
}

/**
 * Strip verbatim tool outputs from the event before secret sanitisation. Only
 * tool_result.output is dropped; user/assistant messages and every other field
 * are preserved for sanitizeEvent to redact secret-shaped strings within.
 */
export function stripToolOutput(event: SessionEvent, includeToolOutput: boolean): SessionEvent {
  if (includeToolOutput) return event
  if ((event as { type?: string }).type !== 'tool_result') return event
  return {
    ...(event as unknown as Record<string, unknown>),
    output: strippedToolOutput((event as { output?: unknown }).output),
  } as unknown as SessionEvent
}

/**
 * File ids are local daemon capabilities and filenames can contain private
 * context. Public JSON shares do not carry the bytes, so neither field is
 * useful or safe to export.
 */
export function stripAttachmentCapabilities(event: SessionEvent): SessionEvent {
  if (event.type !== 'user_message' || !event.attachments?.length) return event
  const safeEvent = { ...event } as Partial<typeof event>
  delete safeEvent.attachments
  return safeEvent as SessionEvent
}

export function prepareEventForPublicShare(
  event: SessionEvent,
  includeToolOutput: boolean,
): SessionEvent {
  return stripAttachmentCapabilities(stripToolOutput(event, includeToolOutput))
}

export function buildSharedSessionPayload(
  session: SessionMeta,
  events: SessionEvent[],
  options: BuildSharedSessionPayloadOptions = {},
): SharedSessionPayload {
  const includeToolOutput = options.includeToolOutput ?? shareIncludesToolOutput()
  return {
    session: sanitizeEvent(session, SHARE_SANITIZE_OPTIONS),
    events: events.map((event) =>
      sanitizeEvent(prepareEventForPublicShare(event, includeToolOutput), SHARE_SANITIZE_OPTIONS),
    ),
  }
}
