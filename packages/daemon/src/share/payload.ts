import type { SessionEvent, SessionMeta } from '@sepilotd/core'
import { homedir } from 'node:os'
import { credentialValuesFromEnv, sanitizeEvent } from '../sessions/sanitize.js'

export interface SharedSessionPayload {
  session: SessionMeta
  events: SessionEvent[]
}

export interface BuildSharedSessionPayloadOptions {
  includeToolOutput?: boolean
}

const SHARE_SANITIZE_OPTIONS = { home: homedir(), knownSecrets: credentialValuesFromEnv() }

export function shareIncludesToolOutput(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  return env.SEPILOTD_SHARE_INCLUDE_TOOL_OUTPUT === '1'
}

function strippedToolOutput(output: unknown): string {
  const length = typeof output === 'string' ? output.length : 0
  return `[tool output omitted from public share (${length} chars)]`
}

export function stripToolOutput(event: SessionEvent, includeToolOutput: boolean): SessionEvent {
  if (includeToolOutput) return event
  if ((event as { type?: string }).type !== 'tool_result') return event
  return {
    ...(event as unknown as Record<string, unknown>),
    output: strippedToolOutput((event as { output?: unknown }).output),
  } as unknown as SessionEvent
}

export function stripAttachmentCapabilities(event: SessionEvent): SessionEvent {
  if (event.type !== 'user_message' || !event.attachments?.length) return event
  const safeEvent = { ...event } as Partial<typeof event>
  delete safeEvent.attachments
  return safeEvent as SessionEvent
}

export function stripLiveViewStills(event: SessionEvent): SessionEvent {
  if (event.type !== 'assistant_message' || event.views === undefined) return event
  const safeEvent = { ...event } as Partial<typeof event>
  delete safeEvent.views
  return safeEvent as SessionEvent
}

export function prepareEventForPublicShare(
  event: SessionEvent,
  includeToolOutput: boolean,
): SessionEvent {
  return stripLiveViewStills(stripAttachmentCapabilities(stripToolOutput(event, includeToolOutput)))
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
