import {
  deriveExtensionMemoryScopeTags,
  deriveScopeTags,
  withLegacyGlobalMemoryRead,
} from '../memory/scope.js'
import type { RequestAuthContext } from './auth.js'

type HeaderValue = string | string[] | undefined

/**
 * Convert the transport-level X-Memory-Scope-* headers into the canonical
 * scope tags understood by memory tools and scoped file-memory buckets.
 */
export function readChatMemoryScope(
  headers: Record<string, HeaderValue>,
  authContext?: RequestAuthContext,
): string[] {
  if (authContext?.kind === 'extension') {
    return deriveExtensionMemoryScopeTags(authContext.tokenId)
  }
  const headerValue = (key: string): string | undefined => {
    const value = headers[key]
    if (typeof value === 'string') return value.trim() || undefined
    if (Array.isArray(value)) return value[0]?.trim() || undefined
    return undefined
  }

  const rawGroups = headerValue('x-memory-scope-groups')
  const groupIds = rawGroups
    ? rawGroups
        .split(',')
        .map((entry) => entry.trim())
        .filter(Boolean)
    : undefined

  const scopeTags = deriveScopeTags({
    userId: headerValue('x-memory-scope-user-id'),
    channelType: headerValue('x-memory-scope-channel-type'),
    chatId: headerValue('x-memory-scope-channel-id'),
    sessionId: headerValue('x-memory-scope-session-id'),
    groupIds,
  })
  // When daemon auth is disabled there is no authContext, but loopback callers
  // still have the same first-party privileges as a master-token caller.
  // Extensions returned above and can never receive this capability.
  return withLegacyGlobalMemoryRead(scopeTags)
}

/**
 * Session ownership is not yet persisted in SessionMeta. Until it is, an
 * extension may create an isolated one-turn session but cannot name/reuse a
 * session id or replay a cursor that could belong to another principal.
 */
export function isExtensionSessionReuseDenied(
  authContext: RequestAuthContext | undefined,
  requestedSessionId: unknown,
): boolean {
  return authContext?.kind === 'extension' && requestedSessionId !== undefined
}

/**
 * Writing mode resolves an omitted document id through a process-wide active
 * document. Extension principals do not own that registry yet, so permitting
 * the mode would expose or mutate another client's document.
 */
export function isExtensionWritingModeDenied(
  authContext: RequestAuthContext | undefined,
  requestedMode: unknown,
): boolean {
  return authContext?.kind === 'extension' && requestedMode === 'writing'
}
