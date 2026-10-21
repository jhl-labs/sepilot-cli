import {
  deriveExtensionMemoryScopeTags,
  deriveScopeTags,
  withLegacyGlobalMemoryRead,
} from '../memory/scope.js'
import type { RequestAuthContext } from './auth.js'

type HeaderValue = string | string[] | undefined

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
  return withLegacyGlobalMemoryRead(scopeTags)
}

export function isExtensionSessionReuseDenied(
  authContext: RequestAuthContext | undefined,
  requestedSessionId: unknown,
): boolean {
  return authContext?.kind === 'extension' && requestedSessionId !== undefined
}

export function isExtensionWritingModeDenied(
  authContext: RequestAuthContext | undefined,
  requestedMode: unknown,
): boolean {
  return authContext?.kind === 'extension' && requestedMode === 'writing'
}
