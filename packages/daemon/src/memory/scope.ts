import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

export interface MemoryScope {
  userId?: string
  channelType?: string
  chatId?: string
  sessionId?: string
  projectHash?: string
  groupIds?: string[]
}

const SCOPE_PREFIX = 'scope:'
const SCOPE_USER_PREFIX = 'scope:user:'
const SCOPE_CHANNEL_PREFIX = 'scope:channel:'
const SCOPE_SESSION_PREFIX = 'scope:session:'
const SCOPE_GROUP_PREFIX = 'scope:group:'
const SCOPE_EXTENSION_PREFIX = 'scope:extension:'
const SCOPE_PUBLIC_TAG = 'scope:public'
export const SCOPE_PROJECT_PREFIX = 'scope:project:'
export const LEGACY_GLOBAL_MEMORY_READ_CAPABILITY = 'memory-capability:legacy-global-read'

export function withLegacyGlobalMemoryRead(scopeTags: string[]): string[] {
  if (scopeTags.length === 0) return scopeTags
  return [...scopeTags, LEGACY_GLOBAL_MEMORY_READ_CAPABILITY]
}

export function canReadLegacyGlobalMemory(scopeTags: string[] | undefined): boolean {
  return scopeTags?.some(
    (tag) => tag.trim().toLowerCase() === LEGACY_GLOBAL_MEMORY_READ_CAPABILITY,
  ) ?? false
}

export function hashProjectPath(cwd: string): string {
  const absolute = resolve(cwd)
  return createHash('sha256').update(absolute).digest('hex').slice(0, 16)
}

export function projectScopeTag(cwd: string): string {
  return `${SCOPE_PROJECT_PREFIX}${hashProjectPath(cwd)}`
}

export function deriveScopeTags(scope: MemoryScope | undefined): string[] {
  if (!scope) return []
  const tags: string[] = []
  if (scope.userId) tags.push(`${SCOPE_USER_PREFIX}${normalizeId(scope.userId)}`)
  if (scope.channelType && scope.chatId) {
    tags.push(`${SCOPE_CHANNEL_PREFIX}${normalizeId(scope.channelType)}:${normalizeId(scope.chatId)}`)
  }
  if (scope.sessionId) tags.push(`${SCOPE_SESSION_PREFIX}${normalizeId(scope.sessionId)}`)
  if (scope.projectHash) tags.push(`${SCOPE_PROJECT_PREFIX}${normalizeId(scope.projectHash)}`)
  if (scope.groupIds && scope.groupIds.length > 0) {
    for (const groupId of scope.groupIds) {
      const trimmed = groupId.trim()
      if (!trimmed) continue
      tags.push(`${SCOPE_GROUP_PREFIX}${normalizeId(trimmed)}`)
    }
  }
  return tags
}

export function deriveExtensionMemoryScopeTags(tokenId: string): string[] {
  const normalized = normalizeId(tokenId) || 'unknown'
  return [
    `${SCOPE_USER_PREFIX}extension-${normalized}`,
    `${SCOPE_EXTENSION_PREFIX}${normalized}`,
  ]
}

export function isExtensionMemoryScope(scopeTags: string[] | undefined): boolean {
  return scopeTags?.some((tag) =>
    tag.trim().toLowerCase().startsWith(SCOPE_EXTENSION_PREFIX)) ?? false
}

export function parseScopeFromTags(tags: string[] | undefined): MemoryScope {
  const scope: MemoryScope = {}
  const groupIds: string[] = []
  if (!tags) return scope
  for (const raw of tags) {
    const tag = raw.trim().toLowerCase()
    if (!tag.startsWith(SCOPE_PREFIX)) continue
    if (tag.startsWith(SCOPE_USER_PREFIX)) {
      scope.userId = tag.slice(SCOPE_USER_PREFIX.length) || undefined
      continue
    }
    if (tag.startsWith(SCOPE_CHANNEL_PREFIX)) {
      const rest = tag.slice(SCOPE_CHANNEL_PREFIX.length)
      const sep = rest.indexOf(':')
      if (sep > 0 && sep < rest.length - 1) {
        scope.channelType = rest.slice(0, sep)
        scope.chatId = rest.slice(sep + 1)
      }
      continue
    }
    if (tag.startsWith(SCOPE_PROJECT_PREFIX)) {
      scope.projectHash = tag.slice(SCOPE_PROJECT_PREFIX.length) || undefined
      continue
    }
    if (tag.startsWith(SCOPE_SESSION_PREFIX)) {
      scope.sessionId = tag.slice(SCOPE_SESSION_PREFIX.length) || undefined
      continue
    }
    if (tag.startsWith(SCOPE_GROUP_PREFIX)) {
      const groupId = tag.slice(SCOPE_GROUP_PREFIX.length)
      if (groupId) groupIds.push(groupId)
    }
  }
  if (groupIds.length > 0) {
    scope.groupIds = groupIds
  }
  return scope
}

export function hasScopeTag(tags: string[] | undefined): boolean {
  if (!tags) return false
  return tags.some((tag) => tag.toLowerCase().startsWith(SCOPE_PREFIX))
}

export function isMemoryVisibleInScope(
  entryTags: string[] | undefined,
  contextTags: string[] | undefined,
): boolean {
  const namespaces = (tags: string[] | undefined) => tags?.filter(tag => tag.toLowerCase().startsWith('scope:persona:')).map(tag => tag.toLowerCase()) ?? []
  const entryNamespaces = namespaces(entryTags), contextNamespaces = namespaces(contextTags)
  if (entryNamespaces.length || contextNamespaces.length) {
    if (entryNamespaces.length !== 1 || contextNamespaces.length !== 1 || entryNamespaces[0] !== contextNamespaces[0]) return false
  }

  if (!entryTags || !hasScopeTag(entryTags)) {
    return !contextTags || contextTags.length === 0 || canReadLegacyGlobalMemory(contextTags)
  }
  const lower = entryTags.map((t) => t.toLowerCase())
  if (lower.includes(SCOPE_PUBLIC_TAG)) return true
  if (!contextTags || contextTags.length === 0) return false
  const ctxLower = contextTags.map((t) => t.toLowerCase())
  return hasMatchingOwnershipScope(lower, ctxLower)
}

export function isMemoryWritableInScope(
  entryTags: string[] | undefined,
  contextTags: string[] | undefined,
): boolean {
  const namespaces = (tags: string[] | undefined) => tags?.filter(tag => tag.toLowerCase().startsWith('scope:persona:')).map(tag => tag.toLowerCase()) ?? []
  const entryNamespaces = namespaces(entryTags), contextNamespaces = namespaces(contextTags)
  if (entryNamespaces.length || contextNamespaces.length) {
    if (entryNamespaces.length !== 1 || contextNamespaces.length !== 1 || entryNamespaces[0] !== contextNamespaces[0]) return false
  }

  if (!entryTags || !hasScopeTag(entryTags)) return !contextTags || contextTags.length === 0
  if (!contextTags || contextTags.length === 0) return false
  const lower = entryTags.map((t) => t.toLowerCase())
  if (lower.includes(SCOPE_PUBLIC_TAG)) return false
  const ctxLower = contextTags.map((t) => t.toLowerCase())
  return hasMatchingOwnershipScope(lower, ctxLower)
}

function hasMatchingOwnershipScope(entryTags: string[], contextTags: string[]): boolean {
  const ownersAt = (prefix: string) => entryTags.filter((tag) => tag.startsWith(prefix))
  const matches = (owners: string[]) => owners.some((tag) => contextTags.includes(tag))

  const users = ownersAt(SCOPE_USER_PREFIX)
  if (matches(users)) return true

  const groups = ownersAt(SCOPE_GROUP_PREFIX)
  if (matches(groups)) return true

  if (users.length > 0) return false

  const channels = ownersAt(SCOPE_CHANNEL_PREFIX)
  if (channels.length > 0) return matches(channels)

  return entryTags.some((tag) => tag.startsWith(SCOPE_PREFIX) && contextTags.includes(tag))
}

export function attachScopeTags(
  baseTags: string[] | undefined,
  scopeTags: string[] | undefined,
): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  const add = (tag: string) => {
    const trimmed = tag.trim()
    if (!trimmed) return
    const key = trimmed.toLowerCase()
    if (seen.has(key)) return
    seen.add(key)
    out.push(trimmed)
  }
  const hasAuthoritativeScope = scopeTags?.some((tag) =>
    tag.trim().toLowerCase().startsWith(SCOPE_PREFIX)) ?? false
  if (baseTags) {
    for (const tag of baseTags) {
      if (hasAuthoritativeScope && tag.trim().toLowerCase().startsWith(SCOPE_PREFIX)) {
        continue
      }
      add(tag)
    }
  }
  if (scopeTags) {
    for (const tag of scopeTags) {
      if (tag.trim().toLowerCase() === LEGACY_GLOBAL_MEMORY_READ_CAPABILITY) continue
      add(tag)
    }
  }
  return out
}

function normalizeId(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9_\-:.]+/g, '_')
}

export function isMemorySuperseded(tags: string[] | undefined): boolean {
  if (!tags) return false
  return tags.some((tag) => {
    const lower = tag.toLowerCase()
    return lower === 'superseded' || lower.startsWith('superseded-by:')
  })
}

export function isMemoryArchived(tags: string[] | undefined): boolean {
  if (!tags) return false
  return tags.some((tag) => {
    const lower = tag.toLowerCase()
    return lower === 'archived' || lower.startsWith('archived-by:')
  })
}

export function deriveChannelScopeTags(input: {
  channelType?: string
  channelId?: string
  senderId?: string
  sessionId?: string
  groupIds?: string[]
}): string[] {
  return deriveScopeTags({
    channelType: input.channelType,
    chatId: input.channelId,
    userId: input.senderId,
    sessionId: input.sessionId,
    groupIds: input.groupIds,
  })
}
