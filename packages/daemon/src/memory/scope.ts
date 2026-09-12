/**
 * Memory scope helpers.
 *
 * sepilotd shares one MEMORY.md and one semantic index across the whole
 * daemon, but multiple users / channels can converse with it. To keep
 * one user's facts from leaking into another user's recall, every durable
 * memory entry carries canonical scope tags, and retrieval/update tools
 * filter by the active scope.
 *
 * Tag canonical form (lowercased, ascii where possible):
 *   - scope:user:<senderId>           a single human/sender
 *   - scope:channel:<type>:<chatId>   a chat / room / thread
 *   - scope:session:<sessionId>       a single sepilotd session
 *   - scope:project:<projectHash>     a repo/cwd, keyed by a hash of the
 *                                     absolute path (never the raw path — it
 *                                     embeds the OS username)
 *
 * Existing memories without a scope:* tag are legacy-global. They are only
 * visible to unscoped administrators or trusted first-party clients carrying
 * the server-derived legacy read capability. New memories saved through any
 * channel surface are automatically tagged with the active scope.
 */

import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

export interface MemoryScope {
  userId?: string
  channelType?: string
  chatId?: string
  sessionId?: string
  /**
   * Stable hash of the active project/repo absolute path (see
   * `hashProjectPath`). Never the raw path: that leaks the OS username into
   * persistent tags/keys. Callers derive it from a cwd via `hashProjectPath`.
   */
  projectHash?: string
  /**
   * Optional groups the caller belongs to. A user can be in multiple
   * groups simultaneously and any memory tagged with one of those
   * `scope:group:<id>` tags becomes visible to them.
   */
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

/**
 * Derive a stable, privacy-preserving project key from a cwd. The path is
 * resolved to an absolute path (so relative/trailing-slash variants collapse)
 * and hashed with sha256; only a 16-hex-char (64-bit) prefix is kept. The raw
 * path — which typically contains the OS username (e.g. `/home/<user>/...`) —
 * is NEVER returned or stored, so it can never leak into a persistent tag,
 * memory key, or on-disk project-state filename.
 */
export function hashProjectPath(cwd: string): string {
  const absolute = resolve(cwd)
  return createHash('sha256').update(absolute).digest('hex').slice(0, 16)
}

/** Full `scope:project:<hash>` tag for a cwd (hash via `hashProjectPath`). */
export function projectScopeTag(cwd: string): string {
  return `${SCOPE_PROJECT_PREFIX}${hashProjectPath(cwd)}`
}

/** Canonicalize a scope into the tags that should be attached to memories. */
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

/**
 * Server-derived isolation scope for an authenticated extension token.
 *
 * The user tag gives file memory a deterministic private bucket. The
 * extension tag marks the context as strict-isolation so legacy unscoped
 * memories are not treated as globally readable/writable by third-party
 * principals. Callers must never derive this from request headers.
 */
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

/** Pull the scope back out of a tag list. Returns an empty object for global memories. */
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

/** True when the memory is visible to a caller running with `contextTags`.
 *  - Legacy-global (no scope tag) memories require an unscoped administrator
 *    or the server-derived first-party read capability.
 *  - Memories tagged with `scope:public` are always visible.
 *  - Otherwise the entry must share at least one scope tag with the caller. */
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

/** Returns true when the caller is allowed to mutate (update/forget) the entry. */
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

/**
 * Match the most specific durable owner on the entry. A channel tag attached
 * alongside a user tag describes where the personal memory was captured; it
 * must not turn that memory into room-shared state.
 */
function hasMatchingOwnershipScope(entryTags: string[], contextTags: string[]): boolean {
  const ownersAt = (prefix: string) => entryTags.filter((tag) => tag.startsWith(prefix))
  const matches = (owners: string[]) => owners.some((tag) => contextTags.includes(tag))

  const users = ownersAt(SCOPE_USER_PREFIX)
  if (matches(users)) return true

  // Group tags are an explicit shared ACL. They can grant access across users,
  // unlike a channel tag, which only records where a personal memory arose.
  const groups = ownersAt(SCOPE_GROUP_PREFIX)
  if (matches(groups)) return true

  if (users.length > 0) return false

  const channels = ownersAt(SCOPE_CHANNEL_PREFIX)
  if (channels.length > 0) return matches(channels)

  return entryTags.some((tag) => tag.startsWith(SCOPE_PREFIX) && contextTags.includes(tag))
}

/** Merge scope tags into an arbitrary user-supplied tag list, deduping. */
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
      // In a scoped context, ownership comes from the trusted transport or
      // agent context. User/model/import-provided tags must not inject
      // scope:public or another user's scope alongside that authority.
      if (hasAuthoritativeScope && tag.trim().toLowerCase().startsWith(SCOPE_PREFIX)) {
        continue
      }
      add(tag)
    }
  }
  if (scopeTags) {
    for (const tag of scopeTags) {
      // Capabilities authorize the current operation; they are never durable
      // memory ownership tags and must not be persisted on an entry.
      if (tag.trim().toLowerCase() === LEGACY_GLOBAL_MEMORY_READ_CAPABILITY) continue
      add(tag)
    }
  }
  return out
}

function normalizeId(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9_\-:.]+/g, '_')
}

/** True when an entry has been superseded (replaced by a newer contradiction). */
export function isMemorySuperseded(tags: string[] | undefined): boolean {
  if (!tags) return false
  return tags.some((tag) => {
    const lower = tag.toLowerCase()
    return lower === 'superseded' || lower.startsWith('superseded-by:')
  })
}

/** True when an entry has been archived (replaced by a compressed summary). */
export function isMemoryArchived(tags: string[] | undefined): boolean {
  if (!tags) return false
  return tags.some((tag) => {
    const lower = tag.toLowerCase()
    return lower === 'archived' || lower.startsWith('archived-by:')
  })
}

/** Derive scope tags directly from a normalized incoming channel message. */
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
