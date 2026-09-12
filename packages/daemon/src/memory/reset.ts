import type { SessionEvent } from '@sepilotd/core'

/** Durable owner, independent of lossy filesystem bucket names and session tags. */
export function memoryResetOwner(tags: string[] = []): string {
  const normalized = tags.map((tag) => tag.toLowerCase())
  for (const prefix of ['scope:user:', 'scope:channel:', 'scope:group:', 'scope:project:', 'scope:session:']) {
    const owners = normalized.filter((tag) => tag.startsWith(prefix)).sort()
    if (owners.length) return owners[0]!
  }
  // Unknown future scope kinds must never acquire global reset authority.
  return normalized.filter((tag) => tag.startsWith('scope:')).sort()[0] ?? 'global'
}

/** Keep complete turns begun after reset; an answer to the erase request is old context too. */
export function eventsAfterMemoryReset(events: SessionEvent[], resetAt?: string | null): SessionEvent[] {
  if (!resetAt) return events
  const cutoff = Date.parse(resetAt)
  let eligible = false
  return events.filter((event) => {
    if (event.type === 'user_message') eligible = Date.parse(event.timestamp) > cutoff
    return eligible
  })
}
