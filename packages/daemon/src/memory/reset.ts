import type { SessionEvent } from '@sepilotd/core'

export function memoryResetOwner(tags: string[] = []): string {
  const normalized = tags.map((tag) => tag.toLowerCase())
  for (const prefix of ['scope:user:', 'scope:channel:', 'scope:group:', 'scope:project:', 'scope:session:']) {
    const owners = normalized.filter((tag) => tag.startsWith(prefix)).sort()
    if (owners.length) return owners[0]!
  }
  return normalized.filter((tag) => tag.startsWith('scope:')).sort()[0] ?? 'global'
}

export function eventsAfterMemoryReset(events: SessionEvent[], resetAt?: string | null): SessionEvent[] {
  if (!resetAt) return events
  const cutoff = Date.parse(resetAt)
  let eligible = false
  return events.filter((event) => {
    if (event.type === 'user_message') eligible = Date.parse(event.timestamp) > cutoff
    return eligible
  })
}
