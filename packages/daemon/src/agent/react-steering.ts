import type { Message } from '@sepilotd/core'
import { formatActiveUserInstructions, activeUserInstructions, type AgentSteeringNote } from './user-steering.js'

const CHECKPOINT_KEY = 'reactSteeringNotes'

export function withoutSteeringCheckpoint(messages: readonly Message[]): Message[] {
  return messages.filter((message) => message.metadata?.[CHECKPOINT_KEY] === undefined)
}

export function recoveredReactSteering(messages: readonly Message[]): AgentSteeringNote[] {
  const value = messages.findLast((message) => message.role === 'system'
    && message.metadata?.[CHECKPOINT_KEY] !== undefined)?.metadata?.[CHECKPOINT_KEY]
  if (!Array.isArray(value)) return []
  return value.filter((note): note is AgentSteeringNote => Boolean(note)
    && typeof note.id === 'string' && typeof note.message === 'string'
    && (note.kind === 'instruction' || note.kind === 'question')
    && typeof note.createdAt === 'number'
    && (note.consumedAt === undefined || typeof note.consumedAt === 'number')
    && (note.cancelledAt === undefined || typeof note.cancelledAt === 'number'))
    .map((note) => ({ ...note }))
}

/** Checkpoint-only metadata; re-inject active instructions after compaction on every request. */
export function withSteeringCheckpoint(messages: readonly Message[], notes: readonly AgentSteeringNote[] = []): Message[] {
  const result = withoutSteeringCheckpoint(messages)
  if (notes.length > 0) result.push({
    role: 'system',
    content: formatActiveUserInstructions(activeUserInstructions(notes)),
    metadata: { [CHECKPOINT_KEY]: notes.map((note) => ({ ...note })) },
  })
  return result
}
