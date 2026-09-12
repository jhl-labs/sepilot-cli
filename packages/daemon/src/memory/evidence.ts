import { z } from 'zod'
import type { MemoryEntry } from '@sepilotd/core'

export const memoryEvidenceSchema = z.object({
  subject: z.enum(['user', 'persona', 'relationship']).optional(),
  reality: z.enum(['real', 'fictional']).optional(),
  kind: z.enum(['semantic', 'episodic', 'procedural']),
  origin: z.enum(['user', 'observed', 'inferred']),
  observedAt: z.string().datetime({ offset: true }),
  validFrom: z.string().datetime({ offset: true }).optional(),
  validUntil: z.string().datetime({ offset: true }).optional(),
  status: z.enum(['active', 'candidate', 'retracted']),
  sourceIds: z.array(z.string().min(1).max(500)).max(100),
}).strict().refine((value) => !(value.subject === 'user' && value.reality === 'fictional'), 'A fictional character claim cannot be classified as a real user fact').refine((value) => !value.validFrom || !value.validUntil
  || Date.parse(value.validFrom) < Date.parse(value.validUntil), 'validUntil must be after validFrom')

export function evidenceForWrite(entry: MemoryEntry, previous?: MemoryEntry | null, sessionId?: string) {
  const evidence = memoryEvidenceSchema.parse(entry.evidence ?? previous?.evidence ?? {
    kind: entry.source === 'conversation' ? 'episodic' : 'semantic',
    origin: entry.source === 'user' ? 'user' : 'inferred',
    observedAt: new Date().toISOString(), status: 'active',
    sourceIds: sessionId ? [sessionId] : [],
  })
  if (evidence.subject === 'persona' && entry.tags.filter((tag) => tag.toLowerCase().startsWith('scope:persona:')).length !== 1) {
    throw new Error('Persona identity memory requires an isolated persona namespace')
  }
  return evidence
}

export const memoryEvidenceInputSchema = {
  type: 'object', additionalProperties: false,
  description: 'Optional provenance. For a memory based solely on the current user request, omit this object so the runtime supplies the observation timestamp and source context.',
  properties: {
    subject: { type: 'string', enum: ['user', 'persona', 'relationship'], description: 'Whose claim this is. Persona identity requires an isolated persona memory space.' },
    reality: { type: 'string', enum: ['real', 'fictional'], description: 'Distinguish actual events from agreed fictional character settings. Never label invented events as real evidence.' },
    kind: { type: 'string', enum: ['semantic', 'episodic', 'procedural'] },
    origin: { type: 'string', enum: ['user', 'observed', 'inferred'] },
    observedAt: { type: 'string', format: 'date-time', description: 'Full ISO 8601 date-time with timezone, for example 2000-01-01T12:00:00Z. A date without a time is invalid. Use the actual observation time; do not invent precision.' },
    validFrom: { type: 'string', format: 'date-time', description: 'Optional full ISO 8601 date-time with timezone; fact is valid starting here.' },
    validUntil: { type: 'string', format: 'date-time', description: 'Optional exclusive expiry as a full ISO 8601 date-time with timezone, strictly after validFrom.' },
    status: { type: 'string', enum: ['active', 'candidate', 'retracted'] },
    sourceIds: { type: 'array', maxItems: 100, items: { type: 'string', minLength: 1, maxLength: 500 }, description: 'Actual supporting event/memory ids; never invent evidence.' },
  },
  required: ['kind', 'origin', 'observedAt', 'status', 'sourceIds'],
}

export function isEvidenceActive(entry: Pick<MemoryEntry, 'evidence'>, at = Date.now()): boolean {
  const evidence = entry.evidence
  return !evidence || (evidence.status === 'active'
    && (!evidence.validFrom || Date.parse(evidence.validFrom) <= at)
    && (!evidence.validUntil || Date.parse(evidence.validUntil) > at))
}
