import { createHash } from 'node:crypto'
import { z } from 'zod'
import type { ISemanticIndex, SessionEvent } from '@sepilotd/core'
import { attachScopeTags } from './scope.js'
import { looksSensitive } from './sensitive.js'

export const procedureCandidateSchema = z.object({
  conditions: z.string().min(10).max(600),
  steps: z.array(z.string().min(3).max(400)).min(1).max(8),
  evidenceEventIds: z.array(z.string()).min(1).max(20),
})

/** Repeated independent episodes support a procedure; assistant prose alone cannot. */
export async function learnProcedure(
  input: unknown, events: SessionEvent[], scopeTags: string[], index: ISemanticIndex,
): Promise<boolean> {
  const parsed = procedureCandidateSchema.safeParse(input)
  const session = scopeTags.find((tag) => tag.startsWith('scope:session:'))
  if (!parsed.success || !session) return false
  const candidate = parsed.data
  const calls = new Set(events.filter((event) => event.type === 'tool_call').map((event) => event.id))
  const results = events.filter((event) => event.type === 'tool_result'
    && calls.has(event.toolCallId) && candidate.evidenceEventIds.includes(event.id))
  if (results.length !== new Set(candidate.evidenceEventIds).size) return false
  const content = `Conditions: ${candidate.conditions}\nSteps:\n${candidate.steps.map((step, i) => `${i + 1}. ${step}`).join('\n')}`
  if (looksSensitive(content)) return false
  const owner = scopeTags.filter((tag) => tag.startsWith('scope:') && !tag.startsWith('scope:session:')).sort()
  if (!owner.length) return false
  const id = `procedure-${createHash('sha256').update(JSON.stringify([owner, content.normalize('NFKC').replace(/\s+/g, ' ')])).digest('hex').slice(0, 32)}`
  const previous = await index.get(id)
  const sourceIds = [...new Set([...(previous?.evidence?.sourceIds ?? []), `episode:${session}`, ...results.map((event) => `${session}/event:${event.id}`)])].slice(-100)
  const failures = results.some((event) => event.type === 'tool_result' && event.status !== 'success')
  const status = failures || previous?.evidence?.status === 'retracted' ? 'retracted'
    : sourceIds.filter((source) => source.startsWith('episode:')).length >= 2 ? 'active' : 'candidate'
  await index.add({
    id, content, source: 'conversation', tags: attachScopeTags(['learned-procedure'], owner),
    evidence: { kind: 'procedural', origin: 'inferred', observedAt: new Date().toISOString(), status, sourceIds },
  })
  return true
}
