import type { SemanticMemoryStore } from './types.js'
import { isMemoryWritableInScope } from './scope.js'

export class MemorySelectionError extends Error {
  constructor(readonly code: 'INVALID_INPUT_PERMANENT' | 'MEMORY_SCOPE_MISMATCH', message: string) {
    super(message)
    this.name = 'MemorySelectionError'
  }
}

export function validateMemorySelection(ids: unknown): asserts ids is string[] {
  if (!Array.isArray(ids) || ids.length === 0 || ids.length > 100
    || ids.some((id) => typeof id !== 'string' || !id.trim() || id !== id.trim() || id.length > 256)) {
    throw new MemorySelectionError('INVALID_INPUT_PERMANENT', 'ids must contain 1–100 nonempty memory IDs returned by memory.search or memory.list.')
  }
}

export async function forgetSelectedMemory(
  index: Pick<SemanticMemoryStore, 'get' | 'delete' | 'recordAudit'>,
  ids: string[],
  scopeTags: string[],
  options: { dryRun?: boolean } = {},
) {
  validateMemorySelection(ids)
  const selected = await Promise.all([...new Set(ids)].map(async (id) => ({ id, entry: await index.get(id) })))
  if (selected.some(({ entry }) => entry && !isMemoryWritableInScope(entry.tags, scopeTags))) {
    throw new MemorySelectionError('MEMORY_SCOPE_MISMATCH', 'One or more selected memories are outside the caller write scope; nothing was deleted.')
  }
  const present = selected.flatMap(({ entry }) => entry ? [entry] : [])
  const missingIds = selected.filter(({ entry }) => !entry).map(({ id }) => id)
  const retained = ['unselected memories', 'conversation history and current context', 'audit and retraction history', 'documents and daily journals', 'scheduled reminders']
  if (options.dryRun) {
    return { dryRun: true, selectedIds: present.map((entry) => entry.id), missingIds, snapshotOnly: true, retained }
  }
  for (const entry of present) {
    await index.delete(entry.id)
    await index.recordAudit({ memoryId: entry.id, action: 'deleted', actor: 'tool',
      reason: 'Explicit selection via memory.forget',
      before: { id: entry.id, content: entry.content, source: entry.source, tags: entry.tags, evidence: entry.evidence },
    })
  }
  return { deletedIds: present.map((entry) => entry.id), missingIds, retained,
    vectorCleanup: 'Uses the index backend cleanup policy; external vector cleanup may be best-effort. Authoritative entries are removed.' }
}
