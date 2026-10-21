import { journalInventory } from './journal-lifecycle.js'
import { readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import type { SemanticMemoryStore } from './types.js'
import type { ScopedFileMemoryRegistry } from './scoped-file-memory.js'
import { canonicalFileMemoryScopeKey } from './scoped-file-memory.js'
import { memoryResetOwner } from './reset.js'

export async function forgetAllOwnedMemory(index: Pick<SemanticMemoryStore, 'forgetOwnedMemory' | 'previewForgetOwnedMemory'>, registry: ScopedFileMemoryRegistry, scopeTags: string[], options: { dryRun?: boolean } = {}) {
  if (!index.forgetOwnedMemory) throw new Error('Semantic memory reset is unavailable')
  const ownsFileBucket = canonicalFileMemoryScopeKey(scopeTags) !== 'global' || !scopeTags.some((tag) => tag.startsWith('scope:'))
  await registry.discover?.()
  const file = ownsFileBucket
    ? registry.list().find((bucket) => bucket.key === canonicalFileMemoryScopeKey(scopeTags))?.fileMemory ?? registry.get(scopeTags)
    : undefined
  if (file) {
    const manifest = await readFile(join(dirname(file.getMemoryPath()), '.scope.json'), 'utf8').catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') return null
      throw error
    })
    if (manifest) {
      const tags: unknown = JSON.parse(manifest)
      if (!Array.isArray(tags) || !tags.every((tag) => typeof tag === 'string') || memoryResetOwner(tags) !== memoryResetOwner(scopeTags)) {
        throw new Error('File memory ownership is ambiguous; reset was not performed')
      }
    }
  }
  const retained = ['conversation history and current context', 'audit and retraction history', 'scheduled reminders', 'project state', 'other owners and shared memory', ...(!file ? ['shared global file memory'] : [])]
  if (options.dryRun) {
    if (!index.previewForgetOwnedMemory) throw new Error('Memory reset preview is unavailable; no deletion was performed')
    return { ...await index.previewForgetOwnedMemory(scopeTags), dryRun: true,
      fileMemoryWouldClear: Boolean(file), sections: file ? (await file.readMemorySections()).map((section) => section.title) : [],
      journalDates: file ? (await journalInventory(dirname(file.getMemoryPath()))).length : 0,
      retained, snapshotOnly: true }
  }
  const result = await index.forgetOwnedMemory(scopeTags)
  if (file) await file.clearAll(false)
  return { ...result, fileMemoryCleared: Boolean(file), retained }
}
