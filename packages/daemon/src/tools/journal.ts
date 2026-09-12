import { memoryResetOwner } from '../memory/reset.js'
import { dirname } from 'node:path'
import { inspectJournal, openJournalTasks, journalInventory, maintainJournals, setJournalTask, validateJournalDate } from '../memory/journal-lifecycle.js'
import { canonicalFileMemoryScopeKey, type ScopedFileMemoryRegistry } from '../memory/scoped-file-memory.js'
import type { ToolDefinitionRuntime } from './registry.js'

export function createJournalTools(registry: ScopedFileMemoryRegistry): ToolDefinitionRuntime[] {
  return [false, true].map((mutating): ToolDefinitionRuntime => ({
    name: mutating ? 'memory.journal.manage' : 'memory.journal.inspect',
    description: mutating
      ? 'Manage the caller-owned Activity Journal. action=archive losslessly compresses eligible old processed notes (default 30 days), preserving open tasks and unprocessed notes. action=complete or reopen changes one explicit checkbox task by date and taskId from inspect. Never infer completion from age. Does not execute the task or delete promoted memories.'
      : 'Inspect Activity Journal dates, archived state, checkbox task IDs, consolidation provenance and an archive preview. Supply date YYYY-MM-DD to inspect one episode, or state=open to page through unresolved tasks across all retained history. Use memory.daily.read/search to read original text including archived notes. Journals are episodic records, not automatically expiring facts.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: mutating ? 'sequential' : 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object', properties: {
        ...(mutating ? { action: { type: 'string', enum: ['archive', 'complete', 'reopen'] } } : { state: { type: 'string', enum: ['open'], description: 'List open tasks across all retained dates; excludes private sections and code examples.' } }),
        date: { type: 'string', description: 'Exact journal date YYYY-MM-DD.' },
        ...(mutating ? { taskId: { type: 'string', description: 'Exact ID returned by journal.inspect; required for complete/reopen.' } } : {}),
        archiveAfterDays: { type: 'integer', minimum: 2, maximum: 3650, description: 'Age threshold for this archive operation/preview, default 30; archive retains originals losslessly.' },
        offset: { type: 'integer', minimum: 0, description: 'Inventory offset, default 0.' },
      }, ...(mutating ? { required: ['action'] } : {}),
    },
    async execute(input, context) {
      const start = Date.now()
      try {
        const tags = context?.scopeTags ?? []
        if (canonicalFileMemoryScopeKey(tags) === 'global' && tags.some((tag) => tag.startsWith('scope:'))) throw new Error('This scope does not own a separate journal bucket')
        await registry.discover?.()
        const cached = registry.list().find((entry) => entry.key === canonicalFileMemoryScopeKey(tags))
        if (cached?.scopeTags && memoryResetOwner(cached.scopeTags) !== memoryResetOwner(tags)) throw new Error('Journal bucket ownership is ambiguous')
        const file = cached?.fileMemory ?? registry.get(tags)
        const root = dirname(file.getMemoryPath())
        const offset = typeof input.offset === 'number' && Number.isInteger(input.offset) && input.offset >= 0 ? input.offset : 0
        const days = input.archiveAfterDays === undefined ? undefined : Number(input.archiveAfterDays)
        let output: unknown
        if (!mutating && input.state !== undefined) {
          if (input.state !== 'open' || input.date !== undefined) throw new Error('state=open cannot be combined with date')
          const tasks = await openJournalTasks(root)
          output = { tasks: tasks.slice(offset, offset + 50), total: tasks.length, nextOffset: tasks.length > offset + 50 ? offset + 50 : null }
        } else if (mutating && input.action !== 'archive') {
          if (!['complete', 'reopen'].includes(String(input.action)) || typeof input.date !== 'string' || typeof input.taskId !== 'string') throw new Error('complete/reopen requires a date and taskId from inspect')
          await setJournalTask(root, input.date, input.taskId, input.action === 'complete')
          output = await inspectJournal(root, input.date)
        } else if (!mutating && input.date !== undefined) {
          if (typeof input.date !== 'string') throw new Error('Invalid date')
          validateJournalDate(input.date)
          output = await inspectJournal(root, input.date)
        } else {
          if (mutating && (input.date !== undefined || input.taskId !== undefined)) throw new Error('archive operates on eligible dates; do not combine with date/taskId')
          const inventory = await journalInventory(root)
          output = { dates: inventory.slice(offset, offset + 50), nextOffset: inventory.length > offset + 50 ? offset + 50 : null,
            maintenance: await maintainJournals(root, { archiveAfterDays: days, apply: mutating }) }
        }
        return { status: 'success', output: JSON.stringify(output), durationMs: Date.now() - start }
      } catch (error) {
        return { status: 'error', code: 'JOURNAL_OPERATION_FAILED', output: `Journal operation did not complete: ${String(error)}`, durationMs: Date.now() - start }
      }
    },
  }))
}
