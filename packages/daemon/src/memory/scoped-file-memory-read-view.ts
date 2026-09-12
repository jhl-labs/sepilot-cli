import type {
  FileMemory,
  FileMemoryPromptContext,
  FileMemorySection,
} from './file-memory.js'
import type { ScopedFileMemoryRegistry } from './scoped-file-memory.js'
import { canReadLegacyGlobalMemory } from './scope.js'

/**
 * Resolve the read view used by chat prompts and scoped memory UIs.
 *
 * Trusted first-party scoped callers may layer the legacy global MEMORY.md as
 * read-only context when the server grants that capability. Other scoped
 * callers see only their own bucket. Daily notes always come from the scoped
 * bucket because they contain automatic conversation digests. The returned
 * view must not be used as a write target; writes should resolve the scoped
 * bucket directly through ScopedFileMemoryRegistry.
 */
export function resolveScopedFileMemoryReadView(
  legacyGlobal: FileMemory | undefined,
  fileMemoryRegistry: ScopedFileMemoryRegistry | undefined,
  scopeTags: string[],
): FileMemoryReadView | undefined {
  if (scopeTags.length === 0 || !fileMemoryRegistry) return legacyGlobal
  const scoped = fileMemoryRegistry.get(scopeTags)
  const globalFallback = canReadLegacyGlobalMemory(scopeTags)
    ? legacyGlobal
    : undefined
  return new ScopedFileMemoryReadView(globalFallback, scoped)
}

export interface FileMemoryReadView {
  getMemoryPath(): string
  getDailyNotePath(date: Date): string
  readMemory(): Promise<string | undefined>
  readMemorySections(): Promise<FileMemorySection[]>
  readDailyNote(date?: Date): Promise<string | undefined>
  getPromptContext(now?: Date): Promise<FileMemoryPromptContext>
}

export class ScopedFileMemoryReadView implements FileMemoryReadView {
  constructor(
    private readonly legacyGlobal: FileMemory | undefined,
    private readonly scoped: FileMemory,
  ) {}

  getMemoryPath(): string {
    return this.scoped.getMemoryPath()
  }

  getDailyNotePath(date: Date): string {
    return this.scoped.getDailyNotePath(date)
  }

  async readMemory(): Promise<string | undefined> {
    if (!this.legacyGlobal) return this.scoped.readMemory()
    const [globalContent, scopedContent] = await Promise.all([
      this.legacyGlobal.readMemory(),
      this.scoped.readMemory(),
    ])
    return mergeLongTermMemory(globalContent, scopedContent)
  }

  async readMemorySections(): Promise<FileMemorySection[]> {
    if (!this.legacyGlobal) return this.scoped.readMemorySections()
    const [globalSections, scopedSections] = await Promise.all([
      this.legacyGlobal.readMemorySections(),
      this.scoped.readMemorySections(),
    ])
    return mergeMemorySections(globalSections, scopedSections)
  }

  async readDailyNote(date = new Date()): Promise<string | undefined> {
    return this.scoped.readDailyNote(date)
  }

  async getPromptContext(now = new Date()): Promise<FileMemoryPromptContext> {
    if (!this.legacyGlobal) return this.scoped.getPromptContext(now)
    const [globalContext, scopedContext] = await Promise.all([
      this.legacyGlobal.getPromptContext(now),
      this.scoped.getPromptContext(now),
    ])

    return {
      longTermMemory: mergePromptLongTermMemory(
        globalContext.longTermMemory,
        scopedContext.longTermMemory,
      ),
      todayNote: scopedContext.todayNote,
      yesterdayNote: scopedContext.yesterdayNote,
    }
  }
}

function mergeLongTermMemory(
  legacyGlobal: string | undefined,
  scoped: string | undefined,
): string | undefined {
  const globalContent = legacyGlobal?.trim()
  const scopedContent = scoped?.trim()
  if (!globalContent) return scopedContent
  if (!scopedContent || scopedContent === globalContent) return globalContent
  return `Scoped memory:\n${scopedContent}\n\n---\n\nLegacy global memory (read-only):\n${globalContent}`
}

const MERGED_PROMPT_LONG_TERM_MAX_CHARS = 4_000

function mergePromptLongTermMemory(
  legacyGlobal: string | undefined,
  scoped: string | undefined,
): string | undefined {
  const merged = mergeLongTermMemory(legacyGlobal, scoped)
  if (!merged || merged.length <= MERGED_PROMPT_LONG_TERM_MAX_CHARS) return merged
  return `${merged.slice(0, MERGED_PROMPT_LONG_TERM_MAX_CHARS - 15)}\n…[truncated]`
}

function mergeMemorySections(
  legacyGlobal: FileMemorySection[],
  scoped: FileMemorySection[],
): FileMemorySection[] {
  const merged = scoped.map((section) => ({ ...section }))
  const sectionIndexes = new Map(
    merged.map((section, index) => [section.title.trim().toLowerCase(), index]),
  )

  for (const section of legacyGlobal) {
    const key = section.title.trim().toLowerCase()
    const existingIndex = sectionIndexes.get(key)
    if (existingIndex === undefined) {
      sectionIndexes.set(key, merged.length)
      merged.push({ ...section })
      continue
    }
    merged[existingIndex] = {
      title: merged[existingIndex].title,
      content: `${merged[existingIndex].content.trim()}\n\n---\n\nLegacy global memory (read-only):\n${section.content.trim()}`,
    }
  }
  return merged
}
