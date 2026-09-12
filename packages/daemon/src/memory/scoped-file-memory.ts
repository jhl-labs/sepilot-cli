import { readFile, readdir, rename, rm, stat } from 'node:fs/promises'
import { mkdir } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { FileMemory, type FileMemoryOptions } from './file-memory.js'
import { parseScopeFromTags, type MemoryScope } from './scope.js'

const GLOBAL_KEY = 'global'
const BUCKET_KEY_PATTERN = /^[a-z0-9][a-z0-9._-]{0,199}$/

function isValidBucketKey(key: string): boolean {
  if (!key || key === GLOBAL_KEY) return false
  if (key.includes('..') || key.includes('/') || key.includes('\\')) return false
  return BUCKET_KEY_PATTERN.test(key)
}

export interface ScopedFileMemoryRegistry {
  /** Returns the FileMemory bucket the caller's scope writes to. */
  get(scopeTags: string[] | undefined): FileMemory
  /** Returns the global (no-scope) FileMemory bucket. */
  global(): FileMemory
  /** All cached buckets — used by maintenance/export operations. */
  list(): Array<{ key: string; fileMemory: FileMemory; scopeTags?: string[] }>
  /** Discover existing owner buckets after a daemon restart. */
  discover?(): Promise<void>
  /** Remove an entire scoped bucket (the directory + the cache entry).
   *  Refuses to delete the global bucket. Returns true when the
   *  directory was removed. */
  deleteScope(scopeKey: string): Promise<boolean>
  /** Rename a scoped bucket directory. Refuses to touch the global
   *  bucket on either side. Returns true when the rename succeeded. */
  renameScope(fromKey: string, toKey: string): Promise<boolean>
}

/**
 * Decide which scope owns the file-memory bucket for a given tag list.
 *
 * Priority is most-specific first:
 *   user > channel > group > session > global
 *
 * - `user` and `channel` are the common cases (a person, a chat room).
 * - `group` is the "shared team notes" case: a caller scoped purely to
 *   `scope:group:<id>` (no user/channel of their own) reads and writes
 *   one shared MEMORY.md / daily-note bucket. When the caller also has
 *   a user or channel identity, that wins — alice's personal notes
 *   don't bleed into the team bucket just because she's on the team.
 *   If the caller is in several groups, the first (canonically sorted)
 *   group id is chosen so the bucket is deterministic.
 * - `session` is intentionally not its own bucket — sessions are
 *   short-lived and forking would balloon the memory directory.
 *   Session-scoped writes fall back to user / channel / group / global.
 */
export function canonicalFileMemoryScopeKey(
  scopeTags: string[] | undefined,
): string {
  const scope = parseScopeFromTags(scopeTags)
  return canonicalKey(scope)
}

function canonicalKey(scope: MemoryScope): string {
  if (scope.userId) return `user-${sanitizeIdSegment(scope.userId)}`
  if (scope.channelType && scope.chatId) {
    return `channel-${sanitizeIdSegment(scope.channelType)}-${sanitizeIdSegment(scope.chatId)}`
  }
  if (scope.groupIds && scope.groupIds.length > 0) {
    const first = [...scope.groupIds].map((id) => id.trim()).filter(Boolean).sort()[0]
    if (first) return `group-${sanitizeIdSegment(first)}`
  }
  return GLOBAL_KEY
}

function sanitizeIdSegment(value: string): string {
  // Lowercase + replace anything outside POSIX-safe filename chars with '_'.
  return value.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '')
    || 'unknown'
}

export class ScopedFileMemoryRegistryImpl implements ScopedFileMemoryRegistry {
  private changeListener?: (key: string, before: string, after: string) => Promise<void>
  private readListener?: () => Promise<void>

  setProjectionHooks(change: (key: string, before: string, after: string) => Promise<void>, read: () => Promise<void>): void {
    this.changeListener = change
    this.readListener = read
  }

  private readonly ownerScopes = new Map<string, string[]>()
  private readonly cache = new Map<string, FileMemory>()

  constructor(
    private readonly rootDir: string,
    private readonly options: FileMemoryOptions = {},
  ) {}

  get(scopeTags: string[] | undefined): FileMemory {
    const key = canonicalFileMemoryScopeKey(scopeTags)
    const file = this.getByKey(key)
    const tags = (scopeTags ?? []).filter((tag) => tag.startsWith('scope:') && !tag.startsWith('scope:session:'))
    this.ownerScopes.set(key, tags)
    file.setOwnerScopeTags(tags)
    return file
  }

  global(): FileMemory {
    return this.getByKey(GLOBAL_KEY)
  }

  list(): Array<{ key: string; fileMemory: FileMemory; scopeTags?: string[] }> {
    return Array.from(this.cache.entries()).map(([key, fileMemory]) => ({ key, fileMemory, scopeTags: key === GLOBAL_KEY ? [] : this.ownerScopes.get(key) }))
  }

  async discover(): Promise<void> {
    const entries = await readdir(join(this.rootDir, 'scopes'), { withFileTypes: true })
      .catch((error: NodeJS.ErrnoException) => {
        if (error.code === 'ENOENT') return []
        throw error
      })
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      // Never follow symlinks or interpret arbitrary directory names as owners.
      if (!entry.isDirectory() || !isValidBucketKey(entry.name)) continue
      const file = this.getByKey(entry.name)
      try {
        const tags: unknown = JSON.parse(await readFile(join(this.rootDir, 'scopes', entry.name, '.scope.json'), 'utf8'))
        if (Array.isArray(tags) && tags.every((tag) => typeof tag === 'string') && canonicalFileMemoryScopeKey(tags) === entry.name) {
          this.ownerScopes.set(entry.name, tags)
          file.setOwnerScopeTags(tags)
        }
      } catch (error) {
        if (!(error instanceof SyntaxError) && (error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
      }
    }
  }

  async renameScope(fromKey: string, toKey: string): Promise<boolean> {
    const from = fromKey.trim()
    const to = toKey.trim()
    if (!isValidBucketKey(from) || !isValidBucketKey(to)) return false
    if (from === to) return false
    const scopesRoot = resolve(join(this.rootDir, 'scopes'))
    const fromDir = resolve(join(scopesRoot, from))
    const toDir = resolve(join(scopesRoot, to))
    if (!fromDir.startsWith(`${scopesRoot}/`) || !toDir.startsWith(`${scopesRoot}/`)) {
      return false
    }
    try {
      await stat(fromDir)
    } catch {
      // Source doesn't exist — nothing to move; evict cache entry just in case.
      this.cache.delete(from)
      return false
    }
    // Refuse to clobber an existing target bucket.
    try {
      await stat(toDir)
      return false
    } catch {
      // target missing, good
    }
    try {
      await mkdir(join(this.rootDir, 'scopes'), { recursive: true })
      await rename(fromDir, toDir)
    } catch {
      return false
    }
    this.cache.delete(from)
    this.cache.delete(to)
    return true
  }

  async deleteScope(scopeKey: string): Promise<boolean> {
    const key = scopeKey.trim()
    if (!isValidBucketKey(key)) return false
    const scopesRoot = resolve(join(this.rootDir, 'scopes'))
    const dir = resolve(join(scopesRoot, key))
    if (!dir.startsWith(`${scopesRoot}/`)) return false
    try {
      await stat(dir)
    } catch {
      // Directory doesn't exist — treat as a no-op delete.
      this.cache.delete(key)
      return false
    }
    try {
      await rm(dir, { recursive: true, force: true })
    } catch {
      return false
    }
    this.cache.delete(key)
    return true
  }

  private getByKey(key: string): FileMemory {
    const cached = this.cache.get(key)
    if (cached) return cached
    const dir = key === GLOBAL_KEY ? this.rootDir : join(this.rootDir, 'scopes', key)
    const fresh = new FileMemory(dir, { ...this.options, onLongTermChange: (before, after) => this.changeListener?.(key, before, after) ?? Promise.resolve(), beforePromptRead: () => this.readListener?.() ?? Promise.resolve() })
    this.cache.set(key, fresh)
    return fresh
  }
}
