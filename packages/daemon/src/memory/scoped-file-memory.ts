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
  get(scopeTags: string[] | undefined): FileMemory
  global(): FileMemory
  list(): Array<{ key: string; fileMemory: FileMemory; scopeTags?: string[] }>
  discover?(): Promise<void>
  deleteScope(scopeKey: string): Promise<boolean>
  renameScope(fromKey: string, toKey: string): Promise<boolean>
}

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
      this.cache.delete(from)
      return false
    }
    try {
      await stat(toDir)
      return false
    } catch {
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
