import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { z } from 'zod'
import { createLogger } from '../logger.js'
import { isNodeFsError } from '../utils/fs-error.js'

const log = createLogger('extensions.registry')

export const EXTENSION_MANIFEST_SCOPES = [
  'inspect',
  'chat',
  'files',
  'ws',
  'approvals',
  'memory',
  'tools',
  'sessions',
  'rag',
] as const

export type ExtensionManifestScope =
  (typeof EXTENSION_MANIFEST_SCOPES)[number]

export const extensionManifestSchema = z.object({
  schemaVersion: z.literal(1),
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
  description: z.string().trim().min(1),
  mode: z.string().trim().min(1).optional(),
  token: z.object({
    scopes: z.array(z.enum(EXTENSION_MANIFEST_SCOPES)).min(1),
  }),
})

export type ExtensionManifest = z.infer<typeof extensionManifestSchema>

export interface InstalledExtensionRecord {
  manifest: ExtensionManifest
  enabled: boolean
  installedAt: string
  updatedAt: string
}

export interface ExtensionItem {
  id: string
  name: string
  enabled: boolean
  status: 'idle' | 'starting' | 'running' | 'crashed' | 'disabled'
  error: string | null
}

const persistedExtensionRecordSchema = z.object({
  manifest: extensionManifestSchema,
  enabled: z.boolean(),
  installedAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
})

const persistedRegistrySchema = z.object({
  version: z.literal(1),
  records: z.array(persistedExtensionRecordSchema),
})

type PersistedRegistry = z.infer<typeof persistedRegistrySchema>

export class ExtensionRegistry {
  private readonly filePath: string
  private readonly tempPath: string
  private loaded = false
  private records = new Map<string, InstalledExtensionRecord>()

  constructor(dataDir: string) {
    this.filePath = join(dataDir, 'extensions', 'registry.json')
    this.tempPath = `${this.filePath}.tmp`
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) {
      return
    }

    this.loaded = true

    let raw: string
    try {
      raw = await readFile(this.filePath, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return
      }
      log.error('failed to read extension registry', {
        path: this.filePath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }

    try {
      const parsed = persistedRegistrySchema.parse(JSON.parse(raw))
      this.records = new Map(
        parsed.records.map((record) => [record.manifest.id, record]),
      )
    } catch (err) {
      // Same rationale as extension-tokens / audit-logger: silently
      // booting empty would un-register every previously installed
      // extension with no operator-visible reason. Rotate the bad
      // file aside so the operator can inspect it and reinstall.
      const aside = `${this.filePath}.broken-${Date.now()}`
      log.error('extension registry unparseable; rotating aside', {
        path: this.filePath,
        rotated: aside,
        error: err instanceof Error ? err.message : String(err),
      })
      await rename(this.filePath, aside)
    }
  }

  private async persist(): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
    const payload: PersistedRegistry = {
      version: 1,
      records: Array.from(this.records.values()).sort((left, right) =>
        left.manifest.id.localeCompare(right.manifest.id),
      ),
    }
    await writeFile(this.tempPath, JSON.stringify(payload, null, 2), {
      mode: 0o600,
    })
    await rename(this.tempPath, this.filePath)
  }

  async list(): Promise<InstalledExtensionRecord[]> {
    await this.ensureLoaded()
    return Array.from(this.records.values()).sort((left, right) =>
      left.manifest.name.localeCompare(right.manifest.name),
    )
  }

  async get(id: string): Promise<InstalledExtensionRecord | null> {
    await this.ensureLoaded()
    return this.records.get(id) ?? null
  }

  async install(manifest: ExtensionManifest): Promise<InstalledExtensionRecord> {
    await this.ensureLoaded()
    const now = new Date().toISOString()
    const existing = this.records.get(manifest.id)
    const nextRecord: InstalledExtensionRecord = {
      manifest,
      enabled: existing?.enabled ?? true,
      installedAt: existing?.installedAt ?? now,
      updatedAt: now,
    }
    this.records.set(manifest.id, nextRecord)
    await this.persist()
    return nextRecord
  }

  async setEnabled(
    id: string,
    enabled: boolean,
  ): Promise<InstalledExtensionRecord | null> {
    await this.ensureLoaded()
    const existing = this.records.get(id)
    if (!existing) {
      return null
    }

    const nextRecord: InstalledExtensionRecord = {
      ...existing,
      enabled,
      updatedAt: new Date().toISOString(),
    }
    this.records.set(id, nextRecord)
    await this.persist()
    return nextRecord
  }

  async uninstall(id: string): Promise<boolean> {
    await this.ensureLoaded()
    if (!this.records.has(id)) {
      return false
    }
    this.records.delete(id)
    await this.persist()
    return true
  }
}

export function toExtensionItem(
  record: InstalledExtensionRecord,
): ExtensionItem {
  return {
    id: record.manifest.id,
    name: record.manifest.name,
    enabled: record.enabled,
    status: record.enabled ? 'idle' : 'disabled',
    error: null,
  }
}
