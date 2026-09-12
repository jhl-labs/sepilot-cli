import { z } from 'zod'

export const WikiNodeInput = z.object({
  id: z.string().optional(),
  parentId: z.string().nullable().optional(),
  title: z.string().min(1),
  icon: z.string().nullable().optional(),
  group: z.string().nullable().optional(),
  body: z.string().optional(),
})
export type WikiNodeInput = z.infer<typeof WikiNodeInput>

export const WikiMoveInput = z.object({
  parentId: z.string().nullable(),
  order: z.number().int().min(0),
})
export type WikiMoveInput = z.infer<typeof WikiMoveInput>

export interface WikiNode {
  id: string
  parentId: string | null
  title: string
  icon: string | null
  group: string | null
  order: number
  body: string
  updatedAt: number
}

export const WIKI_PORTABLE_FORMAT = 'sepilotd.wiki.backup' as const
export const WIKI_PORTABLE_FORMAT_VERSION = 1 as const
export const WIKI_FOLDER_FORMAT = 'sepilotd.wiki.folder' as const
export const WIKI_FOLDER_FORMAT_VERSION = 1 as const
export const WIKI_FOLDER_MANIFEST_PATH = '.sepilot-wiki.json' as const
export const MAX_WIKI_BACKUP_NODES = 10_000
export const MAX_WIKI_BACKUP_BYTES = 25 * 1024 * 1024
export const MAX_WIKI_MARKDOWN_FILES = 100
export const MAX_WIKI_MARKDOWN_FILE_BYTES = 2 * 1024 * 1024
export const MAX_WIKI_MARKDOWN_TOTAL_BYTES = 10 * 1024 * 1024
export const MAX_WIKI_FOLDER_ENTRIES = MAX_WIKI_BACKUP_NODES * 2 + 1
export const MAX_WIKI_FOLDER_BYTES = 50 * 1024 * 1024
export const MAX_WIKI_FOLDER_PATH_LENGTH = 1_024
export const MAX_WIKI_FOLDER_DEPTH = 32

export const WikiPortableNode = z
  .object({
    // Portable backups must be able to round-trip every value accepted by
    // the canonical Wiki API. The request body limit, node-count limit, and
    // total backup byte limit provide the resource bounds here.
    id: z.string().min(1),
    parentId: z.string().min(1).nullable(),
    title: z.string().min(1),
    icon: z.string().nullable(),
    group: z.string().nullable(),
    order: z.number().int().min(0),
    body: z.string(),
    updatedAt: z.number().int().min(0),
  })
  .strict()
export type WikiPortableNode = z.infer<typeof WikiPortableNode>

export const WikiPortableBackup = z
  .object({
    format: z.literal(WIKI_PORTABLE_FORMAT),
    formatVersion: z.literal(WIKI_PORTABLE_FORMAT_VERSION),
    exportedAt: z.number().int().min(0),
    nodes: z.array(WikiPortableNode).max(MAX_WIKI_BACKUP_NODES),
  })
  .strict()
export type WikiPortableBackup = z.infer<typeof WikiPortableBackup>

export const WikiMarkdownFile = z
  .object({
    name: z.string().min(1).max(255),
    content: z.string(),
  })
  .strict()
export type WikiMarkdownFile = z.infer<typeof WikiMarkdownFile>

export const WikiFolderEntry = z.discriminatedUnion('kind', [
  z
    .object({
      kind: z.literal('directory'),
      relativePath: z.string().min(1).max(MAX_WIKI_FOLDER_PATH_LENGTH),
    })
    .strict(),
  z
    .object({
      kind: z.literal('file'),
      relativePath: z.string().min(1).max(MAX_WIKI_FOLDER_PATH_LENGTH),
      content: z.string(),
    })
    .strict(),
])
export type WikiFolderEntry = z.infer<typeof WikiFolderEntry>

export const WikiFolderManifestNode = WikiPortableNode.omit({ body: true })
  .extend({
    relativePath: z.string().min(1).max(MAX_WIKI_FOLDER_PATH_LENGTH),
    contentSha256: z.string().regex(/^[a-f0-9]{64}$/u),
  })
  .strict()
export type WikiFolderManifestNode = z.infer<typeof WikiFolderManifestNode>

export const WikiFolderManifest = z
  .object({
    format: z.literal(WIKI_FOLDER_FORMAT),
    formatVersion: z.literal(WIKI_FOLDER_FORMAT_VERSION),
    exportedAt: z.number().int().min(0),
    nodes: z.array(WikiFolderManifestNode).max(MAX_WIKI_BACKUP_NODES),
  })
  .strict()
export type WikiFolderManifest = z.infer<typeof WikiFolderManifest>

export interface WikiFolderExport {
  directoryName: string
  entries: WikiFolderEntry[]
}

export const WikiImportStrategy = z.enum(['copy', 'merge', 'replace'])
export type WikiImportStrategy = z.infer<typeof WikiImportStrategy>

export const WikiImportRequest = z
  .object({
    source: z.discriminatedUnion('type', [
      z
        .object({
          type: z.literal('backup'),
          archive: WikiPortableBackup,
        })
        .strict(),
      z
        .object({
          type: z.literal('markdown'),
          files: z
            .array(WikiMarkdownFile)
            .min(1)
            .max(MAX_WIKI_MARKDOWN_FILES),
          parentId: z.string().min(1).nullable().optional(),
        })
        .strict(),
      z
        .object({
          type: z.literal('folder'),
          rootName: z.string().trim().min(1).max(255),
          entries: z
            .array(WikiFolderEntry)
            .max(MAX_WIKI_FOLDER_ENTRIES),
          parentId: z.string().min(1).nullable().optional(),
        })
        .strict(),
    ]),
    strategy: WikiImportStrategy.default('copy'),
    confirmed: z.boolean().default(false),
    previewToken: z.string().regex(/^[a-f0-9]{64}$/u).optional(),
  })
  .strict()
export type WikiImportRequest = z.infer<typeof WikiImportRequest>

export interface WikiImportConflict {
  id: string
  title: string
  reason: 'id_exists'
}

export interface WikiImportPreview {
  /** Digest of both the import source and the current Wiki revision. */
  previewToken: string
  strategy: WikiImportStrategy
  sourceType: 'backup' | 'markdown' | 'folder'
  incomingCount: number
  createCount: number
  updateCount: number
  deleteCount: number
  conflicts: WikiImportConflict[]
  warnings: string[]
}

export interface WikiImportResult extends WikiImportPreview {
  importedNodeIds: string[]
}
