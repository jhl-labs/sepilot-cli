import { createHash } from 'node:crypto'
import type { WikiRepo } from './repo.js'
import {
  MAX_WIKI_BACKUP_BYTES,
  MAX_WIKI_BACKUP_NODES,
  MAX_WIKI_MARKDOWN_FILE_BYTES,
  MAX_WIKI_MARKDOWN_TOTAL_BYTES,
  WIKI_PORTABLE_FORMAT,
  WIKI_PORTABLE_FORMAT_VERSION,
  type WikiImportPreview,
  type WikiImportRequest,
  type WikiImportResult,
  type WikiFolderExport,
  type WikiNode,
  type WikiPortableBackup,
} from './schema.js'
import {
  exportWikiFolder,
  parseWikiFolder,
} from './folder-portable.js'
import {
  bodyFromMarkdown,
  titleFromMarkdown,
  validateMarkdownName,
} from './markdown-portable.js'

export { bodyFromMarkdown, titleFromMarkdown } from './markdown-portable.js'

export class WikiPortableError extends Error {
  readonly statusCode: number

  constructor(
    readonly code:
      | 'BACKUP_TOO_LARGE'
      | 'DUPLICATE_NODE'
      | 'INVALID_MARKDOWN_FILE'
      | 'MARKDOWN_TOO_LARGE'
      | 'MISSING_PARENT'
      | 'NODE_CYCLE'
      | 'PREVIEW_REQUIRED'
      | 'PREVIEW_STALE'
      | 'REPLACE_CONFIRMATION_REQUIRED',
    message: string,
    statusCode = 400,
  ) {
    super(message)
    this.name = 'WikiPortableError'
    this.statusCode = statusCode
  }
}

interface PreparedImport {
  nodes: WikiNode[]
  preview: WikiImportPreview
}

function importPreviewToken(
  request: WikiImportRequest,
  existing: readonly WikiNode[],
): string {
  return createHash('sha256')
    .update(JSON.stringify({
      strategy: request.strategy,
      source: request.source,
      existing,
    }))
    .digest('hex')
}

export interface WikiPortableService {
  exportBackup(): WikiPortableBackup
  exportFolder(): WikiFolderExport
  preview(request: WikiImportRequest): WikiImportPreview
  import(request: WikiImportRequest): WikiImportResult
}

function byteLength(value: string): number {
  return Buffer.byteLength(value, 'utf8')
}

function assertBackupSize(archive: WikiPortableBackup): void {
  if (byteLength(JSON.stringify(archive)) > MAX_WIKI_BACKUP_BYTES) {
    throw new WikiPortableError(
      'BACKUP_TOO_LARGE',
      `Wiki backup exceeds the ${MAX_WIKI_BACKUP_BYTES} byte limit`,
    )
  }
}

/** Validate the complete portable tree and return parents before descendants. */
function sortAndValidateGraph(nodes: readonly WikiNode[]): WikiNode[] {
  const byId = new Map<string, WikiNode>()
  for (const node of nodes) {
    if (byId.has(node.id)) {
      throw new WikiPortableError(
        'DUPLICATE_NODE',
        `Wiki backup contains duplicate node id: ${node.id}`,
      )
    }
    byId.set(node.id, node)
  }

  for (const node of nodes) {
    if (node.parentId !== null && !byId.has(node.parentId)) {
      throw new WikiPortableError(
        'MISSING_PARENT',
        `Wiki node ${node.id} references missing parent ${node.parentId}`,
      )
    }
  }

  const state = new Map<string, 'visiting' | 'visited'>()
  const ordered: WikiNode[] = []
  const visit = (node: WikiNode): void => {
    const current = state.get(node.id)
    if (current === 'visited') return
    if (current === 'visiting') {
      throw new WikiPortableError(
        'NODE_CYCLE',
        `Wiki backup contains a parent cycle involving node ${node.id}`,
      )
    }
    state.set(node.id, 'visiting')
    if (node.parentId !== null) visit(byId.get(node.parentId)!)
    state.set(node.id, 'visited')
    ordered.push(node)
  }
  for (const node of nodes) visit(node)
  return ordered
}

function uniqueId(usedIds: Set<string>): string {
  let id: string
  do {
    id = crypto.randomUUID()
  } while (usedIds.has(id))
  usedIds.add(id)
  return id
}

function copyBackupNodes(
  nodes: readonly WikiNode[],
  existing: readonly WikiNode[],
  targetParentId: string | null = null,
): WikiNode[] {
  const usedIds = new Set(existing.map((node) => node.id))
  const mappedIds = new Map(
    nodes.map((node) => [node.id, uniqueId(usedIds)] as const),
  )
  const roots = nodes.filter((node) => node.parentId === null)
  const minimumRootOrder = roots.length
    ? Math.min(...roots.map((node) => node.order))
    : 0
  const maximumExistingRootOrder = existing
    .filter((node) => node.parentId === targetParentId)
    .reduce((maximum, node) => Math.max(maximum, node.order), -1)
  const rootOrderOffset = maximumExistingRootOrder + 1 - minimumRootOrder

  return nodes.map((node) => ({
    ...node,
    id: mappedIds.get(node.id)!,
    parentId:
      node.parentId === null
        ? targetParentId
        : mappedIds.get(node.parentId)!,
    order:
      node.parentId === null ? node.order + rootOrderOffset : node.order,
  }))
}

function attachRootNodes(
  nodes: readonly WikiNode[],
  existing: readonly WikiNode[],
  targetParentId: string | null,
  appendRoots: boolean,
): WikiNode[] {
  if (targetParentId === null && !appendRoots) {
    return nodes.map((node) => ({ ...node }))
  }
  const roots = nodes.filter((node) => node.parentId === null)
  const minimumRootOrder = roots.length
    ? Math.min(...roots.map((node) => node.order))
    : 0
  const maximumExistingOrder = appendRoots
    ? existing
        .filter((node) => node.parentId === targetParentId)
        .reduce((maximum, node) => Math.max(maximum, node.order), -1)
    : -1
  const rootOrderOffset = maximumExistingOrder + 1 - minimumRootOrder
  return nodes.map((node) =>
    node.parentId === null
      ? {
          ...node,
          parentId: targetParentId,
          order: node.order + rootOrderOffset,
        }
      : { ...node },
  )
}

function markdownNodes(
  request: Extract<WikiImportRequest['source'], { type: 'markdown' }>,
  existing: readonly WikiNode[],
  strategy: WikiImportRequest['strategy'],
): WikiNode[] {
  let totalBytes = 0
  for (const file of request.files) {
    validateMarkdownName(file.name)
    const fileBytes = byteLength(file.content)
    if (fileBytes > MAX_WIKI_MARKDOWN_FILE_BYTES) {
      throw new WikiPortableError(
        'MARKDOWN_TOO_LARGE',
        `Markdown file ${file.name} exceeds the ${MAX_WIKI_MARKDOWN_FILE_BYTES} byte limit`,
      )
    }
    totalBytes += fileBytes
  }
  if (totalBytes > MAX_WIKI_MARKDOWN_TOTAL_BYTES) {
    throw new WikiPortableError(
      'MARKDOWN_TOO_LARGE',
      `Markdown files exceed the ${MAX_WIKI_MARKDOWN_TOTAL_BYTES} byte total limit`,
    )
  }

  const parentId = request.parentId ?? null
  if (parentId !== null && !existing.some((node) => node.id === parentId)) {
    throw new WikiPortableError(
      'MISSING_PARENT',
      `Markdown import parent does not exist: ${parentId}`,
    )
  }
  if (strategy === 'replace' && parentId !== null) {
    throw new WikiPortableError(
      'MISSING_PARENT',
      'Markdown replace import cannot target a parent that will be deleted',
    )
  }

  const usedIds = new Set(existing.map((node) => node.id))
  const firstOrder =
    strategy === 'replace'
      ? 0
      : existing
          .filter((node) => node.parentId === parentId)
          .reduce((maximum, node) => Math.max(maximum, node.order), -1) + 1
  const updatedAt = Date.now()
  return request.files.map((file, index) => ({
    id: uniqueId(usedIds),
    parentId,
    title: titleFromMarkdown(file.name, file.content),
    icon: null,
    group: null,
    order: firstOrder + index,
    body: bodyFromMarkdown(file.content),
    updatedAt,
  }))
}

function prepareImport(
  repo: WikiRepo,
  request: WikiImportRequest,
): PreparedImport {
  const existing = repo.tree()
  let nodes: WikiNode[]
  let folderHasManifest = false
  let folderExtraMarkdownCount = 0
  if (request.source.type === 'backup') {
    assertBackupSize(request.source.archive)
    const validated = sortAndValidateGraph(request.source.archive.nodes)
    nodes =
      request.strategy === 'copy'
        ? copyBackupNodes(validated, existing)
        : validated.map((node) => ({ ...node }))
  } else if (request.source.type === 'markdown') {
    nodes = markdownNodes(request.source, existing, request.strategy)
  } else {
    const targetParentId = request.source.parentId ?? null
    if (
      targetParentId !== null &&
      !existing.some((node) => node.id === targetParentId)
    ) {
      throw new WikiPortableError(
        'MISSING_PARENT',
        `Wiki folder import parent does not exist: ${targetParentId}`,
      )
    }
    if (request.strategy === 'replace' && targetParentId !== null) {
      throw new WikiPortableError(
        'MISSING_PARENT',
        'Wiki folder replace import cannot target a parent that will be deleted',
      )
    }

    const parsed = parseWikiFolder(
      request.source,
      existing.map((node) => node.id),
    )
    folderHasManifest = parsed.hasManifest
    folderExtraMarkdownCount = parsed.extraMarkdownCount
    if (
      request.strategy === 'merge' &&
      targetParentId !== null &&
      parsed.hasManifest &&
      parsed.nodes.some((node) => node.id === targetParentId)
    ) {
      throw new WikiPortableError(
        'NODE_CYCLE',
        'A manifest-backed Wiki folder cannot be merged below one of its own nodes',
      )
    }
    if (request.strategy === 'copy') {
      nodes = copyBackupNodes(parsed.nodes, existing, targetParentId)
    } else if (request.strategy === 'merge') {
      nodes = attachRootNodes(
        parsed.nodes,
        existing,
        targetParentId,
        targetParentId !== null || !parsed.hasManifest,
      )
    } else {
      nodes = parsed.nodes.map((node) => ({ ...node }))
    }
  }

  const existingById = new Map(existing.map((node) => [node.id, node]))
  const conflicts = nodes
    .filter((node) => existingById.has(node.id))
    .map((node) => ({
      id: node.id,
      title: node.title,
      reason: 'id_exists' as const,
    }))
  const isReplace = request.strategy === 'replace'
  const warnings: string[] = []
  if (request.strategy === 'copy' && request.source.type === 'backup') {
    warnings.push('Imported Wiki nodes will receive new IDs.')
  }
  if (
    request.strategy === 'copy' &&
    request.source.type === 'folder' &&
    folderHasManifest
  ) {
    warnings.push('Imported Wiki folder nodes will receive new IDs.')
  }
  if (request.source.type === 'folder' && !folderHasManifest) {
    warnings.push('Folder hierarchy will be imported as new Wiki nodes.')
  }
  if (
    request.source.type === 'folder' &&
    folderHasManifest &&
    folderExtraMarkdownCount > 0
  ) {
    warnings.push(
      `${folderExtraMarkdownCount} Markdown file(s) outside the manifest will be added.`,
    )
  }
  if (request.strategy === 'merge' && conflicts.length > 0) {
    warnings.push(`${conflicts.length} existing Wiki node(s) will be updated.`)
  }
  if (isReplace) {
    warnings.push(
      `All ${existing.length} existing Wiki node(s) will be deleted before import.`,
    )
  }

  return {
    nodes,
    preview: {
      previewToken: importPreviewToken(request, existing),
      strategy: request.strategy,
      sourceType: request.source.type,
      incomingCount: nodes.length,
      createCount: isReplace
        ? nodes.length
        : nodes.filter((node) => !existingById.has(node.id)).length,
      updateCount: isReplace ? 0 : conflicts.length,
      deleteCount: isReplace ? existing.length : 0,
      conflicts,
      warnings,
    },
  }
}

export function createWikiPortableService(
  repo: WikiRepo,
): WikiPortableService {
  return {
    exportBackup() {
      const backup: WikiPortableBackup = {
        format: WIKI_PORTABLE_FORMAT,
        formatVersion: WIKI_PORTABLE_FORMAT_VERSION,
        exportedAt: Date.now(),
        nodes: repo.tree(),
      }
      if (backup.nodes.length > MAX_WIKI_BACKUP_NODES) {
        throw new WikiPortableError(
          'BACKUP_TOO_LARGE',
          `Wiki backup exceeds the ${MAX_WIKI_BACKUP_NODES} node limit`,
        )
      }
      // Never offer a download that this same daemon would refuse to import.
      assertBackupSize(backup)
      return backup
    },
    exportFolder() {
      return exportWikiFolder(repo.tree())
    },
    preview(request) {
      return prepareImport(repo, request).preview
    },
    import(request) {
      if (request.strategy === 'replace' && !request.confirmed) {
        throw new WikiPortableError(
          'REPLACE_CONFIRMATION_REQUIRED',
          'Replacing the Wiki requires confirmed=true',
        )
      }
      const prepared = prepareImport(repo, request)
      if (!request.previewToken) {
        throw new WikiPortableError(
          'PREVIEW_REQUIRED',
          'Preview the Wiki import before applying it',
          409,
        )
      }
      if (request.previewToken !== prepared.preview.previewToken) {
        throw new WikiPortableError(
          'PREVIEW_STALE',
          'The Wiki changed after this import was previewed; preview it again',
          409,
        )
      }
      repo.importNodes(prepared.nodes, {
        replace: request.strategy === 'replace',
      })
      return {
        ...prepared.preview,
        importedNodeIds: prepared.nodes.map((node) => node.id),
      }
    },
  }
}
