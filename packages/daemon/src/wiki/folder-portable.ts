import { createHash, randomUUID } from 'node:crypto'
import { posix } from 'node:path'
import {
  MAX_WIKI_BACKUP_NODES,
  MAX_WIKI_FOLDER_BYTES,
  MAX_WIKI_FOLDER_DEPTH,
  MAX_WIKI_FOLDER_ENTRIES,
  MAX_WIKI_FOLDER_PATH_LENGTH,
  MAX_WIKI_MARKDOWN_FILE_BYTES,
  WIKI_FOLDER_FORMAT,
  WIKI_FOLDER_FORMAT_VERSION,
  WIKI_FOLDER_MANIFEST_PATH,
  WikiFolderManifest,
  type WikiFolderEntry,
  type WikiFolderExport,
  type WikiFolderManifestNode,
  type WikiImportRequest,
  type WikiNode,
} from './schema.js'
import {
  bodyFromMarkdown,
  headingFromMarkdown,
  isMarkdownPath,
  portableMarkdownHeading,
  renderWikiMarkdown,
  titleFromMarkdown,
} from './markdown-portable.js'

type FolderSource = Extract<
  WikiImportRequest['source'],
  { type: 'folder' }
>

type FolderErrorCode =
  | 'DUPLICATE_FOLDER_ENTRY'
  | 'FOLDER_TOO_LARGE'
  | 'INVALID_FOLDER_ENTRY'
  | 'INVALID_FOLDER_MANIFEST'
  | 'MISSING_FOLDER_FILE'
  | 'MISSING_PARENT'
  | 'NODE_CYCLE'

export class WikiFolderPortableError extends Error {
  readonly statusCode: number

  constructor(
    readonly code: FolderErrorCode,
    message: string,
    statusCode = 400,
  ) {
    super(message)
    this.name = 'WikiFolderPortableError'
    this.statusCode = statusCode
  }
}

export interface ParsedWikiFolder {
  nodes: WikiNode[]
  hasManifest: boolean
  extraMarkdownCount: number
}

interface NormalizedFileEntry {
  relativePath: string
  content: string
}

interface NormalizedFolderEntries {
  directories: Set<string>
  files: Map<string, NormalizedFileEntry>
}

const WINDOWS_RESERVED_NAME_RE = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/iu
const INVALID_PORTABLE_SEGMENT_RE = /[<>:"/\\|?*\u0000-\u001f]/u
const INVALID_PORTABLE_SEGMENT_GLOBAL_RE = /[<>:"/\\|?*\u0000-\u001f]/gu
const INDEX_MARKDOWN_RE = /^index\.(?:md|markdown|mdx)$/iu

function byteLength(value: string): number {
  return Buffer.byteLength(value, 'utf8')
}

function sha256(value: string): string {
  return createHash('sha256').update(value).digest('hex')
}

function canonicalPathKey(relativePath: string): string {
  return relativePath.normalize('NFC').toLowerCase()
}

function pathDepth(relativePath: string): number {
  return relativePath.split('/').length
}

function normalizeRelativePath(relativePath: string): string {
  if (
    relativePath.startsWith('/') ||
    relativePath.startsWith('\\') ||
    /^[a-z]:/iu.test(relativePath) ||
    relativePath.includes('\\') ||
    relativePath.includes('\0')
  ) {
    throw new WikiFolderPortableError(
      'INVALID_FOLDER_ENTRY',
      `Folder entry must use a safe relative path: ${relativePath}`,
    )
  }

  const normalized = relativePath.normalize('NFC')
  const segments = normalized.split('/')
  if (
    normalized.length > MAX_WIKI_FOLDER_PATH_LENGTH ||
    byteLength(normalized) > MAX_WIKI_FOLDER_PATH_LENGTH ||
    segments.length > MAX_WIKI_FOLDER_DEPTH ||
    segments.some(
      (segment) =>
        !segment ||
        segment === '.' ||
        segment === '..' ||
        segment.endsWith('.') ||
        segment.endsWith(' ') ||
        byteLength(segment) > 255 ||
        INVALID_PORTABLE_SEGMENT_RE.test(segment) ||
        WINDOWS_RESERVED_NAME_RE.test(segment),
    )
  ) {
    throw new WikiFolderPortableError(
      'INVALID_FOLDER_ENTRY',
      `Folder entry contains an unsafe or overlong path: ${relativePath}`,
    )
  }
  return normalized
}

function parentDirectories(relativePath: string): string[] {
  const segments = relativePath.split('/')
  const directories: string[] = []
  for (let index = 1; index < segments.length; index += 1) {
    directories.push(segments.slice(0, index).join('/'))
  }
  return directories
}

function assertFolderSize(source: FolderSource): void {
  if (source.entries.length > MAX_WIKI_FOLDER_ENTRIES) {
    throw new WikiFolderPortableError(
      'FOLDER_TOO_LARGE',
      `Wiki folder exceeds the ${MAX_WIKI_FOLDER_ENTRIES} entry limit`,
      413,
    )
  }
  if (byteLength(JSON.stringify(source)) > MAX_WIKI_FOLDER_BYTES) {
    throw new WikiFolderPortableError(
      'FOLDER_TOO_LARGE',
      `Wiki folder exceeds the ${MAX_WIKI_FOLDER_BYTES} byte limit`,
      413,
    )
  }
}

function normalizeEntries(source: FolderSource): NormalizedFolderEntries {
  assertFolderSize(source)
  const directories = new Set<string>()
  const files = new Map<string, NormalizedFileEntry>()
  const entryKinds = new Map<string, WikiFolderEntry['kind']>()

  for (const entry of source.entries) {
    const relativePath = normalizeRelativePath(entry.relativePath)
    const key = canonicalPathKey(relativePath)
    const previousKind = entryKinds.get(key)
    if (previousKind) {
      throw new WikiFolderPortableError(
        'DUPLICATE_FOLDER_ENTRY',
        `Wiki folder contains duplicate path ${relativePath}`,
      )
    }
    entryKinds.set(key, entry.kind)

    if (entry.kind === 'directory') {
      if (key === canonicalPathKey(WIKI_FOLDER_MANIFEST_PATH)) {
        throw new WikiFolderPortableError(
          'INVALID_FOLDER_ENTRY',
          `${WIKI_FOLDER_MANIFEST_PATH} must be a file`,
        )
      }
      directories.add(relativePath)
      continue
    }

    if (
      relativePath !== WIKI_FOLDER_MANIFEST_PATH &&
      !isMarkdownPath(relativePath)
    ) {
      throw new WikiFolderPortableError(
        'INVALID_FOLDER_ENTRY',
        `Unsupported Wiki folder file: ${relativePath}`,
      )
    }
    if (
      relativePath !== WIKI_FOLDER_MANIFEST_PATH &&
      byteLength(entry.content) > MAX_WIKI_MARKDOWN_FILE_BYTES
    ) {
      throw new WikiFolderPortableError(
        'FOLDER_TOO_LARGE',
        `Markdown file ${relativePath} exceeds the ${MAX_WIKI_MARKDOWN_FILE_BYTES} byte limit`,
        413,
      )
    }
    files.set(relativePath, { relativePath, content: entry.content })
  }

  const fileKeys = new Set(
    [...files.keys()].map((relativePath) => canonicalPathKey(relativePath)),
  )
  for (const relativePath of [...directories, ...files.keys()]) {
    for (const parent of parentDirectories(relativePath)) {
      if (fileKeys.has(canonicalPathKey(parent))) {
        throw new WikiFolderPortableError(
          'INVALID_FOLDER_ENTRY',
          `Folder entry is nested below a file: ${relativePath}`,
        )
      }
    }
  }
  return { directories, files }
}

function uniqueId(usedIds: Set<string>): string {
  let id: string
  do {
    id = randomUUID()
  } while (usedIds.has(id))
  usedIds.add(id)
  return id
}

function sortAndValidateNodes(
  nodes: readonly WikiNode[],
  errorCode: 'INVALID_FOLDER_MANIFEST' | 'MISSING_PARENT' =
    'INVALID_FOLDER_MANIFEST',
): WikiNode[] {
  const byId = new Map<string, WikiNode>()
  for (const node of nodes) {
    if (byId.has(node.id)) {
      throw new WikiFolderPortableError(
        'INVALID_FOLDER_MANIFEST',
        `Wiki folder manifest contains duplicate node id: ${node.id}`,
      )
    }
    byId.set(node.id, node)
  }
  for (const node of nodes) {
    if (node.parentId !== null && !byId.has(node.parentId)) {
      throw new WikiFolderPortableError(
        errorCode,
        `Wiki folder node ${node.id} references missing parent ${node.parentId}`,
      )
    }
  }

  const state = new Map<string, 'visiting' | 'visited'>()
  const ordered: WikiNode[] = []
  const visit = (node: WikiNode): void => {
    const current = state.get(node.id)
    if (current === 'visited') return
    if (current === 'visiting') {
      throw new WikiFolderPortableError(
        'NODE_CYCLE',
        `Wiki folder contains a parent cycle involving node ${node.id}`,
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

function safePortableStem(title: string, id: string): string {
  const cleaned = portableMarkdownHeading(title)
    .normalize('NFKC')
    .replace(INVALID_PORTABLE_SEGMENT_GLOBAL_RE, '-')
    .replace(/[. ]+$/gu, '')
    .trim()
  const portable =
    !cleaned || WINDOWS_RESERVED_NAME_RE.test(cleaned) ? 'document' : cleaned
  const bounded = Array.from(portable).slice(0, 72).join('')
  return `${bounded}--${sha256(id).slice(0, 12)}`
}

function sortedChildren(
  nodes: readonly WikiNode[],
): Map<string | null, WikiNode[]> {
  const children = new Map<string | null, WikiNode[]>()
  for (const node of nodes) {
    const siblings = children.get(node.parentId) ?? []
    siblings.push(node)
    children.set(node.parentId, siblings)
  }
  for (const siblings of children.values()) {
    siblings.sort(
      (left, right) => left.order - right.order || left.id.localeCompare(right.id),
    )
  }
  return children
}

export function exportWikiFolder(
  nodes: readonly WikiNode[],
  exportedAt = Date.now(),
): WikiFolderExport {
  const ordered = sortAndValidateNodes(nodes, 'MISSING_PARENT')
  const children = sortedChildren(ordered)
  const fileEntries: WikiFolderEntry[] = []
  const directoryPaths = new Set<string>()
  const manifestNodes: WikiFolderManifestNode[] = []
  const usedPaths = new Set<string>()

  const writeNode = (node: WikiNode, parentDirectory: string): void => {
    const stem = safePortableStem(node.title, node.id)
    const hasChildren = (children.get(node.id)?.length ?? 0) > 0
    const nodeDirectory = parentDirectory
      ? `${parentDirectory}/${stem}`
      : stem
    const relativePath = normalizeRelativePath(
      hasChildren
        ? `${nodeDirectory}/index.md`
        : parentDirectory
          ? `${parentDirectory}/${stem}.md`
          : `${stem}.md`,
    )
    if (hasChildren) directoryPaths.add(nodeDirectory)
    const pathKey = canonicalPathKey(relativePath)
    if (usedPaths.has(pathKey)) {
      throw new WikiFolderPortableError(
        'DUPLICATE_FOLDER_ENTRY',
        `Wiki nodes produce duplicate folder path ${relativePath}`,
      )
    }
    usedPaths.add(pathKey)

    const content = renderWikiMarkdown(node.title, node.body)
    if (byteLength(content) > MAX_WIKI_MARKDOWN_FILE_BYTES) {
      throw new WikiFolderPortableError(
        'FOLDER_TOO_LARGE',
        `Wiki node ${node.id} exceeds the ${MAX_WIKI_MARKDOWN_FILE_BYTES} byte Markdown file limit`,
        413,
      )
    }
    fileEntries.push({ kind: 'file', relativePath, content })
    manifestNodes.push({
      id: node.id,
      parentId: node.parentId,
      title: node.title,
      icon: node.icon,
      group: node.group,
      order: node.order,
      updatedAt: node.updatedAt,
      relativePath,
      contentSha256: sha256(content),
    })
    for (const child of children.get(node.id) ?? []) {
      writeNode(child, nodeDirectory)
    }
  }

  for (const root of children.get(null) ?? []) writeNode(root, '')

  const manifest = {
    format: WIKI_FOLDER_FORMAT,
    formatVersion: WIKI_FOLDER_FORMAT_VERSION,
    exportedAt,
    nodes: manifestNodes,
  } satisfies WikiFolderManifest
  const entries: WikiFolderEntry[] = [
    ...[...directoryPaths]
      .sort(
        (left, right) =>
          pathDepth(left) - pathDepth(right) ||
          left.localeCompare(right, 'en'),
      )
      .map((relativePath) => ({
        kind: 'directory' as const,
        relativePath,
      })),
    {
      kind: 'file',
      relativePath: WIKI_FOLDER_MANIFEST_PATH,
      content: `${JSON.stringify(manifest, null, 2)}\n`,
    },
    ...fileEntries,
  ]

  const directoryName = `sepilot-wiki-${new Date(exportedAt)
    .toISOString()
    .slice(0, 10)}`
  const sourceLike = {
    type: 'folder' as const,
    rootName: directoryName,
    entries,
  }
  assertFolderSize(sourceLike)
  return { directoryName, entries }
}

function parseManifest(
  entries: NormalizedFolderEntries,
): WikiFolderManifest | null {
  const entry = entries.files.get(WIKI_FOLDER_MANIFEST_PATH)
  if (!entry) return null
  let raw: unknown
  try {
    raw = JSON.parse(entry.content)
  } catch {
    throw new WikiFolderPortableError(
      'INVALID_FOLDER_MANIFEST',
      `${WIKI_FOLDER_MANIFEST_PATH} is not valid JSON`,
    )
  }
  const parsed = WikiFolderManifest.safeParse(raw)
  if (!parsed.success) {
    throw new WikiFolderPortableError(
      'INVALID_FOLDER_MANIFEST',
      parsed.error.issues[0]?.message ?? 'Invalid Wiki folder manifest',
    )
  }
  return parsed.data
}

function validateManifestTopology(
  manifest: WikiFolderManifest,
  normalizedPaths: ReadonlyMap<string, string>,
): void {
  const byId = new Map(manifest.nodes.map((node) => [node.id, node]))
  const childCounts = new Map<string, number>()
  for (const node of manifest.nodes) {
    if (node.parentId !== null) {
      childCounts.set(node.parentId, (childCounts.get(node.parentId) ?? 0) + 1)
    }
  }

  for (const node of manifest.nodes) {
    const relativePath = normalizedPaths.get(node.id)!
    const hasChildren = (childCounts.get(node.id) ?? 0) > 0
    if (hasChildren && posix.basename(relativePath).toLowerCase() !== 'index.md') {
      throw new WikiFolderPortableError(
        'INVALID_FOLDER_MANIFEST',
        `Wiki folder parent node ${node.id} must use an index.md file`,
      )
    }
    const physicalParent = hasChildren
      ? posix.dirname(posix.dirname(relativePath))
      : posix.dirname(relativePath)
    const expectedParent =
      node.parentId === null
        ? '.'
        : posix.dirname(normalizedPaths.get(node.parentId)!)
    if (physicalParent !== expectedParent) {
      throw new WikiFolderPortableError(
        'INVALID_FOLDER_MANIFEST',
        `Wiki folder path does not match node hierarchy: ${relativePath}`,
      )
    }
    if (node.parentId !== null && !byId.has(node.parentId)) {
      throw new WikiFolderPortableError(
        'INVALID_FOLDER_MANIFEST',
        `Wiki folder node ${node.id} references missing parent ${node.parentId}`,
      )
    }
  }
}

function nodesFromManifest(
  manifest: WikiFolderManifest,
  entries: NormalizedFolderEntries,
  now: number,
): {
  nodes: WikiNode[]
  referencedPaths: Set<string>
  directoryAnchors: Map<string, string>
} {
  const normalizedPaths = new Map<string, string>()
  const referencedPaths = new Set<string>()
  const pathOwners = new Map<string, string>()

  for (const node of manifest.nodes) {
    const relativePath = normalizeRelativePath(node.relativePath)
    if (!isMarkdownPath(relativePath)) {
      throw new WikiFolderPortableError(
        'INVALID_FOLDER_MANIFEST',
        `Wiki folder manifest references a non-Markdown file: ${relativePath}`,
      )
    }
    const key = canonicalPathKey(relativePath)
    const previousOwner = pathOwners.get(key)
    if (previousOwner) {
      throw new WikiFolderPortableError(
        'INVALID_FOLDER_MANIFEST',
        `Wiki folder manifest reuses path ${relativePath}`,
      )
    }
    pathOwners.set(key, node.id)
    normalizedPaths.set(node.id, relativePath)
    referencedPaths.add(key)
    if (!entries.files.has(relativePath)) {
      throw new WikiFolderPortableError(
        'MISSING_FOLDER_FILE',
        `Wiki folder is missing manifest file ${relativePath}`,
      )
    }
  }

  const graphNodes = manifest.nodes.map((node) => ({
    id: node.id,
    parentId: node.parentId,
    title: node.title,
    icon: node.icon,
    group: node.group,
    order: node.order,
    body: '',
    updatedAt: node.updatedAt,
  }))
  sortAndValidateNodes(graphNodes)
  validateManifestTopology(manifest, normalizedPaths)

  const nodes = manifest.nodes.map((node) => {
    const relativePath = normalizedPaths.get(node.id)!
    const file = entries.files.get(relativePath)!
    const currentHash = sha256(file.content)
    const changed = currentHash !== node.contentSha256
    const editedHeading = changed ? headingFromMarkdown(file.content) : null
    return {
      id: node.id,
      parentId: node.parentId,
      title: editedHeading?.slice(0, 500) || node.title,
      icon: node.icon,
      group: node.group,
      order: node.order,
      body: bodyFromMarkdown(file.content),
      updatedAt: changed ? now : node.updatedAt,
    }
  })
  const ordered = sortAndValidateNodes(nodes)
  const directoryAnchors = new Map<string, string>()
  for (const node of manifest.nodes) {
    const relativePath = normalizedPaths.get(node.id)!
    if (posix.basename(relativePath).toLowerCase() === 'index.md') {
      directoryAnchors.set(posix.dirname(relativePath), node.id)
    }
  }
  return { nodes: ordered, referencedPaths, directoryAnchors }
}

function indexFileForDirectory(
  directory: string,
  files: readonly NormalizedFileEntry[],
): NormalizedFileEntry | null {
  const matches = files.filter((file) => {
    const fileDirectory = posix.dirname(file.relativePath)
    return (
      fileDirectory === (directory || '.') &&
      INDEX_MARKDOWN_RE.test(posix.basename(file.relativePath))
    )
  })
  if (matches.length > 1) {
    throw new WikiFolderPortableError(
      'INVALID_FOLDER_ENTRY',
      `Wiki folder has multiple index Markdown files in ${directory || '.'}`,
    )
  }
  return matches[0] ?? null
}

function extraFolderNodes(
  source: FolderSource,
  entries: NormalizedFolderEntries,
  options: {
    baseNodes: readonly WikiNode[]
    directoryAnchors: ReadonlyMap<string, string>
    referencedPaths: ReadonlySet<string>
    usedIds: Set<string>
    wrapRoot: boolean
    now: number
  },
): { nodes: WikiNode[]; extraMarkdownCount: number } {
  const extraFiles = [...entries.files.values()]
    .filter(
      (file) =>
        file.relativePath !== WIKI_FOLDER_MANIFEST_PATH &&
        !options.referencedPaths.has(canonicalPathKey(file.relativePath)),
    )
    .sort((left, right) =>
      left.relativePath.localeCompare(right.relativePath, 'en'),
    )
  const consumedFiles = new Set<string>()
  const generated: WikiNode[] = []
  const directoryIds = new Map(options.directoryAnchors)

  const nextOrderByParent = new Map<string | null, number>()
  for (const node of options.baseNodes) {
    nextOrderByParent.set(
      node.parentId,
      Math.max(nextOrderByParent.get(node.parentId) ?? 0, node.order + 1),
    )
  }
  const nextOrder = (parentId: string | null): number => {
    const order = nextOrderByParent.get(parentId) ?? 0
    nextOrderByParent.set(parentId, order + 1)
    return order
  }

  if (options.wrapRoot) {
    const rootIndex = indexFileForDirectory('', extraFiles)
    if (rootIndex) consumedFiles.add(canonicalPathKey(rootIndex.relativePath))
    const rootId = uniqueId(options.usedIds)
    generated.push({
      id: rootId,
      parentId: null,
      title:
        (rootIndex && headingFromMarkdown(rootIndex.content)?.slice(0, 500)) ||
        source.rootName,
      icon: null,
      group: null,
      order: nextOrder(null),
      body: rootIndex ? bodyFromMarkdown(rootIndex.content) : '',
      updatedAt: options.now,
    })
    directoryIds.set('', rootId)
  }

  const neededDirectories = new Set<string>()
  for (const directory of entries.directories) neededDirectories.add(directory)
  for (const file of extraFiles) {
    for (const directory of parentDirectories(file.relativePath)) {
      neededDirectories.add(directory)
    }
  }
  const orderedDirectories = [...neededDirectories].sort(
    (left, right) =>
      pathDepth(left) - pathDepth(right) || left.localeCompare(right, 'en'),
  )

  const resolveParentId = (directory: string): string | null => {
    let cursor = posix.dirname(directory)
    while (cursor !== '.') {
      const parentId = directoryIds.get(cursor)
      if (parentId) return parentId
      cursor = posix.dirname(cursor)
    }
    return directoryIds.get('') ?? null
  }

  for (const directory of orderedDirectories) {
    if (directoryIds.has(directory)) continue
    const indexFile = indexFileForDirectory(directory, extraFiles)
    if (indexFile) consumedFiles.add(canonicalPathKey(indexFile.relativePath))
    const parentId = resolveParentId(directory)
    const id = uniqueId(options.usedIds)
    generated.push({
      id,
      parentId,
      title:
        (indexFile &&
          headingFromMarkdown(indexFile.content)?.slice(0, 500)) ||
        posix.basename(directory),
      icon: null,
      group: null,
      order: nextOrder(parentId),
      body: indexFile ? bodyFromMarkdown(indexFile.content) : '',
      updatedAt: options.now,
    })
    directoryIds.set(directory, id)
  }

  for (const file of extraFiles) {
    if (consumedFiles.has(canonicalPathKey(file.relativePath))) continue
    const directory = posix.dirname(file.relativePath)
    const parentId =
      directory === '.'
        ? (directoryIds.get('') ?? null)
        : (directoryIds.get(directory) ?? resolveParentId(directory))
    generated.push({
      id: uniqueId(options.usedIds),
      parentId,
      title: titleFromMarkdown(posix.basename(file.relativePath), file.content),
      icon: null,
      group: null,
      order: nextOrder(parentId),
      body: bodyFromMarkdown(file.content),
      updatedAt: options.now,
    })
  }
  return { nodes: generated, extraMarkdownCount: extraFiles.length }
}

export function parseWikiFolder(
  source: FolderSource,
  existingIds: Iterable<string>,
  now = Date.now(),
): ParsedWikiFolder {
  const entries = normalizeEntries(source)
  const manifest = parseManifest(entries)
  const usedIds = new Set(existingIds)

  if (!manifest) {
    const extra = extraFolderNodes(source, entries, {
      baseNodes: [],
      directoryAnchors: new Map(),
      referencedPaths: new Set(),
      usedIds,
      wrapRoot: true,
      now,
    })
    if (extra.nodes.length > MAX_WIKI_BACKUP_NODES) {
      throw new WikiFolderPortableError(
        'FOLDER_TOO_LARGE',
        `Wiki folder produces more than ${MAX_WIKI_BACKUP_NODES} nodes`,
        413,
      )
    }
    return {
      nodes: sortAndValidateNodes(extra.nodes),
      hasManifest: false,
      extraMarkdownCount: extra.extraMarkdownCount,
    }
  }

  const manifestResult = nodesFromManifest(manifest, entries, now)
  for (const node of manifestResult.nodes) usedIds.add(node.id)
  const extra = extraFolderNodes(source, entries, {
    baseNodes: manifestResult.nodes,
    directoryAnchors: manifestResult.directoryAnchors,
    referencedPaths: manifestResult.referencedPaths,
    usedIds,
    wrapRoot: false,
    now,
  })
  const nodes = [...manifestResult.nodes, ...extra.nodes]
  if (nodes.length > MAX_WIKI_BACKUP_NODES) {
    throw new WikiFolderPortableError(
      'FOLDER_TOO_LARGE',
      `Wiki folder produces more than ${MAX_WIKI_BACKUP_NODES} nodes`,
      413,
    )
  }
  return {
    nodes: sortAndValidateNodes(nodes),
    hasManifest: true,
    extraMarkdownCount: extra.extraMarkdownCount,
  }
}
