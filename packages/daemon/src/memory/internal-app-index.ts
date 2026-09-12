const APP_INDEX_TAG = 'app'
const APP_DOCUMENT_ID_PREFIX = 'app:'
const APP_DOCUMENT_PATH_PREFIX = 'apps/'

type AppIndexLike = {
  id?: string
  content?: string
  source?: string
  tags?: readonly string[]
  documentId?: string
  documentPath?: string
  path?: string
}

export function hasAppIndexTag(tags?: readonly string[]): boolean {
  return Boolean(tags?.some((tag) => tag === APP_INDEX_TAG || tag.startsWith(`${APP_INDEX_TAG}:`)))
}

export function isAppIndexDocumentId(id?: string | null): boolean {
  return typeof id === 'string' && id.startsWith(APP_DOCUMENT_ID_PREFIX)
}

export function isAppIndexDocumentPath(path?: string | null): boolean {
  return typeof path === 'string' && path.replaceAll('\\', '/').startsWith(APP_DOCUMENT_PATH_PREFIX)
}

export function isAppIndexMemoryEntry(entry: AppIndexLike): boolean {
  const documentId = entry.documentId ?? entry.id
  const documentPath = entry.documentPath ?? entry.path
  const isDocumentIndex =
    entry.source === 'document'
    && (hasAppIndexTag(entry.tags)
      || isAppIndexDocumentId(documentId)
      || isAppIndexDocumentPath(documentPath))
  if (isDocumentIndex) {
    return true
  }

  return Boolean(
    entry.source === 'conversation'
    && entry.tags?.includes('rag-promotion')
    && entry.content?.startsWith('Document recall anchor from "App:'),
  )
}

export function explicitlyRequestsAppIndex(input: {
  tags?: readonly string[]
  documentId?: string
}): boolean {
  return hasAppIndexTag(input.tags) || isAppIndexDocumentId(input.documentId)
}

export function filterAppIndexEntries<T extends AppIndexLike>(
  entries: T[],
  includeAppIndex: boolean,
): T[] {
  return includeAppIndex ? entries : entries.filter((entry) => !isAppIndexMemoryEntry(entry))
}
