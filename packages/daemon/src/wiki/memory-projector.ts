import { createHash } from 'node:crypto'
import type {
  IDocumentMemoryStore,
  MemoryDocument,
} from '@sepilotd/core'
import type { WikiNode } from './schema.js'

const WIKI_TAG = 'wiki'
const WIKI_SOURCE_TAG = 'source:desktop-wiki'
const WIKI_SOURCE_FILE_PREFIX = 'wiki:'
const WIKI_DOCUMENT_ID_PREFIX = 'wiki:'
const GLOBAL_SCOPE_IDENTITY = 'global'

/**
 * listDocuments has no cursor contract, so reconciliation uses a deliberately
 * high but bounded read and refuses to delete anything when that boundary is
 * reached. This avoids silently treating a truncated result as the full set.
 */
export const WIKI_RECONCILE_DOCUMENT_LIMIT = 10_000

export type WikiMemoryDocumentStore = Pick<
  IDocumentMemoryStore,
  'ingestDocument' | 'listDocuments' | 'deleteDocument'
>

export interface WikiMemoryReconcileResult {
  projected: number
  removed: number
}

function durableScopeTags(scopeTags: readonly string[] | undefined): string[] {
  const tags = new Map<string, string>()
  for (const rawTag of scopeTags ?? []) {
    const tag = rawTag.trim().toLowerCase()
    if (!tag.startsWith('scope:') || tag === 'scope:') continue
    tags.set(tag, tag)
  }
  return [...tags.values()].sort()
}

function scopeHash(scopeTags: readonly string[] | undefined): string {
  const tags = durableScopeTags(scopeTags)
  const identity = tags.length > 0 ? JSON.stringify(tags) : GLOBAL_SCOPE_IDENTITY
  return createHash('sha256').update(identity).digest('hex')
}

export function wikiMemoryDocumentId(
  nodeId: string,
  scopeTags: readonly string[] | undefined,
): string {
  return `${WIKI_DOCUMENT_ID_PREFIX}${scopeHash(scopeTags)}:${nodeId}`
}

function wikiMemoryDocumentIdPrefix(scopeTags: readonly string[] | undefined): string {
  return `${WIKI_DOCUMENT_ID_PREFIX}${scopeHash(scopeTags)}:`
}

function wikiDocumentTags(node: WikiNode, scopeTags: readonly string[] | undefined): string[] {
  const tags = [WIKI_SOURCE_TAG, WIKI_TAG]
  const group = node.group?.trim()
  // A Wiki group is user-authored metadata, not an authorization boundary.
  // Never let it masquerade as a trusted scope tag.
  if (group && !group.toLowerCase().startsWith('scope:')) tags.push(group)
  tags.push(...durableScopeTags(scopeTags))
  return [...new Map(tags.map((tag) => [tag.toLowerCase(), tag])).values()]
}

function wikiDocumentContent(node: WikiNode, title: string): string {
  const bodyTitle = node.body
    .replace(/^\uFEFF/u, '')
    .match(/^\s{0,3}#\s+(.+?)\s*#*\s*$/mu)?.[1]
    ?.trim()
  return bodyTitle === title ? node.body : `# ${title}\n\n${node.body}`
}

function isWikiDocumentForScope(document: MemoryDocument, idPrefix: string): boolean {
  const tags = new Set(document.tags.map((tag) => tag.trim().toLowerCase()))
  return document.id.startsWith(idPrefix)
    && document.sourceFileId?.startsWith(WIKI_SOURCE_FILE_PREFIX) === true
    && tags.has(WIKI_SOURCE_TAG)
    && tags.has(WIKI_TAG)
}

/**
 * Projects canonical Wiki nodes into semantic document memory.
 *
 * Wiki remains the source of truth. Errors intentionally propagate so callers
 * can report degraded indexing and retry with reconcile without rolling back a
 * successful Wiki write.
 */
export class WikiMemoryProjector {
  constructor(private readonly documentStore: WikiMemoryDocumentStore) {}

  project(node: WikiNode, scopeTags?: readonly string[]): Promise<MemoryDocument> {
    const title = node.title.trim()
    return this.documentStore.ingestDocument({
      id: wikiMemoryDocumentId(node.id, scopeTags),
      title,
      content: wikiDocumentContent(node, title),
      path: `wiki://node/${encodeURIComponent(node.id)}`,
      mimeType: 'text/markdown',
      sourceFileId: `${WIKI_SOURCE_FILE_PREFIX}${node.id}`,
      tags: wikiDocumentTags(node, scopeTags),
    })
  }

  remove(nodeId: string, scopeTags?: readonly string[]): Promise<boolean> {
    return this.documentStore.deleteDocument(wikiMemoryDocumentId(nodeId, scopeTags))
  }

  async reconcile(
    nodes: readonly WikiNode[],
    scopeTags?: readonly string[],
  ): Promise<WikiMemoryReconcileResult> {
    const expectedIds = new Set<string>()
    for (const node of nodes) {
      expectedIds.add(wikiMemoryDocumentId(node.id, scopeTags))
      await this.project(node, scopeTags)
    }

    const indexedDocuments = await this.documentStore.listDocuments({
      query: WIKI_SOURCE_TAG,
      limit: WIKI_RECONCILE_DOCUMENT_LIMIT,
    })
    if (indexedDocuments.length >= WIKI_RECONCILE_DOCUMENT_LIMIT) {
      throw new Error(
        `Wiki memory reconciliation reached the ${WIKI_RECONCILE_DOCUMENT_LIMIT} document safety limit`,
      )
    }

    const idPrefix = wikiMemoryDocumentIdPrefix(scopeTags)
    const staleIds = indexedDocuments
      .filter((document) => isWikiDocumentForScope(document, idPrefix))
      .map((document) => document.id)
      .filter((id) => !expectedIds.has(id))

    let removed = 0
    for (const id of staleIds) {
      if (await this.documentStore.deleteDocument(id)) removed += 1
    }

    return { projected: nodes.length, removed }
  }
}
