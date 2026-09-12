import { webSourceUrl } from '../../rag/web-sync.js'
import type { FastifyInstance } from 'fastify'
import type { MemoryDocumentChunk } from '@sepilotd/core'
import { z } from 'zod'
import '../fastify-types.js'
import { bindCapability } from '../capabilities/bind.js'
import { createRagStore } from '../../rag/store.js'
import { syncRagSources } from '../../rag/git-sync.js'
import {
  applyRagRerank,
  resolveRagSearchSettings,
  type RagSearchHitForRerank,
} from '../../rag/retrieval.js'
import {
  testRagRerankConnection,
  testRagVectorBackendConnection,
} from '../../rag/connection-test.js'
import type { SemanticIndexRuntimeStatus } from '../../memory/types.js'

function folderIdFromTags(tags: string[]): string {
  const tag = tags.find((item) => item.startsWith('rag-folder:'))
  return tag ? tag.slice('rag-folder:'.length) : 'default'
}

function memoryChunkToRagHit(chunk: MemoryDocumentChunk): RagSearchHitForRerank {
  return {
    documentId: chunk.documentId,
    folderId: folderIdFromTags(chunk.tags),
    title: chunk.documentTitle,
    score: chunk.score ?? 0,
    snippet: chunk.snippet ?? chunk.content.slice(0, 240),
    path: chunk.documentPath,
  }
}

function resolveSearchMode(
  status: SemanticIndexRuntimeStatus | undefined,
): 'hybrid' | 'hybrid-backfilling' | 'keyword-fallback' | 'degraded' | 'memory-vector' {
  if (!status) return 'memory-vector'
  if (status.status === 'disabled' || status.status === 'reindex_required') {
    return 'keyword-fallback'
  }
  if (status.status === 'backfilling') return 'hybrid-backfilling'
  if (status.status === 'degraded') return 'degraded'
  return 'hybrid'
}

function resolveSemanticQueryType(
  searchMode: ReturnType<typeof resolveSearchMode>,
): 'hybrid' | 'keyword' {
  return searchMode === 'keyword-fallback' ? 'keyword' : 'hybrid'
}

export async function registerRagCapabilityRoutes(app: FastifyInstance): Promise<void> {
  const store = createRagStore()
  await bindCapability(
    app,
    {
      name: 'rag',
      version: '1',
      methods: [
        { method: 'GET', path: '/rag/folders' },
        { method: 'POST', path: '/rag/folders' },
        { method: 'DELETE', path: '/rag/folders/:id' },
        { method: 'GET', path: '/rag/documents' },
        { method: 'POST', path: '/rag/documents' },
        { method: 'GET', path: '/rag/documents/:id' },
        { method: 'DELETE', path: '/rag/documents/:id' },
        { method: 'GET', path: '/rag/search' },
        { method: 'POST', path: '/rag/sync' },
        { method: 'GET', path: '/rag/vector-db' },
        { method: 'POST', path: '/rag/vector-db/test' },
        { method: 'POST', path: '/rag/rerank/test' },
      ],
    },
    async (a) => {
      a.get('/rag/folders', async () => store.listFolders())
      a.post('/rag/folders', async (req, reply) => {
        const parsed = z
          .object({
            id: z.string().optional(),
            name: z.string().min(1),
            sourceType: z.enum(['manual', 'git', 'web']).optional(),
            tlsVerify: z.boolean().optional(),
            caCert: z.string().max(100000).optional(),
            path: z.string().min(1).optional(),
            include: z.array(z.string().min(1)).optional(),
            exclude: z.array(z.string().min(1)).optional(),
          })
          .safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        if (parsed.data.sourceType === 'web') {
          if (req.authContext?.kind === 'extension')
            return reply.code(403).send({ error: 'Web sources require a first-party client.' })
          try {
            webSourceUrl(parsed.data.path ?? '')
          } catch (error) {
            return reply.code(400).send({ message: (error as Error).message })
          }
        }
        return store.upsertFolder(parsed.data)
      })
      a.delete('/rag/folders/:id', async (req) => {
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        const docs = store.listDocuments(id)
        store.removeFolder(id)
        const semanticIndex = app.runtime?.semanticIndex
        if (semanticIndex) {
          await Promise.all(docs.map((doc) => semanticIndex.deleteDocument(doc.id)))
        }
        return { ok: true }
      })
      a.get('/rag/documents', async (req, reply) => {
        const parsed = z.object({ folder: z.string().min(1) }).safeParse(req.query)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        return store.listDocuments(parsed.data.folder)
      })
      a.post('/rag/documents', async (req, reply) => {
        const parsed = z
          .object({
            id: z.string().optional(),
            folderId: z.string().min(1),
            title: z.string().min(1),
            body: z.string().default(''),
            path: z.string().optional(),
            sourceFileId: z.string().optional(),
            size: z.number().int().nonnegative().optional(),
          })
          .safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const document = await store.upsertDocument(parsed.data)
        await app.runtime?.semanticIndex.ingestDocument({
          id: document.id,
          title: document.title,
          content: parsed.data.body,
          path: document.path,
          sourceFileId: document.sourceFileId,
          tags: ['rag', `rag-folder:${document.folderId}`],
        })
        return document
      })
      a.get('/rag/documents/:id', async (req, reply) => {
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        const document = store.getDocument(id)
        if (!document) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: `RAG document not found: ${id}`,
            retriable: false,
          })
          return reply
        }
        return document
      })
      a.delete('/rag/documents/:id', async (req) => {
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        store.removeDocument(id)
        await app.runtime?.semanticIndex.deleteDocument(id)
        return { ok: true }
      })
      a.get('/rag/search', async (req, reply) => {
        const parsed = z
          .object({
            q: z.string().min(1),
            limit: z.coerce.number().int().min(1).max(50).optional(),
          })
          .safeParse(req.query)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const settings = resolveRagSearchSettings(app.runtime?.config, parsed.data.limit)
        if (app.runtime?.semanticIndex) {
          const semanticStatus = app.runtime.semanticIndex.getStatus()
          const searchMode = resolveSearchMode(semanticStatus)
          const results = await app.runtime.semanticIndex.searchDocuments(parsed.data.q, {
            type: resolveSemanticQueryType(searchMode),
            limit: settings.candidateLimit,
            tags: ['rag'],
          })
          const hits = results.map((chunk) => ({
            ...memoryChunkToRagHit(chunk),
            searchMode,
          }))
          return (await applyRagRerank(parsed.data.q, hits, settings)).slice(0, settings.limit)
        }
        const hits = (await store.search(parsed.data.q, settings.candidateLimit)).map((hit) => ({
          ...hit,
          searchMode: resolveSearchMode(undefined),
        }))
        return (await applyRagRerank(parsed.data.q, hits, settings)).slice(0, settings.limit)
      })
      a.post('/rag/sync', async (req, reply) => {
        if (req.authContext?.kind === 'extension')
          return reply
            .code(403)
            .send({ error: 'Source synchronization requires a first-party client.' })
        return syncRagSources({
          store,
          semanticIndex: app.runtime?.semanticIndex,
        })
      })
      a.get('/rag/vector-db', async () => {
        const fallback = store.info()
        const status = app.runtime?.semanticIndex.getStatus()
        if (!status) return fallback
        return {
          engine: status.vectorBackend,
          vectorBackend: status.vectorBackend,
          backendAvailable: status.backendAvailable,
          vecAvailable: status.vecAvailable,
          status: status.status,
          configuredProviderId: status.configuredProviderId,
          configuredModel: status.configuredModel,
          indexedProviderId: status.indexedProviderId,
          indexedModel: status.indexedModel,
          dimension: status.dimensions ?? fallback.dimension,
          documents: fallback.documents,
          pendingCount: status.pendingCount,
          failedCount: status.failedCount,
          lastError: status.lastError,
        }
      })
      a.post('/rag/vector-db/test', async () =>
        testRagVectorBackendConnection(app.runtime?.config, app.runtime?.semanticIndex.getStatus()),
      )
      a.post('/rag/rerank/test', async () => testRagRerankConnection(app.runtime?.config))
    },
  )
}
