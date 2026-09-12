import { registerDocsGitRoutes } from '../docs-git/routes.js'
import type { FastifyInstance, FastifyRequest } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../server/capabilities/bind.js'
import { createWikiRepo } from './repo.js'
import {
  MAX_WIKI_BACKUP_BYTES,
  MAX_WIKI_FOLDER_BYTES,
  WikiImportRequest,
  WikiMoveInput,
  WikiNodeInput,
} from './schema.js'
import { rankChatKnowledgeItems, type ChatKnowledgeItem } from '../server/chat-knowledge.js'
import { createWikiPortableService } from './portable.js'
import { WikiMemoryProjector } from './memory-projector.js'
import { registerPersonalKnowledgeRoutes } from '../knowledge/routes.js'
import { createKnowledgeActivityStore } from '../knowledge/activity.js'

const WIKI_IMPORT_BODY_LIMIT_BYTES =
  Math.max(MAX_WIKI_BACKUP_BYTES, MAX_WIKI_FOLDER_BYTES) + 1024 * 1024

function parseImportRequest(body: unknown) {
  const parsed = WikiImportRequest.safeParse(body)
  if (parsed.success) return parsed.data
  const message = parsed.error.issues[0]?.message ?? 'Invalid Wiki import request'
  throw Object.assign(new Error(message), {
    statusCode: 400,
    code: 'INVALID_WIKI_IMPORT',
  })
}

export async function registerWikiRoutes(
  app: FastifyInstance,
): Promise<void> {
  await registerPersonalKnowledgeRoutes(app)
  const activity = createKnowledgeActivityStore()
  function trackedWiki<T>(kind: string, ids: string[], task: () => T): T {
    const runId = activity.begin({ kind, summary: kind, targets: ids.map((id) => ({ id })) })
    try { const result = task(); activity.finish(runId, { ids, preview: result }); return result }
    catch (error) { activity.fail(runId, error); throw error }
  }
  const repo = createWikiRepo()
  const portable = createWikiPortableService(repo)
  const memoryProjector = app.runtime?.semanticIndex
    ? new WikiMemoryProjector(app.runtime.semanticIndex)
    : null
  const pendingMemoryProjections = new Map<string, Promise<void>>()
  let memoryReconciled = false
  let pendingReconcileCount = 0
  let cachedNodes: ReturnType<typeof repo.tree> | null = null
  const readNodes = () => (cachedNodes ??= repo.tree())
  const invalidateNodes = () => {
    cachedNodes = null
  }
  await registerDocsGitRoutes(app, invalidateNodes)
  const scheduleMemoryProjection = (
    request: FastifyRequest,
    operation: string,
    task: (projector: WikiMemoryProjector, scopeTags: string[]) => Promise<unknown>,
    options: { reconcile?: boolean; once?: boolean } = {},
  ): void => {
    // Wiki is currently daemon-global first-party local knowledge. Its Memory
    // projection intentionally uses the same legacy-global visibility model:
    // first-party callers receive the server-derived legacy read capability,
    // while extensions neither trigger nor retrieve this derived copy.
    if (!memoryProjector || request.authContext?.kind === 'extension') return
    if (options.once && (memoryReconciled || pendingReconcileCount > 0)) return
    if (options.reconcile) pendingReconcileCount += 1

    const scopeTags: string[] = []
    const scopeKey = 'daemon-global'
    const previous = pendingMemoryProjections.get(scopeKey) ?? Promise.resolve()
    const current = previous
      .catch(() => undefined)
      .then(async () => {
        await task(memoryProjector, scopeTags)
      })
    pendingMemoryProjections.set(scopeKey, current)
    void current.then(
      () => {
        if (options.reconcile) {
          pendingReconcileCount = Math.max(0, pendingReconcileCount - 1)
          memoryReconciled = true
        }
        if (pendingMemoryProjections.get(scopeKey) === current) {
          pendingMemoryProjections.delete(scopeKey)
        }
      },
      (error) => {
        // Any failed incremental projection makes the derived index suspect;
        // the next tree read must be allowed to run a full reconciliation.
        memoryReconciled = false
        if (options.reconcile) {
          pendingReconcileCount = Math.max(0, pendingReconcileCount - 1)
        }
        request.log.warn(
          { err: error, operation },
          'Wiki was saved but its derived Memory index update failed',
        )
        if (pendingMemoryProjections.get(scopeKey) === current) {
          pendingMemoryProjections.delete(scopeKey)
        }
      },
    )
  }
  app.addHook('onClose', async () => {
    await Promise.allSettled([...pendingMemoryProjections.values()])
  })
  app.chatKnowledgeProviders?.register({
    id: 'desktop-wiki',
    search(query, limit) {
      const nodes = readNodes().filter((node) => node.title.trim() || node.body.trim())
      const items: ChatKnowledgeItem[] = nodes.map((node) => ({
        id: node.id,
        source: 'Wiki',
        title: node.title,
        content: node.body,
        tags: node.group ? [node.group] : undefined,
      }))
      const ranked = rankChatKnowledgeItems(query, items, limit)
      if (ranked.length > 0 || !/(?:\bwiki\b|위키)/iu.test(query)) return ranked
      return [...nodes]
        .sort((left, right) => right.updatedAt - left.updatedAt)
        .slice(0, limit)
        .map((node) => ({
          id: node.id,
          source: 'Wiki',
          title: node.title,
          content: node.body,
          tags: node.group ? [node.group] : undefined,
          score: 0.1,
        }))
    },
  })
  await bindCapability(
    app,
    {
      name: 'wiki',
      version: '1',
      methods: [
        { method: 'GET', path: '/wiki/tree' },
        { method: 'GET', path: '/wiki/search' },
        { method: 'GET', path: '/wiki/export' },
        { method: 'GET', path: '/wiki/export/folder' },
        { method: 'POST', path: '/wiki/import/preview' },
        { method: 'POST', path: '/wiki/import' },
        { method: 'POST', path: '/wiki/nodes' },
        { method: 'DELETE', path: '/wiki/nodes/:id' },
        { method: 'POST', path: '/wiki/nodes/:id/move' },
      ],
    },
    async (a) => {
      a.get('/wiki/tree', async (request) => {
        const nodes = repo.tree()
        scheduleMemoryProjection(request, 'reconcile', (projector, scopeTags) =>
          projector.reconcile(nodes, scopeTags), { reconcile: true, once: true })
        return nodes
      })
      a.get('/wiki/search', async (req) => {
        const { q, limit } = z
          .object({
            q: z.string().min(1),
            limit: z.coerce.number().int().min(1).max(200).default(50),
          })
          .parse(req.query)
        return repo.search(q, limit)
      })
      a.get('/wiki/export', async (_req, reply) => {
        const date = new Date().toISOString().slice(0, 10)
        reply.header('content-type', 'application/json; charset=utf-8')
        reply.header(
          'content-disposition',
          `attachment; filename="sepilot-wiki-${date}.sepilotwiki"`,
        )
        return trackedWiki('wiki-export', [], () => portable.exportBackup())
      })
      a.get('/wiki/export/folder', async (_req, reply) => {
        reply.header('content-type', 'application/json; charset=utf-8')
        return trackedWiki('wiki-export', [], () => portable.exportFolder())
      })
      a.post(
        '/wiki/import/preview',
        { bodyLimit: WIKI_IMPORT_BODY_LIMIT_BYTES },
        async (req) => portable.preview(parseImportRequest(req.body)),
      )
      a.post(
        '/wiki/import',
        { bodyLimit: WIKI_IMPORT_BODY_LIMIT_BYTES },
        async (req) => {
          const result = trackedWiki('wiki-import', [], () => portable.import(parseImportRequest(req.body)))
          invalidateNodes()
          const nodes = repo.tree()
          scheduleMemoryProjection(req, 'import', (projector, scopeTags) =>
            projector.reconcile(nodes, scopeTags), { reconcile: true })
          return result
        },
      )
      a.post('/wiki/nodes', async (req) => {
        const body = WikiNodeInput.parse(req.body)
        const saved = trackedWiki('wiki-write', body.id ? [body.id] : [], () => repo.upsert(body))
        invalidateNodes()
        scheduleMemoryProjection(req, 'upsert', (projector, scopeTags) =>
          projector.project(saved, scopeTags))
        return saved
      })
      a.delete('/wiki/nodes/:id', async (req) => {
        const { id } = z
          .object({ id: z.string().min(1) })
          .parse(req.params)
        trackedWiki('wiki-delete', [id], () => repo.remove(id))
        invalidateNodes()
        const nodes = repo.tree()
        scheduleMemoryProjection(req, 'remove', (projector, scopeTags) =>
          projector.reconcile(nodes, scopeTags), { reconcile: true })
        return { ok: true }
      })
      a.post('/wiki/nodes/:id/move', async (req) => {
        const { id } = z
          .object({ id: z.string().min(1) })
          .parse(req.params)
        const body = WikiMoveInput.parse(req.body)
        const moved = trackedWiki('wiki-move', [id], () => repo.move(id, body))
        invalidateNodes()
        return moved
      })
    },
  )
}
