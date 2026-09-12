import type { FastifyInstance, FastifyReply } from 'fastify'
import { crudRouter } from '../server/crud/crud-router.js'
import { SnippetInput, type Snippet } from './schema.js'
import { createSnippetsRepo } from './repo.js'
import { getAccessToken, getStatus } from '../github/oauth.js'
import { GitHubGistError, pullGistFile, pushSnippetToGist } from '../github/gist.js'
import { z } from 'zod'
import { rankChatKnowledgeItems, type ChatKnowledgeItem } from '../server/chat-knowledge.js'

const SnippetIdParams = z.object({ id: z.string().min(1) })
const ImportGistBody = z.object({
  gistId: z.string().min(1),
  file: z.string().min(1).optional(),
  title: z.string().min(1).optional(),
  tags: z.array(z.string()).optional(),
})

function sendGitHubError(reply: FastifyReply, error: unknown) {
  if (error instanceof GitHubGistError) {
    const status = error.status === 401 || error.status === 403 || error.status === 404
      ? error.status
      : 502
    return reply.status(status).send({
      error: {
        code: 'GITHUB_GIST_ERROR',
        message: error.message,
      },
    })
  }
  const message = error instanceof Error ? error.message : String(error)
  return reply.status(500).send({
    error: {
      code: 'SNIPPET_GIST_ERROR',
      message,
    },
  })
}

export async function registerSnippetsRoutes(
  app: FastifyInstance,
): Promise<void> {
  const repo = createSnippetsRepo()
  let cachedSnippets: ReturnType<typeof repo.list> | null = null
  const readSnippets = () => (cachedSnippets ??= repo.list())
  const invalidateSnippets = () => {
    cachedSnippets = null
  }
  app.chatKnowledgeProviders?.register({
    id: 'desktop-snippets',
    search(query, limit) {
      const snippets = readSnippets()
      const items: ChatKnowledgeItem[] = snippets.map((snippet) => ({
        id: snippet.id,
        source: 'Snippet',
        title: `${snippet.title} (${snippet.language})`,
        content: snippet.body,
        tags: snippet.tags,
      }))
      const ranked = rankChatKnowledgeItems(query, items, limit)
      if (ranked.length > 0 || !/(?:\bsnippets?\b|스니펫)/iu.test(query)) return ranked
      return items.slice(0, limit).map((item) => ({ ...item, score: 0.1 }))
    },
  })
  await crudRouter<Snippet, SnippetInput>(app, {
    list: (query) => repo.list(query),
    upsert: (input) => {
      const saved = repo.upsert(input)
      invalidateSnippets()
      return saved
    },
    remove: (id) => {
      repo.remove(id)
      invalidateSnippets()
    },
  }, {
    capability: {
      name: 'snippets',
      version: '1',
      basePath: '/snippets',
    },
    schema: SnippetInput,
    pickQuery: true,
    extraMethods: [
      { method: 'GET', path: '/snippets/gist/status' },
      { method: 'POST', path: '/snippets/gist/import' },
      { method: 'POST', path: '/snippets/:id/gist/push' },
      { method: 'POST', path: '/snippets/:id/gist/pull' },
      { method: 'DELETE', path: '/snippets/:id/gist' },
    ],
    afterRoutes: async (a) => {
      a.get('/snippets/gist/status', async () => getStatus())

      a.post('/snippets/:id/gist/push', async (req, reply) => {
        const token = getAccessToken()
        if (!token) {
          return reply.status(401).send({
            error: {
              code: 'GITHUB_NOT_CONNECTED',
              message: 'GitHub account is not connected.',
            },
          })
        }
        const { id } = SnippetIdParams.parse(req.params)
        const snippet = repo.get(id)
        if (!snippet) {
          return reply.status(404).send({
            error: {
              code: 'SNIPPET_NOT_FOUND',
              message: `Snippet not found: ${id}`,
            },
          })
        }
        try {
          const link = await pushSnippetToGist(token, snippet)
          return repo.setGistLink(id, link)
        } catch (error) {
          return sendGitHubError(reply, error)
        }
      })

      a.post('/snippets/:id/gist/pull', async (req, reply) => {
        const token = getAccessToken()
        if (!token) {
          return reply.status(401).send({
            error: {
              code: 'GITHUB_NOT_CONNECTED',
              message: 'GitHub account is not connected.',
            },
          })
        }
        const { id } = SnippetIdParams.parse(req.params)
        const snippet = repo.get(id)
        if (!snippet) {
          return reply.status(404).send({
            error: {
              code: 'SNIPPET_NOT_FOUND',
              message: `Snippet not found: ${id}`,
            },
          })
        }
        if (!snippet.gist) {
          return reply.status(409).send({
            error: {
              code: 'SNIPPET_GIST_NOT_LINKED',
              message: 'Snippet is not linked to a GitHub Gist.',
            },
          })
        }
        try {
          const pulled = await pullGistFile(token, snippet.gist.id, snippet.gist.file)
          repo.upsert({
            id: snippet.id,
            title: snippet.title,
            language: pulled.language,
            body: pulled.body,
            tags: snippet.tags,
          })
          invalidateSnippets()
          return repo.setGistLink(snippet.id, pulled.link)
        } catch (error) {
          return sendGitHubError(reply, error)
        }
      })

      a.delete('/snippets/:id/gist', async (req, reply) => {
        const { id } = SnippetIdParams.parse(req.params)
        const snippet = repo.clearGistLink(id)
        if (!snippet) {
          return reply.status(404).send({
            error: {
              code: 'SNIPPET_NOT_FOUND',
              message: `Snippet not found: ${id}`,
            },
          })
        }
        return snippet
      })

      a.post('/snippets/gist/import', async (req, reply) => {
        const token = getAccessToken()
        if (!token) {
          return reply.status(401).send({
            error: {
              code: 'GITHUB_NOT_CONNECTED',
              message: 'GitHub account is not connected.',
            },
          })
        }
        const body = ImportGistBody.parse(req.body ?? {})
        try {
          const pulled = await pullGistFile(token, body.gistId, body.file)
          const existing = repo.findByGist(pulled.link.id, pulled.link.file)
          const saved = repo.upsert({
            id: existing?.id,
            title: body.title ?? existing?.title ?? pulled.title,
            language: pulled.language,
            body: pulled.body,
            tags: body.tags ?? existing?.tags ?? ['gist'],
          })
          invalidateSnippets()
          return repo.setGistLink(saved.id, pulled.link)
        } catch (error) {
          return sendGitHubError(reply, error)
        }
      })
    },
  })
}
