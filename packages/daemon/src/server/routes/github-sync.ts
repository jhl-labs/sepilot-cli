import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import {
  discoverRepos,
  getPolicy,
  inspectRepo,
  listRepos,
  runSync,
  setPolicy,
  setRepos,
} from '../../github/sync.js'
import { getAccessToken } from '../../github/oauth.js'

export async function registerGitHubSyncRoutes(app: FastifyInstance): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'github',
      version: '1',
      methods: [
        { method: 'GET', path: '/github/sync/repos' },
        { method: 'PUT', path: '/github/sync/repos' },
        { method: 'GET', path: '/github/sync/discover' },
        { method: 'POST', path: '/github/repo/inspect' },
        { method: 'GET', path: '/github/sync/policy' },
        { method: 'PUT', path: '/github/sync/policy' },
        { method: 'POST', path: '/github/sync/run' },
      ],
    },
    async (a) => {
      a.get('/github/sync/repos', async () => listRepos())
      a.put('/github/sync/repos', async (req, reply) => {
        const parsed = z
          .array(
            z.object({
              fullName: z.string().min(1),
              enabled: z.boolean(),
              lastSyncedAt: z.number().int().nullable(),
              lastSyncStatus: z.enum(['success', 'error']).nullable().optional(),
              lastSyncError: z.string().nullable().optional(),
              pullRequests: z.number().int().nonnegative().optional(),
              issues: z.number().int().nonnegative().optional(),
              releases: z.number().int().nonnegative().optional(),
            }),
          )
          .safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        setRepos(parsed.data)
        return listRepos()
      })
      a.get('/github/sync/discover', async (_req, reply) => {
        const token = getAccessToken()
        if (!token) {
          void reply.status(401).send({
            code: 'GITHUB_NOT_CONNECTED',
            message: 'GitHub account is not connected.',
            retriable: false,
          })
          return reply
        }
        return discoverRepos(token)
      })
      a.post('/github/repo/inspect', async (req, reply) => {
        const parsed = z
          .object({
            url: z.string().max(2048).optional(),
            owner: z.string().max(200).optional(),
            repo: z.string().max(200).optional(),
          })
          .safeParse(req.body ?? {})
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const token = getAccessToken()
        if (!token) {
          void reply.status(401).send({
            code: 'GITHUB_NOT_CONNECTED',
            message: 'GitHub account is not connected.',
            retriable: false,
          })
          return reply
        }
        try {
          return await inspectRepo(token, parsed.data)
        } catch (error) {
          void reply.status(400).send({
            code: 'GITHUB_REPO_INSPECT_FAILED',
            message:
              error instanceof Error ? error.message : 'GitHub repository inspection failed.',
            retriable: true,
          })
          return reply
        }
      })
      a.get('/github/sync/policy', async () => getPolicy())
      a.put('/github/sync/policy', async (req, reply) => {
        const parsed = z
          .object({
            intervalMin: z.number().int().min(1),
            pullRequests: z.boolean(),
            issues: z.boolean(),
            releases: z.boolean(),
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
        setPolicy(parsed.data)
        return parsed.data
      })
      a.post('/github/sync/run', async (_req, reply) => {
        const token = getAccessToken()
        if (!token) {
          void reply.status(401).send({
            code: 'GITHUB_NOT_CONNECTED',
            message: 'GitHub account is not connected.',
            retriable: false,
          })
          return reply
        }
        return runSync(token)
      })
    },
  )
}
