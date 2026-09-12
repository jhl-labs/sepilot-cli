import type { FastifyInstance, FastifyReply } from 'fastify'
import { z } from 'zod'
import { getAccessToken, getStatus } from '../../github/oauth.js'
import {
  createGitHubProjectDraftCard,
  getGitHubProjectBoard,
  listGitHubProjectsV2,
  moveGitHubProjectCard,
  updateGitHubProjectDraftCard,
  type GitHubProjectRef,
} from '../../github/projects-v2.js'
import { bindCapability } from '../capabilities/bind.js'

const ownerTypeSchema = z.enum(['user', 'organization'])

const projectParamsSchema = z.object({
  ownerType: ownerTypeSchema,
  owner: z.string().trim().min(1),
  number: z.coerce.number().int().positive(),
})

const listProjectsQuerySchema = z.object({
  query: z.string().trim().max(120).optional(),
})

const cardParamsSchema = projectParamsSchema.extend({
  itemId: z.string().trim().min(1),
})

const draftParamsSchema = projectParamsSchema.extend({
  draftIssueId: z.string().trim().min(1),
})

const createCardBodySchema = z.object({
  title: z.string().trim().min(1),
  body: z.string().optional(),
  columnId: z.string().trim().min(1).optional(),
})

const moveCardBodySchema = z.object({
  columnId: z.string().trim().min(1),
})

const updateDraftBodySchema = z.object({
  draftIssueId: z.string().trim().min(1),
  title: z.string().optional(),
  body: z.string().optional(),
})

function sendInvalid(reply: FastifyReply, message: string): FastifyReply {
  void reply.status(400).send({
    code: 'INVALID_REQUEST',
    message,
    retriable: false,
  })
  return reply
}

function parseProjectRef(value: unknown, reply: FastifyReply): GitHubProjectRef | null {
  const parsed = projectParamsSchema.safeParse(value)
  if (!parsed.success) {
    sendInvalid(reply, parsed.error.message)
    return null
  }
  return {
    ownerType: parsed.data.ownerType,
    owner: parsed.data.owner,
    number: parsed.data.number,
  }
}

function tokenOrReply(reply: FastifyReply): string | null {
  const token = getAccessToken()
  if (token) return token
  void reply.status(401).send({
    code: 'GITHUB_NOT_CONNECTED',
    message: 'GitHub account is not connected.',
    retriable: false,
  })
  return null
}

export async function registerGitHubProjectsRoutes(
  app: FastifyInstance,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'github-projects',
      version: '1',
      description: 'Sync kanban boards with GitHub Projects v2 using daemon-held GitHub credentials.',
      methods: [
        { method: 'GET', path: '/github/projects/v2/status' },
        { method: 'GET', path: '/github/projects/v2/projects' },
        { method: 'GET', path: '/github/projects/v2/:ownerType/:owner/:number' },
        { method: 'POST', path: '/github/projects/v2/:ownerType/:owner/:number/cards' },
        { method: 'PUT', path: '/github/projects/v2/:ownerType/:owner/:number/cards/:itemId/move' },
        { method: 'PUT', path: '/github/projects/v2/:ownerType/:owner/:number/drafts/:draftIssueId' },
      ],
    },
    async (a) => {
      a.get('/github/projects/v2/status', async () => getStatus())

      a.get('/github/projects/v2/projects', async (req, reply) => {
        const token = tokenOrReply(reply)
        if (!token) return reply
        const query = listProjectsQuerySchema.safeParse(req.query ?? {})
        if (!query.success) return sendInvalid(reply, query.error.message)
        return listGitHubProjectsV2(token, query.data)
      })

      a.get('/github/projects/v2/:ownerType/:owner/:number', async (req, reply) => {
        const token = tokenOrReply(reply)
        if (!token) return reply
        const ref = parseProjectRef(req.params, reply)
        if (!ref) return reply
        return getGitHubProjectBoard(token, ref)
      })

      a.post('/github/projects/v2/:ownerType/:owner/:number/cards', async (req, reply) => {
        const token = tokenOrReply(reply)
        if (!token) return reply
        const ref = parseProjectRef(req.params, reply)
        if (!ref) return reply
        const body = createCardBodySchema.safeParse(req.body ?? {})
        if (!body.success) return sendInvalid(reply, body.error.message)
        return createGitHubProjectDraftCard(token, ref, body.data)
      })

      a.put('/github/projects/v2/:ownerType/:owner/:number/cards/:itemId/move', async (req, reply) => {
        const token = tokenOrReply(reply)
        if (!token) return reply
        const params = cardParamsSchema.safeParse(req.params)
        if (!params.success) return sendInvalid(reply, params.error.message)
        const body = moveCardBodySchema.safeParse(req.body ?? {})
        if (!body.success) return sendInvalid(reply, body.error.message)
        return moveGitHubProjectCard(
          token,
          {
            ownerType: params.data.ownerType,
            owner: params.data.owner,
            number: params.data.number,
          },
          {
            itemId: params.data.itemId,
            columnId: body.data.columnId,
          },
        )
      })

      a.put('/github/projects/v2/:ownerType/:owner/:number/drafts/:draftIssueId', async (req, reply) => {
        const token = tokenOrReply(reply)
        if (!token) return reply
        const params = draftParamsSchema.safeParse(req.params)
        if (!params.success) return sendInvalid(reply, params.error.message)
        const body = updateDraftBodySchema.safeParse(req.body ?? {})
        if (!body.success) return sendInvalid(reply, body.error.message)
        if (body.data.draftIssueId !== params.data.draftIssueId) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: 'draftIssueId path parameter must match the request body.',
            retriable: false,
          })
          return reply
        }
        return updateGitHubProjectDraftCard(token, body.data)
      })
    },
  )
}
