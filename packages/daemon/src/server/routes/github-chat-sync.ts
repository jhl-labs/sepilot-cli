import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import {
  getChatSyncConfig,
  runChatSync,
  updateChatSyncConfig,
} from '../../github/chat-sync.js'
import { getAccessToken } from '../../github/oauth.js'

const chatSyncConfigUpdateSchema = z.object({
  enabled: z.boolean().optional(),
  repoFullName: z.string().trim().min(1).nullable().optional(),
  branch: z.string().trim().min(1).optional(),
})

export async function registerGitHubChatSyncRoutes(
  app: FastifyInstance,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'github-chat-sync',
      version: '1',
      methods: [
        { method: 'GET', path: '/github/chat-sync' },
        { method: 'PUT', path: '/github/chat-sync' },
        { method: 'POST', path: '/github/chat-sync/run' },
      ],
    },
    async (a) => {
      a.get('/github/chat-sync', async () => getChatSyncConfig())
      a.put('/github/chat-sync', async (req, reply) => {
        const parsed = chatSyncConfigUpdateSchema.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        return updateChatSyncConfig(parsed.data)
      })
      a.post('/github/chat-sync/run', async (_req, reply) => {
        const token = getAccessToken()
        if (!token) {
          void reply.status(401).send({
            code: 'GITHUB_NOT_CONNECTED',
            message: 'GitHub account is not connected.',
            retriable: false,
          })
          return reply
        }
        const runtime = app.runtime
        if (!runtime) {
          void reply.status(503).send({
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized.',
            retriable: true,
          })
          return reply
        }
        return runChatSync({
          token,
          sessions: runtime.sessions,
          encryption: runtime.encryption,
          deviceName: runtime.config.device.name,
        })
      })
    },
  )
}
