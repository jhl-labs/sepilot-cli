import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../server/capabilities/bind.js'
import { createDocsGitService } from './service.js'
import { createDocsGitStore } from './store.js'

export async function registerDocsGitRoutes(app: FastifyInstance, changed: () => void) {
  const service = createDocsGitService(createDocsGitStore(), changed)
  await app.register(async (api) => {
    api.addHook('preHandler', async (request) => {
      if (request.authContext?.kind === 'extension')
        throw Object.assign(new Error('Personal Docs Git requires a first-party client'), {
          statusCode: 403,
        })
    })
    await bindCapability(
      api,
      {
        name: 'docs-git',
        version: '1',
        methods: [
          { method: 'GET', path: '/docs-git' },
          { method: 'POST', path: '/docs-git/connect' },
          { method: 'POST', path: '/docs-git/disconnect' },
          { method: 'POST', path: '/docs-git/preview' },
          { method: 'POST', path: '/docs-git/sync' },
        ],
      },
      async (routes) => {
        routes.get('/docs-git', async () => service.status())
        routes.post('/docs-git/connect', async (req) => {
          await service.connect(req.body)
          return service.status()
        })
        routes.post('/docs-git/disconnect', async () => {
          await service.disconnect()
          return service.status()
        })
        routes.post('/docs-git/preview', async () => service.preview())
        routes.post('/docs-git/sync', async (req) => {
          const input = z
            .object({
              id: z.string().uuid(),
              resolutions: z.record(z.enum(['local', 'remote'])).default({}),
            })
            .strict()
            .parse(req.body)
          await service.sync(input.id, input.resolutions)
          return service.status()
        })
      },
    )
  })
}
