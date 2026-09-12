import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { crudRouter } from '../server/crud/crud-router.js'
import { PersonalDocInput, type PersonalDoc } from './schema.js'
import { createPersonalDocsRepo } from './repo.js'

export async function registerPersonalDocsRoutes(
  app: FastifyInstance,
): Promise<void> {
  const repo = createPersonalDocsRepo()
  await crudRouter<PersonalDoc, PersonalDocInput>(app, repo, {
    capability: {
      name: 'personal-docs',
      version: '1',
      basePath: '/personal-docs',
    },
    schema: PersonalDocInput,
    extraMethods: [
      { method: 'GET', path: '/personal-docs/:id' },
    ],
    afterRoutes: (a) => {
      a.get('/personal-docs/:id', async (req, reply) => {
        const { id } = z
          .object({ id: z.string().min(1) })
          .parse(req.params)
        const document = repo.get(id)
        if (!document) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: `Personal document not found: ${id}`,
            retriable: false,
          })
          return reply
        }
        return document
      })
    },
  })
}
