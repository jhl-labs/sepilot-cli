import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { crudRouter } from '../server/crud/crud-router.js'
import { PersonaInput, type Persona } from './schema.js'
import { createPersonaRepo } from './repo.js'

export async function registerPersonaRoutes(
  app: FastifyInstance,
): Promise<void> {
  const repo = createPersonaRepo()
  await crudRouter<Persona, PersonaInput>(
    app,
    {
      list: () => repo.list(),
      upsert: (input) => repo.upsert(input),
      remove: (id) => repo.remove(id),
    },
    {
      capability: {
        name: 'persona',
        version: '1',
        basePath: '/persona',
      },
      schema: PersonaInput,
      afterRoutes: async (a) => {
        a.post('/persona/:id/activate', async (req) => {
          const { id } = z
            .object({ id: z.string().min(1) })
            .parse(req.params)
          repo.activate(id)
          return { ok: true }
        })
      },
    },
  )
}
