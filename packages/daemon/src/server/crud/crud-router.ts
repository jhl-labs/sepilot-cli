import type { FastifyInstance } from 'fastify'
import { z, type ZodTypeAny } from 'zod'
import { bindCapability } from '../capabilities/bind.js'

export interface CrudRepo<Entity, Upsert> {
  list(query?: { q?: string }): Entity[]
  upsert(input: Upsert): Entity
  remove(id: string): void
}

export interface CrudOptions<_Entity, _Upsert extends { id?: string }> {
  capability: {
    name: string
    version: string
    basePath: string
    description?: string
  }
  schema: ZodTypeAny
  pickQuery?: boolean
  extraMethods?: Array<{
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'WS'
    path: string
  }>
  afterRoutes?: (app: FastifyInstance) => Promise<void> | void
}

export async function crudRouter<E, U extends { id?: string }>(
  app: FastifyInstance,
  repo: CrudRepo<E, U>,
  opts: CrudOptions<E, U>,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: opts.capability.name,
      version: opts.capability.version,
      description: opts.capability.description,
      methods: [
        { method: 'GET', path: opts.capability.basePath },
        { method: 'POST', path: opts.capability.basePath },
        { method: 'DELETE', path: `${opts.capability.basePath}/:id` },
        ...(opts.extraMethods ?? []),
      ],
    },
    async (a) => {
      a.get(opts.capability.basePath, async (req) => {
        const q = opts.pickQuery
          ? z.object({ q: z.string().optional() }).parse(req.query)
          : undefined
        return repo.list(q)
      })
      a.post(opts.capability.basePath, async (req) => {
        const body = opts.schema.parse(req.body) as U
        return repo.upsert(body)
      })
      a.delete(
        `${opts.capability.basePath}/:id`,
        async (req) => {
          const { id } = z
            .object({ id: z.string().min(1) })
            .parse(req.params)
          repo.remove(id)
          return { ok: true }
        },
      )

      if (opts.afterRoutes) await opts.afterRoutes(a)
    },
  )
}
