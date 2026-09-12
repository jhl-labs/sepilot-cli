import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { updateManagedEnvFile } from '../../config/env-file.js'
import { reconfigureRuntimeProviders } from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  configEnvUpdateRequestSchema,
  type ConfigEnvUpdateBody,
} from './config-schema.js'

export function registerConfigEnvRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.put<{ Body: ConfigEnvUpdateBody }>('/config/env', {
    preValidation: zodRequestValidation({
      body: {
        schema: configEnvUpdateRequestSchema,
        message: 'Invalid config env update request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const body = request.body
    const result = await runtime.configMutationService.apply(
      'config.env.update',
      async () => {
        const updateResult = await updateManagedEnvFile(runtime.dataDir, body.updates)
        await reconfigureRuntimeProviders(runtime)
        return updateResult
      },
    )

    return {
      data: {
        updated: result.updated,
        removed: result.removed,
        path: result.path,
      },
    }
  })
}
