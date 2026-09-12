import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import {
  createBackup,
  listBackups,
  removeBackup,
  restoreBackup,
} from '../../backup/service.js'
import { safeIdSchema } from '../../utils/safe-id.js'

export async function registerBackupCapabilityRoutes(
  app: FastifyInstance,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'backup',
      version: '1',
      methods: [
        { method: 'GET', path: '/backup' },
        { method: 'POST', path: '/backup' },
        { method: 'DELETE', path: '/backup/:id' },
        { method: 'POST', path: '/backup/:id/restore' },
      ],
    },
    async (a) => {
      a.get('/backup', async () => listBackups())
      a.post('/backup', async () => createBackup())
      a.delete('/backup/:id', async (req, reply) => {
        const parsed = z.object({ id: safeIdSchema }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: 'invalid backup id',
            retriable: false,
          })
          return reply
        }
        removeBackup(parsed.data.id)
        return { ok: true }
      })
      a.post('/backup/:id/restore', async (req, reply) => {
        const parsed = z.object({ id: safeIdSchema }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: 'invalid backup id',
            retriable: false,
          })
          return reply
        }
        return restoreBackup(parsed.data.id)
      })
    },
  )
}
