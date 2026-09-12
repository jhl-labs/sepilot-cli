import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import type { ImageGenQueue } from '../../media/image-gen/queue.js'
import type { Provider } from '../../media/image-gen/adapter.js'
import { subscribeJobs } from '../../media/image-gen/events.js'
import { readOutput } from '../../media/image-gen/files.js'
import { collectImageGenHardware } from '../../media/image-gen/hardware.js'
import { createImagePromptPreparer } from '../../media/image-gen/prompt-translation.js'
import {
  localPythonEnvironmentStatus,
  localPythonModelCacheStatuses,
} from '../../media/image-gen/adapters/local-python.js'
import { safeIdSchema } from '../../utils/safe-id.js'

const imageGenFileIdSchema = z
  .string()
  .min(1)
  .max(401)
  .refine((v) => {
    if (v.includes('..') || v.includes('/') || v.includes('\\')) return false
    const [a, b] = v.split(':', 2)
    const validPart = (s: string | undefined): boolean =>
      typeof s === 'string' &&
      s.length > 0 &&
      s.length <= 200 &&
      /^[A-Za-z0-9][A-Za-z0-9_.\-]*$/.test(s)
    return validPart(a) && (b === undefined || validPart(b.replace(/\.png$/, '')))
  }, 'invalid image-gen file id')

export async function registerImageGenCapabilityRoutes(
  app: FastifyInstance,
  queue: ImageGenQueue,
  providers: Map<string, Provider>,
): Promise<void> {
  queue.setPromptPreparer(createImagePromptPreparer(() => app.runtime))

  await bindCapability(
    app,
    {
      name: 'image-gen',
      version: '1',
      methods: [
        { method: 'GET', path: '/image-gen/providers' },
        { method: 'GET', path: '/image-gen/environment' },
        { method: 'GET', path: '/image-gen/model-status' },
        { method: 'GET', path: '/image-gen/hardware' },
        { method: 'GET', path: '/image-gen/jobs' },
        { method: 'GET', path: '/image-gen/jobs/:id' },
        { method: 'POST', path: '/image-gen/jobs' },
        { method: 'POST', path: '/image-gen/jobs/:id/cancel' },
        { method: 'DELETE', path: '/image-gen/jobs/:id' },
        { method: 'GET', path: '/image-gen/files/:id' },
        { method: 'WS', path: '/image-gen/events' },
      ],
    },
    async (a) => {
      a.get('/image-gen/providers', async () => Array.from(providers.values()).map((p) => p.info))
      a.get('/image-gen/environment', async (req) => {
        const { providerId } = z
          .object({
            providerId: z.string().min(1).optional(),
          })
          .parse(req.query)
        if (!providers.has('local-python')) return {}
        if (providerId && providerId !== 'local-python') return {}
        return { 'local-python': await localPythonEnvironmentStatus() }
      })
      a.get('/image-gen/model-status', async (req) => {
        const { providerId } = z
          .object({
            providerId: z.string().min(1).optional(),
          })
          .parse(req.query)
        if (providerId && providerId !== 'local-python') return {}
        return { 'local-python': localPythonModelCacheStatuses() }
      })
      a.get('/image-gen/hardware', async () => collectImageGenHardware())
      a.get('/image-gen/jobs', async (req) => {
        const { limit } = z
          .object({
            limit: z.coerce.number().int().min(1).max(500).default(200),
          })
          .parse(req.query)
        return queue.list(limit)
      })
      a.get('/image-gen/jobs/:id', async (req, reply) => {
        const { id } = z.object({ id: safeIdSchema }).parse(req.params)
        const job = queue.get(id)
        if (!job) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'image job not found',
            retriable: false,
          })
          return reply
        }
        return job
      })
      a.post('/image-gen/jobs', async (req, reply) => {
        const parsed = z
          .object({
            providerId: z.string().min(1),
            prompt: z.string().min(1),
            params: z.record(z.string(), z.unknown()).optional(),
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
        return queue.enqueue(parsed.data)
      })
      a.post('/image-gen/jobs/:id/cancel', async (req) => {
        const { id } = z.object({ id: safeIdSchema }).parse(req.params)
        queue.cancel(id)
        return { ok: true }
      })
      a.delete('/image-gen/jobs/:id', async (req) => {
        const { id } = z.object({ id: safeIdSchema }).parse(req.params)
        return { ok: true, deleted: queue.remove(id) }
      })
      a.get('/image-gen/files/:id', async (req, reply) => {
        const { id } = z.object({ id: imageGenFileIdSchema }).parse(req.params)
        const f = readOutput(id)
        if (!f) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'file not found',
            retriable: false,
          })
          return reply
        }
        void reply.header('content-type', f.mime).send(f.bytes)
        return reply
      })
      a.get('/image-gen/events', { websocket: true }, (socket) => {
        const trackedSocket = socket as typeof socket & {
          readyState: number
          send: (data: string) => void
        }
        const send = (payload: unknown) => {
          try {
            if (trackedSocket.readyState === 1) {
              trackedSocket.send(JSON.stringify(payload))
            }
          } catch {
            try {
              trackedSocket.close()
            } catch {
              // Ignore close failures.
            }
          }
        }
        for (const job of queue.list(200).reverse()) send(job)
        const off = subscribeJobs((event) => send(event))
        socket.on('close', off)
        socket.on('error', off)
      })
    },
  )
}
