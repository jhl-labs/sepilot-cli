import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import {
  ExtensionRegistry,
  extensionManifestSchema,
  toExtensionItem,
} from '../../extensions/registry.js'

const extensionIdParamsSchema = z.object({
  id: z.string().trim().min(1),
})

const extensionInstallRequestSchema = z.object({
  manifest: extensionManifestSchema,
})

export async function registerExtensionsCapabilityRoutes(
  app: FastifyInstance,
): Promise<void> {
  let registry: ExtensionRegistry | null = null
  const sockets = new Set<{ readyState: number; send: (data: string) => void }>()

  const getRegistry = (): ExtensionRegistry | null => {
    const runtime = app.runtime
    if (!runtime) {
      return null
    }
    registry ??= new ExtensionRegistry(runtime.dataDir)
    return registry
  }

  const listItems = async () => {
    const currentRegistry = getRegistry()
    if (!currentRegistry) {
      return null
    }
    const records = await currentRegistry.list()
    return records.map(toExtensionItem)
  }

  const broadcastSnapshot = async (): Promise<void> => {
    if (sockets.size === 0) {
      return
    }

    const items = await listItems()
    if (!items) {
      return
    }

    const payload = JSON.stringify(items)
    for (const socket of [...sockets]) {
      try {
        if (socket.readyState === 1) {
          socket.send(payload)
          continue
        }
      } catch {
        // Drop dead sockets below.
      }
      sockets.delete(socket)
    }
  }

  await bindCapability(
    app,
    {
      name: 'extensions',
      version: '1',
      methods: [
        { method: 'GET', path: '/extensions' },
        { method: 'POST', path: '/extensions/install' },
        { method: 'POST', path: '/extensions/:id/enable' },
        { method: 'POST', path: '/extensions/:id/disable' },
        { method: 'DELETE', path: '/extensions/:id' },
        { method: 'GET', path: '/extensions/events' },
      ],
    },
    async (a) => {
      a.get('/extensions', async (_req, reply) => {
        const items = await listItems()
        if (!items) {
          return reply.status(503).send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })
        }
        return items
      })

      a.post('/extensions/install', async (req, reply) => {
        const parsed = extensionInstallRequestSchema.safeParse(req.body)
        if (!parsed.success) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
        }

        const currentRegistry = getRegistry()
        if (!currentRegistry) {
          return reply.status(503).send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })
        }

        await currentRegistry.install(parsed.data.manifest)
        await broadcastSnapshot()
        return { id: parsed.data.manifest.id }
      })

      a.post('/extensions/:id/enable', async (req, reply) => {
        const parsed = extensionIdParamsSchema.safeParse(req.params)
        if (!parsed.success) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
        }

        const currentRegistry = getRegistry()
        if (!currentRegistry) {
          return reply.status(503).send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })
        }

        const record = await currentRegistry.setEnabled(parsed.data.id, true)
        if (!record) {
          return reply.status(404).send({
            code: 'NOT_FOUND',
            message: `Extension not found: ${parsed.data.id}`,
            retriable: false,
          })
        }

        await broadcastSnapshot()
        return toExtensionItem(record)
      })

      a.post('/extensions/:id/disable', async (req, reply) => {
        const parsed = extensionIdParamsSchema.safeParse(req.params)
        if (!parsed.success) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
        }

        const currentRegistry = getRegistry()
        if (!currentRegistry) {
          return reply.status(503).send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })
        }

        const record = await currentRegistry.setEnabled(parsed.data.id, false)
        if (!record) {
          return reply.status(404).send({
            code: 'NOT_FOUND',
            message: `Extension not found: ${parsed.data.id}`,
            retriable: false,
          })
        }

        await broadcastSnapshot()
        return toExtensionItem(record)
      })

      a.delete('/extensions/:id', async (req, reply) => {
        const parsed = extensionIdParamsSchema.safeParse(req.params)
        if (!parsed.success) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
        }

        const currentRegistry = getRegistry()
        if (!currentRegistry) {
          return reply.status(503).send({
            error: {
              code: 'SERVICE_UNAVAILABLE',
              message: 'Runtime not initialized',
            },
          })
        }

        const removed = await currentRegistry.uninstall(parsed.data.id)
        if (!removed) {
          return reply.status(404).send({
            code: 'NOT_FOUND',
            message: `Extension not found: ${parsed.data.id}`,
            retriable: false,
          })
        }

        await broadcastSnapshot()
        return reply.status(204).send()
      })

      a.get('/extensions/events', { websocket: true }, async (socket) => {
        const currentRegistry = getRegistry()
        if (!currentRegistry) {
          try {
            socket.send(JSON.stringify({
              error: {
                code: 'SERVICE_UNAVAILABLE',
                message: 'Runtime not initialized',
              },
            }))
          } finally {
            socket.close()
          }
          return
        }

        const trackedSocket = socket as typeof socket & {
          readyState: number
          send: (data: string) => void
        }
        sockets.add(trackedSocket)
        try {
          const items = (await currentRegistry.list()).map(toExtensionItem)
          trackedSocket.send(JSON.stringify(items))
        } catch {
          sockets.delete(trackedSocket)
          try {
            trackedSocket.close()
          } catch {
            // Ignore close failures.
          }
          return
        }

        socket.on('close', () => {
          sockets.delete(trackedSocket)
        })
      })
    },
  )
}
