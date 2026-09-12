import type { FastifyInstance } from 'fastify'
import { randomUUID } from 'node:crypto'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import { createNotificationsRepo } from '../../notifications/repo.js'
import {
  publishNotification,
  subscribeNotifications,
} from '../../notifications/broker.js'
import { notificationVisibleToSurface } from '../../notifications/audience.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
  trackSseConnection,
} from '../sse-response.js'
import { resolveRequestSurface } from '../request-surface.js'

type NotificationsRepo = ReturnType<typeof createNotificationsRepo>

type NotificationsWatchSnapshot = {
  type: 'snapshot'
  items: ReturnType<NotificationsRepo['list']>
}
type NotificationsSnapshotSubscriber = () => void

export async function registerNotificationsCapabilityRoutes(
  app: FastifyInstance,
): Promise<void> {
  const repo = createNotificationsRepo()
  const snapshotSubscribers = new Set<NotificationsSnapshotSubscriber>()

  function buildWatchSnapshot(surface?: string | null): NotificationsWatchSnapshot {
    return {
      type: 'snapshot',
      items: repo.list({ surface }),
    }
  }

  function publishWatchSnapshot(): void {
    if (snapshotSubscribers.size === 0) return
    for (const subscriber of snapshotSubscribers) subscriber()
  }

  await bindCapability(
    app,
    {
      name: 'notifications',
      version: '1',
      methods: [
        { method: 'GET', path: '/notifications' },
        { method: 'GET', path: '/notifications/watch' },
        { method: 'GET', path: '/notifications/inventory' },
        { method: 'GET', path: '/notifications/:id' },
        { method: 'POST', path: '/notifications' },
        { method: 'POST', path: '/notifications/:id/read' },
        { method: 'POST', path: '/notifications/read-all' },
        { method: 'GET', path: '/notifications/settings' },
        { method: 'PUT', path: '/notifications/settings' },
      ],
    },
    async (a) => {
      a.get('/notifications', async (req, reply) => {
        const parsed = z.object({
          unread: z.enum(['true', 'false']).optional(),
          limit: z.coerce.number().int().min(1).max(200).optional(),
        }).safeParse(req.query)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        return repo.list({
          surface: resolveRequestSurface(req),
          unreadOnly: parsed.data.unread === 'true',
          limit: parsed.data.limit,
        })
      })
      a.get('/notifications/watch', async (req, reply) => {
        const surface = resolveRequestSurface(req)
        reply.hijack()
        reply.raw.writeHead(200, buildSseResponseHeaders(req, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'X-Request-ID': req.requestId ?? randomUUID(),
        }))

        let closed = false
        let unsubscribe: () => void = () => {}
        const send = (event: string, data: unknown) => {
          if (closed) return
          reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
        }
        const heartbeat = setInterval(() => {
          send('notifications', {
            type: 'heartbeat',
            timestamp: new Date().toISOString(),
          })
        }, 15_000)
        heartbeat.unref?.()
        const sendSnapshot = () => send('notifications', buildWatchSnapshot(surface))
        const close = () => {
          if (closed) return
          closed = true
          unsubscribe()
          snapshotSubscribers.delete(sendSnapshot)
          clearInterval(heartbeat)
          if (!reply.raw.destroyed && !reply.raw.writableEnded) {
            reply.raw.end()
          }
        }

        unsubscribe = subscribeNotifications((item) => {
          if (!notificationVisibleToSurface(item.audience, surface)) return
          send('notification', {
            type: 'item',
            item,
          })
        })
        snapshotSubscribers.add(sendSnapshot)

        registerSseDisconnectHandler(req, reply, close)
        trackSseConnection(app, req, reply, { label: 'notifications-watch' })
        send('notifications', buildWatchSnapshot(surface))
      })
      a.get('/notifications/inventory', async (req) => (
        repo.inventory({ surface: resolveRequestSurface(req) })
      ))
      a.get('/notifications/:id', async (req, reply) => {
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        const item = repo.get(id, { surface: resolveRequestSurface(req) })
        if (!item) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'notification not found',
            retriable: false,
          })
          return reply
        }
        return item
      })
      a.post('/notifications', async (req, reply) => {
        const parsed = z
          .object({
            id: z.string().optional(),
            title: z.string().min(1),
            body: z.string().default(''),
            url: z.string().url().nullable().default(null),
            topic: z.string().min(1).nullable().optional(),
            audience: z.array(z.string().min(1)).nullable().optional(),
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
        const item = repo.upsert({
          id: parsed.data.id ?? '',
          title: parsed.data.title,
          body: parsed.data.body,
          url: parsed.data.url,
          topic: parsed.data.topic ?? null,
          audience: parsed.data.audience ?? null,
        })
        publishNotification(item)
        return item
      })
      a.post('/notifications/:id/read', async (req, reply) => {
        const { id } = z
          .object({ id: z.string().min(1) })
          .parse(req.params)
        if (!repo.markRead(id, { surface: resolveRequestSurface(req) })) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'notification not found',
            retriable: false,
          })
          return reply
        }
        publishWatchSnapshot()
        return { ok: true }
      })
      a.post('/notifications/read-all', async (req) => {
        const marked = repo.markAllRead({ surface: resolveRequestSurface(req) })
        if (marked > 0) publishWatchSnapshot()
        return { marked }
      })
      a.get('/notifications/settings', async () => ({
        channels: repo.channelSettings(),
      }))
      a.put('/notifications/settings', async (req, reply) => {
        const parsed = z
          .object({
            channels: z.array(
              z.object({
                id: z.string().min(1),
                enabled: z.boolean(),
              }),
            ),
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
        for (const c of parsed.data.channels) {
          repo.setChannel(c.id, c.enabled)
        }
        return { channels: repo.channelSettings() }
      })
    },
  )
}
