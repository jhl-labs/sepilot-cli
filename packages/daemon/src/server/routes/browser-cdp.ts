import type { FastifyInstance, FastifyRequest } from 'fastify'
import { randomUUID } from 'node:crypto'
import { z } from 'zod'
import type { RequestAuthContext } from '../auth.js'
import {
  CDP_ENABLE_INSTRUCTION,
  CdpDiscoveryError,
  discoverCdpEndpoint,
  type DiscoverCdpOptions,
} from '../../tools/browser-cdp-discovery.js'
import {
  CdpConnection,
  type CdpConnectFn,
  type CdpTab,
} from '../../tools/browser-cdp-transport.js'
import { remoteBrowserBridge } from '../../tools/browser-remote.js'
import { zodRequestValidation } from './utils.js'

export interface BrowserCdpRouteOptions {
  prefix?: string
  discover?: (options: DiscoverCdpOptions) => Promise<{ wsEndpoint: string; userDataDir: string }>
  connect?: CdpConnectFn
}

const HANDLE_TTL_MS = 2 * 60_000
const connectSchema = z.object({ label: z.string().trim().min(1).max(100).optional() })
const attachSchema = z.object({ id: z.string().uuid(), targetId: z.string().min(1).max(200) })
const idSchema = z.object({ id: z.string().uuid() })

interface PendingHandle {
  connection: CdpConnection
  label: string
  tabs: CdpTab[]
  expiresAt: number
}

type StatusReason = 'disabled' | 'not_enabled' | 'unsupported_platform' | 'invalid_endpoint'

function statusReason(error: unknown): StatusReason {
  if (error instanceof CdpDiscoveryError) {
    if (error.code === 'CDP_NOT_ENABLED') return 'not_enabled'
    if (error.code === 'CDP_UNSUPPORTED_PLATFORM') return 'unsupported_platform'
  }
  return 'invalid_endpoint'
}

export async function browserCdpRoutes(app: FastifyInstance, options: BrowserCdpRouteOptions) {
  const discover = options.discover ?? discoverCdpEndpoint
  const handles = new Map<string, PendingHandle>()
  const cdpConfig = () => app.runtime?.config?.browser?.cdp ?? { enabled: true, channel: 'stable' as const }
  const bridge = () => {
    if (!app.runtime?.toolRegistry) throw new Error('Runtime not initialized')
    return remoteBrowserBridge(app.runtime.toolRegistry)
  }
  const expireHandles = () => {
    const now = Date.now()
    for (const [id, handle] of handles) {
      if (handle.expiresAt <= now) {
        handles.delete(id)
        void handle.connection.close()
      }
    }
  }
  app.addHook('onClose', async () => {
    for (const handle of handles.values()) await handle.connection.close()
    handles.clear()
  })
  app.addHook('preHandler', async (request, reply) => {
    const auth = (request as FastifyRequest & { authContext?: RequestAuthContext }).authContext
    if (!auth) return reply.code(401).send({ error: { message: 'Browser control requires authentication' } })
    if (auth.kind !== 'master') return reply.code(403).send({ error: { message: 'Desktop authorization required' } })
  })
  app.setErrorHandler((error, _request, reply) => {
    const detail = error as { statusCode?: number; message?: string }
    reply.code(detail.statusCode ?? 409).send({ error: { message: detail.message ?? 'Browser request failed' } })
  })
  const validate = (schema: z.ZodTypeAny) =>
    zodRequestValidation({ body: { schema, message: 'Invalid browser request' } })

  app.get('/browser/cdp/status', async () => {
    const config = cdpConfig()
    if (!config.enabled) return { available: false, reason: 'disabled' satisfies StatusReason }
    try {
      const { userDataDir } = await discover({ channel: config.channel, userDataDir: config.userDataDir })
      return { available: true, userDataDir }
    } catch (error) {
      return {
        available: false,
        reason: statusReason(error),
        instruction: CDP_ENABLE_INSTRUCTION,
      }
    }
  })

  app.post<{ Body: z.infer<typeof connectSchema> }>(
    '/browser/cdp/connect',
    { preValidation: validate(connectSchema) },
    async (request) => {
      const body = connectSchema.parse(request.body)
      expireHandles()
      const config = cdpConfig()
      if (!config.enabled) throw new Error('Chrome auto-connect is disabled in config (browser.cdp.enabled)')
      const { wsEndpoint } = await discover({ channel: config.channel, userDataDir: config.userDataDir })
      const connection = await CdpConnection.open(wsEndpoint, options.connect)
      const tabs = await connection.listTabs()
      const id = randomUUID()
      handles.set(id, {
        connection,
        label: body.label ?? 'Chrome',
        tabs,
        expiresAt: Date.now() + HANDLE_TTL_MS,
      })
      setTimeout(expireHandles, HANDLE_TTL_MS + 1000).unref?.()
      return { id, tabs }
    },
  )

  app.post<{ Body: z.infer<typeof attachSchema> }>(
    '/browser/cdp/attach',
    { preValidation: validate(attachSchema) },
    async (request) => {
      const body = attachSchema.parse(request.body)
      expireHandles()
      const handle = handles.get(body.id)
      if (!handle) throw new Error('Connect handle expired or already used; connect again')
      handles.delete(body.id)
      const transport = await handle.connection.attach(body.targetId)
      return bridge().connectTransport('master', handle.label, transport)
    },
  )

  app.post<{ Body: z.infer<typeof idSchema> }>(
    '/browser/cdp/cancel',
    { preValidation: validate(idSchema) },
    async (request) => {
      const body = idSchema.parse(request.body)
      const handle = handles.get(body.id)
      handles.delete(body.id)
      if (handle) await handle.connection.close()
      return { ok: true }
    },
  )
}
