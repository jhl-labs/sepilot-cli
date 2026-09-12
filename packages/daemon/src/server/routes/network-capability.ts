import type { FastifyInstance, FastifyReply } from 'fastify'
import type { Dispatcher } from 'undici'
import { z } from 'zod'
import {
  DEFAULT_NETWORK_CONFIG,
  hasLegacyUnsafeProxyMigration,
  networkConfigSchema,
  type SepilotdConfig,
} from '../../config/schema.js'
import { readConfigYaml, writeConfigYamlAtomic } from '../../config/yaml-store.js'
import {
  getProviderDispatcherPlan,
  getProviderHttpDispatcher,
  getProviderNetworkIgnoredOverrideSources,
  getProviderNetworkOverrideSources,
  LEGACY_PROXY_UNSAFE_DEGRADED_REASON,
  planProviderDispatcher,
  ProviderNetworkConfigurationError,
} from '../../providers/http-timeout.js'
import { bindCapability } from '../capabilities/bind.js'
import { applyAndPersistRuntimeUpdate } from '../runtime/config-runtime.js'
import { openJsonWatch } from './watch-sse.js'

export type NetworkConfig = SepilotdConfig['network']
export interface NetworkStatus {
  active: boolean
  effective: {
    proxyMode: NetworkConfig['proxyMode']
    useProxy: boolean
    timeoutMs: number
    customCaPath: string | null
    tlsRejectUnauthorized: boolean
    degradedReason: string | null
  }
  overrides: ReturnType<typeof getProviderNetworkOverrideSources>
  ignoredOverrides: ReturnType<typeof getProviderNetworkIgnoredOverrideSources>
}
type NetworkWatchPayload =
  | { type: 'snapshot'; config: NetworkConfig }
  | { type: 'heartbeat'; timestamp: string }

function readNetwork(app: FastifyInstance): NetworkConfig {
  if (app.runtime) return structuredClone(app.runtime.config.network)
  const raw = readConfigYaml().network
  const parsed = networkConfigSchema.safeParse(raw ?? DEFAULT_NETWORK_CONFIG)
  return parsed.success ? parsed.data : networkConfigSchema.parse(DEFAULT_NETWORK_CONFIG)
}

function readNetworkStatus(app: FastifyInstance): NetworkStatus {
  const configured = readNetwork(app)
  const activePlan = getProviderDispatcherPlan()
  const effective = activePlan ?? planProviderDispatcher(configured)
  return {
    active: activePlan !== null,
    effective: {
      proxyMode: effective.proxyMode,
      useProxy: effective.useProxy,
      timeoutMs: effective.timeoutMs,
      customCaPath: effective.customCaPath,
      tlsRejectUnauthorized: effective.rejectUnauthorized,
      degradedReason: effective.degradedReason,
    },
    overrides: getProviderNetworkOverrideSources(configured, process.env, effective),
    ignoredOverrides: getProviderNetworkIgnoredOverrideSources(configured, effective),
  }
}

async function writeNetwork(app: FastifyInstance, next: NetworkConfig): Promise<NetworkConfig> {
  // Resolve proxy and CA inputs before touching disk or replacing a working
  // global dispatcher. This turns unreadable/invalid CA files into a clear 400.
  planProviderDispatcher(next)

  const runtime = app.runtime
  if (!runtime) {
    const config = readConfigYaml()
    config.network = next
    await writeConfigYamlAtomic(config)
    return next
  }
  if (runtime.configLoadFailed) {
    throw Object.assign(
      new Error(
        'Config was loaded in degraded read-only mode; refusing to overwrite config.yaml until the startup config error is fixed.',
      ),
      { code: 'CONFIG_LOAD_FAILED' as const },
    )
  }

  await runtime.configMutationService.apply('network.update', async () => {
    runtime.config.network = structuredClone(next)
    await applyAndPersistRuntimeUpdate(
      runtime,
      new Set(['network']),
      { network: next },
    )
  })
  return structuredClone(runtime.config.network)
}

export interface NetworkProbeFn {
  (input: { url: string }): Promise<{
    ok: boolean
    reachable?: boolean
    latencyMs?: number
    status?: number
    reason?: string
  }>
}

function networkConfigurationError(reply: FastifyReply, error: unknown) {
  if (error instanceof ProviderNetworkConfigurationError) {
    return reply.status(400).send({
      code: error.code,
      message: error.message,
      retriable: false,
    })
  }
  if (
    error instanceof Error
    && 'code' in error
    && error.code === 'CONFIG_LOAD_FAILED'
  ) {
    return reply.status(409).send({
      code: 'CONFIG_LOAD_FAILED',
      message: error.message,
      retriable: false,
    })
  }
  throw error
}

export async function registerNetworkCapabilityRoutes(
  app: FastifyInstance,
  probe: NetworkProbeFn,
): Promise<void> {
  const watchSubscribers = new Set<(payload: NetworkWatchPayload) => void>()

  function buildWatchSnapshot(): NetworkWatchPayload {
    return {
      type: 'snapshot',
      config: readNetwork(app),
    }
  }

  function publishWatchSnapshot(): void {
    if (watchSubscribers.size === 0) return
    const payload = buildWatchSnapshot()
    for (const subscriber of watchSubscribers) subscriber(payload)
  }

  await bindCapability(
    app,
    {
      name: 'network',
      version: '1',
      methods: [
        { method: 'GET', path: '/network' },
        { method: 'GET', path: '/network/status' },
        { method: 'GET', path: '/network/watch' },
        { method: 'PUT', path: '/network' },
        { method: 'POST', path: '/network/probe' },
      ],
    },
    async (a) => {
      a.get('/network', async () => readNetwork(app))
      a.get('/network/status', async () => readNetworkStatus(app))
      a.get('/network/watch', async (req, reply) => {
        openJsonWatch(req, reply, {
          eventName: 'network',
          subscribers: watchSubscribers,
          buildSnapshot: buildWatchSnapshot,
          buildHeartbeat: () => ({ type: 'heartbeat' as const, timestamp: new Date().toISOString() }),
        })
      })
      a.put('/network', async (req, reply) => {
        const parsed = networkConfigSchema.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        if (hasLegacyUnsafeProxyMigration(parsed.data)) {
          return reply.status(400).send({
            code: LEGACY_PROXY_UNSAFE_DEGRADED_REASON,
            message: 'Choose and save an explicit environment, direct, or manual proxy mode.',
            retriable: false,
          })
        }
        try {
          const saved = await writeNetwork(app, parsed.data)
          publishWatchSnapshot()
          return saved
        } catch (error) {
          return networkConfigurationError(reply, error)
        }
      })
      const ProbeBody = z.object({
        url: z.string().url().superRefine((value, context) => {
          // Zod keeps running refinements after `.url()` has recorded an
          // invalid string. Keep malformed requests on the 400 path.
          let url: URL
          try {
            url = new URL(value)
          } catch {
            return
          }
          if (url.protocol !== 'http:' && url.protocol !== 'https:') {
            context.addIssue({
              code: z.ZodIssueCode.custom,
              message: 'url must use http:// or https://',
            })
          }
          if (url.username || url.password) {
            context.addIssue({
              code: z.ZodIssueCode.custom,
              message: 'url must not contain credentials',
            })
          }
        }),
      })
      a.post('/network/probe', async (req, reply) => {
        const parsed = ProbeBody.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        return probe(parsed.data)
      })
    },
  )
}

type FetchInitWithDispatcher = RequestInit & { dispatcher?: Dispatcher }

export const defaultNetworkProbe: NetworkProbeFn = async ({ url }) => {
  const startedAt = Date.now()
  const plan = getProviderDispatcherPlan()
  const controller = new AbortController()
  const configuredTimeoutMs = plan?.timeoutMs ?? DEFAULT_NETWORK_CONFIG.timeoutMs
  const timeoutMs = configuredTimeoutMs > 0
    ? Math.min(configuredTimeoutMs, 30_000)
    : 30_000
  const timeout = setTimeout(
    () => controller.abort(new Error(`Network probe timed out after ${timeoutMs}ms`)),
    timeoutMs,
  )
  timeout?.unref?.()
  try {
    const dispatcher = getProviderHttpDispatcher()
    const init: FetchInitWithDispatcher = {
      method: 'GET',
      signal: controller.signal,
      ...(dispatcher ? { dispatcher } : {}),
    }
    const response = await fetch(url, init)
    await response.body?.cancel().catch(() => undefined)
    return {
      ok: response.ok,
      reachable: true,
      status: response.status,
      latencyMs: Date.now() - startedAt,
    }
  } catch (error) {
    return {
      ok: false,
      reachable: false,
      latencyMs: Date.now() - startedAt,
      reason: error instanceof Error ? error.message : String(error),
    }
  } finally {
    clearTimeout(timeout)
  }
}
