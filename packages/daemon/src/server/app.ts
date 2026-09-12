import Fastify, { type FastifyInstance } from 'fastify'
import { PassThrough } from 'node:stream'
import { join } from 'node:path'
import fastifyWebsocket from '@fastify/websocket'
import {
  generateOpenApiSpec,
  mergeOpenApiComponents,
  mergeOpenApiOverrides,
  recordDiscoveredOpenApiRoute,
  type DiscoveredOpenApiRoute,
} from './openapi.js'
import { requestIdPlugin } from './request-id.js'
import { requestLoggerPlugin } from './request-logger.js'
import { corsPlugin } from './cors.js'
import { createAuthPlugin, type DaemonAuthTokenResolver } from './auth.js'
import { daemonRateLimitPlugin } from './rate-limiter.js'
import { healthOpenApiComponents, healthOpenApiOverrides, healthRoutes } from './routes/health.js'
import { systemRoutes } from './routes/system.js'
import { registerDesktopAgentRoutes } from './routes/desktop-agents.js'
import { ConnectionRegistry } from './runtime/connection-registry.js'
import {
  metricsOpenApiComponents,
  metricsOpenApiOverrides,
  metricsRoutes,
} from './routes/metrics.js'
import { prometheusOpenApiOverrides, prometheusRoutes } from './routes/metrics-prometheus.js'
import { chatOpenApiComponents, chatOpenApiOverrides, chatRoutes } from './routes/chat.js'
import { runsRoutes } from './routes/runs.js'
import { voiceOpenApiComponents, voiceOpenApiOverrides } from './routes/voice.js'
import {
  sessionsOpenApiComponents,
  sessionsOpenApiOverrides,
  sessionsRoutes,
} from './routes/sessions.js'
import { sessionPromptRoutes } from './routes/session-prompt.js'
import { contextUsageRoutes } from './routes/context-usage.js'
import { toolsIntrospectRoutes } from './routes/tools-introspect.js'
import { registerDocRoutes } from './routes/doc.js'
import { skillsOpenApiComponents, skillsOpenApiOverrides, skillsRoutes } from './routes/skills.js'
import { memoryOpenApiComponents, memoryOpenApiOverrides, memoryRoutes } from './routes/memory.js'
import { configOpenApiComponents, configOpenApiOverrides, configRoutes } from './routes/config.js'
import {
  channelOpenApiComponents,
  channelOpenApiOverrides,
  channelRoutes,
} from './routes/channels.js'
import {
  devicesOpenApiComponents,
  devicesOpenApiOverrides,
  devicesRoutes,
} from './routes/devices.js'
import { mcpOpenApiComponents, mcpOpenApiOverrides } from './routes/mcp.js'
import { usageOpenApiComponents, usageOpenApiOverrides, usageRoutes } from './routes/usage.js'
import {
  observabilityOpenApiComponents,
  observabilityOpenApiOverrides,
  observabilityRoutes,
} from './routes/observability.js'
import {
  webhookOpenApiComponents,
  webhookOpenApiOverrides,
  webhookRoutes,
} from './routes/webhooks.js'
import { cronOpenApiComponents, cronOpenApiOverrides, cronRoutes } from './routes/cron.js'
import { scheduledTasksRoutes } from './routes/scheduled-tasks.js'
import {
  estimateOpenApiComponents,
  estimateOpenApiOverrides,
  estimateRoutes,
} from './routes/estimate.js'
import {
  personaOpenApiComponents,
  personaOpenApiOverrides,
  personaRoutes,
} from './routes/personas.js'
import { chatStreamOpenApiOverrides, chatStreamRoutes } from './routes/chat-stream.js'
import { doctorOpenApiComponents, doctorOpenApiOverrides, doctorRoutes } from './routes/doctor.js'
import { policyRoutes } from './routes/policy.js'
import {
  sessionShareOpenApiComponents,
  sessionShareOpenApiOverrides,
  sessionShareRoutes,
} from './routes/sessions-share.js'
import {
  sessionImportOpenApiComponents,
  sessionImportOpenApiOverrides,
  sessionImportRoutes,
} from './routes/sessions-import.js'
import { sharePublicRoutes } from './routes/share-public.js'
import {
  sessionQuestionsOpenApiComponents,
  sessionQuestionsOpenApiOverrides,
  sessionQuestionsRoutes,
} from './routes/sessions-questions.js'
import {
  sessionAgentOpenApiComponents,
  sessionAgentOpenApiOverrides,
  sessionAgentRoutes,
} from './routes/sessions-agent.js'
import { agentOpenApiComponents, agentOpenApiOverrides, agentRoutes } from './routes/agents.js'
import {
  commandOpenApiComponents,
  commandOpenApiOverrides,
  commandRoutes,
} from './routes/commands.js'
import {
  artifactOpenApiComponents,
  artifactOpenApiOverrides,
  artifactRoutes,
} from './routes/artifacts.js'
import { coworkOpenApiComponents, coworkOpenApiOverrides, coworkRoutes } from './routes/cowork.js'
import {
  skillStoreOpenApiComponents,
  skillStoreOpenApiOverrides,
  skillStoreRoutes,
} from './routes/skill-store.js'
import { marketplacesRoutes } from './routes/marketplaces.js'
import { fileOpenApiComponents, fileOpenApiOverrides, fileRoutes } from './routes/files.js'
import {
  projectOpenApiComponents,
  projectOpenApiOverrides,
  projectRoutes,
} from './routes/projects.js'
import { servicesRoutes } from './routes/services.js'
import {
  approvalOpenApiComponents,
  approvalOpenApiOverrides,
  approvalRoutes,
} from './routes/approvals.js'
import {
  authTokenOpenApiComponents,
  authTokenOpenApiOverrides,
  authTokenRoutes,
} from './routes/auth-tokens.js'
import { secretsRoutes } from './routes/secrets.js'
import { subagentRoutes } from './routes/subagents.js'
import {
  registerBrowserFeatureRoutes,
  registerVoiceFeatureRoutes,
  registerMcpFeatureRoutes,
  registerPagesFeatureRoutes,
  registerMicroAppsFeatureRoutes,
  registerPluginsFeatureRoutes,
  registerAcpFeatureRoutes,
  registerA2aFeatureRoutes,
  registerSwarmFeatureRoutes,
  getPluginsOpenApiComponents,
  getPluginsOpenApiOverrides,
} from '../generated/feature-registration.js'
import { registerJobsRoutes, type JobsBatchExecutor } from './routes/jobs.js'
import { registerPlanRoutes } from './routes/plans.js'
import { createJobsRepo } from '../jobs/repo.js'
import { createJobRunner } from '../jobs/runner.js'
import { JsonWorkPlanStore } from '../plans/store.js'
import { registerMigrationRoutes } from './routes/migration.js'
import { createMigrationRepo } from '../migration/repo.js'
import { sepilotdHome } from '../storage/home.js'
import { capabilityRoutes } from './capability-routes.js'
import { wsRoutes } from './ws.js'
import { errorHandlerPlugin } from './error-handler.js'
import { createObservabilityRepo } from '../observability/events.js'
import { resolveRequestSurface } from './request-surface.js'
import './fastify-types.js'
import type { RuntimeServices } from './runtime.js'
import { ChatKnowledgeProviderRegistry } from './chat-knowledge.js'

export interface AppOptions {
  port: number
  host: string
  runtime?: RuntimeServices
  authToken?: string | null
  authTokenResolver?: DaemonAuthTokenResolver
  authTokenRequired?: boolean
}

function shouldCaptureRawBody(url: string, method: string): boolean {
  if (method !== 'POST') return false
  // Signature verification needs the exact raw bytes only for incoming webhooks.
  return url.startsWith('/api/v1/webhooks/')
}

function shouldRecordRouteEvent(url: string): boolean {
  return !(
    url.startsWith('/api/v1/observability') ||
    url.startsWith('/api/v1/feedback') ||
    url.startsWith('/api/v1/health') ||
    url.startsWith('/api/v1/metrics') ||
    url.startsWith('/api/v1/openapi')
  )
}

function readPositiveIntegerEnv(name: string, fallback: number): number {
  const raw = process.env[name]
  if (!raw) return fallback
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

function resolveMaxWsConnections(): number {
  return readPositiveIntegerEnv('SEPILOTD_MAX_WS_CONNECTIONS', 256)
}

function resolveWsMaxPayloadBytes(): number {
  return readPositiveIntegerEnv('SEPILOTD_WS_MAX_PAYLOAD_BYTES', 1024 * 1024)
}

/**
 * Bound how long a client may take to deliver a complete request. Fastify
 * defaults this to 0, which *disables* Node's own 5-minute `requestTimeout`
 * and lets a slowloris-style client hold a connection open indefinitely by
 * dribbling out the body. That is harmless on a loopback-only daemon but real
 * once the daemon is bound to a LAN/remote address, which the auth policy
 * explicitly supports.
 *
 * This measures receipt of the *request*, not the lifetime of the response, so
 * SSE and WebSocket streams are unaffected. `connectionTimeout` is deliberately
 * left at Fastify's default (0) — it is a socket-inactivity timer and would
 * tear down idle SSE/WS connections.
 */
function resolveRequestTimeoutMs(): number {
  return readPositiveIntegerEnv('SEPILOTD_REQUEST_TIMEOUT_MS', 300_000)
}

/**
 * Whether to derive `request.ip` from `X-Forwarded-For`.
 *
 * Off by default, and that default is the safe one: an untrusted client can
 * put anything in `X-Forwarded-For`, so trusting it on a directly-reachable
 * daemon would let a single caller spoof a fresh IP per request and walk
 * straight past the rate limiter. Left off, every request behind a reverse
 * proxy instead collapses onto the proxy's own address and shares one token
 * bucket — so one busy client can 429 everyone else.
 *
 * Operators who actually terminate at a proxy opt in with
 * `SEPILOTD_TRUST_PROXY`, and should scope it as tightly as they can:
 *
 *   1 | 2 | ...          number of trusted hops in front of the daemon
 *   10.0.0.1,10.0.0.0/8  trust only these proxy addresses
 *   true                 trust any upstream — only when the proxy is the
 *                        daemon's sole ingress
 */
export function resolveTrustProxy(
  raw = process.env.SEPILOTD_TRUST_PROXY,
): boolean | number | string {
  const value = raw?.trim()
  if (!value) return false

  const normalized = value.toLowerCase()
  if (normalized === 'false' || normalized === '0' || normalized === 'off') return false
  if (normalized === 'true' || normalized === 'on') return true

  const hops = Number.parseInt(value, 10)
  if (String(hops) === value && hops > 0) return hops

  // Anything else is an address/CIDR list, which Fastify parses itself. A
  // malformed list throws at construction — louder, and safer, than silently
  // downgrading to "trust everything".
  return value
}

export async function createApp(options: AppOptions): Promise<FastifyInstance> {
  const app = Fastify({
    logger: false,
    requestTimeout: resolveRequestTimeoutMs(),
    trustProxy: resolveTrustProxy(),
  })
  app.addContentTypeParser(
    'application/x-www-form-urlencoded',
    { parseAs: 'string' },
    (_request, body, done) => {
      done(null, Object.fromEntries(new URLSearchParams(String(body))))
    },
  )
  const observability = createObservabilityRepo()
  const openApiRoutes: DiscoveredOpenApiRoute[] = []
  const openApiOverrides = mergeOpenApiOverrides(
    healthOpenApiOverrides,
    metricsOpenApiOverrides,
    prometheusOpenApiOverrides,
    chatOpenApiOverrides,
    voiceOpenApiOverrides,
    chatStreamOpenApiOverrides,
    sessionsOpenApiOverrides,
    skillsOpenApiOverrides,
    memoryOpenApiOverrides,
    configOpenApiOverrides,
    channelOpenApiOverrides,
    devicesOpenApiOverrides,
    mcpOpenApiOverrides,
    usageOpenApiOverrides,
    observabilityOpenApiOverrides,
    webhookOpenApiOverrides,
    cronOpenApiOverrides,
    estimateOpenApiOverrides,
    personaOpenApiOverrides,
    doctorOpenApiOverrides,
    getPluginsOpenApiOverrides(),
    sessionShareOpenApiOverrides,
    sessionImportOpenApiOverrides,
    sessionQuestionsOpenApiOverrides,
    sessionAgentOpenApiOverrides,
    agentOpenApiOverrides,
    commandOpenApiOverrides,
    artifactOpenApiOverrides,
    coworkOpenApiOverrides,
    skillStoreOpenApiOverrides,
    fileOpenApiOverrides,
    projectOpenApiOverrides,
    approvalOpenApiOverrides,
    authTokenOpenApiOverrides,
  )
  const openApiComponents = mergeOpenApiComponents(
    healthOpenApiComponents,
    metricsOpenApiComponents,
    chatOpenApiComponents,
    voiceOpenApiComponents,
    sessionsOpenApiComponents,
    skillsOpenApiComponents,
    memoryOpenApiComponents,
    configOpenApiComponents,
    channelOpenApiComponents,
    devicesOpenApiComponents,
    mcpOpenApiComponents,
    usageOpenApiComponents,
    observabilityOpenApiComponents,
    webhookOpenApiComponents,
    cronOpenApiComponents,
    estimateOpenApiComponents,
    personaOpenApiComponents,
    doctorOpenApiComponents,
    getPluginsOpenApiComponents(),
    sessionShareOpenApiComponents,
    sessionImportOpenApiComponents,
    sessionQuestionsOpenApiComponents,
    sessionAgentOpenApiComponents,
    agentOpenApiComponents,
    commandOpenApiComponents,
    artifactOpenApiComponents,
    coworkOpenApiComponents,
    skillStoreOpenApiComponents,
    fileOpenApiComponents,
    projectOpenApiComponents,
    approvalOpenApiComponents,
    authTokenOpenApiComponents,
  )

  const connectionRegistry = new ConnectionRegistry({
    maxConnectionsByKind: { ws: resolveMaxWsConnections() },
  })
  const chatKnowledgeProviders = new ChatKnowledgeProviderRegistry()
  app.decorate('runtime', options.runtime)
  app.decorate('authToken', options.authToken ?? null)
  app.decorate(
    'authTokenRequired',
    options.authTokenRequired ?? Boolean(options.authToken || options.authTokenResolver),
  )
  app.decorate('connectionRegistry', connectionRegistry)
  app.decorate('chatKnowledgeProviders', chatKnowledgeProviders)
  // Every successful inbound request (anything that isn't a 5xx) counts as
  // activity so the idle reaper does not kill a daemon that's actively
  // servicing one-shot `sepilot ask` calls or background batch jobs.
  app.addHook('onResponse', async (_request, reply) => {
    if (reply.statusCode < 500) connectionRegistry.bumpActivity()
  })
  app.addHook('onRoute', (routeOptions) => {
    recordDiscoveredOpenApiRoute(openApiRoutes, routeOptions)
  })
  app.addHook('preParsing', (request, _reply, payload, done) => {
    if (!shouldCaptureRawBody(request.url, request.method)) {
      done(null, payload)
      return
    }

    const tee = new PassThrough() as PassThrough & { receivedEncodedLength?: number }
    const chunks: Buffer[] = []
    tee.receivedEncodedLength = 0

    payload.on('data', (chunk) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      chunks.push(buffer)
      tee.receivedEncodedLength = (tee.receivedEncodedLength ?? 0) + buffer.length
      tee.write(buffer)
    })
    payload.on('end', () => {
      request.rawBody = Buffer.concat(chunks).toString('utf8')
      tee.end()
    })
    payload.on('error', (error) => {
      tee.destroy(error)
    })

    done(null, tee)
  })
  app.addHook('onResponse', async (request, reply) => {
    if (!shouldRecordRouteEvent(request.url)) return
    const statusCode = reply.statusCode
    const eventType = statusCode >= 500 ? 'route.error' : 'route.request'
    try {
      observability.recordEvents([
        {
          source: 'daemon',
          surface: resolveRequestSurface(request) ?? undefined,
          eventType,
          severity: statusCode >= 500 ? 'error' : statusCode >= 400 ? 'warning' : 'debug',
          privacy: 'operational',
          attributes: {
            method: request.method,
            route: request.routeOptions.url ?? request.url.split('?')[0],
            statusCode,
            requestId: request.id,
          },
        },
      ])
    } catch {
      // Observability is best-effort and must never affect user traffic.
    }
  })

  // Request ID must be registered first
  await app.register(requestIdPlugin)

  // Request logging after request ID
  await app.register(requestLoggerPlugin)

  // CORS must be registered before auth
  await app.register(corsPlugin)

  // Auth must be registered before routes
  await app.register(createAuthPlugin(options.authToken ?? null, {
    resolveMasterToken: options.authTokenResolver,
    masterTokenRequired: app.authTokenRequired,
  }))

  // Rate limiting after auth
  await app.register(daemonRateLimitPlugin)

  // Global error handler - registered after plugins, before routes
  await app.register(errorHandlerPlugin)
  await app.register(fastifyWebsocket, {
    options: {
      maxPayload: resolveWsMaxPayloadBytes(),
    },
  })

  app.get('/api/v1/openapi.json', async (request) => {
    // Advertise the origin the client actually reached so swagger-ui /
    // generated SDKs target the right host when the daemon is bound to
    // 0.0.0.0 and accessed via LAN. Falls back to loopback.
    const host = request.headers.host
    const serverUrl = host ? `${request.protocol}://${host}` : undefined
    return generateOpenApiSpec(openApiRoutes, openApiOverrides, openApiComponents, { serverUrl })
  })

  await app.register(healthRoutes, { prefix: '/api/v1' })
  await app.register(systemRoutes, { prefix: '/api/v1' })
  await app.register(async (routes) => registerDesktopAgentRoutes(routes), { prefix: '/api/v1' })
  await app.register(metricsRoutes, { prefix: '/api/v1' })
  await app.register(prometheusRoutes, { prefix: '/api/v1' })
  await app.register(chatRoutes, { prefix: '/api/v1' })
  await app.register(runsRoutes, { prefix: '/api/v1' })
  await registerBrowserFeatureRoutes(app, { prefix: '/api/v1' })
  await registerVoiceFeatureRoutes(app, { prefix: '/api/v1' })
  await app.register(sessionsRoutes, { prefix: '/api/v1' })
  await app.register(sessionPromptRoutes, { prefix: '/api/v1' })
  await app.register(contextUsageRoutes, { prefix: '/api/v1' })
  await app.register(toolsIntrospectRoutes, { prefix: '/api/v1' })
  await app.register(
    async (instance) => {
      await registerDocRoutes(instance)
    },
    { prefix: '/api/v1' },
  )
  await app.register(skillsRoutes, { prefix: '/api/v1' })
  await app.register(memoryRoutes, { prefix: '/api/v1' })
  await app.register(configRoutes, { prefix: '/api/v1' })
  await app.register(channelRoutes, { prefix: '/api/v1' })
  await registerMcpFeatureRoutes(app, { prefix: '/api/v1' })
  await app.register(devicesRoutes, { prefix: '/api/v1' })
  await app.register(usageRoutes, { prefix: '/api/v1' })
  await app.register(observabilityRoutes, { prefix: '/api/v1' })
  await app.register(webhookRoutes, { prefix: '/api/v1' })
  await app.register(cronRoutes, { prefix: '/api/v1' })
  await app.register(scheduledTasksRoutes, { prefix: '/api/v1' })
  await app.register(estimateRoutes, { prefix: '/api/v1' })
  await app.register(personaRoutes, { prefix: '/api/v1' })
  await app.register(chatStreamRoutes, { prefix: '/api/v1' })
  await app.register(doctorRoutes, { prefix: '/api/v1' })
  await app.register(policyRoutes, { prefix: '/api/v1' })
  await registerPluginsFeatureRoutes(app, { prefix: '/api/v1' })
  await app.register(sessionShareRoutes, { prefix: '/api/v1' })
  await app.register(sessionImportRoutes, { prefix: '/api/v1' })
  await app.register(sharePublicRoutes)
  await app.register(sessionQuestionsRoutes, { prefix: '/api/v1' })
  await app.register(sessionAgentRoutes, { prefix: '/api/v1' })
  await app.register(agentRoutes, { prefix: '/api/v1' })
  await app.register(commandRoutes, { prefix: '/api/v1' })
  await app.register(artifactRoutes, { prefix: '/api/v1' })
  await app.register(coworkRoutes, { prefix: '/api/v1' })
  await app.register(fileRoutes, { prefix: '/api/v1' })
  await app.register(projectRoutes, { prefix: '/api/v1' })
  await registerPagesFeatureRoutes(app, { prefix: '/api/v1' })
  await app.register(servicesRoutes, { prefix: '/api/v1' })
  await registerMicroAppsFeatureRoutes(app, { prefix: '/api/v1' })
  await app.register(skillStoreRoutes, { prefix: '/api/v1' })
  await app.register(marketplacesRoutes, { prefix: '/api/v1' })
  await app.register(approvalRoutes, { prefix: '/api/v1' })
  await app.register(authTokenRoutes, { prefix: '/api/v1' })
  await app.register(secretsRoutes, { prefix: '/api/v1' })
  await app.register(subagentRoutes, { prefix: '/api/v1' })
  await registerAcpFeatureRoutes(app, { prefix: '/api/v1' })
  await registerA2aFeatureRoutes(app, { prefix: '/api/v1' })
  await registerSwarmFeatureRoutes(app, { prefix: '/api/v1' })

  // Jobs infrastructure (batch + future migration runner).
  const jobsRepo = createJobsRepo()
  // Startup recovery: any job left in pending/running from a previous process
  // can never resume — mark them failed so the API surfaces a terminal status.
  const recoveredJobs = jobsRepo.markInProgressFailed()
  if (recoveredJobs > 0) {
    app.log.warn({ recovered: recoveredJobs }, 'marked in-progress jobs as failed on startup')
  }
  const jobRunner = createJobRunner({
    repo: jobsRepo,
    globalMax: parseInt(process.env.JOBS_GLOBAL_MAX ?? '16', 10),
  })
  const planStore = new JsonWorkPlanStore(join(options.runtime?.dataDir ?? sepilotdHome(), 'plans'))
  await planStore.init()
  const startSubagentPlan = async (input: {
    prompt: string
    system?: string
    category?: string
    agentId?: string
    maxIterations?: number
    tools?: string[]
    model?: string
    parentSessionId?: string
  }) => {
    const dispatcher = app.runtime?.subagentDispatcher
    if (!dispatcher) {
      const err = new Error('Subagent dispatcher not initialized') as Error & {
        statusCode?: number
      }
      err.statusCode = 503
      throw err
    }
    const job = jobsRepo.create({ kind: 'subagent', total: 1, concurrency: 1 })
    jobsRepo.insertItems(job.id, [{ idx: 0, request: input }])
    void jobRunner
      .run({
        jobId: job.id,
        concurrency: 1,
        failureMode: 'abort',
        execute: (item) => dispatcher.dispatch(item.request as typeof input),
      })
      .catch(() => {
        /* runner persists error itself */
      })
    return {
      jobId: job.id,
      status: 'pending',
      total: 1,
      createdAt: job.createdAt,
    }
  }
  const batchExecutor: JobsBatchExecutor = {
    async executeItem({ request }) {
      // Re-enter the daemon's own /chat route in-process so the batch executor
      // benefits from the existing handler (sessions, providers, agent loop).
      const res = await app.inject({
        method: 'POST',
        url: '/api/v1/chat',
        payload: request as Record<string, unknown>,
      })
      if (res.statusCode >= 400) {
        let message = `chat request failed (${res.statusCode})`
        try {
          const body = res.json() as { error?: { message?: string } }
          if (body?.error?.message) message = body.error.message
        } catch {
          /* non-JSON error body */
        }
        throw new Error(message)
      }
      return res.json()
    },
  }
  await app.register(
    async (instance) => {
      registerJobsRoutes(instance, {
        repo: jobsRepo,
        runner: jobRunner,
        executor: batchExecutor,
        subagentDispatcher: () => app.runtime?.subagentDispatcher ?? null,
        jobMaxConcurrency: parseInt(process.env.JOBS_MAX_CONCURRENCY ?? '8', 10),
      })
    },
    { prefix: '/api/v1' },
  )
  await app.register(
    async (instance) => {
      registerPlanRoutes(instance, {
        store: planStore,
        startSubagent: startSubagentPlan,
      })
    },
    { prefix: '/api/v1' },
  )

  // Migration runner (Phase 2): /api/v1/migration/* — daemon-side runner that
  // ports the legacy `cli migrate` step pipeline behind HTTP. Shares jobs.db.
  const migrationRepo = createMigrationRepo()
  const recoveredMigrations = migrationRepo.markInProgressFailed()
  if (recoveredMigrations > 0) {
    app.log.warn(
      { recovered: recoveredMigrations },
      'marked in-progress migrations as failed on startup',
    )
  }
  const migrationCancellers = new Map<string, AbortController>()
  const selfOrigin = `http://127.0.0.1:${options.port}`
  await app.register(
    async (instance) => {
      registerMigrationRoutes(instance, {
        repo: migrationRepo,
        selfOrigin,
        selfToken: options.authToken ?? null,
        targetHome: sepilotdHome(),
        cancellers: migrationCancellers,
      })
    },
    { prefix: '/api/v1' },
  )

  // Phase D0-D6 capability routes (desktop UI/UX parity; no /api/v1 prefix).
  await app.register(capabilityRoutes)
  if (options.runtime) {
    await app.register(wsRoutes)
  }

  return app
}
