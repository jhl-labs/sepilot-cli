import Fastify, { type FastifyInstance } from 'fastify'
import { SessionTitleUpdates } from '../agent/session-title.js'
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
  roleBotRoutes,
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
  sessionPrivilegeRoutes,
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
  registerEventRoutingFeatureRoutes,
  installEventRoutingLlmRunner,
  stopEventRoutingFeature,
  getPluginsOpenApiComponents,
  getPluginsOpenApiOverrides,
} from '../generated/feature-registration.js'
import { registerJobsRoutes, type JobsBatchExecutor } from './routes/jobs.js'
import { registerPlanRoutes } from './routes/plans.js'
import { createJobsRepo } from '../jobs/repo.js'
import { createJobRunner } from '../jobs/runner.js'
import { createBackgroundHooks } from '../jobs/hooks.js'
import { executeSubagentJob, createBackgroundSubagents } from '../jobs/subagent.js'
import { publishStoredNotification } from '../notifications/publish.js'
import { SessionInbox, setSessionInboxWakeListener } from './runtime/session-inbox.js'
import { SessionWakeService, resolveMaxWakesPerHour, sessionWakeEnabledFromEnv } from './runtime/session-wake.js'
import type { SessionRunContext } from './runtime/active-runs.js'
import { registerWorkRoutes } from './routes/work.js'
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

function resolveRequestTimeoutMs(): number {
  return readPositiveIntegerEnv('SEPILOTD_REQUEST_TIMEOUT_MS', 300_000)
}

export function resolveTrustProxy(
  raw = process.env.SEPILOTD_TRUST_PROXY,
): boolean | string {
  const value = raw?.trim()
  if (!value) return false

  const normalized = value.toLowerCase()
  if (normalized === 'false' || normalized === '0' || normalized === 'off') return false
  if (normalized === 'true' || normalized === 'on') return true

  const hops = Number.parseInt(value, 10)
  if (String(hops) === value && hops > 0) return false

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
  const sessionTitleUpdates = new SessionTitleUpdates()
  app.decorate('sessionTitleUpdates', sessionTitleUpdates)
  app.addHook('preClose', async () => { sessionTitleUpdates.close() })
  app.decorate('authToken', options.authToken ?? null)
  app.decorate(
    'authTokenRequired',
    options.authTokenRequired ?? Boolean(options.authToken || options.authTokenResolver),
  )
  app.decorate('connectionRegistry', connectionRegistry)
  app.decorate('chatKnowledgeProviders', chatKnowledgeProviders)
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
    }
  })

  await app.register(requestIdPlugin)

  await app.register(requestLoggerPlugin)

  await app.register(corsPlugin)

  await app.register(createAuthPlugin(options.authToken ?? null, {
    resolveMasterToken: options.authTokenResolver,
    masterTokenRequired: app.authTokenRequired,
  }))

  await app.register(daemonRateLimitPlugin)

  await app.register(errorHandlerPlugin)
  await app.register(fastifyWebsocket, {
    options: {
      maxPayload: resolveWsMaxPayloadBytes(),
    },
  })

  app.get('/api/v1/openapi.json', async (request) => {
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
  await registerEventRoutingFeatureRoutes(app, { prefix: '/api/v1' })
  await app.register(cronRoutes, { prefix: '/api/v1' })
  await app.register(scheduledTasksRoutes, { prefix: '/api/v1' })
  await app.register(estimateRoutes, { prefix: '/api/v1' })
  await app.register(personaRoutes, { prefix: '/api/v1' })
  await app.register(roleBotRoutes, { prefix: '/api/v1' })
  await app.register(chatStreamRoutes, { prefix: '/api/v1' })
  await app.register(doctorRoutes, { prefix: '/api/v1' })
  await app.register(policyRoutes, { prefix: '/api/v1' })
  await registerPluginsFeatureRoutes(app, { prefix: '/api/v1' })
  await app.register(sessionShareRoutes, { prefix: '/api/v1' })
  await app.register(sessionImportRoutes, { prefix: '/api/v1' })
  await app.register(sharePublicRoutes)
  await app.register(sessionQuestionsRoutes, { prefix: '/api/v1' })
  await app.register(sessionAgentRoutes, { prefix: '/api/v1' })
  await app.register(sessionPrivilegeRoutes, { prefix: '/api/v1' })
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

  const jobsRepo = createJobsRepo()
  const sessionInbox = new SessionInbox()
  app.runtime?.activeRuns?.setBackgroundEvidenceReader(id => sessionInbox.evidence(id))
  const sessionBusy = app.runtime?.sessionBusy
  const sessionWake = new SessionWakeService({
    pendingWake: id => sessionInbox.pendingWake(id),
    isBusy: id => sessionBusy?.isBusy(id) ?? false,
    lastRunContext: id => app.runtime?.activeRuns?.lastRunContext(id),
    enabled: () => sessionWakeEnabledFromEnv(),
    maxWakesPerHour: resolveMaxWakesPerHour(),
    startTurn: input => startSessionWakeTurn(app, input),
  })
  installEventRoutingLlmRunner(app)
  setSessionInboxWakeListener(id => sessionWake.request(id))
  const stopIdleObserver = sessionBusy?.onIdle(id => {
    sessionInbox.completeTurn(id)
    sessionWake.turnEnded(id)
  })
  app.addHook('onReady', async () => {
    for (const id of sessionInbox.sessionsWithPendingWake()) sessionWake.request(id)
  })
  app.addHook('preClose', async () => {
    if (app.runtime) app.runtime.eventRoutingLlmRunner = undefined
    try { if (app.runtime) await stopEventRoutingFeature(app.runtime) } catch (error) {
      app.log.warn({ err: error }, 'event routing stop failed during close')
    }
    stopIdleObserver?.()
    setSessionInboxWakeListener(undefined)
    sessionWake.dispose()
  })
  try {
    for (const job of jobsRepo.listInProgress().filter(entry => entry.kind === 'subagent')) {
      const parentSessionId = jobsRepo.listItems(job.id, 0)[0]?.sessionId
      if (!parentSessionId) continue
      const childSessionId = jobsRepo.listActivity(job.id)
        .map(activity => activity.sessionId)
        .find(id => id && id !== parentSessionId)
      sessionInbox.tryPublish({
        sessionId: parentSessionId,
        source: 'subagent',
        sourceId: job.id,
        eventKey: `${job.id}:interrupted`,
        title: 'Background subagent interrupted by daemon restart',
        body: [
          `Background subagent job ${job.id} was running when the daemon restarted and did not finish. Nothing was retried automatically.`,
          childSessionId
            ? `Its completed steps are in child session ${childSessionId}; continue it from its last checkpoint with sepilot sessions resume ${childSessionId}, or dispatch the remaining work again.`
            : 'Dispatch the remaining work again if it is still needed.',
        ].join(' '),
        wake: true,
      })
    }
  } catch (error) {
    app.log.warn({ err: error }, 'could not report interrupted background subagents')
  }
  const recoveredJobs = jobsRepo.markInProgressFailed()
  if (recoveredJobs > 0) {
    app.log.warn({ recovered: recoveredJobs }, 'marked in-progress jobs as failed on startup')
  }
  const jobRunner = createJobRunner({
    repo: jobsRepo,
    globalMax: parseInt(process.env.JOBS_GLOBAL_MAX ?? '16', 10),
  })
  app.addHook('preClose', async () => { jobRunner.shutdown() })
  app.runtime?.hookRegistry?.setBackgroundExecutor?.(createBackgroundHooks({ repo: jobsRepo, runner: jobRunner,
    onFinished: (jobId, sessionId, status) => { sessionInbox.tryPublish({ sessionId, source: 'hook', sourceId: jobId,
      eventKey: jobId, title: `Background hook ${status}`, body: `Collect retained evidence with sepilot jobs resume ${jobId}; this does not rerun the hook.` }) },
  }))
  const onSubagentFinished: NonNullable<Parameters<typeof createBackgroundSubagents>[0]['onFinished']> = (job, parentSessionId) => {
        sessionInbox.tryPublish({ sessionId: parentSessionId, source: 'subagent', sourceId: job.id,
          eventKey: job.id, title: `Background subagent ${job.status}`,
          wake: !parentSessionId.startsWith('api-subagent:'),
          body: `Job ${job.id}. Collect retained results with subagent.job or sepilot jobs resume ${job.id} (no re-execution). For interrupted work inspect the child session in job activity and use sepilot sessions resume <child-session-id> only when a safe checkpoint is available.` })
        publishStoredNotification({
          id: `subagent-job:${job.id}`, title: `Background subagent ${job.status}`,
          body: `Job ${job.id} · parent session ${parentSessionId}. Inspect with sepilot jobs status ${job.id}.`,
          topic: 'subagent-job', audience: ['cli', 'desktop', 'web'],
        })
  }
  if (app.runtime?.subagentDispatcher) {
    app.runtime.backgroundSubagents = createBackgroundSubagents({
      dispatcher: app.runtime.subagentDispatcher, repo: jobsRepo, runner: jobRunner,
      onFinished: onSubagentFinished,
    })
  }
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
        execute: (item, signal) => executeSubagentJob(dispatcher, jobsRepo, item, signal),
      })
      .then(() => {
        const final = jobsRepo.get(job.id)
        if (final && input.parentSessionId) onSubagentFinished(final, input.parentSessionId)
      })
      .catch(() => {
      })
    return {
      jobId: job.id,
      status: 'pending',
      total: 1,
      createdAt: job.createdAt,
    }
  }
  const batchExecutor: JobsBatchExecutor = {
    async executeItem({ request }, signal) {
      const res = await app.inject({
        method: 'POST',
        url: '/api/v1/chat',
        signal,
        payload: request as Record<string, unknown>,
      })
      if (res.statusCode >= 400) {
        let message = `chat request failed (${res.statusCode})`
        try {
          const body = res.json() as { error?: { message?: string } }
          if (body?.error?.message) message = body.error.message
        } catch {
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
        onSubagentFinished,
        jobMaxConcurrency: parseInt(process.env.JOBS_MAX_CONCURRENCY ?? '8', 10),
      })
      registerWorkRoutes(instance, jobsRepo)
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

  await app.register(capabilityRoutes)
  if (options.runtime) {
    await app.register(wsRoutes)
  }

  return app
}

const WAKE_TURN_MODES = new Set(['instant', 'react'])

export function buildWakeTurnRequest(input: {
  sessionId: string
  message: string
  cwd?: string
  context?: SessionRunContext
  authToken?: string | null
}): { payload: Record<string, unknown>; headers: Record<string, string> } {
  const context = input.context
  const payload: Record<string, unknown> = { message: input.message, sessionId: input.sessionId }
  payload.mode = context?.mode && WAKE_TURN_MODES.has(context.mode) ? context.mode : 'react'
  const cwd = context?.cwd ?? input.cwd
  if (cwd) payload.cwd = cwd
  if (context?.workspaceRoot) payload.workspaceRoot = context.workspaceRoot
  if (context?.provider && context.model) {
    payload.provider = context.provider
    payload.model = context.model
  }
  if (context?.autonomy) payload.autonomy = context.autonomy
  if (context?.toolNames) payload.toolNames = context.toolNames
  if (context?.bot) payload.bot = context.bot
  const headers: Record<string, string> = {}
  if (input.authToken) headers.authorization = `Bearer ${input.authToken}`
  if (context?.surface) headers['x-sepilotd-surface'] = context.surface
  return { payload, headers }
}

async function startSessionWakeTurn(
  app: FastifyInstance,
  input: { sessionId: string; message: string; context?: SessionRunContext },
): Promise<void> {
  const session = app.runtime?.sessions?.get
    ? await app.runtime.sessions.get(input.sessionId).catch(() => null)
    : null
  if (!session) throw new Error(`Session ${input.sessionId} is not available for a background wake`)
  const { payload, headers } = buildWakeTurnRequest({
    sessionId: input.sessionId,
    message: input.message,
    cwd: session.cwd,
    context: input.context,
    authToken: (app as unknown as { authToken?: string | null }).authToken,
  })
  const response = await app.inject({ method: 'POST', url: '/api/v1/chat/background', headers, payload })
  if (response.statusCode !== 202) {
    throw new Error(`Background wake turn was rejected (${response.statusCode}): ${response.body.slice(0, 200)}`)
  }
}
