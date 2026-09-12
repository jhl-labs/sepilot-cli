import { resolveRequestSurface } from '../request-surface.js'
import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { randomUUID } from 'node:crypto'
import { zodRequestValidation } from './utils.js'
import { buildHealthExportSnapshot } from '../health-support.js'
import {
  sessionIdParamsSchema,
  sessionsListQuerySchema,
  sessionExportQuerySchema,
  sessionBranchRequestInputSchema,
  sessionCreateRequestInputSchema,
  sessionLocalShellTurnRequestSchema,
  sessionTranscriptTurnRequestSchema,
  sessionUpdateRequestSchema,
  type SessionBranchBody,
  type SessionCreateBody,
  type SessionLocalShellTurnBody,
  type SessionTranscriptTurnBody,
  type SessionUpdateBody,
  type SessionsListQuery,
  type SessionExportQuery,
  type SessionIdParams,
} from './sessions-schema.js'
import {
  buildSessionDetail,
  buildSessionMetaSnapshot,
  buildSessionPendingApprovals,
} from './session-detail.js'
import { buildSessionRunbook } from '../session-runbook.js'
import {
  buildSessionJsonExport,
  serializeSessionMarkdown,
} from './session-export.js'
import { registerSessionCompactRoute } from './session-compact.js'
import {
  cleanupSessionRuntimeState,
  registerSessionManagementRoutes,
} from './session-management.js'
import { registerSessionResumeRoute } from './session-resume.js'
import { registerSessionUndoRoutes } from './session-undo.js'
import { registerSessionSteerRoute } from './session-steer.js'
import { registerSessionCheckpointRoutes } from './session-checkpoints.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
  trackSseConnection,
} from '../sse-response.js'
import { InvalidCwdError, invalidCwdResponse, resolveRequestCwd } from './request-cwd.js'

export {
  sessionsOpenApiComponents,
  sessionsOpenApiOverrides,
} from './sessions-openapi.js'

export async function sessionsRoutes(app: FastifyInstance) {
  const runtime = app.runtime
  registerSessionCompactRoute(app)
  registerSessionManagementRoutes(app)
  registerSessionResumeRoute(app)
  registerSessionUndoRoutes(app)
  registerSessionCheckpointRoutes(app)
  registerSessionSteerRoute(app)

  const invalidateWorkspaceExecutionState = async (sessionId: string): Promise<void> => {
    runtime?.approvalRegistry?.cancelForSession(
      sessionId,
      'Workspace binding changed before approval was answered',
    )
    runtime?.approvalDecisions?.clear({ sessionId, scope: 'session' })
    await Promise.all([
      runtime?.approvalCheckpoints?.deleteForSession(sessionId) ?? Promise.resolve(),
      runtime?.runCheckpoints?.delete(sessionId) ?? Promise.resolve(),
      runtime?.toolExecutions?.delete(sessionId) ?? Promise.resolve(),
      runtime?.approvalDecisions?.flush() ?? Promise.resolve(),
    ])
    runtime?.sessionRuntimeSnapshots?.delete(sessionId)
  }

  const listSessionsWithPrimaryAgent = async (query: SessionsListQuery) => {
    const result = await runtime!.sessions.list({
      page: query.page,
      perPage: query.perPage,
      query: query.query,
      workspaceRoot: query.workspaceRoot,
    })

    // Status filter applied after fetch — ISessionStore.list doesn't
    // know about session.status today, and threading it through every
    // store impl (jsonl, encrypted) for one cli use case is overkill.
    // For dozens-of-sessions deployments the post-filter cost is
    // negligible; if a daemon ever holds 10k+ sessions we'd push it
    // down to the store.
    const filteredItems = query.status
      ? result.items.filter((session) => session.status === query.status)
      : result.items
    const filteredResult = query.status
      ? {
          ...result,
          items: filteredItems,
          totalCount: filteredItems.length,
        }
      : result

    if (!query.metrics) {
      return {
        ...filteredResult,
        items: filteredItems.map((session) => buildSessionMetaSnapshot(runtime!, session)),
      }
    }

    // Opt-in approval counters: scan each session's events log so the
    // cli list view can render an `appr: R/A/D · auto: N` column
    // without N+1 calls back. Costs an extra getEvents() per item;
    // gated on ?metrics=true so the daemon's default list stays
    // light for web/desktop.
    const itemsWithMetrics = await Promise.all(
      filteredItems.map(async (session) => {
        const meta = buildSessionMetaSnapshot(runtime!, session)
        try {
          const events = await runtime!.sessions.getEvents(session.id)
          let approvalsRequested = 0
          let approvalsApproved = 0
          let approvalsDenied = 0
          let autoApprovalsApproved = 0
          for (const event of events) {
            if (event.type === 'approval_request') approvalsRequested += 1
            else if (event.type === 'approval_response') {
              if (event.decision === 'approved') approvalsApproved += 1
              else if (event.decision === 'denied') approvalsDenied += 1
            }
            else if (event.type === 'auto_approval') {
              if (event.decision === 'approved') autoApprovalsApproved += 1
            }
          }
          return {
            ...meta,
            approvalCounters: {
              approvalsRequested,
              approvalsApproved,
              approvalsDenied,
              autoApprovalsApproved,
            },
          }
        } catch {
          // If a single session's event log is unreadable, skip the
          // metrics for that row rather than failing the whole list
          // — partial data beats blanking the page.
          return meta
        }
      }),
    )

    return {
      ...filteredResult,
      items: itemsWithMetrics,
    }
  }

  app.post<{ Body: SessionCreateBody }>('/sessions', {
    preValidation: zodRequestValidation({
      body: {
        schema: sessionCreateRequestInputSchema,
        message: 'Invalid session create request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const body = request.body
    let resolvedCwd: string | undefined
    if (body.cwd) {
      try {
        resolvedCwd = await resolveRequestCwd(body.cwd)
      } catch (error) {
        if (error instanceof InvalidCwdError) {
          return reply.status(400).send(invalidCwdResponse(error))
        }
        throw error
      }
    }
    const provider = runtime.providerRegistry.getDefault()
    if (!provider) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'No provider',
        },
      })
    }

    const createdAt = new Date().toISOString()
    const session = await runtime.sessions.create({
      id: randomUUID(),
      title: body.title ?? '새 대화',
      createdAt,
      updatedAt: createdAt,
      provider: provider.id,
      model: runtime.config.agent.defaultModel ?? provider.models[0]?.id ?? 'default',
      device: runtime.config.device.name,
      status: 'active',
      ...(resolvedCwd ? { cwd: resolvedCwd } : {}),
      workspaceIsolation: resolveRequestSurface(request) === 'cli' ? 'policy' : 'strict',
      tags: [],
    })
    await runtime.pluginEvents?.emit('session.created', {
      sessionId: session.id,
      provider: session.provider,
      model: session.model,
    })

    return {
      data: buildSessionMetaSnapshot(runtime, session),
    }
  })

  // GET /sessions — list sessions
  app.get<{ Querystring: SessionsListQuery }>('/sessions', {
    preValidation: zodRequestValidation({
      query: {
        schema: sessionsListQuerySchema,
        message: 'Invalid sessions query parameters',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const result = await listSessionsWithPrimaryAgent(request.query)
    return { data: result }
  })

  app.get<{ Querystring: SessionsListQuery }>('/sessions/watch', {
    preValidation: zodRequestValidation({
      query: {
        schema: sessionsListQuerySchema,
        message: 'Invalid sessions query parameters',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    if (!runtime.sessionWatchBroker) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Session watch broker is unavailable',
        },
      })
    }

    reply.hijack()
    reply.raw.writeHead(200, buildSseResponseHeaders(request, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Request-ID': request.requestId ?? randomUUID(),
    }))

    let closed = false
    const send = (event: string, data: unknown) => {
      if (closed) return
      reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
    }
    const close = () => {
      if (closed) return
      closed = true
      unsubscribe()
      clearInterval(heartbeat)
      if (!reply.raw.destroyed && !reply.raw.writableEnded) {
        reply.raw.end()
      }
    }
    const sendSnapshot = async () => {
      send('sessions', {
        type: 'snapshot',
        sessions: await listSessionsWithPrimaryAgent(query),
      })
    }

    const unsubscribe = runtime.sessionWatchBroker.subscribeAll(() => {
      void sendSnapshot().catch(() => {
        close()
      })
    })
    const heartbeat = setInterval(() => {
      send('sessions', {
        type: 'heartbeat',
        timestamp: new Date().toISOString(),
      })
    }, 15_000)
    heartbeat.unref?.()

    registerSseDisconnectHandler(request, reply, close)
    trackSseConnection(app, request, reply, { label: 'sessions-watch' })

    await sendSnapshot()
  })

  app.get<{ Params: SessionIdParams }>('/sessions/:id/approvals', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const session = await runtime.sessions.get(params.id)
    if (!session) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
    const events = await runtime.sessions.getEvents(session.id)
    return { data: await buildSessionPendingApprovals(runtime, session.id, events) }
  })

  // GET /sessions/:id — get session detail
  app.get<{ Params: SessionIdParams }>('/sessions/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const session = await runtime.sessions.get(params.id)
    if (!session) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
    return {
      data: await buildSessionDetail(runtime, session),
    }
  })

  app.get<{ Params: SessionIdParams }>('/sessions/:id/runbook', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const session = await runtime.sessions.get(params.id)
    if (!session) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
    return {
      data: buildSessionRunbook(await buildSessionDetail(runtime, session)),
    }
  })

  app.post<{ Params: SessionIdParams; Body: SessionTranscriptTurnBody }>('/sessions/:id/transcript-turn', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
      body: {
        schema: sessionTranscriptTurnRequestSchema,
        message: 'Invalid transcript turn payload',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const body = request.body
    const now = new Date().toISOString()
    const defaultProvider = runtime.providerRegistry.getDefault()
    let session = await runtime.sessions.get(params.id)
    let resolvedCwd: string | undefined
    if (body.cwd) {
      try {
        resolvedCwd = await resolveRequestCwd(body.cwd)
      } catch (error) {
        if (error instanceof InvalidCwdError) {
          return reply.status(400).send(invalidCwdResponse(error))
        }
        throw error
      }
    }

    if (!session) {
      session = await runtime.sessions.create({
        id: params.id,
        title: body.title ?? '새 대화',
        createdAt: now,
        updatedAt: now,
        provider: body.provider ?? defaultProvider?.id ?? 'external',
        model: body.model ?? runtime.config.agent.defaultModel ?? defaultProvider?.models[0]?.id ?? 'external',
        device: runtime.config.device.name,
        status: 'active',
        cwd: resolvedCwd,
        workspaceIsolation: resolveRequestSurface(request) === 'cli' ? 'policy' : 'strict',
        tags: body.tags ?? [],
      })
      await runtime.pluginEvents?.emit('session.created', {
        sessionId: session.id,
        provider: session.provider,
        model: session.model,
      })
    } else if (runtime.sessions.updateMeta) {
      const patch: Partial<typeof session> = {}
      if (resolvedCwd && session.cwd !== resolvedCwd) {
        return reply.status(409).send({
          error: {
            code: 'WORKSPACE_BINDING_MISMATCH',
            message: 'Transcript import cannot change an existing session workspace',
          },
        })
      }
      if (body.provider && session.provider !== body.provider) patch.provider = body.provider
      if (body.model && session.model !== body.model) patch.model = body.model
      if (
        body.title &&
        session.title !== body.title &&
        (session.title === '새 대화' || session.title === 'New Chat')
      ) {
        patch.title = body.title
      }
      if (Object.keys(patch).length > 0) {
        session = (await runtime.sessions.updateMeta(session.id, patch)) ?? session
      }
    }

    await runtime.sessions.appendEvent(params.id, {
      type: 'user_message',
      id: randomUUID(),
      timestamp: now,
      content: body.userContent,
    })

    const assistantContent = body.assistantContent?.trim()
    if (assistantContent) {
      await runtime.sessions.appendEvent(params.id, {
        type: 'assistant_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content: assistantContent,
      })
    }

    return {
      data: buildSessionMetaSnapshot(runtime, (await runtime.sessions.get(params.id)) ?? session),
    }
  })

  app.post<{ Params: SessionIdParams; Body: SessionLocalShellTurnBody }>('/sessions/:id/local-shell-turn', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
      body: {
        schema: sessionLocalShellTurnRequestSchema,
        message: 'Invalid local shell turn payload',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const body = request.body
    const now = new Date().toISOString()
    const defaultProvider = runtime.providerRegistry.getDefault()
    let session = await runtime.sessions.get(params.id)
    let resolvedCwd: string | undefined
    if (body.cwd) {
      try {
        resolvedCwd = await resolveRequestCwd(body.cwd)
      } catch (error) {
        if (error instanceof InvalidCwdError) {
          return reply.status(400).send(invalidCwdResponse(error))
        }
        throw error
      }
    }

    if (!session) {
      session = await runtime.sessions.create({
        id: params.id,
        title: body.title ?? `!${body.command}`,
        createdAt: now,
        updatedAt: now,
        provider: body.provider ?? defaultProvider?.id ?? 'local-shell',
        model: body.model ?? runtime.config.agent.defaultModel ?? defaultProvider?.models[0]?.id ?? 'local-shell',
        device: runtime.config.device.name,
        status: 'active',
        cwd: resolvedCwd,
        workspaceIsolation: resolveRequestSurface(request) === 'cli' ? 'policy' : 'strict',
        tags: body.tags ?? [],
      })
      await runtime.pluginEvents?.emit('session.created', {
        sessionId: session.id,
        provider: session.provider,
        model: session.model,
      })
    } else if (runtime.sessions.updateMeta) {
      const patch: Partial<typeof session> = {}
      if (resolvedCwd && session.cwd !== resolvedCwd) {
        return reply.status(409).send({
          error: {
            code: 'WORKSPACE_BINDING_MISMATCH',
            message: 'Local shell transcript cannot change an existing session workspace',
          },
        })
      }
      if (body.provider && session.provider !== body.provider) patch.provider = body.provider
      if (body.model && session.model !== body.model) patch.model = body.model
      if (
        body.title &&
        session.title !== body.title &&
        (session.title === '새 대화' || session.title === 'New Chat')
      ) {
        patch.title = body.title
      }
      if (Object.keys(patch).length > 0) {
        session = (await runtime.sessions.updateMeta(session.id, patch)) ?? session
      }
    }

    const userTimestamp = now
    const toolCallTimestamp = new Date(Date.parse(userTimestamp) + 1).toISOString()
    const toolResultTimestamp = new Date(Date.parse(userTimestamp) + 2).toISOString()
    const toolCallId = randomUUID()
    const toolOutput = body.stderr?.trim()
      ? [body.stdout ?? '', '[stderr]', body.stderr]
          .filter((part) => part.trim().length > 0)
          .join('\n')
      : (body.stdout ?? '')

    await runtime.sessions.appendEvent(params.id, {
      type: 'user_message',
      id: randomUUID(),
      timestamp: userTimestamp,
      content: `!${body.command}`,
    })
    await runtime.sessions.appendEvent(params.id, {
      type: 'tool_call',
      id: toolCallId,
      timestamp: toolCallTimestamp,
      tool: 'terminal.run',
      input: {
        command: body.command,
        ...(body.shell ? { executable: body.shell } : {}),
        ...(body.args ? { args: body.args } : {}),
        ...(resolvedCwd ? { cwd: resolvedCwd } : {}),
      },
      status: 'executing',
    })
    await runtime.sessions.appendEvent(params.id, {
      type: 'tool_result',
      id: randomUUID(),
      timestamp: toolResultTimestamp,
      toolCallId,
      output: toolOutput,
      status: body.exitCode === 0 ? 'success' : 'error',
      duration_ms: body.durationMs,
      metadata: {
        exitCode: body.exitCode,
        signal: body.signal ?? null,
        timedOut: body.timedOut ?? false,
        maxBufferExceeded: body.maxBufferExceeded ?? false,
      },
    })

    return {
      data: buildSessionMetaSnapshot(runtime, (await runtime.sessions.get(params.id)) ?? session),
    }
  })

  app.get<{ Params: SessionIdParams }>('/sessions/:id/watch', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    if (!runtime.sessionWatchBroker) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Session watch broker is unavailable',
        },
      })
    }

    const session = await runtime.sessions.get(params.id)
    if (!session) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })

    reply.hijack()
    reply.raw.writeHead(200, buildSseResponseHeaders(request, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Request-ID': request.requestId ?? randomUUID(),
    }))

    let closed = false
    const send = (event: string, data: unknown) => {
      if (closed) return
      reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
    }
    const close = () => {
      if (closed) return
      closed = true
      unsubscribe()
      clearInterval(heartbeat)
      if (!reply.raw.destroyed && !reply.raw.writableEnded) {
        reply.raw.end()
      }
    }
    const sendSnapshot = async () => {
      const nextSession = await runtime.sessions.get(params.id)
      if (!nextSession) {
        send('session', {
          type: 'deleted',
          sessionId: params.id,
        })
        close()
        return
      }

      send('session', {
        type: 'snapshot',
        session: await buildSessionDetail(runtime, nextSession),
      })
    }

    const unsubscribe = runtime.sessionWatchBroker.subscribe(params.id, () => {
      void sendSnapshot().catch(() => {
        close()
      })
    })
    const heartbeat = setInterval(() => {
      send('session', {
        type: 'heartbeat',
        sessionId: params.id,
        timestamp: new Date().toISOString(),
      })
    }, 15_000)
    heartbeat.unref?.()

    registerSseDisconnectHandler(request, reply, close)
    trackSseConnection(app, request, reply, { label: 'session-watch' })

    await sendSnapshot()
  })

  // PATCH /sessions/:id — rename or otherwise patch session metadata
  app.patch<{ Params: SessionIdParams; Body: SessionUpdateBody }>('/sessions/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
      body: {
        schema: sessionUpdateRequestSchema,
        message: 'Invalid session update payload',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    if (!runtime.sessions.updateMeta) {
      return reply.status(501).send({ error: { code: 'NOT_IMPLEMENTED', message: 'Session store does not support metadata updates' } })
    }
    const params = request.params
    const body = request.body
    let resolvedCwd: string | null | undefined
    if (body.cwd === null) {
      resolvedCwd = null
    } else if (body.cwd) {
      try {
        resolvedCwd = await resolveRequestCwd(body.cwd)
      } catch (error) {
        if (error instanceof InvalidCwdError) {
          return reply.status(400).send(invalidCwdResponse(error))
        }
        throw error
      }
    }

    // A workspace mutation participates in the same per-session lock as every
    // chat transport. Claim first, then re-read the session while holding the
    // lease so a turn cannot start between an isBusy check and updateMeta.
    const workspaceBindingRequested = body.cwd !== undefined || body.workspaceIsolation !== undefined
    const sessionLease = workspaceBindingRequested && runtime.sessionBusy
      ? runtime.sessionBusy.tryAcquireLease(params.id)
      : undefined
    if (workspaceBindingRequested && runtime.sessionBusy && !sessionLease) {
      return reply.status(409).send({
        error: {
          code: 'SESSION_BUSY',
          message: 'Stop the active turn before changing its workspace',
        },
      })
    }

    try {
      const existing = await runtime.sessions.get(params.id)
      if (!existing) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
      }
      const workspaceBindingChanged = (resolvedCwd !== undefined
        && existing.cwd !== (resolvedCwd ?? undefined))
        || (body.workspaceIsolation !== undefined
          && (existing.workspaceIsolation ?? 'strict') !== body.workspaceIsolation)
      if (workspaceBindingChanged) {
        await invalidateWorkspaceExecutionState(params.id)
      }
      const updated = await runtime.sessions.updateMeta(params.id, {
        ...body,
        ...(resolvedCwd !== undefined ? { cwd: resolvedCwd } : {}),
      })
      if (!updated) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
      return reply.status(200).send(buildSessionMetaSnapshot(runtime, updated))
    } finally {
      sessionLease?.release()
    }
  })

  // DELETE /sessions/:id — delete session
  app.delete<{ Params: SessionIdParams }>('/sessions/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const session = await runtime.sessions.get(params.id)
    if (!session) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
    await cleanupSessionRuntimeState(runtime, params.id)
    await runtime.sessions.delete(params.id)
    await runtime.pluginEvents?.emit('session.deleted', { sessionId: params.id })
    return reply.status(204).send()
  })

  // GET /sessions/:id/export — export session as markdown or JSON
  app.get<{ Params: SessionIdParams; Querystring: SessionExportQuery }>('/sessions/:id/export', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
      query: {
        schema: sessionExportQuerySchema,
        message: 'Invalid export format',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const query = request.query
    const session = await runtime.sessions.get(params.id)
    if (!session) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })

    const events = await runtime.sessions.getEvents(params.id)
    const sessionSnapshot = buildSessionMetaSnapshot(runtime, session)
    const format = query.format ?? 'markdown'
    const healthSnapshot = await buildHealthExportSnapshot(app).catch(() => null)

    let sessionForOutput = sessionSnapshot
    let eventsForOutput = events
    if (query.sanitize === '1') {
      const { sanitizeText, sanitizeEvent } = await import('../../sessions/sanitize.js')
      const home = process.env.HOME ?? ''
      sessionForOutput = {
        ...sessionSnapshot,
        title: sanitizeText(session.title, { home }),
      }
      eventsForOutput = events.map((ev) => sanitizeEvent(ev, { home }))
    }

    if (format === 'json') {
      return { data: buildSessionJsonExport(sessionForOutput, eventsForOutput, healthSnapshot) }
    }

    reply.header('Content-Type', 'text/markdown')
    return serializeSessionMarkdown(sessionForOutput, eventsForOutput, healthSnapshot)
  })

  // POST /sessions/:id/branch — fork a session
  app.post<{ Params: SessionIdParams; Body: SessionBranchBody }>('/sessions/:id/branch', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
      body: {
        schema: sessionBranchRequestInputSchema,
        message: 'Invalid branch request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const body = request.body

    const source = await runtime.sessions.get(params.id)
    if (!source) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })

    const events = await runtime.sessions.getEvents(params.id)
    const fromIndex = body.fromEventIndex ?? events.length

    // Create new session
    const { randomUUID } = await import('node:crypto')
    const branchId = randomUUID()
    // A branch is a fresh JSONL file, so event ids do not need to be re-minted
    // to stay unique. Re-minting them (previously `{ ...event, id: randomUUID() }`)
    // silently broke every cross-reference in the copy: tool_result.toolCallId,
    // approval_request.toolCallId and approval_response.requestId still pointed at
    // the *old* ids, so the first branched turn shipped orphan tool_results and
    // dangling approvals (provider 400). Copy events verbatim to preserve pairing.
    await runtime.sessions.create({
      id: branchId,
      title: `[Branch] ${source.title}`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      provider: source.provider,
      model: source.model,
      device: source.device,
      cwd: source.cwd,
      workspaceIsolation: source.workspaceIsolation,
      personaIds: source.personaIds,
      memoryNamespace: source.memoryNamespace,
      status: 'active',
      tags: [...(source.tags ?? []), `branch:${params.id}`],
    })

    // Copy events up to the fork point (verbatim — preserve original ids so
    // tool_call/tool_result and approval id links survive into the branch).
    for (const event of events.slice(0, fromIndex)) {
      await runtime.sessions.appendEvent(branchId, event)
    }

    return { data: { branchId, sourceId: params.id, copiedEvents: Math.min(fromIndex, events.length) } }
  })

}
