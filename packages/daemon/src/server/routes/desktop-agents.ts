import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { DesktopAgentSessions, DESKTOP_AGENT_PRESETS } from '../../terminal/desktop-agent.js'
import { desktopAgentSshSchema, remoteAgentCwd } from '../../terminal/ssh-launch.js'
import { probeDesktopAgent } from '../../terminal/agent-availability.js'
import { resolveRequestCwd } from './request-cwd.js'
import { buildSseResponseHeaders } from '../sse-response.js'

const idSchema = z.object({ id: z.string().uuid() })
const sizeSchema = z
  .object({ cols: z.number().int().min(2).max(400), rows: z.number().int().min(2).max(160) })
  .strict()
const createSchema = z
  .object({
    agent: z.enum(['claude', 'codex', 'gemini', 'opencode']),
    cwd: z.string().trim().min(1),
    executionPolicy: z.literal('external-cli'),
    ssh: desktopAgentSshSchema.optional(),
    cols: sizeSchema.shape.cols.default(120),
    rows: sizeSchema.shape.rows.default(36),
  })
  .strict()
const inputSchema = z.object({ data: z.string().min(1).max(65536) }).strict()

/** Protected by the daemon's normal authenticated API; never by a token in a URL. */
export function registerDesktopAgentRoutes(
  app: FastifyInstance,
  sessions = new DesktopAgentSessions(),
  probe = probeDesktopAgent,
): void {
  app.addHook('preClose', async () => {
    sessions.closeAll()
  })
  app.get('/desktop-agents', async () => ({
    agents: await Promise.all(Object.entries(DESKTOP_AGENT_PRESETS).map(async ([id, preset]) => ({
      id, ...preset, ...await probe(id as keyof typeof DESKTOP_AGENT_PRESETS),
    }))),
    sessions: sessions.list(),
    executionHost: 'daemon',
    transport: 'pty',
    executionPolicy: 'external-cli',
  }))
  app.post('/desktop-agents/sessions', async (request, reply) => {
    const parsed = createSchema.safeParse(request.body)
    if (!parsed.success)
      return reply.code(400).send({
        error: 'CLI, 절대 작업 경로, external-cli 실행 정책 및 올바른 터미널 크기가 필요합니다.',
      })
    try {
      const cwd = parsed.data.ssh
        ? remoteAgentCwd(parsed.data.cwd)
        : await resolveRequestCwd(parsed.data.cwd)
      if (!cwd) throw new Error('작업 폴더가 필요합니다.')
      if (!parsed.data.ssh) {
        const status = await probe(parsed.data.agent)
        if (!status.available) return reply.code(409).send({ error: status.reason ?? 'CLI 실행 불가' })
      }
      return sessions.create(
        parsed.data.agent,
        cwd,
        parsed.data.cols,
        parsed.data.rows,
        parsed.data.ssh,
      )
    } catch (error) {
      return reply.code(400).send({ error: (error as Error).message })
    }
  })
  app.get('/desktop-agents/sessions/:id', async (request, reply) => {
    const params = idSchema.safeParse(request.params)
    if (!params.success) return reply.code(400).send({ error: 'Invalid session id.' })
    try {
      return sessions.get(params.data.id)
    } catch (error) {
      return reply.code(404).send({ error: (error as Error).message })
    }
  })
  app.get('/desktop-agents/sessions/:id/events', async (request, reply) => {
    const params = idSchema.safeParse(request.params)
    if (!params.success) return reply.code(400).send({ error: 'Invalid session id.' })
    try {
      sessions.get(params.data.id)
    } catch (error) {
      return reply.code(404).send({ error: (error as Error).message })
    }
    const registry = app.hasDecorator('connectionRegistry') ? app.connectionRegistry : undefined
    let connectionId: string | undefined
    try {
      connectionId = registry?.add({ kind: 'sse', label: 'desktop-agent-terminal' })
    } catch (error) {
      return reply.code(429).send({ error: (error as Error).message })
    }
    reply.hijack()
    for (const [name, value] of Object.entries(reply.getHeaders()))
      if (value !== undefined) reply.raw.setHeader(name, value)
    reply.raw.writeHead(200, buildSseResponseHeaders(request, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    }))
    let closed = false
    let unsubscribe: (() => void) | undefined
    const heartbeat = setInterval(() => {
      if (!closed) reply.raw.write(': keepalive\n\n')
    }, 15000)
    reply.raw.on('close', () => {
      closed = true
      if (connectionId) registry?.remove(connectionId)
      clearInterval(heartbeat)
      unsubscribe?.()
    })
    try {
      unsubscribe = await sessions.subscribe(params.data.id, (event) => {
        if (closed) return
        reply.raw.write(`data: ${JSON.stringify(event)}\n\n`)
        // A slow reader reconnects to a fresh screen snapshot; never drop arbitrary ANSI bytes.
        if (reply.raw.writableLength > 4 * 1024 * 1024) reply.raw.destroy()
        if (event.type === 'exit') reply.raw.end()
      })
      if (closed) unsubscribe()
    } catch {
      reply.raw.end()
    }
  })
  app.post('/desktop-agents/sessions/:id/input', async (request, reply) => {
    const params = idSchema.safeParse(request.params)
    const body = inputSchema.safeParse(request.body)
    if (!params.success || !body.success)
      return reply.code(400).send({ error: 'Invalid terminal input.' })
    try {
      sessions.write(params.data.id, body.data.data)
      return { accepted: true }
    } catch (error) {
      return reply.code(409).send({ error: (error as Error).message })
    }
  })
  app.post('/desktop-agents/sessions/:id/resize', async (request, reply) => {
    const params = idSchema.safeParse(request.params)
    const body = sizeSchema.safeParse(request.body)
    if (!params.success || !body.success)
      return reply.code(400).send({ error: 'Invalid terminal size.' })
    try {
      await sessions.resize(params.data.id, body.data.cols, body.data.rows)
      return { accepted: true }
    } catch (error) {
      return reply.code(409).send({ error: (error as Error).message })
    }
  })
  app.delete('/desktop-agents/sessions/:id', async (request, reply) => {
    const params = idSchema.safeParse(request.params)
    if (!params.success) return reply.code(400).send({ error: 'Invalid session id.' })
    try {
      sessions.close(params.data.id)
      return { closed: true }
    } catch (error) {
      return reply.code(404).send({ error: (error as Error).message })
    }
  })
}
