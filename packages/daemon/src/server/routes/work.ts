import type { FastifyInstance } from 'fastify'
import type { WorkItem } from '@sepilotd/api-client'
import type { JobsRepo } from '../../jobs/repo.js'
import { z } from 'zod'
import { zodRequestValidation } from './utils.js'

export function registerWorkRoutes(app: FastifyInstance, repo: JobsRepo): void {
  app.get('/work', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime) return reply.code(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime unavailable' } })
    const items: WorkItem[] = []
    const unavailable: string[] = []
    const truncated: string[] = []
    for (const [source, value] of Object.entries({ run: runtime.activeRuns, process: runtime.managedProcesses, schedule: runtime.jobStore, approval: runtime.approvalRegistry, service: runtime.serviceSupervisor })) {
      if (!value) unavailable.push(source)
    }
    const add = (source: string, rows: WorkItem[]) => {
      if (rows.length > 100) truncated.push(source)
      items.push(...rows.slice(0, 100))
    }
    const jobs = repo.list({ limit: 100 })
    if (jobs.length === 100) truncated.push('job')
    add('job', jobs.map(j => ({ key: `job:${j.id}`, kind: 'job', id: j.id, status: j.status, title: `${j.kind} job`, detail: `${j.succeeded}/${j.total} succeeded, ${j.failed} failed`, action: `sepilot jobs status ${j.id}` })))
    add('run', (runtime.activeRuns?.list() ?? []).map(r => ({ key: `run:${r.sessionId}`, kind: 'run', id: r.sessionId, sessionId: r.sessionId, status: 'running', title: r.graphId ?? 'Agent run', detail: `${r.iteration}/${r.maxIterations} iterations`, action: `sepilot tasks cancel run:${r.sessionId}` })))
    add('process', (runtime.managedProcesses?.list() ?? []).map(p => ({ key: `process:${p.id}`, kind: 'process', id: p.id, status: p.status, title: `Process ${p.pid ?? ''}`, detail: p.terminationReason ?? '', action: `sepilot tasks logs ${p.id}` })))
    add('schedule', (runtime.jobStore?.list() ?? []).map(j => ({ key: `schedule:${j.id}`, kind: 'schedule', id: j.id, sessionId: j.parentSessionId ?? undefined, status: j.enabled ? j.status : 'paused', title: j.name, detail: j.enabled ? `Next: ${new Date(j.nextRunAt).toISOString()}` : 'No future run scheduled', action: `sepilot tasks cancel schedule:${j.id}` })))
    add('approval', (runtime.approvalRegistry?.listAll() ?? []).map(a => ({ key: `approval:${a.requestId}`, kind: 'approval', id: a.requestId, sessionId: a.sessionId, status: 'waiting', title: a.tool, detail: `Expires ${a.expiresAt}`, action: `sepilot tasks inspect approval:${a.requestId}` })))
    try {
      add('service', (await runtime.serviceSupervisor?.inventory() ?? []).map(s => ({ key: `service:${s.id}`, kind: 'service', id: s.id, status: s.status, title: s.name, detail: `Last observed ${s.updatedAt}`, action: `sepilot tasks cancel service:${s.id}` })))
    } catch { unavailable.push('service') }
    return { data: { items, unavailable, truncated } }
  })
  const params = z.object({ id: z.string().min(1) })
  const query = z.object({ stdoutOffset: z.coerce.number().int().min(0).optional(), stderrOffset: z.coerce.number().int().min(0).optional() }).strict()
  app.get<{ Params: { id: string }; Querystring: z.infer<typeof query> }>('/work/process/:id', {
    preValidation: zodRequestValidation({ params: { schema: params, message: 'Invalid process id' }, query: { schema: query, message: 'Invalid output offset' } }),
  }, async (request, reply) => {
    const params = request.params
    const query = request.query
    const registry = app.runtime?.managedProcesses
    await registry?.flushPty(params.id)
    const result = registry?.read(params.id, query.stdoutOffset ?? 0, query.stderrOffset ?? 0)
    if (!result) return reply.code(404).send({ error: { code: 'NOT_FOUND', message: 'Managed process not found' } })
    const stdout = result.stdout.slice(0, 64_000)
    const stderr = result.stderr.slice(0, 64_000)
    return { data: { status: result.process.status, stdout, stderr, screen: result.screen?.slice(0, 64_000),
      nextStdoutOffset: result.nextStdoutOffset - result.stdout.length + stdout.length,
      nextStderrOffset: result.nextStderrOffset - result.stderr.length + stderr.length } }
  })
  app.delete<{ Params: { id: string } }>('/work/process/:id', {
    preValidation: zodRequestValidation({ params: { schema: params, message: 'Invalid process id' } }),
  }, async (request, reply) => {
    const params = request.params
    const result = await app.runtime?.managedProcesses?.stopAndWait(params.id)
    if (!result) return reply.code(404).send({ error: { code: 'NOT_FOUND', message: 'Managed process not found' } })
    return { data: { status: result.status } }
  })
  app.post<{ Params: { id: string }; Body: { text: string; end?: boolean } }>('/work/process/:id/input', {
    preValidation: zodRequestValidation({ params: { schema: params, message: 'Invalid process id' }, body: { schema: z.object({ text: z.string().max(16384), end: z.boolean().optional() }).strict(), message: 'Invalid process input' } }),
  }, async (request, reply) => {
    const params = request.params
    const body = request.body
    if (!app.runtime?.managedProcesses?.get(params.id)) return reply.code(404).send({ error: { code: 'NOT_FOUND', message: 'Managed process not found' } })
    try {
      await app.runtime.managedProcesses.writeInput(params.id, body.text, body.end)
      return { data: { accepted: true } }
    } catch (error) { return reply.code(409).send({ error: { code: 'PROCESS_INPUT_UNAVAILABLE', message: error instanceof Error ? error.message : String(error) } }) }
  })
}
