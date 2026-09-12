import { createKnowledgeBudget, DailyTokenLimit } from './budget.js'
import { createKnowledgeLifecycle, VerificationInput, RestoreInput } from './lifecycle.js'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../server/capabilities/bind.js'
import { redactSensitive } from '../memory/sensitive.js'
import { createWikiRepo } from '../wiki/repo.js'
import { createKnowledgeChatTools } from './chat-tools.js'
import { createKnowledgeMaintenance, ChatUpdateMode } from './maintenance.js'
import { createKnowledgeRepo } from './repo.js'
import { KnowledgeCapture, KnowledgeWrite, knowledgeError, type KnowledgeSource } from './schema.js'
import { extractPersonalKnowledge, suggestKnowledgeOrganization } from './curation.js'
import { retrievePersonalKnowledge } from './retrieval.js'
import { createKnowledgeActivityStore } from './activity.js'
import type { KnowledgeActivity } from '@sepilotd/api-client/daemon/types'

export async function registerPersonalKnowledgeRoutes(app: FastifyInstance) {
  const repo = createKnowledgeRepo()
  const activity = createKnowledgeActivityStore()
  const lifecycle = createKnowledgeLifecycle(repo)
  const budget = createKnowledgeBudget()
  const maintenance = createKnowledgeMaintenance(repo, activity, (id) => scheduleOrganization(id))
  activity.recover()
  async function track<T>(
    kind: string,
    summary: string,
    task: (id: string) => Promise<T>,
    targets: KnowledgeActivity['targets'] = [],
  ) {
    const id = activity.begin({ kind, summary, targets })
    try {
      const result = await task(id)
      activity.finish(id, {
        ids: Array.isArray(result)
          ? result.flatMap((r) => (r && typeof r === 'object' && 'id' in r ? [String(r.id)] : []))
          : result && typeof result === 'object' && 'id' in result
            ? [String(result.id)]
            : [],
        preview: result,
      })
      return result
    } catch (error) {
      activity.fail(id, error, shutdown.signal.aborted)
      throw error
    }
  }
  const unregister = app.chatKnowledgeProviders?.register({
    id: 'personal-knowledge',
    search: (query, limit) => {
      const id = activity.begin({
        kind: 'retrieve',
        trigger: 'automatic',
        summary: '대화에 활용할 영구 지식 검색',
      })
      try {
        const records = repo.list()
        const result = retrievePersonalKnowledge(records, query, limit).map((hit) => {
          const record = records.find((r) => r.id === hit.id)!
          const check = lifecycle.status(record)
          return {
            ...hit,
            contextNotes: [
              `Verification: ${check.state}. Due, changed or unverified records need source checking before presenting them as verified facts.`,
              ...(hit.contextNotes ?? []),
            ],
            content: `Verification: ${check.state}; last verified revision: ${check.verifiedRevision ?? 'none'}; next review: ${check.nextReviewAt === null ? 'not scheduled' : new Date(check.nextReviewAt).toISOString()}. Due or changed knowledge needs source rechecking before presenting it as current fact.\n${hit.content}`,
          }
        })
        activity.patch(id, {
          targets: result.map((hit) => ({
            id: hit.id,
            title: hit.title,
            revision: records.find((r) => r.id === hit.id)?.revision,
          })),
        })
        activity.finish(id, {
          ids: result.map((r) => r.id),
          preview: { query, matches: result.map((r) => ({ id: r.id, title: r.title })) },
        })
        return result
      } catch (error) {
        activity.fail(id, error)
        throw error
      }
    },
  })

  // Single bounded auxiliary operation per daemon profile; no overlapping costly extraction jobs.
  let busy = false
  let pendingActivityId: string | null = null
  let closing = false
  let background: Promise<void> | undefined
  const shutdown = new AbortController()
  // A crash cannot leave the persisted review inbox permanently in 'running'.
  if (repo.review().status === 'running') repo.saveReview({ status: 'idle' })
  function scheduleOrganization(parentId?: string) {
    if (!repo.review().autoOrganize || closing) return
    if (pendingActivityId && parentId)
      activity.event(pendingActivityId, `추가 변경 반영: ${parentId}`)
    pendingActivityId ??= activity.begin({
      kind: 'organize',
      trigger: 'automatic',
      parentId,
      queued: true,
      summary: '지식 변경 후 분류·관계 자동 개선',
    })
    startOrganization()
  }
  function startOrganization() {
    if (busy || !pendingActivityId || closing) return
    const activityId = pendingActivityId
    pendingActivityId = null
    activity.patch(activityId, { status: 'running', startedAt: Date.now() })
    activity.event(activityId, '자동 분류 작업 시작')
    busy = true
    repo.saveReview({ status: 'running', error: null })
    background = (async () => {
      try {
        const { provider, model } = providerConfig(activityId)
        const suggestions = await suggestKnowledgeOrganization(
          provider,
          model,
          repo.list(),
          shutdown.signal,
          {
            afterId: repo.organizationCursor(),
            complete: repo.advanceOrganization,
            selected: (records) => {
              activity.patch(activityId, {
                targets: records.map(({ id, revision, title }) => ({ id, revision, title })),
              })
              activity.event(activityId, `순환 분류 검토: ${records.length}개 (실행당 최대 150개)`)
            },
          },
        )
        repo.saveReview({ status: 'ready', suggestions })
        activity.finish(activityId, { ids: suggestions.map((s) => s.id), preview: suggestions })
      } catch (error) {
        activity.fail(activityId, error, shutdown.signal.aborted)
        repo.saveReview({
          status: 'failed',
          error: redactSensitive((error as Error).message).redacted,
        })
      } finally {
        busy = false
        background = undefined
        startOrganization()
      }
    })()
  }
  const chatTools = createKnowledgeChatTools({
    repo,
    activity,
    events: async (sessionId) => {
      if (!app.runtime) knowledgeError('Runtime unavailable', 503)
      return app.runtime.sessions.getEvents(sessionId)
    },
    changed: scheduleOrganization,
  })
  for (const tool of chatTools) app.runtime?.toolRegistry?.register(tool)
  app.addHook('onClose', async () => {
    closing = true
    shutdown.abort()
    if (pendingActivityId) activity.fail(pendingActivityId, 'Daemon shutdown while queued', true)
    unregister?.()
    for (const tool of chatTools) app.runtime?.toolRegistry?.unregister(tool.name)
    await background
  })
  async function curate<T>(operation: () => Promise<T>): Promise<T> {
    if (busy) knowledgeError('다른 지식 제안을 처리 중입니다. 잠시 후 다시 시도하세요.', 409)
    busy = true
    try {
      return await operation()
    } finally {
      busy = false
      startOrganization()
    }
  }
  const providerConfig = (activityId: string) => {
    const runtime = app.runtime
    const provider = runtime?.providerRegistry.getDefault()
    if (!runtime || !provider) knowledgeError('먼저 LLM 제공자를 설정하세요.', 503)
    const model = runtime.config.agent.defaultModel ?? provider.models[0]?.id
    if (!model) knowledgeError('LLM 모델을 설정하세요.', 503)
    activity.event(activityId, 'LLM 요청 준비')
    return { provider: activity.observe(provider, activityId, runtime.usageTracker), model }
  }
  await app.register(async (a) => {
    a.setErrorHandler((error, _request, reply) => {
      const status =
        error instanceof z.ZodError
          ? 400
          : error instanceof Error && 'statusCode' in error && typeof error.statusCode === 'number'
            ? error.statusCode
            : 500
      return reply.code(status).send({
        error: {
          code: status === 409 ? 'KNOWLEDGE_CONFLICT' : 'KNOWLEDGE_ERROR',
          message:
            error instanceof z.ZodError
              ? '지식 요청 형식을 확인하세요.'
              : redactSensitive(
                  error instanceof Error ? error.message : 'Knowledge operation failed',
                ).redacted,
        },
      })
    })
    a.addHook('onRequest', async (request, reply) => {
      if (request.authContext?.kind === 'extension')
        return reply.code(403).send({
          error: {
            code: 'FORBIDDEN',
            message: 'Personal knowledge is first-party profile data.',
          },
        })
    })
    await bindCapability(
      a,
      {
        name: 'personal-knowledge',
        version: '1',
        methods: [
          { method: 'GET', path: '/knowledge/lifecycle' },
          { method: 'POST', path: '/knowledge/verify' },
          { method: 'POST', path: '/knowledge/budget' },
          { method: 'POST', path: '/knowledge/restore' },
          { method: 'GET', path: '/knowledge/maintenance' },
          { method: 'POST', path: '/knowledge/maintenance' },
          { method: 'POST', path: '/knowledge/proposals/:id' },
          { method: 'GET', path: '/knowledge/activity' },
          { method: 'GET', path: '/knowledge/activity/:id' },
          { method: 'GET', path: '/knowledge' },
          { method: 'GET', path: '/knowledge/review' },
          { method: 'POST', path: '/knowledge/review/dismiss' },
          { method: 'POST', path: '/knowledge/review' },
          { method: 'POST', path: '/knowledge' },
          { method: 'GET', path: '/knowledge/export' },
          { method: 'GET', path: '/knowledge/:id/history' },
          { method: 'POST', path: '/knowledge/capture' },
          { method: 'POST', path: '/knowledge/organize' },
        ],
      },
      async (api) => {
        api.get('/knowledge/lifecycle', async () => ({
          verifications: lifecycle.list(),
          budget: budget.state(),
        }))
        api.post('/knowledge/verify', async (req) => {
          const input = VerificationInput.parse(req.body)
          return track(
            'verify',
            input.action === 'verify' ? '사용자가 지식 재검증' : '지식 재검증 시점 설정',
            async () => lifecycle.verify(input),
            [{ id: input.id, revision: input.expectedRevision }],
          )
        })
        api.post('/knowledge/budget', async (req) => {
          const { limit } = z.object({ limit: DailyTokenLimit }).strict().parse(req.body)
          return track('settings', '자동 Wiki 작업 일일 토큰 예산 변경', async () =>
            budget.configure(limit),
          )
        })
        api.post('/knowledge/restore', async (req) => {
          const input = RestoreInput.parse(req.body)
          return track(
            'restore',
            '백업에서 선택한 지식을 후보로 복원',
            async (id) => {
              const saved = lifecycle.restore(input)
              scheduleOrganization(id)
              return saved
            },
            [{ id: input.snapshot.id, revision: input.expectedRevision ?? undefined }],
          )
        })
        api.get('/knowledge', async () => repo.list())
        api.get('/knowledge/maintenance', async () => ({
          mode: maintenance.mode(),
          recordsVersion: maintenance.recordsVersion(),
          proposals: maintenance.list(),
        }))
        api.post('/knowledge/maintenance', async (req) => {
          const input = z.object({ mode: ChatUpdateMode }).strict().parse(req.body)
          maintenance.configure(input.mode)
          return {
            mode: maintenance.mode(),
            recordsVersion: maintenance.recordsVersion(),
            proposals: maintenance.list(),
          }
        })
        api.post('/knowledge/proposals/:id', async (req) => {
          const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
          const { decision } = z
            .object({ decision: z.enum(['accept', 'reject']) })
            .strict()
            .parse(req.body)
          return maintenance.decide(id, decision)
        })
        api.get('/knowledge/activity', async (req) => {
          const query = z
            .object({
              limit: z.coerce.number().int().min(1).max(100).default(30),
              before: z.coerce.number().int().positive().optional(),
            })
            .parse(req.query)
          return activity.page(query.limit, query.before)
        })
        api.get('/knowledge/activity/:id', async (req) => {
          const { id } = z.object({ id: z.string().uuid() }).parse(req.params)
          const record = activity.get(id)
          if (!record) knowledgeError('실행 이력을 찾을 수 없습니다.', 404)
          return { activity: record, calls: activity.calls(id) }
        })
        api.get('/knowledge/review', async () => repo.review())
        api.post('/knowledge/review/dismiss', async (req) => {
          const input = z
            .object({ id: z.string().min(1), expectedRevision: z.number().int().positive() })
            .strict()
            .parse(req.body)
          return track(
            'review',
            '분류·관계 제안 제외',
            async () => {
              const current = repo.review()
              if (
                !current.suggestions.some(
                  (s) => s.id === input.id && s.expectedRevision === input.expectedRevision,
                )
              )
                knowledgeError('제안이 변경되었습니다. 새로고침하세요.', 409)
              return repo.saveReview({
                suggestions: current.suggestions.filter(
                  (s) => s.id !== input.id || s.expectedRevision !== input.expectedRevision,
                ),
              })
            },
            [{ id: input.id, revision: input.expectedRevision }],
          )
        })
        api.post('/knowledge/review', async (req) => {
          const { autoOrganize } = z.object({ autoOrganize: z.boolean() }).strict().parse(req.body)
          return track('settings', '자동 분류 설정 변경', async (id) => {
            repo.saveReview({ autoOrganize })
            if (autoOrganize) scheduleOrganization(id)
            else if (pendingActivityId) {
              activity.fail(pendingActivityId, '자동 분류를 꺼 대기 작업을 취소했습니다.', true)
              pendingActivityId = null
            }
            return repo.review()
          })
        })
        api.get('/knowledge/export', async (_request, reply) => {
          reply.header('content-disposition', 'attachment; filename="sepilot-knowledge.json"')
          const id = activity.begin({ kind: 'export', summary: '지식·수정·실행 이력 내보내기' })
          try {
            const records = repo.list()
            const history = records.flatMap((r) => repo.history(r.id))
            activity.finish(id, { preview: { records: records.length, revisions: history.length } })
            return {
              format: 'sepilotd.personal-knowledge',
              version: 1,
              exportedAt: Date.now(),
              records,
              history,
              activity: activity.export(),
              maintenance: maintenance.export(),
              verifications: lifecycle.export(),
            }
          } catch (error) {
            activity.fail(id, error)
            throw error
          }
        })
        api.get('/knowledge/:id/history', async (req) => {
          const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
          return repo.history(id)
        })
        api.post('/knowledge', async (req) => {
          const input = KnowledgeWrite.parse(req.body)
          return track(
            'write',
            input.reason,
            async (id) => {
              const saved = repo.write(input)
              activity.event(id, `지식 저장: ${saved.id} / v${saved.revision} / ${saved.status}`)
              scheduleOrganization(id)
              return saved
            },
            input.id
              ? [{ id: input.id, revision: input.expectedRevision, title: input.title }]
              : [],
          )
        })
        api.post('/knowledge/capture', async (req) => {
          const { source: input, extract } = z
            .object({ source: KnowledgeCapture, extract: z.boolean().default(true) })
            .strict()
            .parse(req.body)
          return track(
            'capture',
            extract ? '선택한 정보에서 LLM 지식 후보 추출' : '선택한 원문을 지식 후보로 보존',
            (activityId) =>
              curate(async () => {
                activity.event(activityId, `출처 읽기: ${input.kind} / ${input.id}`)
                let source: KnowledgeSource
                if (input.kind === 'conversation') {
                  const runtime = app.runtime
                  if (!runtime) knowledgeError('Runtime unavailable', 503)
                  const session = await runtime.sessions.get(input.id)
                  if (!session) knowledgeError('대화를 찾을 수 없습니다.', 404)
                  const events = await runtime.sessions.getEvents(input.id)
                  // Whole recent messages only; label original event offsets so evidence remains traceable.
                  const messages: string[] = []
                  const roles = new Set<string>()
                  let used = 0
                  for (let index = events.length - 1; index >= 0; index--) {
                    const event = events[index]!
                    if (event.type !== 'user_message' && event.type !== 'assistant_message')
                      continue
                    if (typeof event.content !== 'string' || !event.content.trim()) continue
                    const message = `[event ${index} ${event.type}] ${event.content}`
                    if (used + message.length + 1 > 24000) break
                    roles.add(event.type)
                    messages.unshift(message)
                    used += message.length + 1
                  }
                  if (!roles.has('user_message') || !roles.has('assistant_message'))
                    knowledgeError(
                      '질문과 답변이 함께 있는 최근 대화가 필요합니다. 긴 내용은 발췌해서 추가하세요.',
                    )
                  source = {
                    kind: input.kind,
                    id: input.id,
                    title: session.title ?? input.id,
                    excerpt: messages.join('\n'),
                    capturedAt: Date.now(),
                  }
                } else if (input.kind === 'wiki') {
                  const node = createWikiRepo()
                    .tree()
                    .find((n) => n.id === input.id)
                  if (!node) knowledgeError('문서를 찾을 수 없습니다.', 404)
                  if (!node.body.trim()) knowledgeError('본문이 있는 문서를 선택하세요.')
                  if (node.body.length > 24000)
                    knowledgeError('긴 문서는 필요한 부분을 발췌해서 추가하세요.')
                  source = {
                    kind: input.kind,
                    id: input.id,
                    title: node.title,
                    excerpt: node.body,
                    capturedAt: Date.now(),
                  }
                } else source = { ...input, capturedAt: Date.now() }
                source = {
                  ...source,
                  title: redactSensitive(source.title).redacted,
                  excerpt: redactSensitive(source.excerpt).redacted,
                }
                activity.event(
                  activityId,
                  `원문 준비 완료: ${source.excerpt.length}자 · 민감정보 필터 적용`,
                )
                activity.patch(activityId, {
                  targets: [{ id: `${source.kind}:${source.id}`, title: source.title }],
                })
                // Memory/manual captures are explicitly supplied snapshots, not server-verified provenance.
                if (!extract)
                  return repo.capture(source, [
                    {
                      title: source.title.slice(0, 300),
                      body: source.excerpt,
                      kind: 'fact',
                      status: 'candidate',
                      parentId: null,
                      tags: [],
                      relations: [],
                      reason: '원문을 보존한 미검증 후보. 내용을 검토한 뒤 승격하세요.',
                    },
                  ])
                const { provider, model } = providerConfig(activityId)
                const drafts = await extractPersonalKnowledge(provider, model, source, repo.list())
                return repo.capture(source, drafts)
              }).then((records) => {
                activity.event(activityId, `검증·저장 완료: ${records.length}개 후보`)
                scheduleOrganization(activityId)
                return records
              }),
            [{ id: `${input.kind}:${input.id}` }],
          )
        })
        api.post('/knowledge/organize', async () =>
          track('organize', '분류·관계 개선 제안 요청', (activityId) =>
            curate(async () => {
              const { provider, model } = providerConfig(activityId)
              const suggestions = await suggestKnowledgeOrganization(
                provider,
                model,
                repo.list(),
                shutdown.signal,
                {
                  afterId: repo.organizationCursor(),
                  complete: repo.advanceOrganization,
                  selected: (records) => {
                    activity.patch(activityId, {
                      targets: records.map(({ id, revision, title }) => ({ id, revision, title })),
                    })
                    activity.event(
                      activityId,
                      `순환 분류 검토: ${records.length}개 (실행당 최대 150개)`,
                    )
                  },
                },
              )
              repo.saveReview({ suggestions, status: 'ready', error: null })
              return suggestions
            }),
          ),
        )
      },
    )
  })
}
