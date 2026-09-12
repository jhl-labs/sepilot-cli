import type { FastifyInstance } from 'fastify'
import type { SubagentDispatchInput } from '../../agent/subagent-dispatcher.js'
import type {
  WorkPlan,
  WorkPlanCreateInput,
  WorkPlanPatchInput,
  WorkPlanStore,
} from '../../plans/store.js'

export interface PlanStartJob {
  jobId: string
  status: string
  total: number
  createdAt: number
}

export interface PlanStartInput {
  category?: string
  agentId?: string
  model?: string
  tools?: string[]
  maxIterations?: number
}

export interface RegisterPlanRoutesDeps {
  store: WorkPlanStore
  startSubagent?: (input: SubagentDispatchInput) => Promise<PlanStartJob>
}

function httpError(statusCode: number, message: string): Error & { statusCode: number } {
  const err = new Error(message) as Error & { statusCode: number }
  err.statusCode = statusCode
  return err
}

function assertGoal(goal: unknown): string {
  if (typeof goal !== 'string' || goal.trim().length === 0) {
    throw httpError(400, 'goal required')
  }
  return goal
}

function renderSteps(steps: WorkPlan['steps'], depth = 0): string[] {
  const pad = '  '.repeat(depth)
  return steps.flatMap((step) => [
    `${pad}- [${step.status}] ${step.id}: ${step.title}${step.detail ? ` - ${step.detail}` : ''}`,
    ...renderSteps(step.children ?? [], depth + 1),
  ])
}

function buildPlanStartPrompt(plan: WorkPlan): string {
  const sections = [`Execute this sepilotd work plan.\n\nPlan: ${plan.title}\nGoal: ${plan.goal}`]
  if (plan.acceptanceCriteria.length > 0) {
    sections.push(
      `Acceptance criteria:\n${plan.acceptanceCriteria.map((item) => `- ${item}`).join('\n')}`,
    )
  }
  if (plan.steps.length > 0) {
    sections.push(`Steps:\n${renderSteps(plan.steps).join('\n')}`)
  }
  if (plan.risks.length > 0) {
    sections.push(`Risks:\n${plan.risks.map((item) => `- ${item}`).join('\n')}`)
  }
  if (plan.decisions.length > 0) {
    sections.push(`Decisions:\n${plan.decisions.map((item) => `- ${item}`).join('\n')}`)
  }
  sections.push(
    [
      'Execution contract:',
      '- Start by inspecting the current state relevant to the plan, then derive a concrete checklist from the steps and acceptance criteria before making changes.',
      '- Keep the work scoped to this plan; if the plan is stale, unsafe, or underspecified, report the blocker instead of silently changing scope.',
      '- For any UI, website, app, game, visual, or rendered artifact work, review the result from the user perspective, run/build it when practical, inspect rendered desktop and mobile output with screenshots/browser automation when available, and iterate on visible layout/interaction issues before finalizing.',
      '- Do not claim visual quality from source review alone when rendered UI verification is feasible; if it is not feasible, state exactly what prevented it.',
      '- Final output must include concrete files or artifacts changed, commands/tests run, verification evidence, blockers, and a MET/UNMET judgment for each acceptance criterion.',
    ].join('\n'),
  )
  return sections.join('\n\n')
}

export function registerPlanRoutes(app: FastifyInstance, deps: RegisterPlanRoutesDeps): void {
  app.get('/plans', async () => {
    return { plans: await deps.store.list() }
  })

  app.post<{ Body: WorkPlanCreateInput }>('/plans', async (req, reply) => {
    const body = req.body
    const plan = await deps.store.create({
      ...body,
      goal: assertGoal(body?.goal),
    })
    return reply.status(201).send(plan)
  })

  app.get<{ Params: { id: string } }>('/plans/:id', async (req) => {
    const plan = await deps.store.get(req.params.id)
    if (!plan) throw httpError(404, 'plan not found')
    return plan
  })

  app.patch<{ Params: { id: string }; Body: WorkPlanPatchInput }>('/plans/:id', async (req) => {
    const plan = await deps.store.patch(req.params.id, req.body)
    if (!plan) throw httpError(404, 'plan not found')
    return plan
  })

  app.post<{ Params: { id: string }; Body: PlanStartInput }>(
    '/plans/:id/start',
    async (req, reply) => {
      if (!deps.startSubagent) throw httpError(503, 'subagent starter not configured')
      const plan = await deps.store.get(req.params.id)
      if (!plan) throw httpError(404, 'plan not found')

      const body = req.body ?? {}
      const job = await deps.startSubagent({
        prompt: buildPlanStartPrompt(plan),
        category: body.category ?? 'implementation',
        agentId: body.agentId,
        model: body.model,
        tools: body.tools,
        maxIterations: body.maxIterations,
      })
      const updated = await deps.store.markStarted(plan.id, job.jobId)
      return reply.status(202).send({ plan: updated ?? plan, job })
    },
  )
}
