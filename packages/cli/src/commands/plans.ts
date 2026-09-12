import chalk from 'chalk'
import type {
  WorkPlan,
  WorkPlanCreateInput,
  WorkPlanListResult,
  WorkPlanStartInput,
  WorkPlanStartResult,
  WorkPlanStep,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface PlansClientLike {
  createPlan(input: WorkPlanCreateInput): Promise<WorkPlan>
  listPlans(): Promise<WorkPlanListResult>
  getPlan(id: string): Promise<WorkPlan>
  startPlan(id: string, input?: WorkPlanStartInput): Promise<WorkPlanStartResult>
}

export interface PlanCreateOptions {
  url?: string
  title?: string
  step?: string[]
  acceptance?: string[]
  risk?: string[]
  decision?: string[]
  sourceSession?: string
  ready?: boolean
}

export interface PlanStartOptions {
  url?: string
  category?: string
  agent?: string
  model?: string
  tools?: string
  maxIterations?: string
}

function collect(value: string, previous: string[] = []): string[] {
  return [...previous, value]
}

export const collectPlanOption = collect

function parseToolList(raw?: string): string[] | undefined {
  if (!raw) return undefined
  const tools = raw
    .split(',')
    .map((tool) => tool.trim())
    .filter(Boolean)
  return tools.length > 0 ? tools : undefined
}

function parseMaxIterations(raw?: string): number | undefined {
  if (!raw) return undefined
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--max-iterations must be a positive integer (got ${raw})`)
  }
  return parsed
}

function stepsFromTitles(titles: string[] | undefined): WorkPlanStep[] {
  return (titles ?? []).map((title, index) => ({
    id: String(index + 1),
    title,
    status: 'pending',
  }))
}

function renderPlan(plan: WorkPlan): string {
  const lines = [
    `${chalk.cyan(plan.id)} ${chalk.bold(plan.title)} [${plan.status}]`,
    `goal: ${plan.goal}`,
  ]
  if (plan.steps.length > 0) {
    lines.push('steps:')
    for (const step of plan.steps) {
      lines.push(`  - [${step.status}] ${step.id}: ${step.title}`)
    }
  }
  if (plan.acceptanceCriteria.length > 0) {
    lines.push('acceptance:')
    for (const item of plan.acceptanceCriteria) lines.push(`  - ${item}`)
  }
  if (plan.lastJobId) {
    lines.push(`lastJobId: ${plan.lastJobId}`)
    lines.push(...renderJobFollowUp(plan.lastJobId, {
      includeCancel: plan.status === 'running',
      indent: '  ',
    }))
  }
  return lines.join('\n')
}

function renderJobFollowUp(
  jobId: string,
  options: { includeCancel?: boolean; indent?: string } = {},
): string[] {
  const indent = options.indent ?? ''
  const lines = [
    `${indent}status: sepilot jobs status ${jobId}`,
    `${indent}wait: sepilot jobs resume ${jobId}`,
  ]
  if (options.includeCancel ?? true) {
    lines.push(`${indent}cancel: sepilot jobs cancel ${jobId}`)
  }
  return lines
}

export async function runPlanCreate(input: {
  goal: string
  options: PlanCreateOptions
  client: PlansClientLike
  emit?: (plan: WorkPlan) => void
}): Promise<WorkPlan> {
  const plan = await input.client.createPlan({
    title: input.options.title,
    goal: input.goal,
    steps: stepsFromTitles(input.options.step),
    acceptanceCriteria: input.options.acceptance,
    risks: input.options.risk,
    decisions: input.options.decision,
    sourceSessionId: input.options.sourceSession,
    status: input.options.ready ? 'ready' : 'draft',
  })
  ;(input.emit ?? ((created) => output(created, renderPlan)))(plan)
  return plan
}

export async function runPlanList(input: {
  client: PlansClientLike
  emit?: (result: WorkPlanListResult) => void
}): Promise<WorkPlanListResult> {
  const result = await input.client.listPlans()
  ;(
    input.emit ??
    ((listed) => {
      output(listed, (value) => {
        if (value.plans.length === 0) return 'No plans.'
        return value.plans
          .map((plan) => `${chalk.cyan(plan.id)} ${plan.title} [${plan.status}]`)
          .join('\n')
      })
    })
  )(result)
  return result
}

export async function runPlanShow(input: {
  id: string
  client: PlansClientLike
  emit?: (plan: WorkPlan) => void
}): Promise<WorkPlan> {
  const plan = await input.client.getPlan(input.id)
  ;(input.emit ?? ((fetched) => output(fetched, renderPlan)))(plan)
  return plan
}

export async function runPlanStart(input: {
  id: string
  options: PlanStartOptions
  client: PlansClientLike
  emit?: (result: WorkPlanStartResult) => void
}): Promise<WorkPlanStartResult> {
  const result = await input.client.startPlan(input.id, {
    category: input.options.category,
    agentId: input.options.agent,
    model: input.options.model,
    tools: parseToolList(input.options.tools),
    maxIterations: parseMaxIterations(input.options.maxIterations),
  })
  ;(
    input.emit ??
    ((started) => {
      output(
        started,
        (value) => [
          `${chalk.cyan(`[plan ${value.plan.id}]`)} started background job`,
          `  job: ${value.job.jobId}`,
          `  plan: sepilot plans show ${value.plan.id}`,
          ...renderJobFollowUp(value.job.jobId, { indent: '  ' }),
        ].join('\n'),
      )
    })
  )(result)
  return result
}

export async function plansCreateCommand(
  goalParts: string[],
  options: PlanCreateOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  await runPlanCreate({ goal: goalParts.join(' '), options, client })
}

export async function plansListCommand(options: { url?: string }): Promise<void> {
  const client = new DaemonClient(options.url)
  await runPlanList({ client })
}

export async function plansShowCommand(id: string, options: { url?: string }): Promise<void> {
  const client = new DaemonClient(options.url)
  await runPlanShow({ id, client })
}

export async function plansStartCommand(id: string, options: PlanStartOptions): Promise<void> {
  const client = new DaemonClient(options.url)
  await runPlanStart({ id, options, client })
}
