import type { AgentRunContract } from '@sepilotd/core'
import { contractIterationFloor } from './graph/iteration-budget.js'
import { resolveMaxContinuationCycles } from './task-contract.js'

export const DEFAULT_AGENT_MAX_ITERATIONS = 50
export const DEFAULT_SCHEDULER_MAX_ITERATIONS = 40
export const DEFAULT_SUBAGENT_MAX_ITERATIONS = 24
export const SUBAGENT_PARENT_SHARE = 0.35
export const DEFAULT_DIRECT_MAX_ITERATIONS = DEFAULT_AGENT_MAX_ITERATIONS

export type RunBudgetSurface = 'chat' | 'channel' | 'scheduler' | 'subagent' | 'direct'

export interface RunIterationBudget {
  maxIterations: number
  maxContinuationCycles: number
  hard: boolean
}

export interface ResolveRunIterationBudgetInput {
  surface: RunBudgetSurface
  requested?: number
  graphPreset?: number
  contract?: AgentRunContract
  hardMaxIterations?: boolean
  maxContinuationCycles?: number
  parentBudget?: number
  schedulerMaxIterations?: number
}

const DEFAULT_CONTINUATION_CYCLES: Record<RunBudgetSurface, number> = {
  chat: 6,
  channel: 6,
  scheduler: 3,
  subagent: 0,
  direct: 6,
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 1
    ? Math.floor(value)
    : undefined
}

function envPositiveInteger(name: string): number | undefined {
  const raw = process.env[name]
  if (!raw) return undefined
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed >= 1 ? parsed : undefined
}

export function resolveAgentMaxIterations(
  input: { maxIterations?: number } = {},
): number {
  const requested = positiveInteger(input.maxIterations)
  if (requested !== undefined) return requested
  return envPositiveInteger('SEPILOTD_CHAT_MAX_ITERATIONS') ?? DEFAULT_AGENT_MAX_ITERATIONS
}

export function resolveSubagentIterationBudget(parentBudget?: number): number {
  const parent = positiveInteger(parentBudget)
  if (parent === undefined) return DEFAULT_SUBAGENT_MAX_ITERATIONS
  const target = Math.max(
    DEFAULT_SUBAGENT_MAX_ITERATIONS,
    Math.ceil(parent * SUBAGENT_PARENT_SHARE),
  )
  return Math.max(1, Math.min(parent, target))
}

function surfaceDefault(input: ResolveRunIterationBudgetInput): number {
  switch (input.surface) {
    case 'chat':
    case 'channel':
      return resolveAgentMaxIterations()
    case 'scheduler':
      return positiveInteger(input.schedulerMaxIterations) ?? DEFAULT_SCHEDULER_MAX_ITERATIONS
    case 'subagent':
      return resolveSubagentIterationBudget(input.parentBudget)
    case 'direct':
      return DEFAULT_DIRECT_MAX_ITERATIONS
  }
}

export function resolveRunIterationBudget(
  input: ResolveRunIterationBudgetInput,
): RunIterationBudget {
  const hard = input.hardMaxIterations === true
  const requested = positiveInteger(input.requested)

  let maxIterations: number
  if (hard && requested !== undefined) {
    maxIterations = requested
  } else {
    const candidates = [
      requested,
      positiveInteger(input.graphPreset),
      contractIterationFloor(input.contract),
    ].filter((value): value is number => typeof value === 'number')
    maxIterations = candidates.length > 0
      ? Math.max(...candidates)
      : surfaceDefault(input)
  }

  if (input.surface === 'subagent') {
    const parent = positiveInteger(input.parentBudget)
    if (parent !== undefined) maxIterations = Math.min(maxIterations, parent)
  }

  const maxContinuationCycles = hard
    ? 0
    : resolveMaxContinuationCycles(
        input.maxContinuationCycles,
        DEFAULT_CONTINUATION_CYCLES[input.surface],
      )

  return { maxIterations, maxContinuationCycles, hard }
}
