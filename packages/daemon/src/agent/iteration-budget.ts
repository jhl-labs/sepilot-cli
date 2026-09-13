import type { AgentRunContract } from '@sepilotd/core'
import { contractIterationFloor } from './graph/iteration-budget.js'
import { resolveMaxContinuationCycles } from './task-contract.js'

/**
 * Single source of truth for per-run iteration budgets.
 *
 * One iteration is one LLM round, which may emit one or more parallel tool
 * calls. Every execution surface (interactive chat, channel turns, scheduled
 * jobs, isolated subagents, direct `AgentEngine` callers) resolves its budget
 * here so the defaults, the env overrides, and the contract-derived floors
 * cannot drift apart per call site.
 */

/** Default per-turn cap for user-facing chat surfaces (`SEPILOTD_CHAT_MAX_ITERATIONS`). */
export const DEFAULT_AGENT_MAX_ITERATIONS = 50
/** Scheduler default; mirrors `agent.scheduler.maxIterations` in config. */
export const DEFAULT_SCHEDULER_MAX_ITERATIONS = 40
/** Floor for an isolated subagent dispatch and the auto-decompose fan-out budget. */
export const DEFAULT_SUBAGENT_MAX_ITERATIONS = 24
/** Share of the parent budget an isolated subagent may claim before the parent cap applies. */
export const SUBAGENT_PARENT_SHARE = 0.35
/** Fallback for a direct `AgentEngine` caller that passed no budget. */
export const DEFAULT_DIRECT_MAX_ITERATIONS = DEFAULT_AGENT_MAX_ITERATIONS

export type RunBudgetSurface = 'chat' | 'channel' | 'scheduler' | 'subagent' | 'direct'

export interface RunIterationBudget {
  maxIterations: number
  /** Automatic continuation cycles after the iteration cap; 0 when `hard`. */
  maxContinuationCycles: number
  /** True only when the caller explicitly demanded a hard cap. */
  hard: boolean
}

export interface ResolveRunIterationBudgetInput {
  surface: RunBudgetSurface
  /** Client- or caller-supplied budget. A request, not a hard cap, unless `hardMaxIterations`. */
  requested?: number
  /** Graph preset iteration count (small presets are subgraph minimums, not run caps). */
  graphPreset?: number
  contract?: AgentRunContract
  /** Explicit hard cap flag for benches/automation. Disables continuation cycles. */
  hardMaxIterations?: boolean
  /** Explicit continuation override (wins over the env and surface default). */
  maxContinuationCycles?: number
  /** Parent run budget for `subagent` — the child claims a share of it, capped by it. */
  parentBudget?: number
  /** `agent.scheduler.maxIterations` for `scheduler`. */
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

/** Chat/channel default: `SEPILOTD_CHAT_MAX_ITERATIONS` or 50. */
export function resolveAgentMaxIterations(
  input: { maxIterations?: number } = {},
): number {
  const requested = positiveInteger(input.maxIterations)
  if (requested !== undefined) return requested
  return envPositiveInteger('SEPILOTD_CHAT_MAX_ITERATIONS') ?? DEFAULT_AGENT_MAX_ITERATIONS
}

/** Subagent default: `max(24, ceil(parent * 0.35))`, never above the parent budget. */
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
    // A declared hard cap is exact: no preset or contract floor raises it.
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
