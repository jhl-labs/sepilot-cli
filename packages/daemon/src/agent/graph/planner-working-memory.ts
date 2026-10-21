import { z } from 'zod'
import type {
  PlannerAbandonedAlternative,
  PlannerHierarchicalStep,
  PlannerOpenAssumption,
  PlannerWorkingMemory,
} from '@sepilotd/core'

export const MAX_PLANNER_SCRATCH = 30

export function resolvePlannerScratchCap(): number {
  const raw = process.env.SEPILOTD_PLANNER_SCRATCH_CAP
  if (!raw) return MAX_PLANNER_SCRATCH
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed)) return MAX_PLANNER_SCRATCH
  return Math.max(5, Math.min(200, parsed))
}

function digestFoldedCount(text: string): number {
  const match = text.match(/(\d+)/)
  return match ? Number(match[1]) : 1
}

export function capScratchWithDigest<T extends { digest?: boolean }>(
  items: T[],
  max: number,
  makeDigest: (foldedCount: number) => T,
  countText: (item: T) => string,
): T[] {
  if (items.length <= max) return items
  const overflow = items.length - max
  const folded = items.slice(0, overflow)
  const kept = items.slice(overflow)
  let foldedCount = 0
  for (const entry of folded) {
    foldedCount += entry.digest ? digestFoldedCount(countText(entry)) : 1
  }
  return [makeDigest(foldedCount), ...kept]
}

export function capPlannerScratch<T extends { text: string; digest?: boolean }>(
  items: T[],
  max: number = resolvePlannerScratchCap(),
): T[] {
  return capScratchWithDigest(
    items,
    max,
    (foldedCount) => ({ text: `earlier: ${foldedCount} more (folded)`, digest: true } as T),
    (item) => item.text,
  )
}

function capAbandonedAlternatives(
  items: PlannerAbandonedAlternative[],
  max: number,
): PlannerAbandonedAlternative[] {
  return capScratchWithDigest(
    items,
    max,
    (foldedCount) => ({
      description: `earlier: ${foldedCount} more (folded)`,
      reason: 'folded',
      digest: true,
    }),
    (item) => item.description,
  )
}

function capOpenAssumptions(
  items: PlannerOpenAssumption[],
  max: number,
): PlannerOpenAssumption[] {
  return capScratchWithDigest(
    items,
    max,
    (foldedCount) => ({ text: `earlier: ${foldedCount} more (folded)`, verified: true, digest: true }),
    (item) => item.text,
  )
}

const StepStatusSchema = z.enum([
  'pending',
  'in_progress',
  'done',
  'blocked',
  'skipped',
])

const StepSchema: z.ZodType<PlannerHierarchicalStep> = z.lazy(() =>
  z.object({
    id: z.string().min(1),
    title: z.string().min(1),
    status: StepStatusSchema,
    detail: z.string().optional(),
    children: z.array(StepSchema).optional(),
  }),
)

const DecisionSchema = z.object({
  at: z.string().optional(),
  text: z.string().min(1),
})

const RiskSchema = z.object({
  severity: z.enum(['low', 'medium', 'high']),
  text: z.string().min(1),
})

const AbandonedAlternativeSchema = z.object({
  description: z.string().min(1),
  reason: z.string().min(1),
})

const OpenAssumptionSchema = z.object({
  text: z.string().min(1),
  verified: z.boolean().optional(),
})

const PayloadSchema = z.object({
  taskSummary: z.string().min(1).optional(),
  currentSubtaskId: z.string().optional(),
  plan: z.array(StepSchema).optional(),
  decisions: z.array(DecisionSchema).optional(),
  risks: z.array(RiskSchema).optional(),
  abandonedAlternatives: z.array(AbandonedAlternativeSchema).optional(),
  openAssumptions: z.array(OpenAssumptionSchema).optional(),
  currentStepRationale: z.string().min(1).optional(),
})

const FENCE_REGEX = /```(?:planner|json:planner)\s*\n([\s\S]*?)\n?```/i

export function extractPlannerWorkingMemory(
  content: string,
  prev?: PlannerWorkingMemory,
): PlannerWorkingMemory | undefined {
  const match = content.match(FENCE_REGEX)
  if (!match) return prev
  let raw: unknown
  try {
    raw = JSON.parse(match[1])
  } catch {
    return prev
  }
  const parsed = PayloadSchema.safeParse(raw)
  if (!parsed.success) return prev
  const payload = parsed.data
  const now = new Date().toISOString()
  const decisions = (payload.decisions ?? []).map((d) => ({
    at: d.at ?? now,
    text: d.text,
  }))
  const abandonedFromPayload = payload.abandonedAlternatives ?? []
  const mergedAbandoned = [
    ...(prev?.abandonedAlternatives ?? []),
    ...abandonedFromPayload.filter(
      (incoming) =>
        !(prev?.abandonedAlternatives ?? []).some(
          (existing) => existing.description === incoming.description,
        ),
    ),
  ]

  const assumptionsFromPayload = payload.openAssumptions ?? []
  const mergedAssumptions = (() => {
    if (assumptionsFromPayload.length === 0) {
      return prev?.openAssumptions
    }
    const byText = new Map<string, { text: string; verified?: boolean }>()
    for (const a of prev?.openAssumptions ?? []) byText.set(a.text, a)
    for (const a of assumptionsFromPayload) byText.set(a.text, a)
    return [...byText.values()]
  })()

  const cap = resolvePlannerScratchCap()
  return {
    taskSummary: payload.taskSummary ?? prev?.taskSummary ?? '',
    currentSubtaskId: payload.currentSubtaskId ?? prev?.currentSubtaskId,
    plan: payload.plan ?? prev?.plan ?? [],
    decisions: decisions.length > 0 ? decisions : prev?.decisions ?? [],
    risks: payload.risks ?? prev?.risks ?? [],
    abandonedAlternatives:
      mergedAbandoned.length > 0
        ? capAbandonedAlternatives(mergedAbandoned, cap)
        : prev?.abandonedAlternatives,
    openAssumptions: mergedAssumptions
      ? capOpenAssumptions(mergedAssumptions, cap)
      : mergedAssumptions,
    currentStepRationale:
      payload.currentStepRationale ?? prev?.currentStepRationale,
    updatedAt: now,
  }
}

export interface PlannerBoardPlanStep {
  id: string
  title: string
  status: string
  depth: number
}

export interface PlannerWorkingMemoryBoardSections {
  plan: PlannerBoardPlanStep[]
  decisions: string[]
  abandoned: Array<{ description: string; reason: string }>
  assumptions: Array<{ text: string; verified: boolean }>
  openQuestions: string[]
}

function flattenSteps(
  steps: PlannerHierarchicalStep[] | undefined,
  depth = 0,
  out: PlannerBoardPlanStep[] = [],
): PlannerBoardPlanStep[] {
  for (const step of steps ?? []) {
    out.push({ id: step.id, title: step.title, status: step.status, depth })
    if (step.children?.length) flattenSteps(step.children, depth + 1, out)
  }
  return out
}

export function formatPlannerWorkingMemory(
  pwm: PlannerWorkingMemory | undefined,
): PlannerWorkingMemoryBoardSections {
  const assumptions = (pwm?.openAssumptions ?? []).map((assumption) => ({
    text: assumption.text,
    verified: assumption.verified === true,
  }))
  return {
    plan: flattenSteps(pwm?.plan),
    decisions: (pwm?.decisions ?? []).map((decision) => decision.text),
    abandoned: (pwm?.abandonedAlternatives ?? []).map((alt) => ({
      description: alt.description,
      reason: alt.reason,
    })),
    assumptions,
    openQuestions: assumptions
      .filter((assumption) => !assumption.verified)
      .map((assumption) => assumption.text.trim())
      .filter(Boolean),
  }
}
