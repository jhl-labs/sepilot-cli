import { z } from 'zod'
import type { ILLMProvider } from '@sepilotd/core'

export const modelRoutingHintSchema = z
  .object({
    taskType: z.enum(['general', 'simple', 'complex', 'code', 'creative', 'analysis']).optional(),
    preference: z.enum(['configured', 'lowest-cost']).optional(),
    minContextWindow: z.number().int().positive().max(100_000_000).optional(),
    requireToolUse: z.boolean().optional(),
    maxInputCostPer1k: z.number().finite().nonnegative().optional(),
    maxOutputCostPer1k: z.number().finite().nonnegative().optional(),
  })
  .strict()
export type ModelRoutingHints = z.infer<typeof modelRoutingHintSchema>
export const modelRoutingPolicySchema = z
  .object({
    routes: z
      .array(
        z
          .object({
            taskType: modelRoutingHintSchema.shape.taskType.unwrap(),
            preferredModels: z.array(z.string().trim().min(1)).min(1).max(64),
            providerId: z.string().trim().min(1).optional(),
          })
          .strict(),
      )
      .max(32)
      .optional(),
    defaults: modelRoutingHintSchema.optional(),
  })
  .strict()

export function resolveModelRoutingHints(
  defaults: ModelRoutingHints = {},
  request: ModelRoutingHints = {},
): ModelRoutingHints {
  const minimum = (a?: number, b?: number) =>
    a === undefined ? b : b === undefined ? a : Math.min(a, b)
  return {
    ...defaults,
    ...request,
    minContextWindow:
      Math.max(defaults.minContextWindow ?? 0, request.minContextWindow ?? 0) || undefined,
    requireToolUse: defaults.requireToolUse || request.requireToolUse,
    maxInputCostPer1k: minimum(defaults.maxInputCostPer1k, request.maxInputCostPer1k),
    maxOutputCostPer1k: minimum(defaults.maxOutputCostPer1k, request.maxOutputCostPer1k),
  }
}

export function modelMeetsRoutingRequirements(
  model: ILLMProvider['models'][number],
  hints: ModelRoutingHints,
): boolean {
  const within = (cost: number | undefined, cap: number | undefined) =>
    cap === undefined || (cost !== undefined && Number.isFinite(cost) && cost >= 0 && cost <= cap)
  return (
    (!hints.requireToolUse || model.capabilities.toolUse === true) &&
    (hints.minContextWindow === undefined || model.contextWindow >= hints.minContextWindow) &&
    within(model.inputCostPer1k, hints.maxInputCostPer1k) &&
    within(model.outputCostPer1k, hints.maxOutputCostPer1k)
  )
}

export function modelCatalogCost(model: ILLMProvider['models'][number]): number {
  const values = [model.inputCostPer1k, model.outputCostPer1k]
  return values.every((value) => value !== undefined && Number.isFinite(value) && value >= 0)
    ? values[0]! + values[1]!
    : Number.POSITIVE_INFINITY
}
