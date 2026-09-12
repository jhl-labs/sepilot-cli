import type { AgentRunContract } from '@sepilotd/core'
import type { AgentState } from './types.js'

const CONTRACT_ITERATION_FLOOR_CAP = 160

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 1
    ? Math.floor(value)
    : undefined
}

export function contractIterationFloor(contract: AgentRunContract | undefined): number | undefined {
  if (!contract) {
    return undefined
  }

  let workloadScore = 0
  const requiredArtifacts = contract.requiredArtifacts?.length ?? 0
  if (requiredArtifacts > 0) {
    workloadScore += 16 + requiredArtifacts * 4
  }

  const evidenceRequirements = contract.evidenceRequirements ?? []
  if (evidenceRequirements.length > 0) {
    workloadScore += 12
  }
  for (const requirement of evidenceRequirements) {
    if (typeof requirement.minSourceFiles === 'number' && Number.isFinite(requirement.minSourceFiles)) {
      workloadScore += Math.min(30, Math.max(0, requirement.minSourceFiles)) * 2
    }
    if (typeof requirement.minSourceScopes === 'number' && Number.isFinite(requirement.minSourceScopes)) {
      workloadScore += Math.min(12, Math.max(0, requirement.minSourceScopes)) * 5
    }
    if (requirement.requiresArtifactEvidenceMap) {
      workloadScore += 8
    }
    if (requirement.requiresArtifactSelfReview) {
      workloadScore += 6
    }
    if (requirement.requiresSearch) {
      workloadScore += 4
    }
  }

  const requiredSections = (contract.artifactSections ?? [])
    .filter((section) => section.required !== false)
    .length
  if (requiredSections > 0) {
    workloadScore += 20 + requiredSections * 5
  }

  if (requiredArtifacts === 0 && evidenceRequirements.length === 0 && requiredSections === 0) {
    return undefined
  }

  if (contract.acceptanceCriteria.length > 3) {
    workloadScore += (contract.acceptanceCriteria.length - 3) * 3
  }

  if (workloadScore <= 0) {
    return undefined
  }
  return Math.min(CONTRACT_ITERATION_FLOOR_CAP, Math.max(50, workloadScore))
}

export function resolveRunIterationBudget(input: {
  requestedIterations?: number
  graphIterations?: number
  runContract?: AgentRunContract
  defaultIterations?: number
}): number {
  const candidates = [
    positiveInteger(input.requestedIterations),
    positiveInteger(input.graphIterations),
    contractIterationFloor(input.runContract),
  ].filter((value): value is number => typeof value === 'number')
  return candidates.length > 0
    ? Math.max(...candidates)
    : positiveInteger(input.defaultIterations) ?? 10
}

export function childIterationBudget(
  parent: Pick<AgentState, 'maxIterations' | 'seedContract'>,
  options: {
    min: number
    legacyMax: number
    contractMin?: number
    parentShare?: number
    hardCap?: number
  },
): number {
  const parentBudget = positiveInteger(parent.maxIterations) ?? options.legacyMax
  const min = Math.max(1, Math.floor(options.min))
  const legacyMax = Math.max(min, Math.floor(options.legacyMax))
  const contractFloor = contractIterationFloor(parent.seedContract)

  if (!contractFloor) {
    return Math.max(min, Math.min(parentBudget, legacyMax))
  }

  const share = options.parentShare ?? 0.35
  const contractMin = options.contractMin ?? legacyMax
  const hardCap = options.hardCap ?? parentBudget
  const target = Math.max(
    legacyMax,
    contractMin,
    Math.ceil(parentBudget * share),
  )
  return Math.max(min, Math.min(parentBudget, hardCap, target))
}
