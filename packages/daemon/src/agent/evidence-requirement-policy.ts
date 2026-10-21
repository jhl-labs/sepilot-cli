import type { AgentEvidenceRequirement } from '@sepilotd/core'

export function evidenceRequirementMinSourceFiles(
  requirement: AgentEvidenceRequirement,
): number {
  return requirement.minSourceFiles ?? (requirement.kind === 'repository' ? 4 : 0)
}

export function evidenceRequirementMinSourceObservations(
  requirement: AgentEvidenceRequirement,
): number {
  return requirement.minSourceObservations ?? (requirement.kind === 'source' ? 1 : 0)
}

export function evidenceRequirementMinSourceScopes(
  requirement: AgentEvidenceRequirement,
): number {
  return requirement.minSourceScopes ?? 0
}

export function evidenceRequirementRequiresSearch(
  requirement: AgentEvidenceRequirement,
): boolean {
  return requirement.requiresSearch ?? requirement.kind === 'repository'
}

export function evidenceRequirementUsesRepositoryBreadth(
  requirement: AgentEvidenceRequirement,
): boolean {
  return requirement.kind === 'repository'
    || evidenceRequirementMinSourceFiles(requirement) > 0
    || evidenceRequirementMinSourceScopes(requirement) > 0
}

export function sourceToolMatchesEvidenceRequirement(
  requirement: AgentEvidenceRequirement,
  tool: string,
): boolean {
  const names = requirement.sourceToolNames
  return !names?.length || names.includes(tool)
}
