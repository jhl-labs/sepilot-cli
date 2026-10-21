export type PrimaryAgentId = 'plan' | 'build'

export type PlanToggleDecision =
  | { kind: 'switch'; to: PrimaryAgentId }
  | { kind: 'confirm-exit'; to: 'build' }

export function resolvePlanModeToggle(
  current: PrimaryAgentId,
  exitConfirmArmed: boolean,
): PlanToggleDecision {
  if (current === 'build') {
    return { kind: 'switch', to: 'plan' }
  }
  return exitConfirmArmed ? { kind: 'switch', to: 'build' } : { kind: 'confirm-exit', to: 'build' }
}
