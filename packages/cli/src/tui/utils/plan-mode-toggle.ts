export type PrimaryAgentId = 'plan' | 'build'

export type PlanToggleDecision =
  | { kind: 'switch'; to: PrimaryAgentId }
  | { kind: 'confirm-exit'; to: 'build' }

/**
 * Decide what a plan/build toggle should do. Exiting plan mode (plan →
 * build) is gated like Claude Code's ExitPlanMode: the first toggle arms a
 * confirmation so the user deliberately reviews the plan before the agent
 * starts making changes; a second toggle while armed performs the switch.
 * Entering plan mode (build → plan) is never gated — restricting the
 * session to read-only is always safe and immediate.
 */
export function resolvePlanModeToggle(
  current: PrimaryAgentId,
  exitConfirmArmed: boolean,
): PlanToggleDecision {
  if (current === 'build') {
    return { kind: 'switch', to: 'plan' }
  }
  // current === 'plan' → leaving plan mode
  return exitConfirmArmed ? { kind: 'switch', to: 'build' } : { kind: 'confirm-exit', to: 'build' }
}
