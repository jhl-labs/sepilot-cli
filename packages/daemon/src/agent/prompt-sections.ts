/**
 * Shared system-prompt section contract between the prompt builder and the
 * provider adapters. Leaf module on purpose: providers import it without
 * pulling the agent runtime in.
 *
 * Everything before the turn-context heading is stable across the turns of a
 * session (tool catalog, guidelines, workspace, environment, repository
 * instructions). Everything after it is retrieved per turn (memory hits,
 * daily notes, sender context) and may differ on every request. Keeping the
 * volatile tail last lets prefix-caching providers reuse the stable prefix,
 * and lets providers with explicit cache breakpoints place the breakpoint at
 * the boundary instead of after the volatile text.
 */
export const TURN_CONTEXT_HEADING = 'Turn context (retrieved for this turn; may differ on the next turn):'

export interface SplitSystemText {
  /** Stable prefix; empty when the prompt is nothing but turn context. */
  staticText: string
  /** Volatile tail including the heading, or undefined when absent. */
  turnText?: string
}

export function splitTurnContext(systemText: string): SplitSystemText {
  const marker = `\n${TURN_CONTEXT_HEADING}`
  const index = systemText.startsWith(TURN_CONTEXT_HEADING) ? 0 : systemText.indexOf(marker)
  if (index < 0) return { staticText: systemText }
  const boundary = index === 0 ? 0 : index + 1
  return {
    staticText: systemText.slice(0, boundary).trimEnd(),
    turnText: systemText.slice(boundary),
  }
}
