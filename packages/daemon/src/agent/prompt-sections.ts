export const TURN_CONTEXT_HEADING = 'Turn context (retrieved for this turn; may differ on the next turn):'

export interface SplitSystemText {
  staticText: string
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

export const TURN_SCOPED_SYSTEM_METADATA_KEY = 'sepilot.turnScopedSystem'

export const RUNTIME_CONTROLS_METADATA_KEY = 'sepilot.runtimeControls'

export const RUN_STABLE_SYSTEM_METADATA_KEY = 'sepilot.runStableSystem'

export function isRunStableSystemMessage(message: { role: string; metadata?: Record<string, unknown> }): boolean {
  return message.role === 'system' && message.metadata?.[RUN_STABLE_SYSTEM_METADATA_KEY] === true
}

export function isTurnScopedSystemMessage(message: { role: string; metadata?: Record<string, unknown> }): boolean {
  return message.role === 'system' && message.metadata?.[TURN_SCOPED_SYSTEM_METADATA_KEY] === true
}
