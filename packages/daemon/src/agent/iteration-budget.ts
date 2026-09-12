/**
 * Shared per-turn agent-loop iteration cap for user-facing chat surfaces.
 *
 * One iteration is one LLM round, which may emit one or more parallel tool
 * calls. Keep the env name stable for existing operators even though the
 * budget now applies to channel turns as well as direct /chat routes.
 */
export const DEFAULT_AGENT_MAX_ITERATIONS = 50

export function resolveAgentMaxIterations(
  input: { maxIterations?: number } = {},
): number {
  if (typeof input.maxIterations === 'number' && input.maxIterations >= 1) {
    return input.maxIterations
  }
  const envRaw = process.env.SEPILOTD_CHAT_MAX_ITERATIONS
  if (envRaw) {
    const envValue = Number.parseInt(envRaw, 10)
    if (Number.isFinite(envValue) && envValue >= 1) return envValue
  }
  return DEFAULT_AGENT_MAX_ITERATIONS
}
