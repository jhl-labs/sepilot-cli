// Apply shared model and permission defaults while preserving CLI-owned mode selection.

import type { DaemonConfigDefaults } from './daemon-bootstrap.js'

// Just enough of ChatAction to spell out the 5 dispatches this
// adapter emits. Kept as a structural type so we don't import the
// reducer module here (the reducer's full ChatAction union pulls in
// surface/agent types that don't belong on this seam).
export type DaemonConfigDispatchAction =
  | { type: 'SET_MODE'; mode: DaemonConfigDefaults['agent']['mode'] }
  | {
      type: 'SET_THINKING_LEVEL'
      thinkingLevel: DaemonConfigDefaults['agent']['thinkingLevel']
    }
  | {
      type: 'SET_AUTONOMY'
      autonomy: DaemonConfigDefaults['agent']['autonomy']
    }
  | { type: 'SET_MODEL'; model: string }
  | { type: 'SET_PROVIDER'; provider: string }

export interface CliConfigOverrides {
  model?: string
  provider?: string
}

/**
 * Apply a daemon config snapshot (and any --model / --provider cli
 * overrides) onto the chat reducer. Order matters:
 *
 *   1. Thinking/autonomy come from daemon policy. Mode remains owned by
 *      the CLI: desktop personal-assistant modes must not replace its general loop.
 *   2. Model and provider resolve cli > daemon-default > skip; only
 *      one dispatch each, so we never double-fire and never have to
 *      reason about reducer ordering.
 *
 * No-ops cleanly when daemonConfig is null and the cli didn't pass a
 * model/provider; useful for the "daemon up but config endpoint
 * failed" branch where we still want connect-on-mount to keep going.
 */
export function applyDaemonConfigToState(opts: {
  daemonConfig: DaemonConfigDefaults | null
  cliOverrides: CliConfigOverrides
  dispatch: (action: DaemonConfigDispatchAction) => void
}): void {
  const { daemonConfig, cliOverrides, dispatch } = opts

  if (daemonConfig) {
    dispatch({
      type: 'SET_THINKING_LEVEL',
      thinkingLevel: daemonConfig.agent.thinkingLevel,
    })
    dispatch({ type: 'SET_AUTONOMY', autonomy: daemonConfig.agent.autonomy })
  }

  const resolvedModel =
    cliOverrides.model
    ?? (daemonConfig?.agent.defaultModel || null)
  if (resolvedModel) {
    dispatch({ type: 'SET_MODEL', model: resolvedModel })
  }

  const resolvedProvider =
    cliOverrides.provider
    ?? (daemonConfig?.agent.defaultProvider || null)
  if (resolvedProvider) {
    dispatch({ type: 'SET_PROVIDER', provider: resolvedProvider })
  }
}
