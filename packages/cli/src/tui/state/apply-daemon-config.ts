import type { DaemonConfigDefaults } from './daemon-bootstrap.js'

export type DaemonConfigDispatchAction =
  | { type: 'SET_MODE'; mode: DaemonConfigDefaults['agent']['mode'] }
  | {
      type: 'SET_THINKING_LEVEL'
      thinkingLevel: DaemonConfigDefaults['agent']['thinkingLevel']
      source: 'default'
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
      source: 'default',
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
