import { createDelegateTool } from '../../tools/delegate.js'
import type { FeatureRuntimeToolRegistrar, FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry, deps) => {
  registry.register(createDelegateTool(deps.delegator!))
}

export const registerRuntimeTool: FeatureRuntimeToolRegistrar = (registry, deps) => {
  registry.register(
    createDelegateTool(deps.delegator, {
      appendDelegationResultEvent: (sessionId, event) =>
        deps.sessions.appendEvent(sessionId, event),
    }),
  )
}
