import { createDelegateTool } from '../../tools/delegate.js'
import type { FeatureRuntimeToolRegistrar, FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry, deps) => {
  registry.register(createDelegateTool(deps.delegator!))
}

/**
 * Second-pass registration inside buildRuntime: re-registers the delegate tool
 * (same tool name, overriding the buildToolRegistry one) wired to append the
 * delegation result event onto the originating session.
 */
export const registerRuntimeTool: FeatureRuntimeToolRegistrar = (registry, deps) => {
  registry.register(
    createDelegateTool(deps.delegator, {
      appendDelegationResultEvent: (sessionId, event) =>
        deps.sessions.appendEvent(sessionId, event),
    }),
  )
}
