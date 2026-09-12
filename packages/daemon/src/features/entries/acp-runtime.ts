import { ExternalAcpAgentDispatcher } from '../../acp/external-agent.js'
import { createExternalAcpRunTool } from '../../tools/external-acp-run.js'
import type { ToolRegistry } from '../../tools/registry.js'
import type { FeatureRuntimeToolDeps } from '../types.js'

/**
 * Creates the external ACP agent dispatcher, registers the `external_acp.run`
 * tool, and returns the dispatcher so buildRuntime can expose it on the runtime
 * object. Only imported (and thus only bundled) when the `acp` feature is
 * enabled.
 */
export function createAcpRuntime(
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
): ExternalAcpAgentDispatcher {
  const dispatcher = new ExternalAcpAgentDispatcher({
    sessions: deps.sessions,
    deviceName: deps.deviceName,
    dreaming: deps.dreaming,
  })
  registry.register(createExternalAcpRunTool(dispatcher))
  return dispatcher
}
