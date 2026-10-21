import { ExternalAcpAgentDispatcher, type ExternalAgentRuntime } from '../../acp/external-agent.js'
import { createExternalAcpRunTool } from '../../tools/external-acp-run.js'
import { createExternalCliRunTool } from '../../tools/external-cli-run.js'
import { ExternalCliDispatcher } from '../../external-cli/dispatcher.js'
import { createLogger } from '../../logger.js'
import type { ToolRegistry } from '../../tools/registry.js'
import type { FeatureRuntimeToolDeps } from '../types.js'

const log = createLogger('external-cli')

export function createAcpRuntime(
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
): ExternalAgentRuntime {
  const dispatcher = new ExternalAcpAgentDispatcher({
    sessions: deps.sessions,
    deviceName: deps.deviceName,
    dreaming: deps.dreaming,
  })
  registry.register(createExternalAcpRunTool(dispatcher))
  const cli = new ExternalCliDispatcher({
    sessions: deps.sessions,
    deviceName: deps.deviceName,
    usageTracker: deps.usageTracker,
    runOptions: { log: (event, fields) => log.info(event, fields) },
  })
  registry.register(createExternalCliRunTool(cli))
  return { acp: dispatcher, cli }
}
