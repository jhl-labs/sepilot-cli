import type { SwarmRunRegistry } from '../run/run-registry.js'
import type { TmuxSessionPool } from '../tmux/pool.js'
import type { AgentLauncher } from '../launcher/agent-launcher.js'
import type { SwarmAgentRuntimeAdapter } from '../launcher/runtime-adapter.js'

export interface SwarmToolContext {
  registry: SwarmRunRegistry
  pool: TmuxSessionPool
  launcher: AgentLauncher
  runtime?: SwarmAgentRuntimeAdapter
  env?: NodeJS.ProcessEnv
  resolveRunId: (sessionId: string | undefined) => string | null
}
