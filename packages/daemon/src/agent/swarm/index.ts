import type { ToolRegistry } from '../../tools/registry.js'
import type { SwarmToolContext } from './tools/context.js'
import {
  createPreflightTool,
  createSpawnAgentTool,
  createKillAgentTool,
  createListAgentsTool,
  createRecoverTool,
  createSendTool,
  createInterruptTool,
  createCaptureTool,
  createSetActiveTool,
  createWaitIdleTool,
  createDriveTool,
} from './tools/tmux-tools.js'

export { SwarmRunRegistry } from './run/run-registry.js'
export { SwarmRunStore } from './run/run-store.js'
export { TmuxSessionPool } from './tmux/pool.js'
export { AgentLauncher } from './launcher/agent-launcher.js'
export {
  A2ASwarmAgentRuntimeAdapter,
  AcpSwarmAgentRuntimeAdapter,
  MultiplexSwarmAgentRuntimeAdapter,
  TmuxSwarmAgentRuntimeAdapter,
  createDefaultSwarmAgentRuntimeAdapter,
} from './launcher/runtime-adapter.js'
export type {
  SwarmAgentIdleResult,
  SwarmAgentRuntimeAdapter,
} from './launcher/runtime-adapter.js'
export {
  driveSwarmAgent,
  driveResultToToolResult,
} from './controller/driver.js'
export type {
  SwarmDriveInput,
  SwarmDriveRequiredAction,
  SwarmDriveRequiredActionType,
  SwarmDriveResult,
  SwarmDriveStatus,
  SwarmDriveTurn,
  SwarmDriveTurnStatus,
} from './controller/driver.js'
export { WorktreeManager } from './launcher/worktree.js'
export type { SwarmToolContext } from './tools/context.js'

export function registerSwarmTools(registry: ToolRegistry, ctx: SwarmToolContext): void {
  registry.register(createPreflightTool(ctx))
  registry.register(createSpawnAgentTool(ctx))
  registry.register(createKillAgentTool(ctx))
  registry.register(createListAgentsTool(ctx))
  registry.register(createRecoverTool(ctx))
  registry.register(createSendTool(ctx))
  registry.register(createInterruptTool(ctx))
  registry.register(createCaptureTool(ctx))
  registry.register(createSetActiveTool(ctx))
  registry.register(createWaitIdleTool(ctx))
  registry.register(createDriveTool(ctx))
}
