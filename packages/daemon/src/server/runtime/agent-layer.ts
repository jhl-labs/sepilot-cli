import {
  buildLlmCache,
  buildModelRouter,
  buildProviderCircuitBreaker,
  buildRunLimiter,
} from './services.js'
import { ModelFallbackState } from '../../providers/model-fallback.js'
import { join } from 'node:path'
import { EditSnapshotStore } from '../../agent/edit-rollback/store.js'
import { ToolStatsStore } from '../../agent/tool-learning/store.js'
import type { ToolOutcomeObservation } from '../../agent/tool-learning/store.js'
import { WorkspaceMutationTracker } from '../../agent/workspace-mutation/tracker.js'
import { SessionUndoStack } from './undo-stack.js'
import type { buildProviderRegistry } from './providers.js'

type ProviderRegistryInstance = ReturnType<typeof buildProviderRegistry>

export interface AgentLayer {
  llmCache: ReturnType<typeof buildLlmCache>
  providerCircuitBreaker: ReturnType<typeof buildProviderCircuitBreaker>
  modelFallbackState: ModelFallbackState
  runLimiter: ReturnType<typeof buildRunLimiter>
  modelRouter: ReturnType<typeof buildModelRouter>
  editSnapshotStore: EditSnapshotStore
  toolStatsStore: ToolStatsStore
  workspaceMutationTracker: WorkspaceMutationTracker
  sessionUndoStack: SessionUndoStack
}

export function assembleAgentLayer(args: {
  providerRegistry: ProviderRegistryInstance
  modelRouting?: Parameters<typeof buildModelRouter>[1]
  dataDir?: string
  onToolOutcome?: (outcome: ToolOutcomeObservation) => void
}): AgentLayer {
  const llmCache = buildLlmCache()
  const providerCircuitBreaker = buildProviderCircuitBreaker()
  const modelFallbackState = new ModelFallbackState()
  const runLimiter = buildRunLimiter()
  const modelRouter = buildModelRouter(args.providerRegistry, args.modelRouting)
  const editSnapshotStore = new EditSnapshotStore(
    args.dataDir ? { persistDir: join(args.dataDir, 'sessions') } : {},
  )
  const toolStatsStore = new ToolStatsStore(args.onToolOutcome)
  const workspaceMutationTracker = new WorkspaceMutationTracker()
  const sessionUndoStack = new SessionUndoStack()

  return {
    llmCache,
    providerCircuitBreaker,
    modelFallbackState,
    runLimiter,
    modelRouter,
    editSnapshotStore,
    toolStatsStore,
    workspaceMutationTracker,
    sessionUndoStack,
  }
}
