import type { AgentEvent } from '@sepilotd/core'
import type { AgentState, GraphExecutionContext } from '../types.js'

export function openEditCheckpointNode(label = 'agent-turn') {
  return async function* (
    state: AgentState,
    context?: GraphExecutionContext,
  ): AsyncGenerator<AgentEvent, AgentState, void> {
    if (
      context?.editSnapshotStore
      && !state.currentEditCheckpointId
    ) {
      const checkpointId = context.editSnapshotStore.openCheckpoint(
        context.agentContext.sessionId,
        label,
      )
      state.currentEditCheckpointId = checkpointId
      const summary = context.editSnapshotStore.get(
        context.agentContext.sessionId,
        checkpointId,
      )
      if (summary) {
        yield { type: 'edit_checkpoint_opened', checkpoint: summary }
      }
    }
    return state
  }
}
