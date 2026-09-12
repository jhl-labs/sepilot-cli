import type { ToolExecutionPosture } from '@sepilotd/core'

const RUNTIME_ATTESTED_DYNAMIC_OBSERVATION_TOOLS = new Set([
  'terminal.run',
  'process.start',
])

export interface ReadOnlyObservationEvidenceInput {
  tool: string
  securityEffect?: unknown
  executionObserved?: boolean
  actionPurpose?: unknown
  executionPosture?: ToolExecutionPosture
}

/**
 * Decide whether one successful result may be cited as read-only observation
 * evidence. Static observe tools need the executor marker. Dynamic evidence is
 * limited to the two built-ins whose runtimes enforce and attest immutable
 * filesystem capabilities; a plugin cannot become evidence by returning a
 * lookalike posture object or declaring its own dynamic effect.
 */
export function isExecutorConfirmedReadOnlyObservation(
  input: ReadOnlyObservationEvidenceInput,
): boolean {
  if (input.executionObserved === true && input.securityEffect === 'observe') {
    return true
  }
  return input.securityEffect === 'dynamic'
    && RUNTIME_ATTESTED_DYNAMIC_OBSERVATION_TOOLS.has(input.tool)
    && input.actionPurpose === 'observe'
    && input.executionPosture?.filesystem.readOnly === true
}
