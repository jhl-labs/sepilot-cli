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

export function isExecutorConfirmedReadOnlyObservation(
  input: ReadOnlyObservationEvidenceInput,
): boolean {
  if (input.tool === 'tool.output.read') return false
  if (input.executionObserved === true && input.securityEffect === 'observe') {
    return true
  }
  return input.securityEffect === 'dynamic'
    && RUNTIME_ATTESTED_DYNAMIC_OBSERVATION_TOOLS.has(input.tool)
    && input.actionPurpose === 'observe'
    && input.executionPosture?.filesystem.readOnly === true
}
