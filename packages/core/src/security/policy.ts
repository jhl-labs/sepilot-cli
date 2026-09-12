import type { AutonomyLevel } from '../agent/types.js'
import type { ToolSecurityDescriptor } from '../agent/types.js'
import type { PolicyCheckResult } from './types.js'

export interface ToolExecRequest {
  tool: string
  input: Record<string, unknown>
  cwd?: string
  /** Immutable strict filesystem boundary for the active turn. */
  workspaceRoot?: string
  /** Runtime registration provenance used to fail closed for unaudited tools. */
  registrationSource?: 'builtin' | 'plugin'
  /** Registry-owned effect metadata. Missing descriptors fail closed. */
  security?: ToolSecurityDescriptor
}

export interface IToolPolicy {
  check(request: ToolExecRequest, autonomy: AutonomyLevel): PolicyCheckResult
  reload(): Promise<void>
}
