import type { AutonomyLevel } from '../agent/types.js'
import type { ToolSecurityDescriptor } from '../agent/types.js'
import type { PolicyCheckResult } from './types.js'

export interface ToolExecRequest {
  tool: string
  input: Record<string, unknown>
  cwd?: string
  workspaceRoot?: string
  registrationSource?: 'builtin' | 'plugin'
  security?: ToolSecurityDescriptor
}

export interface IToolPolicy {
  check(request: ToolExecRequest, autonomy: AutonomyLevel): PolicyCheckResult
  reload(): Promise<void>
}
