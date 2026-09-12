/** A real interactive CLI running on the connected daemon host. */
export type DesktopAgentId = 'claude' | 'codex' | 'gemini' | 'opencode'
export type DesktopAgentState = 'running' | 'exited'
export interface DesktopAgentEvent {
  seq: number
  type: 'snapshot' | 'output' | 'resize' | 'exit'
  data?: string
  cols?: number
  rows?: number
  exitCode?: number
}
export interface DesktopAgentSshTarget {
  /** Hostname, IP address or alias in the daemon user's ~/.ssh/config. */
  host: string
  user?: string
  port?: number
}
export interface DesktopAgentSession {
  ssh?: DesktopAgentSshTarget
  id: string
  agent: DesktopAgentId
  cwd: string
  command: string
  pid: number
  state: DesktopAgentState
  cols: number
  rows: number
  createdAt: string
  exitCode?: number
}
