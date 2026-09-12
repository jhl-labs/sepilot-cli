export const SWARM_AGENT_NAMES = ['claude', 'codex', 'gemini', 'opencode'] as const
export type SwarmAgentName = typeof SWARM_AGENT_NAMES[number]

export type SwarmAgentStatus =
  | 'starting'
  | 'idle'
  | 'busy'
  | 'blocked'
  | 'dead'
  | 'killed'

export type SwarmAgentRuntimeName =
  | 'tmux'
  | 'acp'
  | 'a2a'
  | 'unknown'

export type SwarmPreflightStatus =
  | 'ready'
  | 'degraded'
  | 'blocked'

export type SwarmPreflightCheckStatus =
  | 'ok'
  | 'warn'
  | 'blocked'

export interface SwarmPreflightCheck {
  id: string
  status: SwarmPreflightCheckStatus
  summary: string
  detail?: string
}

export interface SwarmPreflightReport {
  status: SwarmPreflightStatus
  agent: SwarmAgentName
  runtime: SwarmAgentRuntimeName
  cwd: string
  checks: SwarmPreflightCheck[]
}

export type SwarmAgentStartupState =
  | 'spawning'
  | 'trust_required'
  | 'tool_permission_required'
  | 'ready_for_prompt'
  | 'running'
  | 'finished'
  | 'failed'

export type SwarmAgentStartupFailureKind =
  | 'configuration'
  | 'process_spawn'
  | 'transport'
  | 'protocol'
  | 'timeout'
  | 'unknown'

export type SwarmAgentRecoveryScenario =
  | 'trust_prompt_unresolved'
  | 'tool_permission_unresolved'
  | 'prompt_misdelivery'
  | 'configuration_failure'
  | 'transport_failure'
  | 'protocol_failure'
  | 'provider_failure'
  | 'unknown_failure'

export type SwarmAgentRecoveryStep =
  | 'inspect_worker_output'
  | 'review_trust_prompt'
  | 'review_tool_permission'
  | 'resend_prompt'
  | 'fix_configuration'
  | 'check_transport'
  | 'inspect_protocol_logs'
  | 'restart_worker'
  | 'escalate_to_human'

export type SwarmAgentRecoveryEscalationPolicy =
  | 'alert_human'
  | 'abort'
  | 'log_and_continue'

export interface SwarmAgentRecoveryHint {
  scenario: SwarmAgentRecoveryScenario
  steps: SwarmAgentRecoveryStep[]
  maxAttempts: number
  escalationPolicy: SwarmAgentRecoveryEscalationPolicy
  reason: string
}

export type SwarmAgentRecoveryStatus =
  | 'succeeded'
  | 'failed'
  | 'escalated'

export interface SwarmAgentStartupEvidence {
  handle: string
  agent: SwarmAgentName
  runtime: SwarmAgentRuntimeName
  lifecycleState: SwarmAgentStartupState
  cwd: string
  paneCommand?: string
  startedAt: number
  readyAt?: number
  promptSentAt?: number
  promptAccepted?: boolean
  trustPromptDetected: boolean
  toolPermissionPromptDetected: boolean
  transportHealthy: boolean
  elapsedMs: number
  failureKind?: SwarmAgentStartupFailureKind
  failureMessage?: string
  lastOutputPreview?: string
  recoveryHint?: SwarmAgentRecoveryHint
}

export type SwarmRunStatus =
  | 'pending'
  | 'running'
  | 'done'
  | 'error'
  | 'cancelled'
  | 'interrupted'

export interface SwarmAgentHandle {
  handle: string
  agent: SwarmAgentName
  role?: string
  tmuxSessionName: string
  cwd: string
  status: SwarmAgentStatus
  spawnedAt: number
  runtime?: SwarmAgentRuntimeName
  startupEvidence?: SwarmAgentStartupEvidence
}

export interface SwarmRun {
  id: string
  goal: string
  status: SwarmRunStatus
  worktree: { path: string; createdByDaemon: boolean }
  activeHandle?: string
  agents: SwarmAgentHandle[]
  createdAt: number
  endedAt?: number
}

export type SwarmEvent =
  | { type: 'run.started'; runId: string; goal: string; ts: number }
  | { type: 'agent.spawned'; runId: string; agent: SwarmAgentHandle; ts: number }
  | { type: 'agent.startup'; runId: string; handle: string; evidence: SwarmAgentStartupEvidence; ts: number }
  | { type: 'agent.status'; runId: string; handle: string; status: SwarmAgentStatus; ts: number }
  | {
      type: 'agent.recovery'
      runId: string
      handle: string
      scenario: SwarmAgentRecoveryScenario
      action: SwarmAgentRecoveryStep
      attempt: number
      maxAttempts: number
      status: SwarmAgentRecoveryStatus
      message: string
      replacementHandle?: string
      outputPreview?: string
      ts: number
    }
  | { type: 'agent.active'; runId: string; handle: string; ts: number }
  | { type: 'agent.killed'; runId: string; handle: string; ts: number }
  | { type: 'tool.call'; runId: string; tool: string; input: unknown; ts: number }
  | { type: 'tool.result'; runId: string; tool: string; outputPreview: string; ts: number }
  | { type: 'pane.snapshot'; runId: string; handle: string; text: string; ts: number }
  | { type: 'supervisor.message'; runId: string; text: string; ts: number }
  | { type: 'run.ended'; runId: string; status: SwarmRunStatus; ts: number }
