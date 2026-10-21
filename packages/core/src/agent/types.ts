import type { ApiError } from '../types/errors.js'
import type { MemoryContextItem } from '../memory/semantic-index.js'
import type { ApprovalDecisionStatus, ApprovalRule } from '../security/approval-decisions.js'
import type { LiveView, LiveViewStill } from '../views/types.js'

export enum AutonomyLevel {
  ReadOnly = 'readonly',
  AcceptEdits = 'accept-edits',
  WorkspaceWrite = 'workspace-write',
  Supervised = 'supervised',
  Autonomous = 'autonomous',
}

export enum ThinkingLevel {
  Auto = 'auto',
  Off = 'off',
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Max = 'max',
}

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool'

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; source: ImageSource }
  | { type: 'document'; source: DocumentSource }

export interface ImageSource {
  type: 'base64' | 'url'
  mediaType: string
  data: string
}

export interface DocumentSource {
  type: 'base64' | 'url'
  mediaType: string
  data: string
}

export interface Message {
  role: MessageRole
  content: string | ContentPart[]
  toolCallId?: string
  toolCalls?: ToolCall[]
  name?: string
  metadata?: Record<string, unknown>
}

export interface ToolDefinition {
  name: string
  description: string
  inputSchema: Record<string, unknown>
}

export type ToolSecurityEffect =
  | 'observe'
  | 'internal-state'
  | 'workspace-write'
  | 'external-write'
  | 'process-lifecycle'
  | 'dynamic'
  | 'unknown'

export type ToolReach = 'workspace' | 'host' | 'network'

export interface ToolSecurityDescriptor {
  effect: ToolSecurityEffect
  executionCapabilities?: AgentExecutionCapability[]
  rationale?: string
  mutationResultBoundary?: 'required' | 'commutative'
  reach?: ToolReach
  egressGuard?: 'allowlist'
  writesArtifact?: true
}

export interface AgentExecutionPolicy {
  requestedAutonomy?: AutonomyLevel
  configuredAutonomy: AutonomyLevel
  effectiveAutonomy: AutonomyLevel
  clamped: boolean
  clampReason?: string
  agentMode: string
  primaryAgentId?: string
  workspaceBoundary: 'strict' | 'unrestricted'
  freshApprovalRequired: boolean
}

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
}

export interface AgentAcceptanceCriterion {
  id: string
  text: string
}

export interface AgentRequiredArtifact {
  path: string
  kind: 'file' | 'directory' | 'document' | 'other' | (string & {})
  description?: string
}

export interface AgentEvidenceRequirement {
  kind: 'source' | 'repository' | 'artifact' | 'validation' | 'other' | (string & {})
  description: string
  minSourceObservations?: number
  minSourceFiles?: number
  minSourceScopes?: number
  sourceToolNames?: string[]
  requiresArtifactEvidenceMap?: boolean
  requiresArtifactSelfReview?: boolean
  requiresSearch?: boolean
}

export interface AgentArtifactSection {
  id: string
  title: string
  description?: string
  artifactPath?: string
  required?: boolean
}

export type AgentExecutionIntentKind =
  | 'operational-action'
  | 'workspace-change'
  | 'inspection'
  | 'artifact-production'
  | 'conversation'

export type AgentWorkspaceMutationIntent = 'forbidden' | 'allowed' | 'required'

export type AgentExecutionCapability =
  | 'process'
  | 'service'
  | 'terminal'
  | 'browser'
  | 'filesystem-read'
  | 'filesystem-write'
  | 'network'
  | 'application-state'

export interface AgentRequestedTerminalCommand {
  executable: string
  args: string[]
}

export interface AgentRequestedProcessStart extends AgentRequestedTerminalCommand {
  cwd?: string
  ttlMs?: number
  lifetime?: 'bounded' | 'session'
  network?: 'none' | {
    mode: 'none' | 'loopback'
    ports?: number[]
  }
}

export interface AgentExecutionIntent {
  kind: AgentExecutionIntentKind
  workspaceMutation: AgentWorkspaceMutationIntent
  capabilities: AgentExecutionCapability[]
  capabilityPolicy?: 'closed'
  allowedTools?: string[]
  toolSequence?: string[]
  retryPolicy?: 'forbidden'
  authorizedWriteTargets?: string[]
  protectedWriteTargets?: string[]
  requestedTerminalCommand?: AgentRequestedTerminalCommand
  requestedProcessStart?: AgentRequestedProcessStart
  constrainedProcessStart?: AgentRequestedProcessStart
}

export interface AgentRunContract {
  summary: string
  acceptanceCriteria: AgentAcceptanceCriterion[]
  constraints: string[]
  outOfScope: string[]
  requiredArtifacts?: AgentRequiredArtifact[]
  evidenceRequirements?: AgentEvidenceRequirement[]
  artifactSections?: AgentArtifactSection[]
  executionIntent?: AgentExecutionIntent
  reasoningComplexity?: 'routine' | 'standard' | 'complex'
  source: 'planner' | 'fallback'
}

export interface AgentFailedAttempt {
  signature: string
  tool: string
  reason: string
  ts: number
}

export interface AgentOpenQuestion {
  id: string
  text: string
  blocking: boolean
  askedAt?: number
}

export interface AgentStateBoardPlanStep {
  id: string
  title: string
  status: string
  depth: number
}

export interface ToolExecutionPosture {
  sandbox: {
    requested: boolean
    active: boolean
    mode: 'host' | 'docker' | 'ssh' | 'unknown' | (string & {})
    fallbackReason?: string
  }
  filesystem: {
    boundary:
      | 'working-directory'
      | 'session_cwd'
      | 'process_cwd'
      | 'workspace_policy'
      | 'tool_specific'
      | 'unknown'
      | (string & {})
    cwd?: string
    isolated: boolean
    readOnly?: boolean
    note?: string
  }
  network: {
    isolated: boolean
    mode: 'host' | 'none' | 'bridge' | 'disabled' | 'unknown' | (string & {})
  }
}

export interface ToolResultMetadata {
  userActionRequired?: string
  executionPosture?: ToolExecutionPosture
  managedProcess?: {
    id: string
    startedAt: string
    status: 'running' | 'exited'
    exitCode: number | null
    signal: string | null
    terminationReason?: 'requested' | 'ttl' | 'daemon-shutdown'
  }
  [key: string]: unknown
}

export interface ChatRequest {
  model: string
  messages: Message[]
  systemPrompt?: string
  tools?: ToolDefinition[]
  toolChoice?: 'auto' | 'required' | 'none'
  temperature?: number
  maxTokens?: number
  thinkingLevel?: ThinkingLevel
  stopSequences?: string[]
  timeoutMs?: number
}

export interface TokenUsage {
  inputTokens: number
  outputTokens: number
  thinkingTokens?: number
  cacheReadTokens?: number
  cacheCreationTokens?: number
  estimatedCost?: number
  generationMs?: number
}

export interface AgentContextUsage {
  inputTokens: number
  contextWindowTokens?: number
  reservedOutputTokens?: number
  outputTokens?: number
  generationMs?: number
  iteration: number
  source: 'estimated' | 'provider'
}

export interface ChatResponse {
  message: Message
  thinking?: string
  usage: TokenUsage
  finishReason: 'stop' | 'length' | 'tool_use' | 'content_filter'
  finishDetail?: 'truncated_output' | 'incomplete_stream'
  truncatedToolCalls?: Array<{ name: string; streamedChars: number }>
  raw?: Record<string, unknown>
}

export interface LlmRequestDigest {
  maxOutputTokens?: number
  thinkingLevel?: string
  model: string
  providerId?: string
  source?: string
  startedAt?: number
  timeoutMs?: number
  auxiliary?: boolean
  messageCount: number
  systemPromptChars: number
  toolSchemaChars?: number
  toolNames: string[]
  traceRef?: string
}

export type StreamChunk =
  | { type: 'text'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_call_start'; toolCall: Partial<ToolCall> }
  | { type: 'tool_call_delta'; toolCallId: string; delta: string }
  | { type: 'tool_call_end'; toolCallId: string }
  | { type: 'usage'; usage: TokenUsage }
  | { type: 'done'; finishReason: ChatResponse['finishReason'] }
  | { type: 'error'; error: ApiError }

export type ModelToolTransport = 'auto' | 'native' | 'prompt-react' | 'adaptive'
export type ModelAnswerProtocol = 'auto' | 'repair-left-truncated-answer-stem'
export type ModelReasoningEffortAliases = Partial<Record<'low' | 'medium' | 'high', string>>

export interface ModelCompatibilityProfile {
  toolTransport?: ModelToolTransport
  reasoningEffortOffValue?: 'none' | 'minimal'
  reasoningEffortAliases?: ModelReasoningEffortAliases
  auxiliaryThinkingOffFallback?: 'low' | 'medium' | 'high' | 'max'
  repairThinkingOffFallback?: 'low' | 'medium' | 'high' | 'max'
  answerProtocol?: ModelAnswerProtocol
  notes?: string[]
}

export interface ModelInfo {
  id: string
  name: string
  contextWindow: number
  maxOutputTokens: number
  capabilities: {
    vision: boolean
    toolUse: boolean
    streaming: boolean
    embedding: boolean
    thinking: boolean
    thinkingControl?: 'reasoning-effort' | 'chat-template-kwargs' | 'think-levels'
    thinkingEffort?: 'graded' | 'binary'
    adaptivePromptReact?: boolean
    promptReactPreferred?: boolean
    deepCoderAnalysis?: boolean
  }
  compatibility?: ModelCompatibilityProfile
  inputCostPer1k?: number
  outputCostPer1k?: number
}

export type AgentState = 'idle' | 'thinking' | 'acting' | 'observing' | 'done' | 'error'

export type AgentRecoveryScope =
  | 'provider_protocol'
  | 'tool_execution'
  | 'output_synthesis'
  | 'session_resume'
  | (string & {})

export type AgentRecoveryKind =
  | 'missing_tool_result'
  | 'thinking_block_order'
  | 'thinking_disabled_violation'
  | 'empty_message_content'
  | 'journal_replay'
  | 'probe_recovery'
  | 'synthetic_final_message'
  | (string & {})

export type AgentRecoveryAction =
  | 'retry_sanitized_messages'
  | 'retry_sanitized_messages_without_thinking'
  | 'reuse_journaled_tool_result'
  | 'probe_side_effect'
  | 'append_synthetic_message'
  | (string & {})

export type RunStopKind = 'completed' | 'incomplete' | 'blocked' | 'cancelled' | 'error'

export type RunStopCode =
  | 'completed'
  | 'iteration_budget'
  | 'node_budget'
  | 'observation_budget'
  | 'no_progress'
  | 'stuck_repeat'
  | 'completion_gate'
  | 'cost_gate'
  | 'spend_budget'
  | 'approval_denied'
  | 'approval_timeout'
  | 'user_action_required'
  | 'policy_blocked'
  | 'wall_clock'
  | 'inactivity'
  | 'user_abort'
  | 'provider_error'
  | 'agent_incomplete'

export type RunStopNextAction =
  | 'resume'
  | 'switch_autonomy'
  | 'approve_pending'
  | 'raise_budget'
  | 'retry'
  | 'escalate'

export interface RunStopReason {
  kind: RunStopKind
  code: RunStopCode
  summary?: string
  detail?: {
    budget?: number
    used?: number
    tool?: string
    criteria?: string[]
    requestId?: string
    layer?: string
    escalateMode?: string
    escalateBot?: string
  }
  resumable: boolean
  nextActions: RunStopNextAction[]
}

export type AgentEvent =
  | { type: 'state_change'; state: AgentState }
  | { type: 'execution_policy'; policy: AgentExecutionPolicy }
  | { type: 'memory_context'; id: string; items: MemoryContextItem[] }
  | {
      type: 'llm_request'
      turnId: string
      iteration: number
      requestDigest: LlmRequestDigest
    }
  | {
      type: 'context_usage'
      context: AgentContextUsage
    }
  | {
      type: 'skills_loaded'
      skills: Array<{ id: string; source: 'selected' | 'router' | 'bot' | 'auto' }>
    }
  | {
      type: 'mode_route_decision'
      chosen: string
      persona?: string
      candidates?: string[]
      reason?: string
      confidence?: number
      fallback: boolean
    }
  | {
      type: 'quality_gate_verdict'
      phase: string
      decision: 'pass' | 'retry' | 'incomplete'
      blockingReason?: string
      backtrackCount: number
    }
  | {
      type: 'backtrack'
      phase: string
      reason: string
      attempt: number
    }
  | {
      type: 'node_trace'
      node: string
      durationMs: number
      nextEdge?: string
    }
  | { type: 'thinking'; content: string }
  | {
      type: 'reasoning_step'
      label: string
      detail?: string
    }
  | {
      type: 'action_progress'
      summary: string
      nextStep: string
      toolNames: string[]
    }
  | { type: 'text_delta'; text: string }
  | { type: 'message'; content: string }
  | {
      type: 'question_request'
      questionId: string
      prompt: string
      choices?: string[]
    }
  | {
      type: 'recovery'
      scope: AgentRecoveryScope
      kind: AgentRecoveryKind
      action: AgentRecoveryAction
      message: string
      recoverable: boolean
      details?: Record<string, unknown>
    }
  | { type: 'tool_call'; toolCall: ToolCall }
  | {
      type: 'tool_result'
      toolCallId: string
      output: string
      status: 'success' | 'error'
      contentParts?: ContentPart[]
      recovery?: 'journal' | 'probe'
      executionPosture?: ToolExecutionPosture
      metadata?: ToolResultMetadata
    }
  | {
      type: 'approval_request'
      toolCall: ToolCall
      requestId: string
      suggestedRule?: ApprovalRule
      previewDiff?: string
      context?: string
    }
  | {
      type: 'auto_approval'
      toolCall: ToolCall
      requestId: string
      decision: 'approved' | 'denied'
      rule: ApprovalRule
      scope: 'session' | 'always' | 'run' | 'session-all'
    }
  | {
      type: 'approval_response'
      requestId: string
      decision: ApprovalDecisionStatus
      approved: boolean
      note?: string
    }
  | {
      type: 'cowork_plan'
      plan: Array<{ role: string; instruction: string }>
    }
  | {
      type: 'run_contract'
      contract: AgentRunContract
    }
  | {
      type: 'cowork_task_start'
      role: string
      instruction: string
    }
  | {
      type: 'cowork_task_complete'
      role: string
      instruction: string
      result: string
    }
  | {
      type: 'cowork_task_failed'
      role: string
      instruction: string
      error: string
    }
  | {
      type: 'cowork_synthesizing'
      summary: string
    }
  | {
      type: 'cowork_discuss_request'
      prompt: string
      choices?: string[]
    }
  | {
      type: 'cowork_discuss_response'
      prompt: string
      response: string
    }
  | { type: 'view_attached'; view: LiveView }
  | { type: 'view_still'; view: LiveView; still: LiveViewStill }
  | { type: 'view_detached'; viewId: string; reason: string }
  | {
      type: 'panel_open'
      personas: Array<{ id: string; name: string; description?: string }>
    }
  | {
      type: 'panel_turn_start'
      personaId: string
      personaName: string
    }
  | {
      type: 'panel_turn_complete'
      personaId: string
      personaName: string
      text: string
    }
  | {
      type: 'panel_turn_failed'
      personaId: string
      personaName: string
      error: string
    }
  | {
      type: 'panel_synthesizing'
      panelists: number
    }
  | {
      type: 'edit_checkpoint_opened'
      checkpoint: EditCheckpointSummary
    }
  | {
      type: 'edit_checkpoint_resolved'
      checkpoint: EditCheckpointSummary
    }
  | {
      type: 'debate_round'
      round: DebateRoundSummary
    }
  | {
      type: 'planner_working_memory_updated'
      workingMemory: PlannerWorkingMemory
    }
  | {
      type: 'phase_change'
      enteredPhase: string | null
      closedPhase?: { phase: string; usage: TokenUsage }
      phaseUsages?: Record<string, TokenUsage>
    }
  | {
      type: 'post_edit_findings'
      editedFiles: string[]
      impactedExternalModules: string[]
      impactedLocalModules: string[]
      reverseCallers: string[]
      diagnostics: Array<{ file: string; summary: string }>
      analyzedAt: string
    }
  | {
      type: 'context_compact'
      summary: string
      beforeTokens: number
      afterTokens: number
    }
  | {
      type: 'done'
      usage: TokenUsage
      stopReason?: RunStopReason
    }
  | { type: 'error'; error: ApiError }
  | {
      type: 'subagent_progress'
      subagentId: string
      label?: string
      inner: AgentEvent
    }
  | {
      type: 'router_decision'
      id: string
      decision: {
        mode: string
        persona: string
        skillIds: string[]
        toolGroups?: string[]
        reason: string
        confidence: 'high' | 'medium' | 'low'
        fallback: boolean
      }
    }
  | {
      type: 'state_board'
      criteriaTotal: number
      planTotal: number
      planDone: number
      todosTotal: number
      todosDone: number
      currentNode?: string
      nodeState?: AgentState
      iteration?: number
      todos?: Array<{ content: string; status: string }>
      plan?: Array<{ title: string; status: string; depth: number }>
    }
  | {
      type: 'steering_consumed'
      id: string
      noteId: string
      message: string
      kind: 'instruction' | 'question'
      iteration: number
    }

export interface EditCheckpointFile {
  path: string
  hadFileBefore: boolean
  bytesBefore: number
}

export interface EditCheckpointSummary {
  checkpointId: string
  label: string
  status: 'open' | 'committed' | 'reverted'
  files: EditCheckpointFile[]
  createdAt: string
  closedAt?: string
  revertedAt?: string
  revertReason?: string
}

export type DebateRole = 'proposer' | 'critic' | 'resolver'

export interface DebateRoundEntry {
  role: DebateRole
  content: string
  tokensUsed: number
  startedAt: string
  endedAt: string
}

export type DebateDecision = 'accept' | 'revise' | 'reject'

export interface DebateRoundSummary {
  roundId: string
  topic: string
  entries: DebateRoundEntry[]
  finalDecision: DebateDecision
  rationale: string
  createdAt: string
}

export type PlannerStepStatus = 'pending' | 'in_progress' | 'done' | 'blocked' | 'skipped'

export interface PlannerHierarchicalStep {
  id: string
  title: string
  status: PlannerStepStatus
  detail?: string
  children?: PlannerHierarchicalStep[]
}

export interface PlannerDecision {
  at: string
  text: string
}

export interface PlannerRisk {
  severity: 'low' | 'medium' | 'high'
  text: string
}

export interface PlannerAbandonedAlternative {
  description: string
  reason: string
  digest?: boolean
}

export interface PlannerOpenAssumption {
  text: string
  verified?: boolean
  digest?: boolean
}

export interface PlannerWorkingMemory {
  taskSummary: string
  currentSubtaskId?: string
  plan: PlannerHierarchicalStep[]
  decisions: PlannerDecision[]
  risks: PlannerRisk[]
  abandonedAlternatives?: PlannerAbandonedAlternative[]
  openAssumptions?: PlannerOpenAssumption[]
  currentStepRationale?: string
  updatedAt: string
}
