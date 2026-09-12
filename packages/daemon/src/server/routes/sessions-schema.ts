import { z } from 'zod'
import type { SessionEvidenceArtifactKind } from '@sepilotd/core'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import { safeIdSchema } from '../../utils/safe-id.js'

export const sessionTotalsSchema = z.object({
  input: z.number().int(),
  output: z.number().int(),
})

const sessionPersonaIdSchema = z.string().trim().min(1).max(120)

export const sessionPersonaIdsSchema = z
  .array(sessionPersonaIdSchema)
  .transform((ids) => [...new Set(ids)])
  .pipe(z.array(sessionPersonaIdSchema).max(6))

export const sessionMetaSchema = z.object({
  id: z.string(),
  title: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  provider: z.string(),
  model: z.string(),
  device: z.string(),
  status: z.enum(['active', 'completed', 'abandoned']),
  cwd: z.string().optional(),
  workspaceIsolation: z.enum(['policy', 'strict']).optional(),
  messageCount: z.number().int(),
  totalTokens: sessionTotalsSchema,
  totalCost: z.number(),
  tags: z.array(z.string()),
  personaIds: sessionPersonaIdsSchema.optional(),
  primaryAgentId: z.string().optional(),
  isRunning: z.boolean(),
})

export const sessionEventSchema = z.object({
  id: z.string(),
  timestamp: z.string(),
  type: z.string(),
}).catchall(z.unknown())

export const pendingApprovalSchema = z.object({
  requestId: z.string(),
  sessionId: z.string(),
  toolCallId: z.string(),
  tool: z.string(),
  input: z.record(z.unknown()),
  requestedAt: z.string(),
  expiresAt: z.string(),
  state: z.enum(['live', 'stale']),
  resumeAvailable: z.boolean().optional(),
})

export const resumableRunSchema = z.object({
  stage: z.enum(['thinking', 'acting', 'observing']),
  checkpointedAt: z.string(),
  mode: z.enum(['exact', 'replay-safe', 'replay-risky']),
  forceRequired: z.boolean(),
  currentTool: z.string().optional(),
  currentToolCount: z.number().int().positive().optional(),
  currentTools: z.array(z.string()).optional(),
  journaledResultAvailable: z.boolean().optional(),
  recoveryProbeAvailable: z.boolean().optional(),
})

export const resumableRunIssueSchema = z.object({
  status: z.enum(['corrupt', 'unreadable']),
  message: z.string(),
})

export const sessionDelegationSchema = z.object({
  delegationId: z.string(),
  targetDevice: z.string(),
  claimHealth: z.enum(['healthy', 'degraded', 'lost']),
  startedAt: z.string(),
  updatedAt: z.string(),
  degradedSince: z.string().optional(),
  lastHeartbeatAt: z.string().optional(),
  lastError: z.string().optional(),
  leaseLossSource: z.enum(['gateway', 'comments', 'transport']).optional(),
})

export const sessionTraceMetricsSchema = z.object({
  startedAt: z.string(),
  lastEventAt: z.string(),
  runDurationMs: z.number().int().nonnegative(),
  totalEvents: z.number().int().nonnegative(),
  userMessages: z.number().int().nonnegative(),
  assistantMessages: z.number().int().nonnegative(),
  toolCalls: z.number().int().nonnegative(),
  toolResults: z.number().int().nonnegative(),
  toolSuccesses: z.number().int().nonnegative(),
  toolFailures: z.number().int().nonnegative(),
  approvalRequests: z.number().int().nonnegative(),
  approvalApproved: z.number().int().nonnegative(),
  approvalFeedback: z.number().int().nonnegative(),
  approvalDenied: z.number().int().nonnegative(),
  autoApprovalsApproved: z.number().int().nonnegative(),
  autoApprovalsDenied: z.number().int().nonnegative(),
  contextCompactions: z.number().int().nonnegative(),
  contextCompactionTokensBefore: z.number().int().nonnegative(),
  contextCompactionTokensAfter: z.number().int().nonnegative(),
  contextCompactionTokensSaved: z.number().int().nonnegative(),
  lastContextCompactedAt: z.string().optional(),
  memoryContextEvents: z.number().int().nonnegative(),
  memoryContextItems: z.number().int().nonnegative(),
  memorySummaryEvents: z.number().int().nonnegative(),
  memorySummaryLightCaptures: z.number().int().nonnegative(),
  memorySummarySemanticExtractions: z.number().int().nonnegative(),
  memorySummaryRagPromotions: z.number().int().nonnegative(),
  lastMemorySummarizedAt: z.string().optional(),
  coworkTasksStarted: z.number().int().nonnegative(),
  coworkTasksCompleted: z.number().int().nonnegative(),
  coworkTasksFailed: z.number().int().nonnegative(),
  coworkDiscussRequests: z.number().int().nonnegative(),
  coworkDiscussResponses: z.number().int().nonnegative(),
  todoUpdates: z.number().int().nonnegative(),
  providerAttemptEvents: z.number().int().nonnegative(),
  providerAttemptsStarted: z.number().int().nonnegative(),
  providerAttemptFailures: z.number().int().nonnegative(),
  providerAttemptSuccesses: z.number().int().nonnegative(),
  providerFallbacks: z.number().int().nonnegative(),
  lastProviderFallbackAt: z.string().optional(),
  finalProvider: z.string(),
  finalModel: z.string(),
  timeToFirstAssistantMessageMs: z.number().int().nonnegative().optional(),
  timeToFirstToolCallMs: z.number().int().nonnegative().optional(),
  timeToFirstApprovalRequestMs: z.number().int().nonnegative().optional(),
  timeToFirstCoworkTaskMs: z.number().int().nonnegative().optional(),
})

export const sessionChecklistItemSchema = z.object({
  label: z.string(),
  status: z.enum(['pending', 'in_progress', 'completed', 'blocked', 'skipped']),
  detail: z.string().optional(),
  source: z.enum(['todo', 'tool', 'approval', 'question', 'session']).optional(),
})

export const sessionCompletionChecklistSchema = z.object({
  status: z.enum(['not_started', 'in_progress', 'completed', 'blocked']),
  items: z.array(sessionChecklistItemSchema),
  lastUpdatedAt: z.string().optional(),
})

export const sessionWorkingMemoryDecisionSchema = z.object({
  type: z.enum(['approval', 'question', 'delegation']),
  summary: z.string(),
  timestamp: z.string(),
})

export const sessionWorkingMemoryToolOutcomeSchema = z.object({
  tool: z.string(),
  status: z.enum(['success', 'error']),
  output: z.string(),
  timestamp: z.string(),
})

export const sessionWorkingMemoryFileChangeSchema = z.object({
  path: z.string(),
  tool: z.string(),
  kind: z.enum(['write', 'edit', 'patch', 'unknown']),
  timestamp: z.string(),
})

export const sessionWorkingMemorySchema = z.object({
  taskSummary: z.string(),
  latestPlanStep: z.string().optional(),
  activeTodo: z.string().optional(),
  keyDecisions: z.array(sessionWorkingMemoryDecisionSchema),
  recentToolOutcomes: z.array(sessionWorkingMemoryToolOutcomeSchema),
  fileChanges: z.array(sessionWorkingMemoryFileChangeSchema),
  openQuestions: z.array(z.string()),
  lastUpdatedAt: z.string().optional(),
})

export const sessionRunContractSchema = z.object({
  summary: z.string(),
  acceptanceCriteria: z.array(z.object({
    id: z.string(),
    text: z.string(),
  })),
  constraints: z.array(z.string()),
  outOfScope: z.array(z.string()),
  requiredArtifacts: z.array(z.object({
    path: z.string(),
    kind: z.string(),
    description: z.string().optional(),
  })).optional(),
  evidenceRequirements: z.array(z.object({
    kind: z.string(),
    description: z.string(),
    minSourceObservations: z.number().optional(),
    minSourceFiles: z.number().optional(),
    minSourceScopes: z.number().optional(),
    sourceToolNames: z.array(z.string()).optional(),
    requiresArtifactEvidenceMap: z.boolean().optional(),
    requiresArtifactSelfReview: z.boolean().optional(),
    requiresSearch: z.boolean().optional(),
  })).optional(),
  artifactSections: z.array(z.object({
    id: z.string(),
    title: z.string(),
    description: z.string().optional(),
    artifactPath: z.string().optional(),
    required: z.boolean().optional(),
  })).optional(),
  source: z.enum(['planner', 'fallback']),
})

const SESSION_EVIDENCE_ARTIFACT_SCHEMA_KIND_VALUES = [
  'run_contract',
  'completion_verdict',
  'observation',
  'action_receipt',
  'validation',
  'file_change',
  'artifact_readback',
  'approval',
  'todo',
  'assistant_claim',
  'post_edit_findings',
  'session_end',
] as const satisfies readonly SessionEvidenceArtifactKind[]

// `satisfies` rejects unknown schema values; this conditional rejects a
// missing core value. Keep the runtime tuple local so direct daemon tests do
// not depend on a previously built @sepilotd/core dist artifact.
export const SESSION_EVIDENCE_ARTIFACT_SCHEMA_KINDS: Exclude<
  SessionEvidenceArtifactKind,
  (typeof SESSION_EVIDENCE_ARTIFACT_SCHEMA_KIND_VALUES)[number]
> extends never
  ? typeof SESSION_EVIDENCE_ARTIFACT_SCHEMA_KIND_VALUES
  : never = SESSION_EVIDENCE_ARTIFACT_SCHEMA_KIND_VALUES

export const sessionEvidenceArtifactSchema = z.object({
  id: z.string(),
  kind: z.enum(SESSION_EVIDENCE_ARTIFACT_SCHEMA_KINDS),
  label: z.string(),
  summary: z.string(),
  status: z.enum(['pending', 'success', 'error', 'warning', 'info']),
  timestamp: z.string(),
  sourceEventIds: z.array(z.string()),
  toolCallId: z.string().optional(),
  tool: z.string().optional(),
  path: z.string().optional(),
  hash: z.string().optional(),
  currentRevision: z.boolean().optional(),
  relatedAcceptanceCriteriaIds: z.array(z.string()).optional(),
})

export const sessionAcceptanceEvidenceSchema = z.object({
  id: z.string(),
  text: z.string(),
  status: z.enum(['supported', 'failed', 'blocked', 'unverified']),
  evidenceIds: z.array(z.string()),
  reason: z.string(),
})

export const sessionEvidenceRiskSchema = z.object({
  code: z.enum([
    'no_run_contract',
    'acceptance_criteria_unverified',
    'validation_failed',
    'changes_without_validation',
    'pending_tool_results',
    'failed_tool_results',
    'post_edit_findings',
    'assistant_marked_unverified',
  ]),
  severity: z.enum(['info', 'warning']),
  message: z.string(),
  evidenceIds: z.array(z.string()).optional(),
})

export const sessionContractLedgerEntrySchema = z.object({
  id: z.string(),
  key: z.string(),
  value: z.string(),
  source: z.enum([
    'user_goal',
    'run_contract',
    'session_runtime',
    'repo_fact',
    'evidence',
    'conservative_default',
    'blocker',
  ]),
  status: z.enum([
    'missing',
    'weak',
    'defaulted',
    'inferred',
    'confirmed',
    'blocked',
  ]),
  confidence: z.number().min(0).max(1),
  reversible: z.boolean(),
  rationale: z.string(),
  eventIds: z.array(z.string()).optional(),
  evidenceIds: z.array(z.string()).optional(),
})

export const sessionContractLedgerSectionSchema = z.object({
  name: z.enum([
    'goal',
    'scope',
    'acceptance_criteria',
    'deliverables',
    'evidence_requirements',
    'verification_plan',
    'runtime_context',
    'blockers',
  ]),
  status: z.enum([
    'missing',
    'weak',
    'defaulted',
    'inferred',
    'confirmed',
    'blocked',
  ]),
  summary: z.string(),
  entries: z.array(sessionContractLedgerEntrySchema),
})

export const sessionContractLedgerBlockerSchema = z.object({
  code: z.enum([
    'credential_or_secret',
    'destructive_production_action',
    'external_side_effect',
    'billing_authority',
    'legal_or_medical_judgment',
  ]),
  severity: z.enum(['warning', 'blocker']),
  message: z.string(),
  eventIds: z.array(z.string()),
})

export const sessionContractLedgerSchema = z.object({
  schemaVersion: z.literal(1),
  status: z.enum(['empty', 'ready', 'needs_attention', 'blocked']),
  revision: z.string(),
  fingerprint: z.string(),
  eventCount: z.number().int().nonnegative(),
  lastEventId: z.string().optional(),
  lastEventAt: z.string(),
  summary: z.object({
    sections: z.number().int().nonnegative(),
    confirmedSections: z.number().int().nonnegative(),
    defaultedSections: z.number().int().nonnegative(),
    inferredSections: z.number().int().nonnegative(),
    missingSections: z.number().int().nonnegative(),
    blockedSections: z.number().int().nonnegative(),
    blockers: z.number().int().nonnegative(),
    safeDefaults: z.number().int().nonnegative(),
  }),
  sections: z.array(sessionContractLedgerSectionSchema),
  blockers: z.array(sessionContractLedgerBlockerSchema),
  lastUpdatedAt: z.string(),
})

export const sessionEvidenceManifestSchema = z.object({
  schemaVersion: z.literal(1),
  status: z.enum(['empty', 'collecting', 'ready', 'attention_needed']),
  revision: z.string(),
  fingerprint: z.string(),
  eventCount: z.number().int().nonnegative(),
  lastEventId: z.string().optional(),
  lastEventAt: z.string(),
  summary: z.object({
    artifacts: z.number().int().nonnegative(),
    toolCalls: z.number().int().nonnegative(),
    toolResults: z.number().int().nonnegative(),
    validationRuns: z.number().int().nonnegative(),
    validationFailures: z.number().int().nonnegative(),
    filesChanged: z.number().int().nonnegative(),
    approvals: z.number().int().nonnegative(),
    acceptanceCriteria: z.number().int().nonnegative(),
    supportedAcceptanceCriteria: z.number().int().nonnegative(),
    failedAcceptanceCriteria: z.number().int().nonnegative(),
    blockedAcceptanceCriteria: z.number().int().nonnegative(),
    unverifiedAcceptanceCriteria: z.number().int().nonnegative(),
    risks: z.number().int().nonnegative(),
  }),
  acceptanceCriteria: z.array(sessionAcceptanceEvidenceSchema),
  artifacts: z.array(sessionEvidenceArtifactSchema),
  risks: z.array(sessionEvidenceRiskSchema),
  lastUpdatedAt: z.string(),
})

export const sessionEvaluationStageSchema = z.object({
  status: z.enum([
    'not_started',
    'running',
    'passed',
    'failed',
    'blocked',
    'skipped',
    'unverified',
  ]),
  summary: z.string(),
  evidenceIds: z.array(z.string()),
})

export const sessionEvaluationArtifactFileSchema = z.object({
  id: z.string(),
  path: z.string(),
  artifactId: z.string(),
  status: z.enum(['pending', 'success', 'error', 'warning', 'info']),
  operation: z.string(),
  contentState: z.enum(['hashed', 'metadata_only']),
  sourceEventIds: z.array(z.string()),
  tool: z.string().optional(),
  sizeBytes: z.number().int().nonnegative().optional(),
  contentHash: z.string().optional(),
})

export const sessionEvaluationArtifactSkipSchema = z.object({
  reason: z.enum([
    'missing_path',
    'path_traversal',
    'absolute_path_without_cwd',
    'outside_cwd',
    'generated_or_vendor',
    'duplicate_path',
    'max_files_exceeded',
    'file_missing',
    'not_file',
    'file_too_large',
    'total_budget_exceeded',
    'read_error',
  ]),
  message: z.string(),
  artifactId: z.string().optional(),
  path: z.string().optional(),
  sourceEventIds: z.array(z.string()).optional(),
})

export const sessionEvaluationArtifactBundleSchema = z.object({
  schemaVersion: z.literal(1),
  status: z.enum(['empty', 'metadata_only', 'ready', 'partial']),
  revision: z.string(),
  fingerprint: z.string(),
  eventCount: z.number().int().nonnegative(),
  cwd: z.string().optional(),
  limits: z.object({
    maxFiles: z.number().int().positive(),
    maxFileBytes: z.number().int().positive(),
    maxTotalBytes: z.number().int().positive(),
  }),
  summary: z.object({
    files: z.number().int().nonnegative(),
    hashedFiles: z.number().int().nonnegative(),
    metadataOnlyFiles: z.number().int().nonnegative(),
    skippedFiles: z.number().int().nonnegative(),
    totalBytes: z.number().int().nonnegative(),
    validationArtifacts: z.number().int().nonnegative(),
    acceptanceCriteria: z.number().int().nonnegative(),
  }),
  files: z.array(sessionEvaluationArtifactFileSchema),
  skipped: z.array(sessionEvaluationArtifactSkipSchema),
  validationEvidenceIds: z.array(z.string()),
  acceptanceEvidenceIds: z.array(z.string()),
  lastUpdatedAt: z.string(),
})

export const sessionConsensusTriggerSchema = z.object({
  code: z.enum([
    'validation_failed',
    'post_edit_findings',
    'assistant_marked_unverified',
    'mechanical_validation_missing',
    'semantic_acceptance_unverified',
    'artifact_bundle_partial',
    'large_change_set',
    'missing_run_contract',
  ]),
  priority: z.number().int().positive(),
  fired: z.boolean(),
  severity: z.enum(['info', 'warning']),
  message: z.string(),
  evidenceIds: z.array(z.string()).optional(),
})

export const sessionConsensusTriggerMatrixSchema = z.object({
  required: z.boolean(),
  primaryTrigger: sessionConsensusTriggerSchema.optional(),
  triggers: z.array(sessionConsensusTriggerSchema),
})

export const sessionAcceptanceAssertionSchema = z.object({
  id: z.string(),
  acceptanceCriterionId: z.string(),
  tier: z.enum(['constant', 'structural', 'behavioral', 'subjective']),
  kind: z.enum(['file_exists', 'symbol_exists', 'text_match', 'validation_required', 'human_review']),
  description: z.string(),
  pattern: z.string().optional(),
  expectedValue: z.string().optional(),
  fileHint: z.string().optional(),
  confidence: z.number().min(0).max(1),
})

export const sessionAcceptanceAssertionResultSchema = z.object({
  assertion: sessionAcceptanceAssertionSchema,
  status: z.enum(['verified', 'failed', 'unverified', 'skipped']),
  detail: z.string(),
  evidenceIds: z.array(z.string()),
  path: z.string().optional(),
  actualValue: z.string().optional(),
})

export const sessionAcceptanceVerificationReportSchema = z.object({
  acceptanceCriterionId: z.string(),
  acceptanceCriterionText: z.string(),
  status: z.enum(['verified', 'failed', 'unverified', 'skipped']),
  results: z.array(sessionAcceptanceAssertionResultSchema),
  evidenceIds: z.array(z.string()),
  reason: z.string(),
})

export const sessionAcceptanceVerificationSchema = z.object({
  schemaVersion: z.literal(1),
  status: z.enum(['empty', 'passed', 'failed', 'unverified', 'skipped']),
  revision: z.string(),
  fingerprint: z.string(),
  eventCount: z.number().int().nonnegative(),
  summary: z.object({
    acceptanceCriteria: z.number().int().nonnegative(),
    assertions: z.number().int().nonnegative(),
    verifiedAssertions: z.number().int().nonnegative(),
    failedAssertions: z.number().int().nonnegative(),
    unverifiedAssertions: z.number().int().nonnegative(),
    skippedAssertions: z.number().int().nonnegative(),
    constantAssertions: z.number().int().nonnegative(),
    structuralAssertions: z.number().int().nonnegative(),
    behavioralAssertions: z.number().int().nonnegative(),
    subjectiveAssertions: z.number().int().nonnegative(),
  }),
  reports: z.array(sessionAcceptanceVerificationReportSchema),
  lastUpdatedAt: z.string(),
})

export const sessionEvaluationGateRiskSchema = z.object({
  code: z.enum([
    'execution_complete_without_evaluation',
    'mechanical_validation_missing',
    'mechanical_validation_failed',
    'semantic_acceptance_unverified',
    'semantic_acceptance_failed',
    'pending_tool_results',
    'consensus_required',
  ]),
  severity: z.enum(['info', 'warning']),
  message: z.string(),
  evidenceIds: z.array(z.string()).optional(),
})

export const sessionEvaluationGateSchema = z.object({
  schemaVersion: z.literal(1),
  status: z.enum(['empty', 'not_started', 'running', 'passed', 'failed', 'blocked', 'unverified']),
  revision: z.string(),
  fingerprint: z.string(),
  eventCount: z.number().int().nonnegative(),
  lastEventId: z.string().optional(),
  lastEventAt: z.string(),
  stages: z.object({
    mechanical: sessionEvaluationStageSchema,
    semantic: sessionEvaluationStageSchema,
    consensus: sessionEvaluationStageSchema,
  }),
  artifactBundle: sessionEvaluationArtifactBundleSchema,
  acceptanceVerification: sessionAcceptanceVerificationSchema,
  consensusTriggers: sessionConsensusTriggerMatrixSchema,
  signals: z.object({
    executionComplete: z.boolean(),
    runContractPresent: z.boolean(),
    fileChanges: z.number().int().nonnegative(),
    validationRuns: z.number().int().nonnegative(),
    validationFailures: z.number().int().nonnegative(),
    acceptanceCriteria: z.number().int().nonnegative(),
    supportedAcceptanceCriteria: z.number().int().nonnegative(),
    failedAcceptanceCriteria: z.number().int().nonnegative(),
    blockedAcceptanceCriteria: z.number().int().nonnegative(),
    unverifiedAcceptanceCriteria: z.number().int().nonnegative(),
    pendingToolResults: z.number().int().nonnegative(),
    assistantMarkedVerified: z.boolean(),
    assistantMarkedUnverified: z.boolean(),
    consensusRequired: z.boolean(),
  }),
  verdict: z.object({
    approved: z.boolean(),
    reason: z.string(),
  }),
  risks: z.array(sessionEvaluationGateRiskSchema),
  lastUpdatedAt: z.string(),
})

export const sessionRunbookSignalSchema = z.object({
  id: z.string(),
  severity: z.enum(['info', 'warning', 'critical']),
  title: z.string(),
  detail: z.string(),
  source: z.enum(['evidence', 'runtime', 'history', 'checklist', 'session']),
  evidenceIds: z.array(z.string()).optional(),
})

export const sessionRunbookActionSchema = z.object({
  id: z.string(),
  priority: z.enum(['now', 'next', 'optional']),
  title: z.string(),
  command: z.string(),
  description: z.string(),
  destructive: z.boolean(),
  requiresReview: z.boolean(),
})

export const sessionRunbookMechanicalValidationCommandSchema = z.object({
  kind: z.enum(['lint', 'typecheck', 'build', 'test', 'coverage', 'static']),
  label: z.string(),
  command: z.string(),
  reason: z.string(),
})

export const sessionRunbookMechanicalValidationSchema = z.object({
  status: z.enum([
    'not_needed',
    'already_validated',
    'ready',
    'missing_project_context',
    'unsupported_project',
  ]),
  reason: z.string(),
  cwd: z.string().optional(),
  toolchain: z.enum([
    'node-pnpm-turbo',
    'node-pnpm',
    'node-npm',
    'node-yarn',
    'node-bun',
    'python-uv',
    'python',
    'rust',
    'go',
    'zig',
  ]).optional(),
  commands: z.array(sessionRunbookMechanicalValidationCommandSchema),
})

export const sessionRunbookSchema = z.object({
  schemaVersion: z.literal(1),
  generatedAt: z.string(),
  status: z.enum(['ready', 'needs_attention', 'blocked', 'recoverable']),
  headline: z.string(),
  session: z.object({
    id: z.string(),
    title: z.string(),
    status: z.enum(['active', 'completed', 'abandoned']),
    provider: z.string(),
    model: z.string(),
    createdAt: z.string(),
    updatedAt: z.string(),
    eventCount: z.number().int().nonnegative(),
  }),
  summary: z.object({
    signals: z.number().int().nonnegative(),
    criticalSignals: z.number().int().nonnegative(),
    warningSignals: z.number().int().nonnegative(),
    pendingApprovals: z.number().int().nonnegative(),
    pendingQuestions: z.number().int().nonnegative(),
    evidenceRisks: z.number().int().nonnegative(),
    validationRuns: z.number().int().nonnegative(),
    validationFailures: z.number().int().nonnegative(),
    filesChanged: z.number().int().nonnegative(),
    acceptanceCriteria: z.number().int().nonnegative(),
    supportedAcceptanceCriteria: z.number().int().nonnegative(),
    failedAcceptanceCriteria: z.number().int().nonnegative(),
    blockedAcceptanceCriteria: z.number().int().nonnegative(),
    unverifiedAcceptanceCriteria: z.number().int().nonnegative(),
    validationSuggestions: z.number().int().nonnegative(),
  }),
  signals: z.array(sessionRunbookSignalSchema),
  actions: z.array(sessionRunbookActionSchema),
  mechanicalValidation: sessionRunbookMechanicalValidationSchema,
  evidence: z.object({
    manifestRevision: z.string().optional(),
    manifestStatus: z.enum(['empty', 'collecting', 'ready', 'attention_needed']).optional(),
    acceptanceCriteria: z.array(sessionAcceptanceEvidenceSchema),
    risks: z.array(sessionEvidenceRiskSchema),
    artifacts: z.array(sessionEvidenceArtifactSchema),
  }),
  supportBundle: z.object({
    recommended: z.boolean(),
    command: z.string(),
    reason: z.string(),
  }),
})

export const sessionHistoryManagementRiskSchema = z.object({
  code: z.enum([
    'semantic_index_unavailable',
    'semantic_recall_not_observed',
    'compaction_not_observed',
    'compaction_stale',
    'memory_summary_not_observed',
    'memory_summary_stale',
    'dreaming_provider_missing',
    'memory_lifecycle_attention',
  ]),
  severity: z.enum(['info', 'warning']),
  message: z.string(),
})

export const sessionContextEngineSchema = z.object({
  schemaVersion: z.literal(1),
  status: z.enum(['empty', 'warming', 'versioned']),
  revision: z.string(),
  fingerprint: z.string(),
  eventCount: z.number().int().nonnegative(),
  lastEventId: z.string().optional(),
  lastEventAt: z.string(),
  sources: z.object({
    memoryContext: z.object({
      events: z.number().int().nonnegative(),
      items: z.number().int().nonnegative(),
      lastAt: z.string().optional(),
    }),
    compaction: z.object({
      events: z.number().int().nonnegative(),
      tokensSaved: z.number().int().nonnegative(),
      lastAt: z.string().optional(),
    }),
    memorySummary: z.object({
      events: z.number().int().nonnegative(),
      semanticExtractions: z.number().int().nonnegative(),
      ragPromotions: z.number().int().nonnegative(),
      lastAt: z.string().optional(),
    }),
    workingMemory: z.object({
      decisions: z.number().int().nonnegative(),
      fileChanges: z.number().int().nonnegative(),
      openQuestions: z.number().int().nonnegative(),
      lastUpdatedAt: z.string().optional(),
    }),
    runContract: z.object({
      present: z.boolean(),
      source: z.enum(['planner', 'fallback']).optional(),
      acceptanceCriteria: z.number().int().nonnegative(),
    }),
  }),
})

export const sessionHistoryManagementSchema = z.object({
  status: z.enum(['empty', 'warming', 'managed', 'attention_needed']),
  compact: z.object({
    compactions: z.number().int().nonnegative(),
    tokensBefore: z.number().int().nonnegative(),
    tokensAfter: z.number().int().nonnegative(),
    tokensSaved: z.number().int().nonnegative(),
    lastCompactedAt: z.string().optional(),
    assistantMessagesSinceLastCompaction: z.number().int().nonnegative(),
    lastStrategy: z.enum(['preserve_tail', 'summary_only']).optional(),
    lastRemovedMessageCount: z.number().int().nonnegative().optional(),
    lastPreservedMessageCount: z.number().int().nonnegative().optional(),
  }),
  semanticRecall: z.object({
    contextEvents: z.number().int().nonnegative(),
    contextItems: z.number().int().nonnegative(),
    memoryItems: z.number().int().nonnegative(),
    documentItems: z.number().int().nonnegative(),
    lastContextAt: z.string().optional(),
    lastMemoryItemAt: z.string().optional(),
    lastDocumentContextAt: z.string().optional(),
  }),
  memorySummary: z.object({
    events: z.number().int().nonnegative(),
    lightCaptures: z.number().int().nonnegative(),
    semanticExtractions: z.number().int().nonnegative(),
    ragPromotions: z.number().int().nonnegative(),
    lastSummarizedAt: z.string().optional(),
    assistantMessagesSinceLastSummary: z.number().int().nonnegative(),
    sources: z.record(z.number().int().nonnegative()),
  }),
  runtime: z.object({
    semanticIndex: z.object({
      status: z.string(),
      pendingCount: z.number().int().nonnegative().optional(),
      failedCount: z.number().int().nonnegative().optional(),
      vecAvailable: z.boolean().optional(),
      backendAvailable: z.boolean().optional(),
      vectorBackend: z.string().optional(),
      lastError: z.string().optional(),
    }).optional(),
    dreaming: z.object({
      enabled: z.boolean(),
      running: z.boolean(),
      providerConfigured: z.boolean(),
      model: z.string().optional(),
      fileMemoryEnabled: z.boolean(),
    }).optional(),
    memoryLifecycle: z.object({
      totalMemories: z.number().int().nonnegative(),
      staleConversationMemories: z.number().int().nonnegative(),
      lowImportanceConversationMemories: z.number().int().nonnegative(),
      pruneCandidateMemories: z.number().int().nonnegative(),
      pendingEmbeddings: z.number().int().nonnegative(),
      failedEmbeddings: z.number().int().nonnegative(),
      lastAuditAt: z.string().optional(),
    }).optional(),
  }),
  risks: z.array(sessionHistoryManagementRiskSchema),
  lastUpdatedAt: z.string(),
})

export const sessionBranchRequestSchema = z.object({
  fromEventIndex: z.number().int().optional(),
})

export const sessionCreateRequestSchema = z.object({
  title: z.string().trim().min(1).max(200).optional(),
  /** Absolute, existing workspace root copied from the Desktop new-session default. */
  cwd: z.string().trim().min(1).optional(),
})

export const sessionTranscriptTurnRequestSchema = z.object({
  userContent: z.string().trim().min(1),
  assistantContent: z.string().optional(),
  title: z.string().trim().min(1).max(200).optional(),
  provider: z.string().trim().min(1).max(80).optional(),
  model: z.string().trim().min(1).max(120).optional(),
  cwd: z.string().trim().min(1).optional(),
  tags: z.array(z.string().trim().min(1).max(64)).max(16).optional(),
})

export const sessionLocalShellTurnRequestSchema = z.object({
  command: z.string().trim().min(1),
  cwd: z.string().trim().min(1).optional(),
  shell: z.string().trim().min(1).optional(),
  args: z.array(z.string()).optional(),
  stdout: z.string().optional(),
  stderr: z.string().optional(),
  exitCode: z.number().int(),
  signal: z.string().nullable().optional(),
  durationMs: z.number().int().nonnegative(),
  timedOut: z.boolean().optional(),
  maxBufferExceeded: z.boolean().optional(),
  title: z.string().trim().min(1).max(200).optional(),
  provider: z.string().trim().min(1).max(80).optional(),
  model: z.string().trim().min(1).max(120).optional(),
  tags: z.array(z.string().trim().min(1).max(64)).max(16).optional(),
})

export const sessionUpdateRequestSchema = z.object({
  title: z.string().trim().min(1).max(200).optional(),
  status: z.enum(['active', 'completed', 'abandoned']).optional(),
  /** Absolute, existing workspace root bound to this session. */
  cwd: z.string().trim().min(1).nullable().optional(),
  /** Explicit per-session execution boundary; never inferred from a prompt. */
  workspaceIsolation: z.enum(['policy', 'strict']).optional(),
  /** Ordered persona roster; an explicit empty array restores the default assistant. */
  personaIds: sessionPersonaIdsSchema.optional(),
  starred: z.boolean().optional(),
  tags: z.array(z.string().trim().min(1).max(64)).max(16).optional(),
}).refine(
  (value) => Object.keys(value).length > 0,
  { message: 'At least one updatable field is required' },
)

export const sessionsListQuerySchema = z.object({
  page: z.coerce.number().int().min(1).optional(),
  perPage: z.coerce.number().int().min(1).optional(),
  query: z.string().optional(),
  /** Exact immutable workspace binding used by workspace-scoped resume UIs. */
  workspaceRoot: z.string().trim().min(1).optional(),
  /**
   * Opt-in approval counters per item — costs an extra getEvents() per
   * session in the page, so default off for backward compat. cli
   * `sessions list` always passes true so the audit column shows up;
   * web/desktop list views can stay light by leaving it false.
   */
  metrics: z.coerce.boolean().optional(),
  /**
   * Narrow the page to a single status. Filter applies after the
   * underlying sessions.list() returns, so totalCount reflects the
   * filtered set — operators eyeballing list output expect "showing 5
   * of 5 active" not "showing 5 of 247 (filtered from 200 idle)".
   */
  status: z.enum(['active', 'completed', 'abandoned']).optional(),
})

export const sessionExportQuerySchema = z.object({
  format: z.enum(['markdown', 'json']).optional(),
  sanitize: z.enum(['0', '1']).optional(),
})

export const sessionIdParamsSchema = z.object({
  id: safeIdSchema,
})

export const sessionResumeRequestSchema = z.object({
  force: z.boolean().optional(),
})

export const sessionsOpenApiZodComponents = openApiComponentsFromZod({
  schemas: {
    SessionTotals: sessionTotalsSchema,
    SessionMeta: sessionMetaSchema,
    SessionEvent: sessionEventSchema,
    PendingApproval: pendingApprovalSchema,
    ResumableRun: resumableRunSchema,
    ResumableRunIssue: resumableRunIssueSchema,
    SessionDelegation: sessionDelegationSchema,
    SessionTraceMetrics: sessionTraceMetricsSchema,
    SessionContextEngine: sessionContextEngineSchema,
    SessionChecklistItem: sessionChecklistItemSchema,
    SessionCompletionChecklist: sessionCompletionChecklistSchema,
    SessionWorkingMemoryDecision: sessionWorkingMemoryDecisionSchema,
    SessionWorkingMemoryToolOutcome: sessionWorkingMemoryToolOutcomeSchema,
    SessionWorkingMemoryFileChange: sessionWorkingMemoryFileChangeSchema,
    SessionWorkingMemory: sessionWorkingMemorySchema,
    SessionRunContract: sessionRunContractSchema,
    SessionEvidenceArtifact: sessionEvidenceArtifactSchema,
    SessionAcceptanceEvidence: sessionAcceptanceEvidenceSchema,
    SessionEvidenceRisk: sessionEvidenceRiskSchema,
    SessionContractLedgerEntry: sessionContractLedgerEntrySchema,
    SessionContractLedgerSection: sessionContractLedgerSectionSchema,
    SessionContractLedgerBlocker: sessionContractLedgerBlockerSchema,
    SessionContractLedger: sessionContractLedgerSchema,
    SessionEvidenceManifest: sessionEvidenceManifestSchema,
    SessionEvaluationStage: sessionEvaluationStageSchema,
    SessionEvaluationArtifactFile: sessionEvaluationArtifactFileSchema,
    SessionEvaluationArtifactSkip: sessionEvaluationArtifactSkipSchema,
    SessionEvaluationArtifactBundle: sessionEvaluationArtifactBundleSchema,
    SessionConsensusTrigger: sessionConsensusTriggerSchema,
    SessionConsensusTriggerMatrix: sessionConsensusTriggerMatrixSchema,
    SessionAcceptanceAssertion: sessionAcceptanceAssertionSchema,
    SessionAcceptanceAssertionResult: sessionAcceptanceAssertionResultSchema,
    SessionAcceptanceVerificationReport: sessionAcceptanceVerificationReportSchema,
    SessionAcceptanceVerification: sessionAcceptanceVerificationSchema,
    SessionEvaluationGateRisk: sessionEvaluationGateRiskSchema,
    SessionEvaluationGate: sessionEvaluationGateSchema,
    SessionRunbookSignal: sessionRunbookSignalSchema,
    SessionRunbookAction: sessionRunbookActionSchema,
    SessionRunbookMechanicalValidationCommand: sessionRunbookMechanicalValidationCommandSchema,
    SessionRunbookMechanicalValidation: sessionRunbookMechanicalValidationSchema,
    SessionRunbook: sessionRunbookSchema,
    SessionHistoryManagementRisk: sessionHistoryManagementRiskSchema,
    SessionHistoryManagement: sessionHistoryManagementSchema,
    SessionBranchRequest: sessionBranchRequestSchema,
    SessionCreateRequest: sessionCreateRequestSchema,
    SessionTranscriptTurnRequest: sessionTranscriptTurnRequestSchema,
    SessionLocalShellTurnRequest: sessionLocalShellTurnRequestSchema,
    SessionUpdateRequest: sessionUpdateRequestSchema,
    SessionResumeRequest: sessionResumeRequestSchema,
  },
  parameters: {
    SessionsPageParam: {
      name: 'page',
      in: 'query',
      schema: z.number().int(),
    },
    SessionsPerPageParam: {
      name: 'perPage',
      in: 'query',
      schema: z.number().int(),
    },
    SessionsQueryParam: {
      name: 'query',
      in: 'query',
      schema: z.string(),
    },
    SessionsWorkspaceRootParam: {
      name: 'workspaceRoot',
      in: 'query',
      schema: z.string(),
    },
    SessionExportFormatParam: {
      name: 'format',
      in: 'query',
      schema: z.enum(['markdown', 'json']),
    },
    SessionIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: sessionIdParamsSchema.shape.id,
    },
  },
})

export type SessionBranchBody = z.infer<typeof sessionBranchRequestSchema>
export type SessionCreateBody = z.infer<typeof sessionCreateRequestSchema>
export type SessionTranscriptTurnBody = z.infer<typeof sessionTranscriptTurnRequestSchema>
export type SessionLocalShellTurnBody = z.infer<typeof sessionLocalShellTurnRequestSchema>
export type SessionUpdateBody = z.infer<typeof sessionUpdateRequestSchema>
export type SessionsListQuery = z.infer<typeof sessionsListQuerySchema>
export type SessionExportQuery = z.infer<typeof sessionExportQuerySchema>
export type SessionIdParams = z.infer<typeof sessionIdParamsSchema>
export type SessionResumeBody = z.infer<typeof sessionResumeRequestSchema>

export const sessionBranchRequestInputSchema = sessionBranchRequestSchema
  .optional()
  .transform((value): SessionBranchBody => value ?? {})

export const sessionCreateRequestInputSchema = sessionCreateRequestSchema
  .optional()
  .transform((value): SessionCreateBody => value ?? {})

export const sessionResumeRequestInputSchema = sessionResumeRequestSchema
  .optional()
  .transform((value): SessionResumeBody => value ?? {})
