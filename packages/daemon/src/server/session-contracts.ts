import { createHash } from 'node:crypto'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import type { Dirent } from 'node:fs'
import { basename, isAbsolute, normalize, relative, resolve } from 'node:path'
import type {
  AgentRunContract,
  EditCheckpointSummary,
  SessionEvent,
  SessionEvidenceArtifactKind as CoreSessionEvidenceArtifactKind,
  SessionMeta,
  TodoStatus,
} from '@sepilotd/core'
import {
  terminalRunHasReliableSuccessStatus,
  toolProvidesContentReadEvidence,
  toolProvidesSearchEvidence,
  toolCallRepresentsProductMutation,
} from '../agent/graph/evidence-ledger.js'
import {
  evidenceRequirementMinSourceFiles,
  evidenceRequirementMinSourceObservations,
  evidenceRequirementMinSourceScopes,
  evidenceRequirementRequiresSearch,
  evidenceRequirementUsesRepositoryBreadth,
  sourceToolMatchesEvidenceRequirement,
} from '../agent/evidence-requirement-policy.js'
import {
  TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY,
  TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY,
} from '../agent/policy-failure.js'
import { isExecutorConfirmedReadOnlyObservation } from '../agent/read-only-observation-evidence.js'
import { isExecutorConfirmedExternalActionReceipt } from '../agent/external-action-receipt-evidence.js'
import { inputPositiveCapabilityScope, isDocumentArtifactPath } from '../agent/task-contract.js'
import { sanitizeText } from '../sessions/sanitize.js'

export type ChecklistItemStatus =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'blocked'
  | 'skipped'

export type ChecklistItemSource =
  | 'todo'
  | 'tool'
  | 'approval'
  | 'question'
  | 'session'

export interface ChecklistItem {
  label: string
  status: ChecklistItemStatus
  detail?: string
  source?: ChecklistItemSource
}

export interface PendingQuestionLike {
  id: string
  prompt: string
  choices?: string[]
}

export interface PendingApprovalLike {
  requestId: string
  state: 'live' | 'stale'
  resumeAvailable?: boolean
}

export interface SessionHistorySemanticRuntimeStatus {
  status: string
  pendingCount?: number
  failedCount?: number
  vecAvailable?: boolean
  backendAvailable?: boolean
  vectorBackend?: string
  lastError?: string
}

export interface SessionHistoryDreamingRuntimeStatus {
  enabled: boolean
  running: boolean
  providerConfigured: boolean
  model?: string
  fileMemoryEnabled: boolean
}

export interface SessionHistoryMemoryLifecycleRuntimeStatus {
  totalMemories: number
  staleConversationMemories: number
  lowImportanceConversationMemories: number
  pruneCandidateMemories: number
  pendingEmbeddings: number
  failedEmbeddings: number
  lastAuditAt?: string
}

export interface SessionHistoryManagementRuntime {
  semanticIndex?: SessionHistorySemanticRuntimeStatus
  dreaming?: SessionHistoryDreamingRuntimeStatus
  memoryLifecycle?: SessionHistoryMemoryLifecycleRuntimeStatus
}

export type SessionHistoryManagementStatus =
  | 'empty'
  | 'warming'
  | 'managed'
  | 'attention_needed'

export interface SessionHistoryManagementRisk {
  code:
    | 'semantic_index_unavailable'
    | 'semantic_recall_not_observed'
    | 'compaction_not_observed'
    | 'compaction_stale'
    | 'memory_summary_not_observed'
    | 'memory_summary_stale'
    | 'dreaming_provider_missing'
    | 'memory_lifecycle_attention'
  severity: 'info' | 'warning'
  message: string
}

export type SessionContextEngineStatus = 'empty' | 'warming' | 'versioned'

export interface SessionContextEngine {
  schemaVersion: 1
  status: SessionContextEngineStatus
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  sources: {
    memoryContext: {
      events: number
      items: number
      lastAt?: string
    }
    compaction: {
      events: number
      tokensSaved: number
      lastAt?: string
    }
    memorySummary: {
      events: number
      semanticExtractions: number
      ragPromotions: number
      lastAt?: string
    }
    workingMemory: {
      decisions: number
      fileChanges: number
      openQuestions: number
      lastUpdatedAt?: string
    }
    runContract: {
      present: boolean
      source?: AgentRunContract['source']
      acceptanceCriteria: number
    }
  }
}

export type SessionContractLedgerStatus =
  | 'empty'
  | 'ready'
  | 'needs_attention'
  | 'blocked'

export type SessionContractLedgerSectionName =
  | 'goal'
  | 'scope'
  | 'acceptance_criteria'
  | 'deliverables'
  | 'evidence_requirements'
  | 'verification_plan'
  | 'runtime_context'
  | 'blockers'

export type SessionContractLedgerEntrySource =
  | 'user_goal'
  | 'run_contract'
  | 'session_runtime'
  | 'repo_fact'
  | 'evidence'
  | 'conservative_default'
  | 'blocker'

export type SessionContractLedgerEntryStatus =
  | 'missing'
  | 'weak'
  | 'defaulted'
  | 'inferred'
  | 'confirmed'
  | 'blocked'

export type SessionContractLedgerBlockerCode =
  | 'credential_or_secret'
  | 'destructive_production_action'
  | 'external_side_effect'
  | 'billing_authority'
  | 'legal_or_medical_judgment'

export interface SessionContractLedgerEntry {
  id: string
  key: string
  value: string
  source: SessionContractLedgerEntrySource
  status: SessionContractLedgerEntryStatus
  confidence: number
  reversible: boolean
  rationale: string
  eventIds?: string[]
  evidenceIds?: string[]
}

export interface SessionContractLedgerSection {
  name: SessionContractLedgerSectionName
  status: SessionContractLedgerEntryStatus
  summary: string
  entries: SessionContractLedgerEntry[]
}

export interface SessionContractLedgerBlocker {
  code: SessionContractLedgerBlockerCode
  severity: 'warning' | 'blocker'
  message: string
  eventIds: string[]
}

export interface SessionContractLedger {
  schemaVersion: 1
  status: SessionContractLedgerStatus
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  summary: {
    sections: number
    confirmedSections: number
    defaultedSections: number
    inferredSections: number
    missingSections: number
    blockedSections: number
    blockers: number
    safeDefaults: number
  }
  sections: SessionContractLedgerSection[]
  blockers: SessionContractLedgerBlocker[]
  lastUpdatedAt: string
}

export type SessionEvidenceManifestStatus =
  | 'empty'
  | 'collecting'
  | 'ready'
  | 'attention_needed'

export type SessionEvidenceArtifactKind = CoreSessionEvidenceArtifactKind

export type SessionEvidenceArtifactStatus =
  | 'pending'
  | 'success'
  | 'error'
  | 'warning'
  | 'info'

export interface SessionEvidenceArtifact {
  id: string
  kind: SessionEvidenceArtifactKind
  label: string
  summary: string
  status: SessionEvidenceArtifactStatus
  timestamp: string
  sourceEventIds: string[]
  toolCallId?: string
  tool?: string
  path?: string
  hash?: string
  /** Validation result belongs to the latest validation-relevant workspace revision. */
  currentRevision?: boolean
  relatedAcceptanceCriteriaIds?: string[]
}

export interface SessionAcceptanceEvidence {
  id: string
  text: string
  status: 'supported' | 'failed' | 'blocked' | 'unverified'
  evidenceIds: string[]
  reason: string
}

export interface SessionEvidenceRisk {
  code:
    | 'no_run_contract'
    | 'acceptance_criteria_unverified'
    | 'validation_failed'
    | 'changes_without_validation'
    | 'pending_tool_results'
    | 'failed_tool_results'
    | 'post_edit_findings'
    | 'assistant_marked_unverified'
  severity: 'info' | 'warning'
  message: string
  evidenceIds?: string[]
}

export interface SessionEvidenceManifest {
  schemaVersion: 1
  status: SessionEvidenceManifestStatus
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  summary: {
    artifacts: number
    toolCalls: number
    toolResults: number
    validationRuns: number
    validationFailures: number
    filesChanged: number
    approvals: number
    acceptanceCriteria: number
    supportedAcceptanceCriteria: number
    failedAcceptanceCriteria: number
    blockedAcceptanceCriteria: number
    unverifiedAcceptanceCriteria: number
    risks: number
  }
  acceptanceCriteria: SessionAcceptanceEvidence[]
  artifacts: SessionEvidenceArtifact[]
  risks: SessionEvidenceRisk[]
  lastUpdatedAt: string
}

export type SessionEvaluationGateStatus =
  | 'empty'
  | 'not_started'
  | 'running'
  | 'passed'
  | 'failed'
  | 'blocked'
  | 'unverified'

export type SessionEvaluationStageStatus =
  | 'not_started'
  | 'running'
  | 'passed'
  | 'failed'
  | 'blocked'
  | 'skipped'
  | 'unverified'

export interface SessionEvaluationStage {
  status: SessionEvaluationStageStatus
  summary: string
  evidenceIds: string[]
}

export type SessionEvaluationArtifactBundleStatus =
  | 'empty'
  | 'metadata_only'
  | 'ready'
  | 'partial'

export type SessionEvaluationArtifactContentState =
  | 'hashed'
  | 'metadata_only'

export type SessionEvaluationArtifactSkipReason =
  | 'missing_path'
  | 'path_traversal'
  | 'absolute_path_without_cwd'
  | 'outside_cwd'
  | 'generated_or_vendor'
  | 'duplicate_path'
  | 'max_files_exceeded'
  | 'file_missing'
  | 'not_file'
  | 'file_too_large'
  | 'total_budget_exceeded'
  | 'read_error'

export interface SessionEvaluationArtifactFile {
  id: string
  path: string
  artifactId: string
  status: SessionEvidenceArtifactStatus
  operation: string
  contentState: SessionEvaluationArtifactContentState
  sourceEventIds: string[]
  tool?: string
  sizeBytes?: number
  contentHash?: string
}

export interface SessionEvaluationArtifactSkip {
  reason: SessionEvaluationArtifactSkipReason
  message: string
  artifactId?: string
  path?: string
  sourceEventIds?: string[]
}

export interface SessionEvaluationArtifactBundle {
  schemaVersion: 1
  status: SessionEvaluationArtifactBundleStatus
  revision: string
  fingerprint: string
  eventCount: number
  cwd?: string
  limits: {
    maxFiles: number
    maxFileBytes: number
    maxTotalBytes: number
  }
  summary: {
    files: number
    hashedFiles: number
    metadataOnlyFiles: number
    skippedFiles: number
    totalBytes: number
    validationArtifacts: number
    acceptanceCriteria: number
  }
  files: SessionEvaluationArtifactFile[]
  skipped: SessionEvaluationArtifactSkip[]
  validationEvidenceIds: string[]
  acceptanceEvidenceIds: string[]
  lastUpdatedAt: string
}

export type SessionConsensusTriggerCode =
  | 'validation_failed'
  | 'post_edit_findings'
  | 'assistant_marked_unverified'
  | 'mechanical_validation_missing'
  | 'semantic_acceptance_unverified'
  | 'artifact_bundle_partial'
  | 'large_change_set'
  | 'missing_run_contract'

export type SessionAcceptanceAssertionTier =
  | 'constant'
  | 'structural'
  | 'behavioral'
  | 'subjective'

export type SessionAcceptanceAssertionKind =
  | 'file_exists'
  | 'symbol_exists'
  | 'text_match'
  | 'validation_required'
  | 'human_review'

export type SessionAcceptanceAssertionStatus =
  | 'verified'
  | 'failed'
  | 'unverified'
  | 'skipped'

export type SessionAcceptanceVerificationStatus =
  | 'empty'
  | 'passed'
  | 'failed'
  | 'unverified'
  | 'skipped'

export interface SessionAcceptanceAssertion {
  id: string
  acceptanceCriterionId: string
  tier: SessionAcceptanceAssertionTier
  kind: SessionAcceptanceAssertionKind
  description: string
  pattern?: string
  expectedValue?: string
  fileHint?: string
  confidence: number
}

export interface SessionAcceptanceAssertionResult {
  assertion: SessionAcceptanceAssertion
  status: SessionAcceptanceAssertionStatus
  detail: string
  evidenceIds: string[]
  path?: string
  actualValue?: string
}

export interface SessionAcceptanceVerificationReport {
  acceptanceCriterionId: string
  acceptanceCriterionText: string
  status: SessionAcceptanceAssertionStatus
  results: SessionAcceptanceAssertionResult[]
  evidenceIds: string[]
  reason: string
}

export interface SessionAcceptanceVerification {
  schemaVersion: 1
  status: SessionAcceptanceVerificationStatus
  revision: string
  fingerprint: string
  eventCount: number
  summary: {
    acceptanceCriteria: number
    assertions: number
    verifiedAssertions: number
    failedAssertions: number
    unverifiedAssertions: number
    skippedAssertions: number
    constantAssertions: number
    structuralAssertions: number
    behavioralAssertions: number
    subjectiveAssertions: number
  }
  reports: SessionAcceptanceVerificationReport[]
  lastUpdatedAt: string
}

export interface SessionConsensusTrigger {
  code: SessionConsensusTriggerCode
  priority: number
  fired: boolean
  severity: 'info' | 'warning'
  message: string
  evidenceIds?: string[]
}

export interface SessionConsensusTriggerMatrix {
  required: boolean
  primaryTrigger?: SessionConsensusTrigger
  triggers: SessionConsensusTrigger[]
}

export interface SessionEvaluationGateRisk {
  code:
    | 'execution_complete_without_evaluation'
    | 'mechanical_validation_missing'
    | 'mechanical_validation_failed'
    | 'semantic_acceptance_unverified'
    | 'semantic_acceptance_failed'
    | 'pending_tool_results'
    | 'consensus_required'
  severity: 'info' | 'warning'
  message: string
  evidenceIds?: string[]
}

export interface SessionEvaluationGate {
  schemaVersion: 1
  status: SessionEvaluationGateStatus
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  stages: {
    mechanical: SessionEvaluationStage
    semantic: SessionEvaluationStage
    consensus: SessionEvaluationStage
  }
  artifactBundle: SessionEvaluationArtifactBundle
  acceptanceVerification: SessionAcceptanceVerification
  consensusTriggers: SessionConsensusTriggerMatrix
  signals: {
    executionComplete: boolean
    runContractPresent: boolean
    fileChanges: number
    validationRuns: number
    validationFailures: number
    acceptanceCriteria: number
    supportedAcceptanceCriteria: number
    failedAcceptanceCriteria: number
    blockedAcceptanceCriteria: number
    unverifiedAcceptanceCriteria: number
    pendingToolResults: number
    assistantMarkedVerified: boolean
    assistantMarkedUnverified: boolean
    consensusRequired: boolean
  }
  verdict: {
    approved: boolean
    reason: string
  }
  risks: SessionEvaluationGateRisk[]
  lastUpdatedAt: string
}

function truncateText(value: string, max = 240): string {
  const trimmed = value.trim()
  return trimmed.length > max
    ? `${trimmed.slice(0, max - 1)}…`
    : trimmed
}

function evidenceSnippet(value: string, max = 180): string {
  return truncateText(sanitizeText(value, { home: process.env.HOME ?? '' }), max)
}

function parseTimestamp(value: string | undefined): number | null {
  if (!value) {
    return null
  }
  const timestamp = Date.parse(value)
  return Number.isFinite(timestamp) ? timestamp : null
}

function diffMs(
  startTimestamp: string | undefined,
  endTimestamp: string | undefined,
): number | undefined {
  const start = parseTimestamp(startTimestamp)
  const end = parseTimestamp(endTimestamp)
  if (start === null || end === null || end < start) {
    return undefined
  }
  return end - start
}

function latestTimestamp(
  session: SessionMeta,
  events: SessionEvent[],
): string {
  return events.at(-1)?.timestamp ?? session.updatedAt
}

function stableFingerprint(payload: unknown): string {
  return createHash('sha256')
    .update(JSON.stringify(payload))
    .digest('hex')
    .slice(0, 16)
}

function stableDigest(value: string | Buffer): string {
  return createHash('sha256')
    .update(value)
    .digest('hex')
    .slice(0, 16)
}

const EVALUATION_BUNDLE_MAX_FILES = 30
const EVALUATION_BUNDLE_MAX_FILE_BYTES = 50 * 1024
const EVALUATION_BUNDLE_MAX_TOTAL_BYTES = 150_000

const GENERATED_OR_VENDOR_PATH_SEGMENTS = new Set([
  '.cache',
  '.git',
  '.next',
  '.nuxt',
  '.turbo',
  '.venv',
  '__pycache__',
  'build',
  'coverage',
  'dist',
  'node_modules',
  'out',
  'target',
  'venv',
])

function normalizeArtifactPath(value: string): string {
  return value.replaceAll('\\', '/').replace(/^\.\//, '')
}

function hasPathTraversal(value: string): boolean {
  return normalizeArtifactPath(value).split('/').some((segment) => segment === '..')
}

function isGeneratedOrVendorPath(path: string): boolean {
  const normalizedPath = normalizeArtifactPath(path)
  const segments = normalizedPath.split('/').filter(Boolean)
  if (segments.some((segment) => GENERATED_OR_VENDOR_PATH_SEGMENTS.has(segment))) {
    return true
  }
  const filename = segments.at(-1) ?? normalizedPath
  return filename.endsWith('.map')
    || filename.endsWith('.min.css')
    || filename.endsWith('.min.js')
}

function artifactOperation(artifact: SessionEvidenceArtifact): string {
  const operation = artifact.summary.match(/^(write|edit|patch|unknown)\b/)?.[1]
  return operation ?? 'unknown'
}

function artifactSkip(
  reason: SessionEvaluationArtifactSkipReason,
  message: string,
  artifact?: SessionEvidenceArtifact,
  path: string | undefined = artifact?.path,
): SessionEvaluationArtifactSkip {
  return {
    reason,
    message,
    artifactId: artifact?.id,
    path,
    sourceEventIds: artifact?.sourceEventIds,
  }
}

function safeEvaluationArtifactPath(
  rawPath: string | undefined,
  cwd: string | undefined,
): {
  path?: string
  absolutePath?: string
  skip?: SessionEvaluationArtifactSkip
} {
  const trimmedPath = rawPath?.trim()
  if (!trimmedPath) {
    return {
      skip: artifactSkip('missing_path', 'File-change evidence did not include a path.'),
    }
  }
  if (trimmedPath.includes('\0') || hasPathTraversal(trimmedPath)) {
    return {
      skip: artifactSkip(
        'path_traversal',
        'File-change path contains traversal or null-byte segments.',
        undefined,
        trimmedPath,
      ),
    }
  }
  if (!cwd && isAbsolute(trimmedPath)) {
    return {
      skip: artifactSkip(
        'absolute_path_without_cwd',
        'Absolute file-change paths require a session cwd before they can be evaluated.',
        undefined,
        trimmedPath,
      ),
    }
  }

  const root = cwd ? resolve(cwd) : undefined
  const absolutePath = root
    ? isAbsolute(trimmedPath)
      ? resolve(trimmedPath)
      : resolve(root, trimmedPath)
    : undefined
  const path = root && absolutePath
    ? normalizeArtifactPath(relative(root, absolutePath))
    : normalizeArtifactPath(normalize(trimmedPath))

  if (!path || path === '.') {
    return {
      skip: artifactSkip('missing_path', 'File-change evidence resolved to an empty path.', undefined, trimmedPath),
    }
  }
  if (root && absolutePath) {
    const relativePath = relative(root, absolutePath)
    const relativePathSegments = normalizeArtifactPath(relativePath).split('/')
    if (relativePathSegments[0] === '..' || isAbsolute(relativePath)) {
      return {
        skip: artifactSkip(
          'outside_cwd',
          'File-change path resolves outside the session cwd.',
          undefined,
          trimmedPath,
        ),
      }
    }
  }
  if (isGeneratedOrVendorPath(path)) {
    return {
      skip: artifactSkip(
        'generated_or_vendor',
        'Generated, dependency, or vendor paths are excluded from evaluation bundles.',
        undefined,
        path,
      ),
    }
  }

  return { path, absolutePath }
}

function checklistStatusFromTodoStatus(value: TodoStatus): ChecklistItemStatus {
  if (value === 'completed') {
    return 'completed'
  }
  if (value === 'blocked') {
    return 'blocked'
  }
  if (value === 'cancelled') {
    return 'skipped'
  }
  if (value === 'in_progress') {
    return 'in_progress'
  }
  return 'pending'
}

function extractApplyPatchPaths(patch: string): string[] {
  const paths = new Set<string>()
  for (const match of patch.matchAll(/^\*\*\* (?:Add|Update|Delete) File: (.+)$/gm)) {
    const path = match[1]?.trim()
    if (path) {
      paths.add(path)
    }
  }
  return [...paths]
}

function extractToolCallPaths(
  event: Extract<SessionEvent, { type: 'tool_call' }>,
): Array<{ path: string; kind: 'write' | 'edit' | 'patch' | 'unknown' }> {
  const inputPath = typeof event.input.path === 'string' ? event.input.path : null
  const inputFile = typeof event.input.file === 'string' ? event.input.file : null
  const directPath = inputPath ?? inputFile
  if (directPath) {
    if (event.tool === 'fs.write') {
      return [{ path: directPath, kind: 'write' }]
    }
    if (event.tool === 'fs.append') {
      return [{ path: directPath, kind: 'write' }]
    }
    if (event.tool === 'fs.edit') {
      return [{ path: directPath, kind: 'edit' }]
    }
    if (event.tool === 'apply_patch') {
      return [{ path: directPath, kind: 'patch' }]
    }
    if (
      event.tool === 'fs.read'
      || event.tool === 'fs.glob'
      || event.tool === 'fs.search'
      || event.tool.startsWith('git.')
      || event.tool.startsWith('code.')
    ) {
      return []
    }
    return [{ path: directPath, kind: 'unknown' }]
  }

  if (event.tool !== 'apply_patch' || typeof event.input.patch !== 'string') {
    return []
  }

  return extractApplyPatchPaths(event.input.patch).map((path) => ({
    path,
    kind: 'patch' as const,
  }))
}

function commandFromToolInput(input: Record<string, unknown>): string | undefined {
  if (typeof input.cmd === 'string') return input.cmd
  if (typeof input.command === 'string') return input.command
  if (Array.isArray(input.command) && input.command.every((item) => typeof item === 'string')) {
    return input.command.join(' ')
  }
  if (typeof input.executable === 'string') {
    const args = Array.isArray(input.args)
      ? input.args.filter((item): item is string => typeof item === 'string')
      : []
    return [input.executable, ...args].join(' ')
  }
  return undefined
}

function splitCommandWords(command: string): string[] {
  const words: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaped = false
  for (const char of command.trim()) {
    if (escaped) {
      current += char
      escaped = false
      continue
    }
    if (char === '\\' && quote !== "'") {
      escaped = true
      continue
    }
    if (quote) {
      if (char === quote) quote = null
      else current += char
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (char === ';' || char === '|' || char === '&') {
      if (current) words.push(current)
      current = ''
      words.push(char)
      continue
    }
    if (/\s/u.test(char)) {
      if (current) words.push(current)
      current = ''
      continue
    }
    current += char
  }
  if (escaped) current += '\\'
  if (current) words.push(current)
  return words
}

function commandArgvFromToolInput(input: Record<string, unknown>): string[] {
  if (typeof input.executable === 'string' && input.executable.trim()) {
    const args = Array.isArray(input.args)
      ? input.args.filter((item): item is string => typeof item === 'string')
      : []
    return [input.executable.trim(), ...args]
  }
  if (Array.isArray(input.command) && input.command.every((item) => typeof item === 'string')) {
    return [...input.command]
  }
  const command = typeof input.cmd === 'string'
    ? input.cmd
    : typeof input.command === 'string'
      ? input.command
      : ''
  return command ? splitCommandWords(command) : []
}

const DIRECT_VALIDATION_EXECUTABLES = new Set([
  'biome',
  'build',
  'check',
  'eslint',
  'jest',
  'lint',
  'mocha',
  'mypy',
  'prettier',
  'pytest',
  'rspec',
  'ruff',
  'test',
  'tsc',
  'typecheck',
  'verify',
  'vitest',
])

const VALIDATION_SCRIPT_NAMES = new Set([
  'build',
  'check',
  'lint',
  'test',
  'typecheck',
  'verify',
])

function normalizedCommandExecutable(value: string): string {
  return basename(value.replace(/\\/g, '/')).toLowerCase().replace(/\.exe$/u, '')
}

function firstNonOptionArg(args: readonly string[]): string | undefined {
  return args.find((arg) => !arg.startsWith('-'))?.toLowerCase()
}

function isValidationArgv(argv: readonly string[], depth = 0): boolean {
  if (argv.length === 0 || depth > 2) return false
  const segments: string[][] = [[]]
  for (const token of argv) {
    if (token === ';' || token === '|' || token === '&') {
      if ((segments.at(-1)?.length ?? 0) > 0) segments.push([])
      continue
    }
    segments.at(-1)?.push(token)
  }
  const commandSegments = segments.filter((segment) => segment.length > 0)
  if (commandSegments.length > 1) {
    return commandSegments.some((segment) => isValidationArgv(segment, depth + 1))
  }
  const executable = normalizedCommandExecutable(argv[0] ?? '')
  const args = argv.slice(1)
  if (DIRECT_VALIDATION_EXECUTABLES.has(executable)) return true

  if (executable === 'bash' || executable === 'sh' || executable === 'zsh') {
    const commandIndex = args.findIndex((arg) => arg === '-c' || arg === '-lc')
    const script = commandIndex >= 0 ? args[commandIndex + 1] : undefined
    return typeof script === 'string'
      && isValidationArgv(splitCommandWords(script), depth + 1)
  }

  if (executable === 'node') {
    return args.some((arg) => arg === '--test' || arg.startsWith('--test='))
  }
  if (executable === 'python' || executable === 'python3' || /^python\d+(?:\.\d+)?$/u.test(executable)) {
    const moduleIndex = args.indexOf('-m')
    return moduleIndex >= 0
      && ['pytest', 'mypy', 'ruff'].includes((args[moduleIndex + 1] ?? '').toLowerCase())
  }
  if (executable === 'go') {
    return ['build', 'test', 'vet'].includes(firstNonOptionArg(args) ?? '')
  }
  if (executable === 'cargo') {
    return ['build', 'check', 'clippy', 'test'].includes(firstNonOptionArg(args) ?? '')
  }
  if (executable === 'dotnet') {
    return ['build', 'test'].includes(firstNonOptionArg(args) ?? '')
  }
  if (executable === 'mvn' || executable === 'mvnw') {
    return args.some((arg) => ['package', 'test', 'verify'].includes(arg.toLowerCase()))
  }
  if (executable === 'gradle' || executable === 'gradlew' || executable === 'make') {
    return args.some((arg) => VALIDATION_SCRIPT_NAMES.has(arg.toLowerCase()))
  }
  if (executable === 'npm' || executable === 'pnpm' || executable === 'yarn' || executable === 'bun') {
    const normalizedArgs = args.map((arg) => arg.toLowerCase())
    const execIndex = normalizedArgs.findIndex((arg) => arg === 'exec' || arg === 'x')
    if (execIndex >= 0 && args[execIndex + 1]) {
      return isValidationArgv(args.slice(execIndex + 1), depth + 1)
    }
    return normalizedArgs.some((arg) => (
      VALIDATION_SCRIPT_NAMES.has(arg)
      || [...VALIDATION_SCRIPT_NAMES].some((name) => arg.startsWith(`${name}:`))
    ))
  }
  return false
}

function isValidationCommand(_tool: string, input: Record<string, unknown>): boolean {
  return input.actionPurpose === 'validate'
    || isValidationArgv(commandArgvFromToolInput(input))
}

function isReadOnlyObservationResult(
  _contract: AgentRunContract | null,
  call: Extract<SessionEvent, { type: 'tool_call' }>,
  result: Extract<SessionEvent, { type: 'tool_result' }>,
): boolean {
  return isExecutorConfirmedReadOnlyObservation({
      tool: call.tool,
      securityEffect:
        typeof result.metadata?.[TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY] === 'string'
          ? result.metadata[TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY]
          : undefined,
      executionObserved:
        result.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
      actionPurpose: call.input.actionPurpose,
      executionPosture: result.executionPosture,
    })
    && result.output.trim().length > 0
    && !isValidationCommand(call.tool, call.input)
}

function isExternalActionReceiptResult(
  result: Extract<SessionEvent, { type: 'tool_result' }>,
): boolean {
  return isExecutorConfirmedExternalActionReceipt({
    status: result.status,
    output: result.output,
    securityEffect: result.metadata?.[TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY],
    executionObserved:
      result.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
  })
}

function eventIndexById(events: SessionEvent[]): Map<string, number> {
  return new Map(events.map((event, index) => [event.id, index]))
}

function evidenceArtifactId(kind: SessionEvidenceArtifactKind, sourceEventIds: string[]): string {
  return `ev-${kind}-${stableDigest(sourceEventIds.join('|'))}`
}

function sourceEventIds(...events: Array<SessionEvent | undefined>): string[] {
  return [...new Set(events.filter((event): event is SessionEvent => Boolean(event)).map((event) => event.id))]
}

function artifactStatusFromToolResult(
  result: Extract<SessionEvent, { type: 'tool_result' }> | undefined,
): SessionEvidenceArtifactStatus {
  if (!result) return 'pending'
  return result.status === 'success' ? 'success' : 'error'
}

function validationArtifactStatus(
  call: Extract<SessionEvent, { type: 'tool_call' }>,
  result: Extract<SessionEvent, { type: 'tool_result' }>,
): SessionEvidenceArtifactStatus {
  // A schema/policy/preflight rejection is still useful audit evidence, but
  // it says nothing about the workspace's mechanical validity because the
  // validator never ran. Preserve it as a warning instead of letting it
  // override a later executed validation for the same revision. Missing
  // metadata remains conservative for historical events.
  if (result.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === false) {
    return 'warning'
  }
  if (result.status !== 'success') return 'error'
  if (
    call.tool === 'terminal.run'
    && !terminalRunHasReliableSuccessStatus(call.input)
  ) {
    return 'warning'
  }
  return 'success'
}

function summarizeValidationArtifact(
  call: Extract<SessionEvent, { type: 'tool_call' }>,
  result: Extract<SessionEvent, { type: 'tool_result' }>,
  status: SessionEvidenceArtifactStatus = validationArtifactStatus(call, result),
): string {
  const command = commandFromToolInput(call.input) ?? call.tool
  const output = evidenceSnippet(result.output, 160)
  const outcome = result.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === false
    ? 'rejected before execution'
    : status === 'warning'
    ? 'diagnostic-only (wrapper can mask validation failure)'
    : result.status
  return output
    ? `${command} -> ${outcome}: ${output}`
    : `${command} -> ${outcome}`
}

function summarizeTodoArtifact(event: Extract<SessionEvent, { type: 'todo_list' }>): string {
  if (event.items.length === 0) {
    return 'No todo items were recorded.'
  }
  const completed = event.items.filter((item) => item.status === 'completed').length
  const inProgress = event.items.filter((item) => item.status === 'in_progress').length
  const pending = event.items.filter((item) => item.status === 'pending').length
  const blocked = event.items.filter((item) => item.status === 'blocked').length
  const cancelled = event.items.filter((item) => item.status === 'cancelled').length
  return `${completed}/${event.items.length} completed`
    + (inProgress > 0 ? `, ${inProgress} in progress` : '')
    + (pending > 0 ? `, ${pending} pending` : '')
    + (blocked > 0 ? `, ${blocked} blocked` : '')
    + (cancelled > 0 ? `, ${cancelled} cancelled` : '')
}

function latestRunContractEvent(
  events: SessionEvent[],
): Extract<SessionEvent, { type: 'run_contract' }> | undefined {
  const contractEvents = events.filter(
    (event): event is Extract<SessionEvent, { type: 'run_contract' }> =>
      event.type === 'run_contract',
  )
  const latest = contractEvents.at(-1)
  if (!latest) return undefined

  // Graph checkpoints and phase handoffs can re-emit the unchanged active
  // contract. The evidence window begins when that semantic contract became
  // active, not at its latest transport copy; otherwise a late duplicate can
  // make earlier edits and validations appear to predate the contract. Walk
  // back only through the trailing run of semantically identical contracts.
  // A genuinely revised contract therefore still starts a new evidence
  // window, including the case where a later revision returns to old text.
  const latestFingerprint = stableFingerprint(latest.contract)
  let active = latest
  for (let index = contractEvents.length - 2; index >= 0; index -= 1) {
    const candidate = contractEvents[index]!
    if (stableFingerprint(candidate.contract) !== latestFingerprint) break
    active = candidate
  }
  return active
}

function latestMatchingCompletionBoard(
  events: SessionEvent[],
  contract: AgentRunContract | null,
  contractIndex: number,
): Extract<SessionEvent, { type: 'state_board' }> | undefined {
  if (!contract) return undefined
  const contractFingerprint = stableFingerprint(contract)
  for (let index = events.length - 1; index > contractIndex; index -= 1) {
    const event = events[index]
    if (
      event?.type === 'state_board'
      && event.board.completion
      && event.board.contract
      && stableFingerprint(event.board.contract) === contractFingerprint
    ) {
      return event
    }
  }
  return undefined
}

function latestAssistantMessage(events: SessionEvent[]): Extract<SessionEvent, { type: 'assistant_message' }> | undefined {
  return events.findLast(
    (event): event is Extract<SessionEvent, { type: 'assistant_message' }> =>
      event.type === 'assistant_message',
  )
}

function assistantContentMarkedUnverified(content: string): boolean {
  return /\bUNVERIFIED\b/i.test(content)
    || /(?:^|\r?\n)\s*INCOMPLETE\s*:/i.test(content)
}

function firstUserMessage(events: SessionEvent[]): Extract<SessionEvent, { type: 'user_message' }> | undefined {
  return events.find(
    (event): event is Extract<SessionEvent, { type: 'user_message' }> =>
      event.type === 'user_message',
  )
}

function countPendingToolResults(events: SessionEvent[]): number {
  const toolCallIds = new Set<string>()
  const completedCallIds = new Set<string>()
  for (const event of events) {
    if (event.type === 'tool_call') {
      toolCallIds.add(event.id)
    } else if (event.type === 'tool_result') {
      completedCallIds.add(event.toolCallId)
    }
  }
  let pending = 0
  for (const toolCallId of toolCallIds) {
    if (!completedCallIds.has(toolCallId)) pending += 1
  }
  return pending
}

export function buildSessionTraceMetrics(
  session: SessionMeta,
  events: SessionEvent[],
) {
  let userMessages = 0
  let assistantMessages = 0
  let toolCalls = 0
  let toolResults = 0
  let toolSuccesses = 0
  let toolFailures = 0
  let approvalRequests = 0
  let approvalApproved = 0
  let approvalFeedback = 0
  let approvalDenied = 0
  // Auto-approval counters track tool calls that ran via remembered
  // session/always rules — operators auditing a long session need to
  // tell the difference between "agent ran 12 tools, 4 with explicit
  // consent, 8 short-circuited by the rule registered at turn 13" vs
  // "agent ran 12 tools with 4 explicit prompts" (current totals).
  let autoApprovalsApproved = 0
  let autoApprovalsDenied = 0
  let contextCompactions = 0
  let contextCompactionTokensBefore = 0
  let contextCompactionTokensAfter = 0
  let contextCompactionTokensSaved = 0
  let lastContextCompactedAt: string | undefined
  let assistantMessagesSinceLastContextCompaction = 0
  let memoryContextEvents = 0
  let memoryContextItems = 0
  let memorySummaryEvents = 0
  let memorySummaryLightCaptures = 0
  let memorySummarySemanticExtractions = 0
  let memorySummaryRagPromotions = 0
  let lastMemorySummarizedAt: string | undefined
  let assistantMessagesSinceLastMemorySummary = 0
  let coworkTasksStarted = 0
  let coworkTasksCompleted = 0
  let coworkTasksFailed = 0
  let coworkDiscussRequests = 0
  let coworkDiscussResponses = 0
  let todoUpdates = 0
  let providerAttemptEvents = 0
  let providerAttemptsStarted = 0
  let providerAttemptFailures = 0
  let providerAttemptSuccesses = 0
  let providerFallbacks = 0
  let lastProviderFallbackAt: string | undefined
  let finalProvider = session.provider
  let finalModel = session.model

  let firstAssistantAt: string | undefined
  let firstToolCallAt: string | undefined
  let firstApprovalAt: string | undefined
  let firstCoworkTaskAt: string | undefined

  for (const event of events) {
    switch (event.type) {
      case 'user_message':
        userMessages += 1
        break
      case 'assistant_message':
        assistantMessages += 1
        assistantMessagesSinceLastContextCompaction += 1
        assistantMessagesSinceLastMemorySummary += 1
        firstAssistantAt ??= event.timestamp
        break
      case 'tool_call':
        toolCalls += 1
        firstToolCallAt ??= event.timestamp
        break
      case 'tool_result':
        toolResults += 1
        if (event.status === 'success') {
          toolSuccesses += 1
        } else {
          toolFailures += 1
        }
        break
      case 'approval_request':
        approvalRequests += 1
        firstApprovalAt ??= event.timestamp
        break
      case 'approval_response':
        if (event.decision === 'approved') {
          approvalApproved += 1
        } else if (event.decision === 'feedback') {
          approvalFeedback += 1
        } else {
          approvalDenied += 1
        }
        break
      case 'auto_approval':
        if (event.decision === 'approved') {
          autoApprovalsApproved += 1
        } else {
          autoApprovalsDenied += 1
        }
        break
      case 'memory_context':
        memoryContextEvents += 1
        memoryContextItems += event.items.length
        break
      case 'memory_summary':
        memorySummaryEvents += 1
        if (event.lightCaptured) {
          memorySummaryLightCaptures += 1
        }
        memorySummarySemanticExtractions += event.semanticMemoriesExtracted
        memorySummaryRagPromotions += event.ragContextPromotions ?? 0
        lastMemorySummarizedAt = event.timestamp
        assistantMessagesSinceLastMemorySummary = 0
        break
      case 'context_compact':
        contextCompactions += 1
        contextCompactionTokensBefore += event.beforeTokens
        contextCompactionTokensAfter += event.afterTokens
        contextCompactionTokensSaved += Math.max(0, event.beforeTokens - event.afterTokens)
        lastContextCompactedAt = event.timestamp
        assistantMessagesSinceLastContextCompaction = 0
        break
      case 'cowork_task_start':
        coworkTasksStarted += 1
        firstCoworkTaskAt ??= event.timestamp
        break
      case 'cowork_task_complete':
        coworkTasksCompleted += 1
        break
      case 'cowork_task_failed':
        coworkTasksFailed += 1
        break
      case 'cowork_discuss_request':
        coworkDiscussRequests += 1
        break
      case 'cowork_discuss_response':
        coworkDiscussResponses += 1
        break
      case 'todo_list':
        todoUpdates += 1
        break
      case 'provider_attempt':
        providerAttemptEvents += 1
        if (event.status === 'started') {
          providerAttemptsStarted += 1
        } else if (event.status === 'failed') {
          providerAttemptFailures += 1
          if (event.retryable && event.nextProvider && event.nextModel) {
            providerFallbacks += 1
            lastProviderFallbackAt = event.timestamp
          }
        } else if (event.status === 'succeeded') {
          providerAttemptSuccesses += 1
          finalProvider = event.provider
          finalModel = event.model
        }
        break
      default:
        break
    }
  }

  const startedAt = events[0]?.timestamp ?? session.createdAt
  const lastEventAt = latestTimestamp(session, events)

  return {
    startedAt,
    lastEventAt,
    runDurationMs: diffMs(startedAt, lastEventAt) ?? 0,
    totalEvents: events.length,
    userMessages,
    assistantMessages,
    toolCalls,
    toolResults,
    toolSuccesses,
    toolFailures,
    approvalRequests,
    approvalApproved,
    approvalFeedback,
    approvalDenied,
    autoApprovalsApproved,
    autoApprovalsDenied,
    contextCompactions,
    contextCompactionTokensBefore,
    contextCompactionTokensAfter,
    contextCompactionTokensSaved,
    lastContextCompactedAt,
    assistantMessagesSinceLastContextCompaction,
    memoryContextEvents,
    memoryContextItems,
    memorySummaryEvents,
    memorySummaryLightCaptures,
    memorySummarySemanticExtractions,
    memorySummaryRagPromotions,
    lastMemorySummarizedAt,
    assistantMessagesSinceLastMemorySummary,
    coworkTasksStarted,
    coworkTasksCompleted,
    coworkTasksFailed,
    coworkDiscussRequests,
    coworkDiscussResponses,
    todoUpdates,
    providerAttemptEvents,
    providerAttemptsStarted,
    providerAttemptFailures,
    providerAttemptSuccesses,
    providerFallbacks,
    lastProviderFallbackAt,
    finalProvider,
    finalModel,
    timeToFirstAssistantMessageMs: diffMs(startedAt, firstAssistantAt),
    timeToFirstToolCallMs: diffMs(startedAt, firstToolCallAt),
    timeToFirstApprovalRequestMs: diffMs(startedAt, firstApprovalAt),
    timeToFirstCoworkTaskMs: diffMs(startedAt, firstCoworkTaskAt),
  }
}

const COMPACTION_OBSERVATION_MESSAGE_THRESHOLD = 12
const SEMANTIC_RECALL_OBSERVATION_MESSAGE_THRESHOLD = 4
const MEMORY_SUMMARY_OBSERVATION_MESSAGE_THRESHOLD = 4

function compactRuntimeStatus(
  runtime: SessionHistoryManagementRuntime | undefined,
): SessionHistoryManagementRuntime {
  return {
    ...(runtime?.semanticIndex
      ? {
          semanticIndex: {
            status: runtime.semanticIndex.status,
            pendingCount: runtime.semanticIndex.pendingCount,
            failedCount: runtime.semanticIndex.failedCount,
            vecAvailable: runtime.semanticIndex.vecAvailable,
            backendAvailable: runtime.semanticIndex.backendAvailable,
            vectorBackend: runtime.semanticIndex.vectorBackend,
            lastError: runtime.semanticIndex.lastError,
          },
        }
      : {}),
    ...(runtime?.dreaming
      ? {
          dreaming: {
            enabled: runtime.dreaming.enabled,
            running: runtime.dreaming.running,
            providerConfigured: runtime.dreaming.providerConfigured,
            model: runtime.dreaming.model,
            fileMemoryEnabled: runtime.dreaming.fileMemoryEnabled,
          },
        }
      : {}),
    ...(runtime?.memoryLifecycle
      ? {
          memoryLifecycle: {
            totalMemories: runtime.memoryLifecycle.totalMemories,
            staleConversationMemories: runtime.memoryLifecycle.staleConversationMemories,
            lowImportanceConversationMemories: runtime.memoryLifecycle.lowImportanceConversationMemories,
            pruneCandidateMemories: runtime.memoryLifecycle.pruneCandidateMemories,
            pendingEmbeddings: runtime.memoryLifecycle.pendingEmbeddings,
            failedEmbeddings: runtime.memoryLifecycle.failedEmbeddings,
            lastAuditAt: runtime.memoryLifecycle.lastAuditAt,
          },
        }
      : {}),
  }
}

export function buildSessionHistoryManagement(
  session: SessionMeta,
  events: SessionEvent[],
  runtime?: SessionHistoryManagementRuntime,
) {
  let memoryItems = 0
  let documentItems = 0
  let lastMemoryContextAt: string | undefined
  let lastDocumentContextAt: string | undefined
  let lastMemoryItemAt: string | undefined
  let lastContextCompactionStrategy: 'preserve_tail' | 'summary_only' | undefined
  let lastRemovedMessageCount: number | undefined
  let lastPreservedMessageCount: number | undefined
  const memorySummarySources: Record<string, number> = {}

  const traceMetrics = buildSessionTraceMetrics(session, events)

  for (const event of events) {
    if (event.type === 'memory_context') {
      lastMemoryContextAt = event.timestamp
      for (const item of event.items) {
        if (item.kind === 'document' || item.source === 'document') {
          documentItems += 1
          lastDocumentContextAt = event.timestamp
        } else {
          memoryItems += 1
          lastMemoryItemAt = event.timestamp
        }
      }
    } else if (event.type === 'context_compact') {
      lastContextCompactionStrategy = event.strategy
      lastRemovedMessageCount = event.removedMessageCount
      lastPreservedMessageCount = event.preservedMessageCount
    } else if (event.type === 'memory_summary') {
      memorySummarySources[event.source] = (memorySummarySources[event.source] ?? 0) + 1
    }
  }

  const risks: SessionHistoryManagementRisk[] = []
  const semanticStatus = runtime?.semanticIndex?.status
  if (
    semanticStatus
    && ['disabled', 'degraded', 'reindex_required'].includes(semanticStatus)
  ) {
    risks.push({
      code: 'semantic_index_unavailable',
      severity: 'warning',
      message: `Semantic index status is ${semanticStatus}.`,
    })
  }

  if (
    traceMetrics.userMessages >= SEMANTIC_RECALL_OBSERVATION_MESSAGE_THRESHOLD
    && traceMetrics.memoryContextEvents === 0
  ) {
    risks.push({
      code: 'semantic_recall_not_observed',
      severity: 'info',
      message: 'No semantic memory or document RAG context has been injected for this session.',
    })
  }

  if (
    traceMetrics.assistantMessages >= COMPACTION_OBSERVATION_MESSAGE_THRESHOLD
    && traceMetrics.contextCompactions === 0
  ) {
    risks.push({
      code: 'compaction_not_observed',
      severity: 'info',
      message: 'No context compaction event has been observed in this long session.',
    })
  } else if (
    traceMetrics.contextCompactions > 0
    && traceMetrics.assistantMessagesSinceLastContextCompaction >= COMPACTION_OBSERVATION_MESSAGE_THRESHOLD
  ) {
    risks.push({
      code: 'compaction_stale',
      severity: 'warning',
      message: `${traceMetrics.assistantMessagesSinceLastContextCompaction} assistant message(s) have accumulated since the last context compaction.`,
    })
  }

  if (
    traceMetrics.assistantMessages >= MEMORY_SUMMARY_OBSERVATION_MESSAGE_THRESHOLD
    && traceMetrics.memorySummaryEvents === 0
  ) {
    risks.push({
      code: 'memory_summary_not_observed',
      severity: 'info',
      message: 'No durable memory summary event has been recorded for this session.',
    })
  } else if (
    traceMetrics.memorySummaryEvents > 0
    && traceMetrics.assistantMessagesSinceLastMemorySummary >= MEMORY_SUMMARY_OBSERVATION_MESSAGE_THRESHOLD
  ) {
    risks.push({
      code: 'memory_summary_stale',
      severity: 'warning',
      message: `${traceMetrics.assistantMessagesSinceLastMemorySummary} assistant message(s) have accumulated since the last durable memory summary.`,
    })
  }

  if (
    runtime?.dreaming?.enabled
    && !runtime.dreaming.providerConfigured
    && traceMetrics.assistantMessages >= MEMORY_SUMMARY_OBSERVATION_MESSAGE_THRESHOLD
  ) {
    risks.push({
      code: 'dreaming_provider_missing',
      severity: 'warning',
      message: 'Dreaming is enabled but no LLM provider is configured for memory extraction.',
    })
  }

  if ((runtime?.memoryLifecycle?.pruneCandidateMemories ?? 0) > 0) {
    risks.push({
      code: 'memory_lifecycle_attention',
      severity: 'warning',
      message: `${runtime?.memoryLifecycle?.pruneCandidateMemories ?? 0} stale low-importance memory item(s) are ready for maintenance.`,
    })
  }

  const hasHistorySignals = traceMetrics.contextCompactions > 0
    || traceMetrics.memoryContextEvents > 0
    || traceMetrics.memorySummaryEvents > 0
  const status: SessionHistoryManagementStatus = traceMetrics.totalEvents === 0
    ? 'empty'
    : risks.some((risk) => risk.severity === 'warning')
      ? 'attention_needed'
      : hasHistorySignals
        ? 'managed'
        : risks.length > 0
          ? 'attention_needed'
          : 'warming'

  return {
    status,
    compact: {
      compactions: traceMetrics.contextCompactions,
      tokensBefore: traceMetrics.contextCompactionTokensBefore,
      tokensAfter: traceMetrics.contextCompactionTokensAfter,
      tokensSaved: traceMetrics.contextCompactionTokensSaved,
      lastCompactedAt: traceMetrics.lastContextCompactedAt,
      assistantMessagesSinceLastCompaction: traceMetrics.assistantMessagesSinceLastContextCompaction,
      lastStrategy: lastContextCompactionStrategy,
      lastRemovedMessageCount,
      lastPreservedMessageCount,
    },
    semanticRecall: {
      contextEvents: traceMetrics.memoryContextEvents,
      contextItems: traceMetrics.memoryContextItems,
      memoryItems,
      documentItems,
      lastContextAt: lastMemoryContextAt,
      lastMemoryItemAt,
      lastDocumentContextAt,
    },
    memorySummary: {
      events: traceMetrics.memorySummaryEvents,
      lightCaptures: traceMetrics.memorySummaryLightCaptures,
      semanticExtractions: traceMetrics.memorySummarySemanticExtractions,
      ragPromotions: traceMetrics.memorySummaryRagPromotions,
      lastSummarizedAt: traceMetrics.lastMemorySummarizedAt,
      assistantMessagesSinceLastSummary: traceMetrics.assistantMessagesSinceLastMemorySummary,
      sources: memorySummarySources,
    },
    runtime: compactRuntimeStatus(runtime),
    risks,
    lastUpdatedAt: traceMetrics.lastEventAt,
  }
}

export function buildSessionCompletionChecklist(
  session: SessionMeta,
  events: SessionEvent[],
  pendingApprovals: PendingApprovalLike[],
  pendingQuestions: PendingQuestionLike[],
) {
  const latestTodo = [...events]
    .reverse()
    .find((event) => event.type === 'todo_list')

  const pendingApprovalCount = pendingApprovals.filter(
    (approval) => approval.state === 'live',
  ).length
  const pendingQuestionCount = pendingQuestions.length
  const toolCalls = events.filter((event) => event.type === 'tool_call').length
  const toolResults = events.filter((event) => event.type === 'tool_result').length
  const approvalRequests = events.filter((event) => event.type === 'approval_request').length
  const approvalResponses = events.filter((event) => event.type === 'approval_response').length
  const assistantMessages = events.filter((event) => event.type === 'assistant_message').length
  const userMessages = events.filter((event) => event.type === 'user_message').length

  const items: ChecklistItem[] = latestTodo
    ? latestTodo.items.map((item) => ({
        label: item.content,
        status: checklistStatusFromTodoStatus(item.status),
        source: 'todo' as const,
      }))
    : [
        {
          label: 'Capture the user request',
          status: userMessages > 0 ? 'completed' : 'pending',
          source: 'session' as const,
        },
        {
          label: 'Produce an assistant response',
          status: assistantMessages > 0
            ? 'completed'
            : userMessages > 0
              ? 'in_progress'
              : 'pending',
          source: 'session' as const,
        },
        {
          label: 'Run tool-assisted work',
          status: toolCalls === 0
            ? 'skipped'
            : toolResults >= toolCalls
              ? 'completed'
              : 'in_progress',
          detail: toolCalls === 0
            ? 'No tool call was needed for this run.'
            : `${toolResults}/${toolCalls} tool calls completed.`,
          source: 'tool' as const,
        },
        {
          label: 'Resolve approval gates',
          status: approvalRequests === 0
            ? 'skipped'
            : pendingApprovalCount > 0
              ? 'blocked'
              : approvalResponses >= approvalRequests
                ? 'completed'
                : 'in_progress',
          detail: approvalRequests === 0
            ? 'No explicit approval request was raised.'
            : `${approvalResponses}/${approvalRequests} approval decisions recorded.`,
          source: 'approval' as const,
        },
      ]

  if (pendingQuestionCount > 0) {
    items.push({
      label: 'Answer pending runtime questions',
      status: 'blocked',
      detail: `${pendingQuestionCount} question(s) still need user input.`,
      source: 'question' as const,
    })
  } else {
    const coworkQuestionCount = events.filter(
      (event) => event.type === 'cowork_discuss_request',
    ).length
    if (coworkQuestionCount > 0) {
      const coworkResponseCount = events.filter(
        (event) => event.type === 'cowork_discuss_response',
      ).length
      items.push({
        label: 'Resolve cowork discussion requests',
        status: coworkResponseCount >= coworkQuestionCount
          ? 'completed'
          : 'in_progress',
        detail: `${coworkResponseCount}/${coworkQuestionCount} discussion response(s) recorded.`,
        source: 'question' as const,
      })
    }
  }

  const statuses = items.map((item) => item.status)
  const status = statuses.includes('blocked')
    ? 'blocked'
    : statuses.some((value) => value === 'pending' || value === 'in_progress')
      ? 'in_progress'
      : statuses.some((value) => value === 'completed')
        ? 'completed'
        : 'not_started'

  return {
    status,
    items,
    lastUpdatedAt: latestTimestamp(session, events),
  }
}

export function buildSessionWorkingMemory(
  session: SessionMeta,
  events: SessionEvent[],
  pendingQuestions: PendingQuestionLike[],
) {
  const toolCalls = new Map<string, Extract<SessionEvent, { type: 'tool_call' }>>()
  const approvalRequests = new Map<
    string,
    Extract<SessionEvent, { type: 'approval_request' }>
  >()

  for (const event of events) {
    if (event.type === 'tool_call') {
      toolCalls.set(event.id, event)
    } else if (event.type === 'approval_request') {
      approvalRequests.set(event.id, event)
    }
  }

  const latestTodo = [...events]
    .reverse()
    .find((event) => event.type === 'todo_list')
  const activeTodo = latestTodo?.items.find((item) => item.status === 'in_progress')
    ?? latestTodo?.items.find((item) => item.status === 'blocked')
    ?? latestTodo?.items.find((item) => item.status === 'pending')

  const taskSummary = [...events]
    .reverse()
    .find((event) => event.type === 'user_message')
  const keyDecisions = [...events]
    .filter((event) =>
      event.type === 'approval_response'
      || event.type === 'auto_approval'
      || event.type === 'cowork_discuss_response'
      || event.type === 'delegation_state',
    )
    .slice(-6)
    .map((event) => {
      if (event.type === 'approval_response') {
        const request = approvalRequests.get(event.requestId)
        const tool = request?.tool ? ` for ${request.tool}` : ''
        return {
          type: 'approval' as const,
          summary: truncateText(
            `Approval ${event.decision}${tool}${event.note ? `: ${event.note}` : ''}`,
          ),
          timestamp: event.timestamp,
        }
      }
      if (event.type === 'auto_approval') {
        // Surfaces the matched scope/pattern so working-memory
        // consumers (web/desktop overlay, /sessions show working
        // memory) can show *why* a tool ran without an explicit
        // prompt. Categorised under 'approval' so existing UIs that
        // group by decision type still see it.
        return {
          type: 'approval' as const,
          summary: truncateText(
            `Auto-${event.decision} for ${event.tool} via ${event.scope} rule '${event.rule.pattern}'`,
          ),
          timestamp: event.timestamp,
        }
      }
      if (event.type === 'cowork_discuss_response') {
        return {
          type: 'question' as const,
          summary: truncateText(`Cowork answer: ${event.response}`),
          timestamp: event.timestamp,
        }
      }
      return {
        type: 'delegation' as const,
        summary: truncateText(
          `Delegation ${event.claimHealth} on ${event.targetDevice}: ${event.detail}`,
        ),
        timestamp: event.timestamp,
      }
    })

  const recentToolOutcomes = [...events]
    .filter((event) => event.type === 'tool_result')
    .slice(-5)
    .map((event) => ({
      tool: toolCalls.get(event.toolCallId)?.tool ?? 'unknown',
      status: event.status,
      output: truncateText(event.output),
      timestamp: event.timestamp,
    }))

  const fileChanges = [...events]
    .filter((event) => event.type === 'tool_call')
    .flatMap((event) =>
      extractToolCallPaths(event).map((change) => ({
        path: change.path,
        tool: event.tool,
        kind: change.kind,
        timestamp: event.timestamp,
      })),
    )
    .slice(-10)

  return {
    taskSummary: truncateText(taskSummary?.content ?? session.title, 400),
    latestPlanStep: activeTodo?.content,
    activeTodo: activeTodo?.content,
    keyDecisions,
    recentToolOutcomes,
    fileChanges,
    openQuestions: pendingQuestions.map((question) => truncateText(question.prompt)),
    lastUpdatedAt: latestTimestamp(session, events),
  }
}

export interface SessionEditRollback {
  checkpointId: string
  status: 'committed' | 'reverted'
  reason?: string
  files: string[]
  resolvedAt: string
}

export function buildSessionEditRollbacks(events: SessionEvent[]): SessionEditRollback[] {
  return events
    .filter(
      (event): event is Extract<SessionEvent, { type: 'edit_checkpoint_resolved' }> =>
        event.type === 'edit_checkpoint_resolved',
    )
    .map((event) => ({
      checkpointId: event.checkpoint.checkpointId,
      status: event.checkpoint.status === 'reverted' ? 'reverted' : 'committed',
      reason: event.checkpoint.revertReason,
      files: event.checkpoint.files.map((f) => f.path),
      resolvedAt:
        event.checkpoint.revertedAt ?? event.checkpoint.closedAt ?? event.timestamp,
    }))
}

/** Same source events as buildSessionEditRollbacks, but returns the raw
 * EditCheckpointSummary the live SSE onEditCheckpointResolved callback
 * already sends per-turn — lets session-history replay feed the exact
 * same client-side store action (appendEditRollback) without an adapter. */
export function buildSessionRawEditCheckpoints(events: SessionEvent[]): EditCheckpointSummary[] {
  return events
    .filter(
      (event): event is Extract<SessionEvent, { type: 'edit_checkpoint_resolved' }> =>
        event.type === 'edit_checkpoint_resolved',
    )
    .map((event) => event.checkpoint)
}

export function buildSessionDebateRounds(events: SessionEvent[]) {
  return events
    .filter(
      (event): event is Extract<SessionEvent, { type: 'debate_round' }> =>
        event.type === 'debate_round',
    )
    .map((event) => event.round)
}

export function buildSessionPlannerWorkingMemory(events: SessionEvent[]) {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i]
    if (event.type === 'planner_working_memory_updated') {
      return event.workingMemory
    }
  }
  return null
}

export function buildSessionRunContract(events: SessionEvent[]): AgentRunContract | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i]
    if (event.type === 'run_contract') {
      return event.contract
    }
  }
  return null
}

const CONTRACT_LEDGER_SECTIONS: SessionContractLedgerSectionName[] = [
  'goal',
  'scope',
  'acceptance_criteria',
  'deliverables',
  'evidence_requirements',
  'verification_plan',
  'runtime_context',
  'blockers',
]

function ledgerEntry(
  section: SessionContractLedgerSectionName,
  key: string,
  input: Omit<SessionContractLedgerEntry, 'id' | 'key'>,
): SessionContractLedgerEntry {
  return {
    id: `ledger-${stableDigest(`${section}|${key}|${input.status}|${input.source}|${input.value}`)}`,
    key,
    ...input,
  }
}

function sectionStatus(entries: SessionContractLedgerEntry[]): SessionContractLedgerEntryStatus {
  if (entries.length === 0) return 'missing'
  const statuses = new Set(entries.map((entry) => entry.status))
  if (statuses.has('blocked')) return 'blocked'
  if (statuses.has('missing')) return 'missing'
  if (statuses.has('weak')) return 'weak'
  if (statuses.has('confirmed')) return 'confirmed'
  if (statuses.has('defaulted')) return 'defaulted'
  if (statuses.has('inferred')) return 'inferred'
  return 'missing'
}

function sectionSummary(
  name: SessionContractLedgerSectionName,
  entries: SessionContractLedgerEntry[],
): string {
  const status = sectionStatus(entries)
  const count = entries.length
  switch (name) {
    case 'goal':
      return status === 'confirmed'
        ? 'A user goal is available as contract input.'
        : 'No explicit user goal was found.'
    case 'scope':
      return status === 'confirmed'
        ? `${count} scope constraint/non-goal item(s) are captured.`
        : 'Scope is bounded by conservative local defaults.'
    case 'acceptance_criteria':
      return status === 'confirmed'
        ? `${count} acceptance criterion/criteria are captured.`
        : 'Acceptance criteria are missing from the contract.'
    case 'deliverables':
      return status === 'confirmed'
        ? `${count} deliverable requirement(s) are evidenced or explicitly not required.`
        : 'One or more required deliverables or artifact sections lack evidence.'
    case 'evidence_requirements':
      return status === 'confirmed'
        ? `${count} evidence requirement(s) are satisfied or explicitly not required.`
        : 'One or more run-contract evidence requirements need more support.'
    case 'verification_plan':
      return status === 'confirmed'
        ? 'Validation evidence or a no-change verification path is available.'
        : 'A conservative local validation plan is assumed but not yet evidenced.'
    case 'runtime_context':
      return status === 'confirmed'
        ? 'Session runtime context is available.'
        : 'Runtime context is incomplete.'
    case 'blockers':
      return status === 'blocked'
        ? `${count} authority blocker(s) require human review.`
        : count > 0
          ? `${count} authority warning(s) were detected.`
          : 'No authority blocker was detected.'
  }
}

function buildLedgerSection(
  name: SessionContractLedgerSectionName,
  entries: SessionContractLedgerEntry[],
): SessionContractLedgerSection {
  const status = name === 'blockers' && entries.length === 0
    ? 'confirmed'
    : sectionStatus(entries)
  return {
    name,
    status,
    summary: sectionSummary(name, entries),
    entries,
  }
}

function ledgerPathsReferToSameFile(left: string, right: string): boolean {
  const normalizedLeft = normalizeArtifactPath(left).replace(/\/+$/g, '')
  const normalizedRight = normalizeArtifactPath(right).replace(/\/+$/g, '')
  return normalizedLeft === normalizedRight
    || normalizedLeft.endsWith(`/${normalizedRight}`)
    || normalizedRight.endsWith(`/${normalizedLeft}`)
}

function requiredDocumentArtifactPaths(
  contract: AgentRunContract | null,
): string[] {
  const artifacts = contract?.requiredArtifacts ?? []
  if (
    artifacts.length === 0
    || artifacts.some((artifact) => (
      artifact.kind === 'directory'
      || !(artifact.kind === 'document' || isDocumentArtifactPath(artifact.path))
    ))
    || (contract?.evidenceRequirements ?? []).some((requirement) => requirement.kind === 'validation')
  ) {
    return []
  }
  return [...new Set(artifacts.map((artifact) => normalizeArtifactPath(artifact.path)))]
}

function currentDocumentArtifactReadBackEvidence(options: {
  contract: AgentRunContract | null
  contractIndex: number
  indexes: Map<string, number>
  fileChanges: readonly SessionEvidenceArtifact[]
  artifactReadBacks: readonly SessionEvidenceArtifact[]
}): SessionEvidenceArtifact[] {
  const requiredPaths = requiredDocumentArtifactPaths(options.contract)
  if (requiredPaths.length === 0) return []

  const successfulChanges = options.fileChanges.filter((artifact) => (
    artifact.status === 'success'
    && artifact.path
    && artifact.sourceEventIds.some((id) => (
      (options.indexes.get(id) ?? -1) > options.contractIndex
    ))
  ))
  if (
    successfulChanges.length === 0
    || successfulChanges.some((artifact) => (
      !artifact.path
      || !requiredPaths.some((path) => ledgerPathsReferToSameFile(artifact.path!, path))
    ))
    || requiredPaths.some((path) => !successfulChanges.some((artifact) => (
      artifact.path && ledgerPathsReferToSameFile(artifact.path, path)
    )))
  ) {
    return []
  }

  const evidence = options.artifactReadBacks.filter((artifact) => (
    artifact.status === 'success'
    && artifact.currentRevision === true
    && artifact.path
    && requiredPaths.some((path) => ledgerPathsReferToSameFile(artifact.path!, path))
  ))
  return requiredPaths.every((path) => evidence.some((artifact) => (
    artifact.path && ledgerPathsReferToSameFile(artifact.path, path)
  )))
    ? evidence
    : []
}

function ledgerArtifactEvidenceForPath(
  artifacts: SessionEvidenceArtifact[],
  path: string,
): SessionEvidenceArtifact[] {
  return artifacts.filter((artifact) =>
    artifact.path && ledgerPathsReferToSameFile(artifact.path, path)
  )
}

function normalizeArtifactHeading(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/^#+\s*/, '')
    .replace(/^[\s\d.)-]+/, '')
    .replace(/[`*_~:：\-–—()[\]{}]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function artifactHasMarkdownSection(content: string, title: string): boolean {
  const expected = normalizeArtifactHeading(title)
  if (!expected) return false
  return content
    .split(/\r?\n/)
    .map((line) => line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/)?.[1] ?? '')
    .filter(Boolean)
    .map(normalizeArtifactHeading)
    .some((heading) =>
      heading === expected || heading.includes(expected) || expected.includes(heading)
    )
}

interface ContractEvidenceStats {
  sourceObservations: Array<{ eventId: string; tool: string }>
  sourceReadPaths: string[]
  sourceScopes: string[]
  searchObservationTools: string[]
  searchCount: number
  validationEvidenceIds: string[]
  fileChangeEvidenceIds: string[]
}

function directorySegmentsForEvidencePath(path: string): string[] {
  const segments = normalizeArtifactPath(path).split('/').filter(Boolean)
  if (segments.length === 0) return []
  const last = segments.at(-1) ?? ''
  return fileExtension(last) ? segments.slice(0, -1) : segments
}

function commonLeadingSegments(paths: string[][]): string[] {
  if (paths.length === 0) return []
  const [first, ...rest] = paths
  const common = [...first]
  for (const path of rest) {
    let index = 0
    while (index < common.length && index < path.length && common[index] === path[index]) {
      index += 1
    }
    common.length = index
    if (common.length === 0) break
  }
  return common
}

function sourceEvidenceScopes(paths: string[]): string[] {
  const directories = paths
    .filter(isSourceLikePath)
    .map(directorySegmentsForEvidencePath)
    .filter((segments) => segments.length > 0)
  const commonPrefix = commonLeadingSegments(directories)
  const scopes = new Set<string>()
  for (const segments of directories) {
    const scopeSegments = segments.length > commonPrefix.length
      ? segments.slice(0, commonPrefix.length + 1)
      : segments
    const scope = scopeSegments.join('/')
    if (scope) scopes.add(scope)
  }
  return Array.from(scopes)
}

function collectContractEvidenceStats(
  events: SessionEvent[],
  contractIndex: number,
  evidenceManifest: SessionEvidenceManifest,
): ContractEvidenceStats {
  const indexes = eventIndexById(events)
  const toolCalls = new Map<string, Extract<SessionEvent, { type: 'tool_call' }>>()
  for (const event of events) {
    if (event.type === 'tool_call') {
      toolCalls.set(event.id, event)
    }
  }

  const sourceReadPaths = new Set<string>()
  const sourceObservations = new Map<string, { eventId: string; tool: string }>()
  const successfulSearches = new Map<string, string>()
  for (const [index, event] of events.entries()) {
    if (index <= contractIndex) continue
    if (event.type !== 'tool_result' || event.status !== 'success') {
      continue
    }
    const call = toolCalls.get(event.toolCallId)
    if (!call || !event.output.trim()) continue
    const executionObserved = event.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY]
    const securityEffect = event.metadata?.[TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY]
    const structurallyObserved = toolProvidesContentReadEvidence(call.tool)
      || toolProvidesSearchEvidence(call.tool)
      || (
        executionObserved === true
        && securityEffect === 'observe'
      )
    if (structurallyObserved && executionObserved !== false) {
      sourceObservations.set(call.id, { eventId: event.id, tool: call.tool })
    }
    if (toolProvidesSearchEvidence(call.tool) && executionObserved !== false) {
      successfulSearches.set(call.id, call.tool)
      continue
    }
    const path = typeof call?.input.path === 'string' ? call.input.path : undefined
    if (
      call.tool === 'fs.read'
      && executionObserved !== false
      && path
      && isSourceLikePath(path)
    ) {
      sourceReadPaths.add(normalizeArtifactPath(path))
    }
  }

  const validationEvidenceIds = evidenceManifest.artifacts
    .filter((artifact) =>
      artifact.kind === 'validation'
      && artifact.sourceEventIds.some((id) => (indexes.get(id) ?? -1) > contractIndex)
    )
    .map((artifact) => artifact.id)
  const fileChangeEvidenceIds = evidenceManifest.artifacts
    .filter((artifact) =>
      artifact.kind === 'file_change'
      && artifact.status === 'success'
      && artifact.sourceEventIds.some((id) => (indexes.get(id) ?? -1) > contractIndex)
    )
    .map((artifact) => artifact.id)

  return {
    sourceObservations: [...sourceObservations.values()],
    sourceReadPaths: [...sourceReadPaths],
    sourceScopes: sourceEvidenceScopes([...sourceReadPaths]),
    searchObservationTools: [...successfulSearches.values()],
    searchCount: successfulSearches.size,
    validationEvidenceIds,
    fileChangeEvidenceIds,
  }
}

function evidenceRequirementSupported(
  requirement: NonNullable<AgentRunContract['evidenceRequirements']>[number],
  stats: ContractEvidenceStats,
): boolean {
  if (requirement.kind === 'source' || requirement.kind === 'repository') {
    const matchingObservations = stats.sourceObservations.filter((entry) =>
      sourceToolMatchesEvidenceRequirement(requirement, entry.tool)
    )
    const minSourceObservations = evidenceRequirementMinSourceObservations(requirement)
    const minSourceFiles = evidenceRequirementMinSourceFiles(requirement)
    const minSourceScopes = evidenceRequirementMinSourceScopes(requirement)
    const repositoryBreadthSupported = !evidenceRequirementUsesRepositoryBreadth(requirement)
      || (
        stats.sourceReadPaths.length >= minSourceFiles
        && stats.sourceScopes.length >= minSourceScopes
      )
    return matchingObservations.length >= minSourceObservations
      && repositoryBreadthSupported
      && (
        !evidenceRequirementRequiresSearch(requirement)
        || stats.searchObservationTools.some((tool) =>
          sourceToolMatchesEvidenceRequirement(requirement, tool)
        )
      )
  }
  if (requirement.kind === 'validation') {
    return stats.validationEvidenceIds.length > 0
  }
  if (requirement.kind === 'artifact') {
    return stats.fileChangeEvidenceIds.length > 0
  }
  return stats.sourceReadPaths.length > 0
    || stats.searchCount > 0
    || stats.validationEvidenceIds.length > 0
    || stats.fileChangeEvidenceIds.length > 0
}

function evidenceTextMentions(content: string, target: string): boolean {
  const normalizedContent = normalizeArtifactPath(content).toLowerCase()
  const normalizedTarget = normalizeArtifactPath(target).toLowerCase()
  return normalizedTarget.length > 0 && normalizedContent.includes(normalizedTarget)
}

function traceableSourceScopeCount(content: string, sourcePaths: string[]): number {
  const scopes = sourceEvidenceScopes(sourcePaths)
  const mentioned = new Set<string>()
  for (const scope of scopes) {
    if (evidenceTextMentions(content, scope)) {
      mentioned.add(scope)
    }
  }
  for (const path of sourcePaths) {
    if (!evidenceTextMentions(content, path)) {
      continue
    }
    const scope = sourceEvidenceScopes([path])[0]
    if (scope) mentioned.add(scope)
  }
  return mentioned.size
}

function artifactEvidenceMapSupported(
  session: SessionMeta,
  contract: AgentRunContract | null,
  requirement: NonNullable<AgentRunContract['evidenceRequirements']>[number],
  stats: ContractEvidenceStats,
): boolean {
  if (
    !requirement.requiresArtifactEvidenceMap
    || (requirement.kind !== 'source' && requirement.kind !== 'repository')
  ) {
    return true
  }
  const requiredScopeCount = Math.min(
    stats.sourceScopes.length,
    Math.max(1, requirement.minSourceScopes ?? 1),
  )
  if (requiredScopeCount <= 0) {
    return false
  }
  for (const artifact of contract?.requiredArtifacts ?? []) {
    const content = readAssertionCandidateFile(session.cwd, artifact.path)
    if (content && traceableSourceScopeCount(content, stats.sourceReadPaths) >= requiredScopeCount) {
      return true
    }
  }
  return false
}

function artifactSelfReviewSupported(
  session: SessionMeta,
  contract: AgentRunContract | null,
  requirement: NonNullable<AgentRunContract['evidenceRequirements']>[number],
): boolean {
  if (!requirement.requiresArtifactSelfReview) {
    return true
  }
  const criteria = contract?.acceptanceCriteria ?? []
  if (criteria.length === 0) {
    return false
  }
  for (const artifact of contract?.requiredArtifacts ?? []) {
    const content = readAssertionCandidateFile(session.cwd, artifact.path)
    if (!content) continue
    const covered = criteria.every((criterion) =>
      new RegExp(`\\b${criterion.id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(content)
    )
    if (covered) {
      return true
    }
  }
  return false
}

function shouldIgnoreAuthorityMatch(
  text: string,
  matchIndex: number,
  matchLength: number,
): boolean {
  const before = text.slice(Math.max(0, matchIndex - 72), matchIndex)
  const after = text.slice(matchIndex + matchLength, matchIndex + matchLength + 72)
  const directPrefix = before.split(/[\n.!?;,]/).at(-1) ?? ''
  const directSuffix = after.split(/[\n.!?;,]/, 1)[0] ?? ''

  const safeQualifierBefore = /(?:\b(?:example|placeholder|dummy|fake|test-token|test token|mock|without|do not|don't|never|avoid|exclude|no real)\b|예시|더미|가짜|테스트|없이|말고|금지|제외|넣지|포함하지|조회하지|출력하지|사용하지)(?:\s+\S+){0,5}\s*$/i
  const safeConstraintAfter = /^(?:\s+(?:values?|material|contents?|data|값|값은|값을|내용|내용은|내용을))?(?:\s+(?:must|should|shall|is|are|은|는|이|가|을|를))?\s*(?:absolutely\s+|절대\s*)?(?:not\b|never\b|금지|제외|하지\s*(?:말|않)|않(?:고|는|도록|음)|말(?:고|라|아)|넣지|포함하지|조회하지|출력하지|사용하지)/i
  const safeControlPlaneReferenceAfter = /^(?:\s*-(?:manager|management)\b|\s+(?:manager|management)\b|\s+(?:store\s+)?(?:integration|configuration|config|health|status|metadata|names?|references?|inventory|sync|synchronization|wiring)\b|\s*(?:매니저|관리)(?:\s|$)|\s*(?:연동|구성|상태|메타데이터|이름|참조|목록|동기화))/iu
  const materialAccessAfter = /\b(?:retrieve|read|export|obtain|reveal|decrypt|use)\b|\b(?:values?|contents?|private material)\b|(?:값|내용|원문)(?:은|는|이|가|을|를)?|(?:조회|읽|내보내|복호화|사용)(?:하|해|해서|하여|할|해줘)/iu

  return safeQualifierBefore.test(directPrefix)
    || safeConstraintAfter.test(directSuffix)
    || (
      safeControlPlaneReferenceAfter.test(directSuffix)
      && !materialAccessAfter.test(directSuffix)
    )
}

function hasActionableAuthorityMatch(text: string, pattern: RegExp): boolean {
  const flags = pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`
  const matcher = new RegExp(pattern.source, flags)
  for (const match of text.matchAll(matcher)) {
    const matchIndex = match.index ?? -1
    if (matchIndex < 0) continue
    if (!shouldIgnoreAuthorityMatch(text, matchIndex, match[0].length)) {
      return true
    }
  }
  return false
}

function detectContractLedgerBlockers(events: SessionEvent[]): SessionContractLedgerBlocker[] {
  const blockers: SessionContractLedgerBlocker[] = []
  const userEvents = events.filter(
    (event): event is Extract<SessionEvent, { type: 'user_message' }> =>
      event.type === 'user_message',
  )
  const specs: Array<{
    code: SessionContractLedgerBlockerCode
    severity: 'warning' | 'blocker'
    pattern: RegExp
    message: string
  }> = [
    {
      code: 'credential_or_secret',
      severity: 'blocker',
      pattern: /\b(api key|apikey|secret|password|passphrase|private key|token|credential|credentials|service account)\b|비밀번호|시크릿|토큰|인증키|개인키/i,
      message: 'The request appears to require a real credential, token, password, or secret value.',
    },
    {
      code: 'destructive_production_action',
      severity: 'blocker',
      pattern: /\b(delete|drop|erase|wipe|destroy|truncate|remove)\b.{0,80}\b(production|prod|live|database|db|bucket|account)\b|\b(production|prod|live)\b.{0,80}\b(delete|drop|erase|wipe|destroy|truncate|remove)\b|운영.{0,40}(삭제|초기화|파괴|제거)/i,
      message: 'The request appears to authorize destructive production or account-level action.',
    },
    {
      code: 'external_side_effect',
      severity: 'warning',
      pattern: /\b(deploy|release|publish|send email|notify users|create account|database migration|go live|push live)\b|배포|릴리스|발행|이메일\s*전송|사용자\s*알림/i,
      message: 'The request may create an external side effect and should keep authority explicit.',
    },
    {
      code: 'billing_authority',
      severity: 'warning',
      pattern: /\b(charge|purchase|subscribe|paid plan|credit card|bank account|invoice|real payment)\b|결제|청구|구독|카드/i,
      message: 'The request touches billing or payment authority that should stay explicit.',
    },
    {
      code: 'legal_or_medical_judgment',
      severity: 'warning',
      pattern: /\b(legal advice|liability|medical advice|clinical|diagnosis|treatment|patient)\b|법률\s*자문|의학\s*진단|치료|환자/i,
      message: 'The request touches legal or medical judgment and should be reviewed as a blocker if authoritative advice is required.',
    },
  ]

  for (const event of userEvents) {
    // Authority blockers describe actions the user affirmatively authorized,
    // not every dangerous noun that appears in the request. Reuse the same
    // bounded negative-capability scope as execution contracts so a leading
    // or trailing prohibition applies across coordinated action lists, while
    // an explicit later contrast/sequence remains actionable.
    const text = inputPositiveCapabilityScope(event.content)
    for (const spec of specs) {
      if (!hasActionableAuthorityMatch(text, spec.pattern)) continue
      if (blockers.some((blocker) => blocker.code === spec.code)) {
        const existing = blockers.find((blocker) => blocker.code === spec.code)
        existing?.eventIds.push(event.id)
        continue
      }
      blockers.push({
        code: spec.code,
        severity: spec.severity,
        message: spec.message,
        eventIds: [event.id],
      })
    }
  }

  return blockers
}

function contractLedgerStatus(
  sections: SessionContractLedgerSection[],
  blockers: SessionContractLedgerBlocker[],
  events: SessionEvent[],
): SessionContractLedgerStatus {
  if (events.length === 0) return 'empty'
  if (blockers.some((blocker) => blocker.severity === 'blocker')) return 'blocked'
  if (sections.some((section) => section.status === 'missing' || section.status === 'weak' || section.status === 'blocked')) {
    return 'needs_attention'
  }
  return 'ready'
}

export function buildSessionContractLedger(
  session: SessionMeta,
  events: SessionEvent[],
  evidenceManifest: SessionEvidenceManifest = buildSessionEvidenceManifest(session, events),
): SessionContractLedger {
  const firstUser = firstUserMessage(events)
  const contractEvent = latestRunContractEvent(events)
  const contract = contractEvent?.contract ?? null
  const indexes = eventIndexById(events)
  const contractIndex = contractEvent ? indexes.get(contractEvent.id) ?? -1 : -1
  const fileChangeArtifacts = evidenceManifest.artifacts
    .filter((artifact) => artifact.kind === 'file_change')
  const fileChangeArtifactsAfterContract = fileChangeArtifacts.filter((artifact) =>
    artifact.sourceEventIds.some((id) => (indexes.get(id) ?? -1) > contractIndex)
  )
  const fileChangeEvidenceIds = fileChangeArtifacts.map((artifact) => artifact.id)
  const validationEvidence = evidenceManifest.artifacts
    .filter((artifact) => artifact.kind === 'validation')
  const contractEvidenceStats = collectContractEvidenceStats(events, contractIndex, evidenceManifest)
  const blockers = detectContractLedgerBlockers(events)
  const sectionEntries = new Map<SessionContractLedgerSectionName, SessionContractLedgerEntry[]>()
  const addEntry = (
    section: SessionContractLedgerSectionName,
    key: string,
    input: Omit<SessionContractLedgerEntry, 'id' | 'key'>,
  ) => {
    const entries = sectionEntries.get(section) ?? []
    entries.push(ledgerEntry(section, key, input))
    sectionEntries.set(section, entries)
  }

  if (firstUser) {
    addEntry('goal', 'goal.primary', {
      value: evidenceSnippet(firstUser.content, 320),
      source: 'user_goal',
      status: 'confirmed',
      confidence: 0.95,
      reversible: false,
      rationale: 'Initial user request captured as the contract goal.',
      eventIds: [firstUser.id],
    })
  } else if (session.title.trim()) {
    addEntry('goal', 'goal.session_title', {
      value: evidenceSnippet(session.title, 220),
      source: 'session_runtime',
      status: 'inferred',
      confidence: 0.5,
      reversible: true,
      rationale: 'No user message was available, so the session title is used as a weak goal proxy.',
    })
  } else {
    addEntry('goal', 'goal.missing', {
      value: 'No explicit user goal was captured.',
      source: 'conservative_default',
      status: 'missing',
      confidence: 0,
      reversible: true,
      rationale: 'A contract cannot be trusted without a goal.',
    })
  }

  if (contract) {
    addEntry('scope', 'scope.summary', {
      value: evidenceSnippet(contract.summary, 320),
      source: 'run_contract',
      status: 'confirmed',
      confidence: 0.9,
      reversible: true,
      rationale: 'Planner emitted a durable run contract summary.',
      eventIds: contractEvent ? [contractEvent.id] : undefined,
    })
    for (const [index, constraint] of contract.constraints.entries()) {
      addEntry('scope', `scope.constraint.${index + 1}`, {
        value: evidenceSnippet(constraint, 240),
        source: 'run_contract',
        status: 'confirmed',
        confidence: 0.86,
        reversible: true,
        rationale: 'Planner emitted this run-contract constraint.',
        eventIds: contractEvent ? [contractEvent.id] : undefined,
      })
    }
    for (const [index, nonGoal] of contract.outOfScope.entries()) {
      addEntry('scope', `scope.non_goal.${index + 1}`, {
        value: evidenceSnippet(nonGoal, 240),
        source: 'run_contract',
        status: 'confirmed',
        confidence: 0.86,
        reversible: true,
        rationale: 'Planner emitted this out-of-scope boundary.',
        eventIds: contractEvent ? [contractEvent.id] : undefined,
      })
    }
  } else {
    addEntry('scope', 'scope.safe_default', {
      value: 'Keep work local, reversible, and aligned with existing project patterns; avoid new dependencies or external side effects unless explicitly approved.',
      source: 'conservative_default',
      status: 'defaulted',
      confidence: 0.62,
      reversible: true,
      rationale: 'No durable run contract exists, so scope is bounded by a conservative default.',
    })
  }

  if (contract && contract.acceptanceCriteria.length > 0) {
    for (const criterion of contract.acceptanceCriteria) {
      addEntry('acceptance_criteria', criterion.id, {
        value: evidenceSnippet(criterion.text, 280),
        source: 'run_contract',
        status: 'confirmed',
        confidence: 0.9,
        reversible: true,
        rationale: 'Planner emitted this acceptance criterion in the durable run contract.',
        eventIds: contractEvent ? [contractEvent.id] : undefined,
      })
    }
  } else {
    addEntry('acceptance_criteria', 'acceptance_criteria.missing', {
      value: 'No acceptance criteria were recorded.',
      source: 'conservative_default',
      status: 'missing',
      confidence: 0,
      reversible: true,
      rationale: 'Work cannot be evaluated semantically without explicit acceptance criteria.',
    })
  }

  if (contract?.requiredArtifacts?.length) {
    for (const [index, artifact] of contract.requiredArtifacts.entries()) {
      const matchingEvidence = ledgerArtifactEvidenceForPath(fileChangeArtifactsAfterContract, artifact.path)
      const successfulEvidence = matchingEvidence.filter((entry) => entry.status === 'success')
      const failedEvidence = matchingEvidence.filter((entry) => entry.status === 'error')
      const status: SessionContractLedgerEntryStatus = successfulEvidence.length > 0
        ? 'confirmed'
        : failedEvidence.length > 0
          ? 'weak'
          : 'missing'
      addEntry('deliverables', `deliverable.${index + 1}`, {
        value: evidenceSnippet(`${artifact.path} (${artifact.kind})${artifact.description ? ` — ${artifact.description}` : ''}`, 320),
        source: 'run_contract',
        status,
        confidence: status === 'confirmed' ? 0.88 : status === 'weak' ? 0.45 : 0.12,
        reversible: true,
        rationale: status === 'confirmed'
          ? 'A successful file-change artifact after the run contract matches this required deliverable.'
          : status === 'weak'
            ? 'A matching file-change artifact exists after the run contract, but it did not succeed.'
            : 'No matching file-change evidence after the run contract was found for this required deliverable.',
        eventIds: contractEvent ? [contractEvent.id] : undefined,
        evidenceIds: matchingEvidence.map((entry) => entry.id),
      })
    }
  } else {
    addEntry('deliverables', 'deliverables.none_declared', {
      value: contract
        ? 'No required file or document artifact was declared by the run contract.'
        : 'No durable run contract declared file or document deliverables.',
      source: contract ? 'run_contract' : 'conservative_default',
      status: contract ? 'confirmed' : 'defaulted',
      confidence: contract ? 0.78 : 0.5,
      reversible: true,
      rationale: contract
        ? 'The run contract did not require a durable artifact.'
        : 'No run contract exists, so deliverable requirements are unknown and conservatively defaulted.',
      eventIds: contractEvent ? [contractEvent.id] : undefined,
    })
  }

  const requiredArtifactSections = contract?.artifactSections?.filter((section) => section.required !== false) ?? []
  for (const [index, section] of requiredArtifactSections.entries()) {
    const artifactPath = section.artifactPath ?? contract?.requiredArtifacts?.[0]?.path
    const matchingEvidence = artifactPath
      ? ledgerArtifactEvidenceForPath(fileChangeArtifactsAfterContract, artifactPath)
      : []
    const artifactContent = artifactPath
      ? readAssertionCandidateFile(session.cwd, artifactPath)
      : null
    const sectionPresent = artifactContent
      ? artifactHasMarkdownSection(artifactContent, section.title)
      : false
    const status: SessionContractLedgerEntryStatus = sectionPresent
      ? 'confirmed'
      : matchingEvidence.some((entry) => entry.status === 'success')
        ? 'weak'
        : 'missing'
    addEntry('deliverables', `artifact_section.${index + 1}`, {
      value: evidenceSnippet(`${artifactPath ?? '(unknown artifact)'} :: ${section.title}${section.description ? ` — ${section.description}` : ''}`, 320),
      source: 'run_contract',
      status,
      confidence: status === 'confirmed' ? 0.9 : status === 'weak' ? 0.48 : 0.12,
      reversible: true,
      rationale: status === 'confirmed'
        ? 'The current artifact file contains a markdown heading matching this required section.'
        : status === 'weak'
          ? 'The artifact was changed, but the current artifact file did not expose the required section heading.'
          : 'No artifact path/content evidence confirmed this required section.',
      eventIds: contractEvent ? [contractEvent.id] : undefined,
      evidenceIds: matchingEvidence.map((entry) => entry.id),
    })
  }

  if (contract?.evidenceRequirements?.length) {
    for (const [index, requirement] of contract.evidenceRequirements.entries()) {
      const supported = evidenceRequirementSupported(requirement, contractEvidenceStats)
        && artifactEvidenceMapSupported(session, contract, requirement, contractEvidenceStats)
        && artifactSelfReviewSupported(session, contract, requirement)
      const details = [
        `${requirement.kind}: ${requirement.description}`,
        typeof requirement.minSourceObservations === 'number'
          ? `minSourceObservations=${requirement.minSourceObservations}`
          : '',
        typeof requirement.minSourceFiles === 'number'
          ? `minSourceFiles=${requirement.minSourceFiles}`
          : '',
        typeof requirement.minSourceScopes === 'number'
          ? `minSourceScopes=${requirement.minSourceScopes}`
          : '',
        requirement.sourceToolNames?.length
          ? `sourceToolNames=${requirement.sourceToolNames.join(',')}`
          : '',
        requirement.requiresArtifactEvidenceMap ? 'requiresArtifactEvidenceMap=true' : '',
        requirement.requiresArtifactSelfReview ? 'requiresArtifactSelfReview=true' : '',
        requirement.requiresSearch ? 'requiresSearch=true' : '',
      ].filter(Boolean).join('; ')
      addEntry('evidence_requirements', `evidence_requirement.${index + 1}`, {
        value: evidenceSnippet(details, 320),
        source: 'run_contract',
        status: supported ? 'confirmed' : 'weak',
        confidence: supported ? 0.82 : 0.36,
        reversible: true,
        rationale: supported
          ? `Observed evidence after the run contract supports this requirement (${contractEvidenceStats.sourceObservations.length} source observation(s), ${contractEvidenceStats.sourceReadPaths.length} source read(s), ${contractEvidenceStats.sourceScopes.length} source scope(s), ${contractEvidenceStats.searchCount} search step(s), ${contractEvidenceStats.validationEvidenceIds.length} validation artifact(s), ${contractEvidenceStats.fileChangeEvidenceIds.length} file change artifact(s)).`
          : `Observed evidence after the run contract is not enough for this requirement (${contractEvidenceStats.sourceObservations.length} source observation(s), ${contractEvidenceStats.sourceReadPaths.length} source read(s), ${contractEvidenceStats.sourceScopes.length} source scope(s), ${contractEvidenceStats.searchCount} search step(s), ${contractEvidenceStats.validationEvidenceIds.length} validation artifact(s), ${contractEvidenceStats.fileChangeEvidenceIds.length} file change artifact(s)).`,
        eventIds: contractEvent ? [contractEvent.id] : undefined,
        evidenceIds: [
          ...contractEvidenceStats.validationEvidenceIds,
          ...contractEvidenceStats.fileChangeEvidenceIds,
        ],
      })
    }
  } else {
    addEntry('evidence_requirements', 'evidence_requirements.none_declared', {
      value: contract
        ? 'No extra source, repository, artifact, or validation evidence requirement was declared.'
        : 'No durable run contract declared evidence requirements.',
      source: contract ? 'run_contract' : 'conservative_default',
      status: contract ? 'confirmed' : 'defaulted',
      confidence: contract ? 0.78 : 0.5,
      reversible: true,
      rationale: contract
        ? 'The run contract did not require additional evidence beyond acceptance criteria.'
        : 'No run contract exists, so evidence requirements are unknown and conservatively defaulted.',
      eventIds: contractEvent ? [contractEvent.id] : undefined,
    })
  }

  if (validationEvidence.length > 0) {
    for (const artifact of validationEvidence) {
      addEntry('verification_plan', artifact.id, {
        value: artifact.label,
        source: 'evidence',
        status: 'confirmed',
        confidence: artifact.status === 'success' ? 0.9 : 0.7,
        reversible: true,
        rationale: artifact.summary,
        eventIds: artifact.sourceEventIds,
        evidenceIds: [artifact.id],
      })
    }
  } else if (evidenceManifest.summary.filesChanged > 0) {
    addEntry('verification_plan', 'verification_plan.safe_default', {
      value: 'Run the narrowest relevant local lint, typecheck, build, or test command for the changed behavior.',
      source: 'conservative_default',
      status: 'defaulted',
      confidence: 0.65,
      reversible: true,
      rationale: 'File changes exist but no validation evidence has been recorded yet.',
      evidenceIds: fileChangeEvidenceIds,
    })
  } else {
    addEntry('verification_plan', 'verification_plan.no_change', {
      value: 'No file-changing work was recorded; mechanical validation can be skipped unless the answer claims executable changes.',
      source: 'evidence',
      status: 'confirmed',
      confidence: 0.78,
      reversible: true,
      rationale: 'The evidence manifest has no file-change artifacts.',
    })
  }

  const runtimeValue = [
    `provider=${session.provider}`,
    `model=${session.model}`,
    `device=${session.device}`,
    session.cwd ? `cwd=${session.cwd}` : 'cwd=(not recorded)',
  ].join(', ')
  addEntry('runtime_context', 'runtime.session', {
    value: runtimeValue,
    source: session.cwd ? 'session_runtime' : 'conservative_default',
    status: session.cwd ? 'confirmed' : 'defaulted',
    confidence: session.cwd ? 0.86 : 0.52,
    reversible: true,
    rationale: session.cwd
      ? 'Session runtime and working directory are recorded.'
      : 'Session runtime is recorded, but repository working directory is missing.',
  })

  if (blockers.length > 0) {
    for (const blocker of blockers) {
      addEntry('blockers', `blocker.${blocker.code}`, {
        value: blocker.message,
        source: 'blocker',
        status: blocker.severity === 'blocker' ? 'blocked' : 'weak',
        confidence: blocker.severity === 'blocker' ? 0.88 : 0.65,
        reversible: true,
        rationale: 'Detected from user-authored session text; confirm authority before executing this part of the request.',
        eventIds: blocker.eventIds,
      })
    }
  }

  const sections = CONTRACT_LEDGER_SECTIONS.map((section) =>
    buildLedgerSection(section, sectionEntries.get(section) ?? []),
  )
  const lastEvent = events.at(-1)
  const lastEventAt = latestTimestamp(session, events)
  const status = contractLedgerStatus(sections, blockers, events)
  const fingerprintPayload = {
    sessionId: session.id,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt,
    status,
    sectionStatuses: sections.map((section) => `${section.name}:${section.status}`),
    blockers: blockers.map((blocker) => blocker.code),
    evidenceRevision: evidenceManifest.revision,
  }
  const fingerprint = stableFingerprint(fingerprintPayload)
  const safeDefaults = sections.flatMap((section) =>
    section.entries.filter((entry) => entry.source === 'conservative_default' && entry.status === 'defaulted'),
  ).length

  return {
    schemaVersion: 1,
    status,
    revision: `contract-ledger-${events.length}-${fingerprint.slice(0, 8)}`,
    fingerprint,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt,
    summary: {
      sections: sections.length,
      confirmedSections: sections.filter((section) => section.status === 'confirmed').length,
      defaultedSections: sections.filter((section) => section.status === 'defaulted').length,
      inferredSections: sections.filter((section) => section.status === 'inferred').length,
      missingSections: sections.filter((section) => section.status === 'missing' || section.status === 'weak').length,
      blockedSections: sections.filter((section) => section.status === 'blocked').length,
      blockers: blockers.length,
      safeDefaults,
    },
    sections,
    blockers,
    lastUpdatedAt: lastEventAt,
  }
}

export function buildSessionEvidenceManifest(
  session: SessionMeta,
  events: SessionEvent[],
): SessionEvidenceManifest {
  const toolCalls = new Map<string, Extract<SessionEvent, { type: 'tool_call' }>>()
  const toolResultsByCall = new Map<string, Array<Extract<SessionEvent, { type: 'tool_result' }>>>()
  const approvalRequests = new Map<string, Extract<SessionEvent, { type: 'approval_request' }>>()
  const artifacts: SessionEvidenceArtifact[] = []
  const risks: SessionEvidenceRisk[] = []
  const indexes = eventIndexById(events)
  const contractEvent = latestRunContractEvent(events)
  const contractIndex = contractEvent ? indexes.get(contractEvent.id) ?? -1 : -1
  const contract = contractEvent?.contract ?? null
  const acceptanceCriterionIds = contract?.acceptanceCriteria.map((criterion) => criterion.id) ?? []
  const completionBoardEvent = latestMatchingCompletionBoard(events, contract, contractIndex)
  const canonicalCriterionIds = new Map(
    acceptanceCriterionIds.map((id) => [id.toUpperCase(), id]),
  )
  const completionVerdicts = new Map<string, {
    verdict: 'met' | 'unmet' | 'not_applicable'
    evidenceToolCallIds: string[]
  }>()
  for (const candidate of completionBoardEvent?.board.completion?.criterionVerdicts ?? []) {
    const id = canonicalCriterionIds.get(candidate.id.toUpperCase())
    if (id) {
      completionVerdicts.set(id, {
        verdict: candidate.verdict,
        evidenceToolCallIds: [...new Set(
          (candidate.evidenceToolCallIds ?? [])
            .filter((toolCallId) => /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(toolCallId)),
        )].slice(0, 16),
      })
    }
  }
  const criterionIdsForEvidenceToolCall = (toolCallId: string): string[] =>
    [...completionVerdicts.entries()]
      .filter(([, verdict]) => verdict.evidenceToolCallIds.includes(toolCallId))
      .map(([criterionId]) => criterionId)

  for (const event of events) {
    if (event.type === 'tool_call') {
      toolCalls.set(event.id, event)
    } else if (event.type === 'tool_result') {
      const results = toolResultsByCall.get(event.toolCallId) ?? []
      results.push(event)
      toolResultsByCall.set(event.toolCallId, results)
    } else if (event.type === 'approval_request') {
      approvalRequests.set(event.id, event)
    }
  }

  if (contractEvent) {
    const ids = sourceEventIds(contractEvent)
    artifacts.push({
      id: evidenceArtifactId('run_contract', ids),
      kind: 'run_contract',
      label: 'Run contract',
      summary: evidenceSnippet(contractEvent.contract.summary),
      status: 'info',
      timestamp: contractEvent.timestamp,
      sourceEventIds: ids,
      relatedAcceptanceCriteriaIds: acceptanceCriterionIds,
    })
  }

  let completionVerdictArtifact: SessionEvidenceArtifact | undefined
  if (completionBoardEvent && completionVerdicts.size > 0) {
    const ids = sourceEventIds(completionBoardEvent)
    const gate = completionBoardEvent.board.completion?.gate
    completionVerdictArtifact = {
      id: evidenceArtifactId('completion_verdict', ids),
      kind: 'completion_verdict',
      label: 'Terminal criterion verdicts',
      summary: evidenceSnippet([
        [...completionVerdicts].map(([id, record]) => (
          `${id}:${record.verdict}${record.evidenceToolCallIds.length > 0
            ? `[evidence=${record.evidenceToolCallIds.join(',')}]`
            : ''}`
        )).join(', '),
        gate ? `gate=${gate.decision}${gate.budgetExhausted ? ':budget-exhausted' : ''}` : '',
      ].filter(Boolean).join('; ')),
      status: [...completionVerdicts.values()].some((record) => record.verdict === 'unmet')
        ? 'error'
        : gate?.decision === 'block' || gate?.budgetExhausted
          ? 'warning'
          : 'info',
      timestamp: completionBoardEvent.timestamp,
      sourceEventIds: ids,
      relatedAcceptanceCriteriaIds: [...completionVerdicts.keys()],
    }
    artifacts.push(completionVerdictArtifact)
  }

  // Mechanical evidence belongs to a concrete workspace revision. A failed
  // check followed by a source edit is stale, while a check from before the
  // latest source/config/test edit cannot validate the current revision. Keep
  // all artifacts for audit, but scope pass/fail semantics to results after
  // the latest validation-relevant successful change.
  let validationBaselineIndex = contractIndex
  for (const call of toolCalls.values()) {
    const hasWorkspaceMutation = extractToolCallPaths(call).length > 0
      || toolCallRepresentsProductMutation(call.tool, call.input)
    if (!hasWorkspaceMutation) continue
    const result = toolResultsByCall.get(call.id)?.at(-1)
    if (!result || result.status !== 'success') continue
    validationBaselineIndex = Math.max(
      validationBaselineIndex,
      indexes.get(result.id) ?? -1,
    )
  }

  const validationArtifacts: SessionEvidenceArtifact[] = []
  for (const event of events) {
    if (event.type !== 'tool_result') continue
    const call = toolCalls.get(event.toolCallId)
    if (!call || !isValidationCommand(call.tool, call.input)) continue
    const ids = sourceEventIds(call, event)
    const status = validationArtifactStatus(call, event)
    const isCurrentRevision = (indexes.get(event.id) ?? -1) > validationBaselineIndex
    const explicitlyLinkedCriterionIds = criterionIdsForEvidenceToolCall(call.id)
    const relatedAcceptanceCriteriaIds = contract && isCurrentRevision
      ? explicitlyLinkedCriterionIds.length > 0
        ? explicitlyLinkedCriterionIds
        : acceptanceCriterionIds.length === 1
          ? acceptanceCriterionIds
          : undefined
      : undefined
    const artifact: SessionEvidenceArtifact = {
      id: evidenceArtifactId('validation', ids),
      kind: 'validation',
      label: commandFromToolInput(call.input) ?? call.tool,
      summary: summarizeValidationArtifact(call, event, status),
      status,
      timestamp: event.timestamp,
      sourceEventIds: ids,
      toolCallId: call.id,
      tool: call.tool,
      hash: stableDigest(event.output),
      currentRevision: isCurrentRevision,
      relatedAcceptanceCriteriaIds,
    }
    validationArtifacts.push(artifact)
    artifacts.push(artifact)
  }

  const observationArtifacts: SessionEvidenceArtifact[] = []
  for (const event of events) {
    if (event.type !== 'tool_result') continue
    const call = toolCalls.get(event.toolCallId)
    if (
      !call
      || (indexes.get(call.id) ?? -1) <= contractIndex
      || (indexes.get(event.id) ?? -1) <= contractIndex
      || !isReadOnlyObservationResult(contract, call, event)
    ) continue
    const ids = sourceEventIds(call, event)
    const command = commandFromToolInput(call.input) ?? call.tool
    const artifact: SessionEvidenceArtifact = {
      id: evidenceArtifactId('observation', ids),
      kind: 'observation',
      label: command,
      summary: evidenceSnippet(`${command} -> ${event.status}: ${event.output}`),
      status: event.status === 'success' ? 'success' : 'error',
      timestamp: event.timestamp,
      sourceEventIds: ids,
      toolCallId: call.id,
      tool: call.tool,
      hash: stableDigest(event.output),
      relatedAcceptanceCriteriaIds: acceptanceCriterionIds,
    }
    observationArtifacts.push(artifact)
    artifacts.push(artifact)
  }

  const actionReceiptArtifacts: SessionEvidenceArtifact[] = []
  for (const event of events) {
    if (event.type !== 'tool_result') continue
    const call = toolCalls.get(event.toolCallId)
    if (
      !call
      || (indexes.get(call.id) ?? -1) <= contractIndex
      || (indexes.get(event.id) ?? -1) <= contractIndex
      || !isExternalActionReceiptResult(event)
    ) continue
    const ids = sourceEventIds(call, event)
    const artifact: SessionEvidenceArtifact = {
      id: evidenceArtifactId('action_receipt', ids),
      kind: 'action_receipt',
      label: call.tool,
      summary: evidenceSnippet(
        `Executor-confirmed tool-boundary success; downstream provider finality is not established: ${event.output}`,
      ),
      status: 'success',
      timestamp: event.timestamp,
      sourceEventIds: ids,
      toolCallId: call.id,
      tool: call.tool,
      hash: stableDigest(event.output),
      relatedAcceptanceCriteriaIds: acceptanceCriterionIds,
    }
    actionReceiptArtifacts.push(artifact)
    artifacts.push(artifact)
  }

  const fileChangeArtifacts: SessionEvidenceArtifact[] = []
  for (const call of toolCalls.values()) {
    const paths = extractToolCallPaths(call)
    if (paths.length === 0) continue
    const result = toolResultsByCall.get(call.id)?.at(-1)
    for (const change of paths) {
      const ids = sourceEventIds(call, result)
      const artifact: SessionEvidenceArtifact = {
        id: evidenceArtifactId('file_change', [...ids, change.path]),
        kind: 'file_change',
        label: change.path,
        summary: `${change.kind} via ${call.tool}`,
        status: artifactStatusFromToolResult(result),
        timestamp: result?.timestamp ?? call.timestamp,
        sourceEventIds: ids,
        toolCallId: call.id,
        tool: call.tool,
        path: change.path,
      }
      fileChangeArtifacts.push(artifact)
      artifacts.push(artifact)
    }
  }

  const artifactReadBackArtifacts: SessionEvidenceArtifact[] = []
  for (const artifact of contract?.requiredArtifacts ?? []) {
    if (!(artifact.kind === 'document' || isDocumentArtifactPath(artifact.path))) continue
    const matchingWrites = fileChangeArtifacts.filter((candidate) => (
      candidate.status === 'success'
      && candidate.path
      && ledgerPathsReferToSameFile(candidate.path, artifact.path)
      && candidate.sourceEventIds.some((id) => (indexes.get(id) ?? -1) > contractIndex)
    ))
    const latestWriteIndex = matchingWrites.reduce((latest, candidate) => Math.max(
      latest,
      ...candidate.sourceEventIds.map((id) => indexes.get(id) ?? -1),
    ), -1)
    if (latestWriteIndex < 0) continue

    const readBack = events.flatMap((event): Array<{
      call: Extract<SessionEvent, { type: 'tool_call' }>
      result: Extract<SessionEvent, { type: 'tool_result' }>
      index: number
    }> => {
      if (
        event.type !== 'tool_result'
        || event.status !== 'success'
        || !event.output.trim()
        || event.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] !== true
      ) return []
      const call = toolCalls.get(event.toolCallId)
      const readPath = typeof call?.input.path === 'string'
        ? call.input.path
        : typeof call?.input.file === 'string'
          ? call.input.file
          : ''
      const index = indexes.get(event.id) ?? -1
      return call?.tool === 'fs.read'
        && index > latestWriteIndex
        && ledgerPathsReferToSameFile(readPath, artifact.path)
        ? [{ call, result: event, index }]
        : []
    }).sort((left, right) => left.index - right.index).at(-1)
    if (!readBack) continue

    const ids = sourceEventIds(readBack.call, readBack.result)
    const readBackArtifact: SessionEvidenceArtifact = {
      id: evidenceArtifactId('artifact_readback', ids),
      kind: 'artifact_readback',
      label: artifact.path,
      summary: `Post-write read-back via ${readBack.call.tool}`,
      status: 'success',
      timestamp: readBack.result.timestamp,
      sourceEventIds: ids,
      toolCallId: readBack.call.id,
      tool: readBack.call.tool,
      path: artifact.path,
      hash: stableDigest(readBack.result.output),
      currentRevision: true,
    }
    artifactReadBackArtifacts.push(readBackArtifact)
    artifacts.push(readBackArtifact)
  }

  for (const event of events) {
    if (event.type === 'approval_response') {
      const request = approvalRequests.get(event.requestId)
      const ids = sourceEventIds(request, event)
      artifacts.push({
        id: evidenceArtifactId('approval', ids),
        kind: 'approval',
        label: request?.tool ?? 'approval',
        summary: evidenceSnippet(
          `${event.decision}${event.note ? `: ${event.note}` : ''}`,
        ),
        status: event.decision === 'approved'
          ? 'success'
          : event.decision === 'feedback'
            ? 'warning'
            : 'error',
        timestamp: event.timestamp,
        sourceEventIds: ids,
        toolCallId: request?.toolCallId,
        tool: request?.tool,
      })
    } else if (event.type === 'auto_approval') {
      const ids = sourceEventIds(event)
      artifacts.push({
        id: evidenceArtifactId('approval', ids),
        kind: 'approval',
        label: event.tool,
        summary: evidenceSnippet(`auto-${event.decision} via ${event.scope} rule '${event.rule.pattern}'`),
        status: event.decision === 'approved' ? 'success' : 'error',
        timestamp: event.timestamp,
        sourceEventIds: ids,
        toolCallId: event.toolCallId,
        tool: event.tool,
      })
    } else if (event.type === 'post_edit_findings') {
      const ids = sourceEventIds(event)
      const diagnostics = event.diagnostics.length
      artifacts.push({
        id: evidenceArtifactId('post_edit_findings', ids),
        kind: 'post_edit_findings',
        label: 'Post-edit findings',
        summary: evidenceSnippet(
          `${event.editedFiles.length} edited file(s), ${diagnostics} diagnostic(s), ${event.reverseCallers.length} likely caller(s)`,
        ),
        status: diagnostics > 0 ? 'warning' : 'info',
        timestamp: event.timestamp,
        sourceEventIds: ids,
        relatedAcceptanceCriteriaIds: contract && (indexes.get(event.id) ?? -1) > contractIndex
          ? acceptanceCriterionIds
          : undefined,
      })
    } else if (event.type === 'session_end') {
      const ids = sourceEventIds(event)
      artifacts.push({
        id: evidenceArtifactId('session_end', ids),
        kind: 'session_end',
        label: 'Session ended',
        summary: `${event.totalTokens.input + event.totalTokens.output} tokens, $${event.totalCost.toFixed(4)}`,
        status: 'success',
        timestamp: event.timestamp,
        sourceEventIds: ids,
      })
    }
  }

  const latestTodo = events.findLast(
    (event): event is Extract<SessionEvent, { type: 'todo_list' }> =>
      event.type === 'todo_list',
  )
  if (latestTodo) {
    const incomplete = latestTodo.items.filter(
      (item) => item.status !== 'completed' && item.status !== 'cancelled',
    ).length
    const ids = sourceEventIds(latestTodo)
    artifacts.push({
      id: evidenceArtifactId('todo', ids),
      kind: 'todo',
      label: 'Todo snapshot',
      summary: summarizeTodoArtifact(latestTodo),
      status: incomplete > 0 ? 'warning' : 'success',
      timestamp: latestTodo.timestamp,
      sourceEventIds: ids,
    })
  }

  const latestAssistant = events.findLast(
    (event): event is Extract<SessionEvent, { type: 'assistant_message' }> =>
      event.type === 'assistant_message',
  )
  if (latestAssistant) {
    const ids = sourceEventIds(latestAssistant)
    const markedUnverified = assistantContentMarkedUnverified(latestAssistant.content)
    artifacts.push({
      id: evidenceArtifactId('assistant_claim', ids),
      kind: 'assistant_claim',
      label: 'Latest assistant message',
      summary: evidenceSnippet(latestAssistant.content),
      status: markedUnverified ? 'warning' : 'info',
      timestamp: latestAssistant.timestamp,
      sourceEventIds: ids,
      relatedAcceptanceCriteriaIds: contract && (indexes.get(latestAssistant.id) ?? -1) > contractIndex
        ? acceptanceCriterionIds
        : undefined,
    })
  }

  const pendingToolCallIds = [...toolCalls.keys()].filter(
    (toolCallId) => !toolResultsByCall.has(toolCallId),
  )
  const failedToolResults = events.filter(
    (event) => event.type === 'tool_result' && event.status !== 'success',
  ) as Array<Extract<SessionEvent, { type: 'tool_result' }>>
  const validationsAfterContract = validationArtifacts.filter(
    (artifact) => artifact.sourceEventIds.some(
      (id) => (indexes.get(id) ?? -1) > validationBaselineIndex,
    ),
  )
  const validationFailuresAfterContract = validationsAfterContract.filter(
    (artifact) => artifact.status === 'error',
  )
  const evidenceScope = validationBaselineIndex > contractIndex
    ? 'after the latest validation-relevant workspace change'
    : contract
      ? 'after the run contract'
      : 'in this session'
  const successfulValidationEvidenceIds = validationsAfterContract
    .filter((artifact) => artifact.status === 'success')
    .map((artifact) => artifact.id)
  const failedValidationEvidenceIds = validationFailuresAfterContract.map((artifact) => artifact.id)
  const fileChangesAfterContract = fileChangeArtifacts.filter(
    (artifact) => artifact.sourceEventIds.some((id) => (indexes.get(id) ?? -1) > contractIndex),
  )
  const currentDocumentReadBackEvidence = currentDocumentArtifactReadBackEvidence({
    contract,
    contractIndex,
    indexes,
    fileChanges: fileChangeArtifacts,
    artifactReadBacks: artifactReadBackArtifacts,
  })
  const criterionEvidenceArtifacts = [
    ...validationsAfterContract,
    ...observationArtifacts,
    ...actionReceiptArtifacts,
  ]

  const acceptanceCriteria: SessionAcceptanceEvidence[] = contract
    ? contract.acceptanceCriteria.map((criterion) => {
        const observationEvidenceIds = observationArtifacts
          .filter((artifact) => artifact.relatedAcceptanceCriteriaIds?.includes(criterion.id))
          .map((artifact) => artifact.id)
        const actionReceiptEvidenceIds = actionReceiptArtifacts
          .filter((artifact) => artifact.relatedAcceptanceCriteriaIds?.includes(criterion.id))
          .map((artifact) => artifact.id)
        const completionVerdict = completionVerdicts.get(criterion.id)
        const completionEvidenceIds = completionVerdictArtifact?.relatedAcceptanceCriteriaIds?.includes(criterion.id)
          ? [completionVerdictArtifact.id]
          : []
        const linkedEvidenceArtifacts = completionVerdict?.evidenceToolCallIds
          .map((toolCallId) => criterionEvidenceArtifacts.find((artifact) => (
            artifact.toolCallId === toolCallId
          )))
          .filter((artifact): artifact is SessionEvidenceArtifact => artifact !== undefined)
          ?? []
        const hasCompleteSuccessfulEvidenceLink = completionVerdict?.evidenceToolCallIds.length
          ? linkedEvidenceArtifacts.length === completionVerdict.evidenceToolCallIds.length
            && linkedEvidenceArtifacts.every((artifact) => artifact.status === 'success')
          : false
        const linkedEvidenceIds = linkedEvidenceArtifacts.map((artifact) => artifact.id)
        if (completionVerdict?.verdict === 'unmet') {
          return {
            id: criterion.id,
            text: criterion.text,
            status: 'failed' as const,
            evidenceIds: completionEvidenceIds,
            reason: 'The accepted terminal completion diagnostics explicitly marked this criterion unmet.',
          }
        }
        if (completionVerdict?.verdict === 'met' && hasCompleteSuccessfulEvidenceLink) {
          return {
            id: criterion.id,
            text: criterion.text,
            status: 'supported' as const,
            evidenceIds: [...completionEvidenceIds, ...linkedEvidenceIds],
            reason: 'Supported by exact successful validation, read-only observation, or external-action receipt tool calls linked from the accepted terminal criterion verdict.',
          }
        }
        if (!completionVerdict && acceptanceCriterionIds.length === 1) {
          if (validationFailuresAfterContract.length > 0) {
            return {
              id: criterion.id,
              text: criterion.text,
              status: 'failed' as const,
              evidenceIds: failedValidationEvidenceIds,
              reason: 'The validation check for the sole acceptance criterion failed on the current workspace revision.',
            }
          }
          if (successfulValidationEvidenceIds.length > 0) {
            return {
              id: criterion.id,
              text: criterion.text,
              status: 'supported' as const,
              evidenceIds: successfulValidationEvidenceIds,
              reason: 'The sole acceptance criterion is supported by current successful validation evidence.',
            }
          }
        }
        const candidateExecutionEvidenceIds = completionVerdict?.evidenceToolCallIds.length
          ? linkedEvidenceIds
          : [...observationEvidenceIds, ...actionReceiptEvidenceIds]
        const candidateEvidenceIds = [...completionEvidenceIds, ...candidateExecutionEvidenceIds]
        return {
          id: criterion.id,
          text: criterion.text,
          status: pendingToolCallIds.length > 0 ? 'blocked' as const : 'unverified' as const,
          evidenceIds: candidateEvidenceIds,
          reason: pendingToolCallIds.length > 0
            ? 'One or more tool calls have not produced a result yet.'
            : completionVerdict?.verdict === 'met' && completionVerdict.evidenceToolCallIds.length > 0
              ? 'The terminal diagnostics marked this criterion met, but one or more linked tool calls were not current successful validation, read-only observation, or external-action receipt evidence for the active contract.'
              : completionVerdict?.verdict === 'met' && candidateExecutionEvidenceIds.length > 0
                ? 'The terminal diagnostics marked this criterion met and executor evidence exists, but no criterion-scoped tool evidence link was recorded.'
                : completionVerdict?.verdict === 'met'
                  ? 'The terminal diagnostics marked this criterion met, but no criterion-scoped tool evidence was recorded.'
                  : completionVerdict?.verdict === 'not_applicable'
                  ? 'The terminal diagnostics marked this criterion not applicable; applicability was not independently verified.'
                  : candidateExecutionEvidenceIds.length > 0
                    ? 'Read-only observation or external-action receipt evidence was recorded, but no criterion-level validation verdict was recorded.'
              : successfulValidationEvidenceIds.length > 0 || failedValidationEvidenceIds.length > 0
                ? 'Validation evidence exists for the run, but no criterion-scoped evidence link was recorded.'
                : 'No validation evidence was recorded after the run contract.',
        }
      })
    : []

  if (!contract && events.length > 0) {
    risks.push({
      code: 'no_run_contract',
      severity: 'info',
      message: 'No run contract was recorded for this session.',
    })
  }
  if (contract && acceptanceCriteria.some((criterion) => criterion.status === 'unverified')) {
    const evidenceIds = acceptanceCriteria
      .filter((criterion) => criterion.status === 'unverified')
      .flatMap((criterion) => criterion.evidenceIds)
    risks.push({
      code: 'acceptance_criteria_unverified',
      severity: 'warning',
      message: 'One or more acceptance criteria have no validation evidence after the run contract.',
      ...(evidenceIds.length > 0 ? { evidenceIds: [...new Set(evidenceIds)] } : {}),
    })
  }
  if (validationFailuresAfterContract.length > 0) {
    risks.push({
      code: 'validation_failed',
      severity: 'warning',
      message: `${validationFailuresAfterContract.length} validation run(s) failed ${evidenceScope}.`,
      evidenceIds: failedValidationEvidenceIds,
    })
  }
  if (
    fileChangesAfterContract.length > 0
    && validationsAfterContract.length === 0
    && currentDocumentReadBackEvidence.length === 0
  ) {
    risks.push({
      code: 'changes_without_validation',
      severity: 'warning',
      message: `${fileChangesAfterContract.length} file change artifact(s) were recorded ${evidenceScope} without validation evidence.`,
      evidenceIds: fileChangesAfterContract.map((artifact) => artifact.id),
    })
  }
  if (pendingToolCallIds.length > 0) {
    risks.push({
      code: 'pending_tool_results',
      severity: 'info',
      message: `${pendingToolCallIds.length} tool call(s) have no recorded result yet.`,
    })
  }
  if (failedToolResults.length > 0) {
    risks.push({
      code: 'failed_tool_results',
      severity: 'warning',
      message: `${failedToolResults.length} tool result(s) ended in error, timeout, or cancellation.`,
      evidenceIds: artifacts
        .filter((artifact) => artifact.status === 'error')
        .map((artifact) => artifact.id),
    })
  }
  const postEditWarnings = artifacts.filter(
    (artifact) => artifact.kind === 'post_edit_findings' && artifact.status === 'warning',
  )
  if (postEditWarnings.length > 0) {
    risks.push({
      code: 'post_edit_findings',
      severity: 'warning',
      message: 'Post-edit findings recorded diagnostics that may need review.',
      evidenceIds: postEditWarnings.map((artifact) => artifact.id),
    })
  }
  const assistantWarning = artifacts.find(
    (artifact) => artifact.kind === 'assistant_claim' && artifact.status === 'warning',
  )
  if (assistantWarning) {
    risks.push({
      code: 'assistant_marked_unverified',
      severity: 'warning',
      message: 'The latest assistant message explicitly marked the result as unverified or incomplete.',
      evidenceIds: [assistantWarning.id],
    })
  }

  artifacts.sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id))

  const supportedAcceptanceCriteria = acceptanceCriteria.filter((criterion) => criterion.status === 'supported').length
  const failedAcceptanceCriteria = acceptanceCriteria.filter((criterion) => criterion.status === 'failed').length
  const blockedAcceptanceCriteria = acceptanceCriteria.filter((criterion) => criterion.status === 'blocked').length
  const unverifiedAcceptanceCriteria = acceptanceCriteria.filter((criterion) => criterion.status === 'unverified').length
  const warningRiskCount = risks.filter((risk) => risk.severity === 'warning').length
  const lastEvent = events.at(-1)
  const lastEventAt = latestTimestamp(session, events)
  const fingerprintPayload = {
    sessionId: session.id,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt,
    artifactIds: artifacts.map((artifact) => artifact.id),
    artifactStatuses: artifacts.map((artifact) => artifact.status),
    acceptanceStatuses: acceptanceCriteria.map((criterion) => `${criterion.id}:${criterion.status}`),
    riskCodes: risks.map((risk) => risk.code),
  }
  const fingerprint = stableFingerprint(fingerprintPayload)
  const status: SessionEvidenceManifestStatus = events.length === 0
    ? 'empty'
    : warningRiskCount > 0
      ? 'attention_needed'
      : artifacts.length > 0
        ? 'ready'
        : 'collecting'

  return {
    schemaVersion: 1,
    status,
    revision: `evidence-${events.length}-${fingerprint.slice(0, 8)}`,
    fingerprint,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt,
    summary: {
      artifacts: artifacts.length,
      toolCalls: toolCalls.size,
      toolResults: events.filter((event) => event.type === 'tool_result').length,
      validationRuns: validationsAfterContract.length,
      validationFailures: validationFailuresAfterContract.length,
      filesChanged: fileChangeArtifacts.length,
      approvals: artifacts.filter((artifact) => artifact.kind === 'approval').length,
      acceptanceCriteria: acceptanceCriteria.length,
      supportedAcceptanceCriteria,
      failedAcceptanceCriteria,
      blockedAcceptanceCriteria,
      unverifiedAcceptanceCriteria,
      risks: risks.length,
    },
    acceptanceCriteria,
    artifacts,
    risks,
    lastUpdatedAt: lastEventAt,
  }
}

function buildSessionEvaluationArtifactBundle(
  session: SessionMeta,
  events: SessionEvent[],
  evidenceManifest: SessionEvidenceManifest,
): SessionEvaluationArtifactBundle {
  const sessionCwd = session.cwd?.trim()
  const cwd = sessionCwd ? resolve(sessionCwd) : undefined
  const fileArtifacts = evidenceManifest.artifacts.filter((artifact) => artifact.kind === 'file_change')
  const validationEvidenceIds = evidenceManifest.artifacts
    .filter((artifact) => artifact.kind === 'validation')
    .map((artifact) => artifact.id)
  const acceptanceEvidenceIds = [
    ...new Set(evidenceManifest.acceptanceCriteria.flatMap((criterion) => criterion.evidenceIds)),
  ]
  const files: SessionEvaluationArtifactFile[] = []
  const skipped: SessionEvaluationArtifactSkip[] = []
  const seenPaths = new Set<string>()
  let totalBytes = 0

  for (const artifact of fileArtifacts) {
    const resolvedPath = safeEvaluationArtifactPath(artifact.path, cwd)
    if (resolvedPath.skip) {
      skipped.push({
        ...resolvedPath.skip,
        artifactId: artifact.id,
        sourceEventIds: artifact.sourceEventIds,
      })
      continue
    }
    const path = resolvedPath.path
    if (!path) {
      skipped.push(artifactSkip('missing_path', 'File-change evidence did not include a path.', artifact))
      continue
    }
    if (seenPaths.has(path)) {
      skipped.push(artifactSkip('duplicate_path', 'Duplicate file-change path was already included.', artifact, path))
      continue
    }
    if (files.length >= EVALUATION_BUNDLE_MAX_FILES) {
      skipped.push(artifactSkip('max_files_exceeded', 'Evaluation bundle file count limit was reached.', artifact, path))
      continue
    }
    seenPaths.add(path)

    const baseFile = {
      id: `bundle-file-${stableDigest(`${artifact.id}|${path}`)}`,
      path,
      artifactId: artifact.id,
      status: artifact.status,
      operation: artifactOperation(artifact),
      sourceEventIds: artifact.sourceEventIds,
      tool: artifact.tool,
    }

    if (!resolvedPath.absolutePath) {
      files.push({
        ...baseFile,
        contentState: 'metadata_only',
      })
      continue
    }

    if (!existsSync(resolvedPath.absolutePath)) {
      skipped.push(artifactSkip('file_missing', 'Changed file is no longer present on disk.', artifact, path))
      continue
    }

    let sizeBytes: number
    try {
      const stats = statSync(resolvedPath.absolutePath)
      if (!stats.isFile()) {
        skipped.push(artifactSkip('not_file', 'Changed path is not a regular file.', artifact, path))
        continue
      }
      sizeBytes = stats.size
    } catch {
      skipped.push(artifactSkip('read_error', 'Changed file metadata could not be read.', artifact, path))
      continue
    }

    if (sizeBytes > EVALUATION_BUNDLE_MAX_FILE_BYTES) {
      skipped.push(artifactSkip('file_too_large', 'Changed file exceeds the per-file evaluation bundle budget.', artifact, path))
      continue
    }
    if (totalBytes + sizeBytes > EVALUATION_BUNDLE_MAX_TOTAL_BYTES) {
      skipped.push(artifactSkip('total_budget_exceeded', 'Changed file exceeds the total evaluation bundle budget.', artifact, path))
      continue
    }

    try {
      const content = readFileSync(resolvedPath.absolutePath)
      totalBytes += content.byteLength
      files.push({
        ...baseFile,
        contentState: 'hashed',
        sizeBytes: content.byteLength,
        contentHash: stableDigest(content),
      })
    } catch {
      skipped.push(artifactSkip('read_error', 'Changed file content could not be read.', artifact, path))
    }
  }

  const hashedFiles = files.filter((file) => file.contentState === 'hashed').length
  const metadataOnlyFiles = files.filter((file) => file.contentState === 'metadata_only').length
  const materialSkippedFiles = skipped.filter((item) => item.reason !== 'duplicate_path').length
  const status: SessionEvaluationArtifactBundleStatus = files.length === 0
    && materialSkippedFiles === 0
    && validationEvidenceIds.length === 0
    && acceptanceEvidenceIds.length === 0
    ? 'empty'
    : files.length > 0 && hashedFiles === 0 && metadataOnlyFiles === files.length && materialSkippedFiles === 0
      ? 'metadata_only'
      : materialSkippedFiles > 0 || metadataOnlyFiles > 0
        ? 'partial'
        : 'ready'
  const lastEvent = events.at(-1)
  const lastEventAt = latestTimestamp(session, events)
  const fingerprintPayload = {
    sessionId: session.id,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt,
    cwd,
    files: files.map((file) => ({
      path: file.path,
      state: file.contentState,
      hash: file.contentHash,
      size: file.sizeBytes,
      artifactId: file.artifactId,
    })),
    skipped: skipped.map((item) => `${item.reason}:${item.path ?? item.artifactId ?? ''}`),
    validationEvidenceIds,
    acceptanceEvidenceIds,
  }
  const fingerprint = stableFingerprint(fingerprintPayload)

  return {
    schemaVersion: 1,
    status,
    revision: `artifact-bundle-${events.length}-${fingerprint.slice(0, 8)}`,
    fingerprint,
    eventCount: events.length,
    cwd,
    limits: {
      maxFiles: EVALUATION_BUNDLE_MAX_FILES,
      maxFileBytes: EVALUATION_BUNDLE_MAX_FILE_BYTES,
      maxTotalBytes: EVALUATION_BUNDLE_MAX_TOTAL_BYTES,
    },
    summary: {
      files: files.length,
      hashedFiles,
      metadataOnlyFiles,
      skippedFiles: skipped.length,
      totalBytes,
      validationArtifacts: validationEvidenceIds.length,
      acceptanceCriteria: evidenceManifest.summary.acceptanceCriteria,
    },
    files,
    skipped,
    validationEvidenceIds,
    acceptanceEvidenceIds,
    lastUpdatedAt: lastEventAt,
  }
}

const ASSERTION_SCAN_MAX_FILES = 240
const ASSERTION_SCAN_MAX_DIRS = 180
const ASSERTION_PATTERN_MAX_LENGTH = 220

const ASSERTION_SOURCE_EXTENSIONS = new Set([
  '.c',
  '.cc',
  '.cpp',
  '.cs',
  '.css',
  '.go',
  '.h',
  '.hpp',
  '.html',
  '.java',
  '.js',
  '.json',
  '.jsx',
  '.kt',
  '.mjs',
  '.py',
  '.rs',
  '.scss',
  '.sh',
  '.sql',
  '.swift',
  '.toml',
  '.ts',
  '.tsx',
  '.yaml',
  '.yml',
  '.md',
])

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function fileExtension(path: string): string {
  const basename = normalizeArtifactPath(path).split('/').at(-1) ?? path
  const index = basename.lastIndexOf('.')
  return index >= 0 ? basename.slice(index).toLowerCase() : ''
}

function isSourceLikePath(path: string): boolean {
  return ASSERTION_SOURCE_EXTENSIONS.has(fileExtension(path))
}

function hasGlobMagic(value: string): boolean {
  return /[*?[\]{}]/.test(value)
}

function globPatternToRegExp(pattern: string): RegExp | null {
  const normalized = normalizeArtifactPath(pattern.trim())
  if (!normalized || normalized.length > ASSERTION_PATTERN_MAX_LENGTH) return null
  const escaped = escapeRegExp(normalized)
    .replaceAll('\\*\\*', '.*')
    .replaceAll('\\*', '[^/]*')
    .replaceAll('\\?', '.')
  try {
    return new RegExp(`^${escaped}$`, 'i')
  } catch {
    return null
  }
}

function matchesFileHint(path: string, hint: string): boolean {
  const regex = globPatternToRegExp(hint)
  if (!regex) return false
  const normalized = normalizeArtifactPath(path)
  const basename = normalized.split('/').at(-1) ?? normalized
  return regex.test(normalized) || regex.test(basename)
}

function normalizeAssertionPath(raw: string): string | null {
  const cleaned = raw
    .trim()
    .replace(/^['"`(<]+|['"`),.>]+$/g, '')
  if (!cleaned || cleaned.includes('\0') || hasPathTraversal(cleaned)) return null
  return normalizeArtifactPath(cleaned)
}

function resolveProjectFilePath(cwd: string | undefined, path: string): string | null {
  if (!cwd) return null
  const safe = safeEvaluationArtifactPath(path, cwd)
  return safe.absolutePath ?? null
}

function readAssertionCandidateFile(
  cwd: string | undefined,
  path: string,
): string | null {
  const absolutePath = resolveProjectFilePath(cwd, path)
  if (!absolutePath || !existsSync(absolutePath)) return null
  try {
    const stats = statSync(absolutePath)
    if (!stats.isFile() || stats.size > EVALUATION_BUNDLE_MAX_FILE_BYTES) return null
    return readFileSync(absolutePath, 'utf8')
  } catch {
    return null
  }
}

function projectFileExists(cwd: string | undefined, path: string): boolean | null {
  const absolutePath = resolveProjectFilePath(cwd, path)
  if (!absolutePath) return null
  try {
    return existsSync(absolutePath) && statSync(absolutePath).isFile()
  } catch {
    return null
  }
}

function collectProjectSourceFiles(cwd: string | undefined): string[] {
  if (!cwd) return []
  const root = resolve(cwd)
  const files: string[] = []
  const stack = ['']
  let visitedDirs = 0

  while (stack.length > 0 && files.length < ASSERTION_SCAN_MAX_FILES && visitedDirs < ASSERTION_SCAN_MAX_DIRS) {
    const current = stack.pop() ?? ''
    const absoluteDir = current ? resolve(root, current) : root
    visitedDirs += 1
    let entries: Dirent[]
    try {
      entries = readdirSync(absoluteDir, { withFileTypes: true })
        .sort((left, right) => left.name.localeCompare(right.name))
    } catch {
      continue
    }

    for (const entry of entries) {
      const path = current ? `${current}/${entry.name}` : entry.name
      if (isGeneratedOrVendorPath(path)) continue
      if (entry.isDirectory()) {
        stack.push(path)
      } else if (entry.isFile() && isSourceLikePath(path)) {
        files.push(normalizeArtifactPath(path))
        if (files.length >= ASSERTION_SCAN_MAX_FILES) break
      }
    }
  }

  return files
}

function extractBacktickTokens(text: string): string[] {
  return [...text.matchAll(/`([^`]+)`/g)]
    .map((match) => match[1]?.trim())
    .filter((value): value is string => Boolean(value))
}

function extractPathTokens(text: string): string[] {
  const extensions = [...ASSERTION_SOURCE_EXTENSIONS]
    .map((extension) => extension.slice(1))
    .join('|')
  const pattern = new RegExp(`(?:^|[\\s"'(])([A-Za-z0-9_@./\\\\-]+\\.(?:${extensions}))(?:$|[\\s"',).:])`, 'gi')
  const paths = new Set<string>()
  for (const token of extractBacktickTokens(text)) {
    const normalized = normalizeAssertionPath(token)
    if (normalized && isSourceLikePath(normalized)) paths.add(normalized)
  }
  for (const match of text.matchAll(pattern)) {
    const normalized = normalizeAssertionPath(match[1] ?? '')
    // A bare dotted token such as a runtime, framework, or protocol name is
    // syntactically indistinguishable from a basename-only source file.
    // Require explicit path syntax for unquoted prose; backticks above remain
    // the unambiguous way to assert a basename-only artifact like package.json.
    if (normalized?.includes('/') && isSourceLikePath(normalized)) {
      paths.add(normalized)
    }
  }
  return [...paths]
}

function isLikelyIdentifier(value: string): boolean {
  return /^[A-Za-z_$][\w$.-]{2,}$/.test(value)
    && !value.includes('/')
    && /[A-Z_$.-]/.test(value)
}

function isSubjectiveCriterion(text: string): boolean {
  return /\b(clean|intuitive|beautiful|polished|nice|simple|easy to use|user friendly|readable)\b|깔끔|자연스|보기\s*좋|사용성|가독성/i.test(text)
}

function isBehavioralCriterion(text: string): boolean {
  return /\b(test|tests|pass|passes|lint|typecheck|build|compile|render|renders|return|returns|handle|handles|reject|rejects|persist|persists|work|works|validate|verified)\b|테스트|검증|빌드|동작|렌더|처리/i.test(text)
}

function supportsRepositoryStructuralAssertions(
  contract: AgentRunContract,
  criterionText: string,
): boolean {
  if (extractPathTokens(criterionText).length > 0) return true
  if (contract.evidenceRequirements?.some((requirement) =>
    requirement.kind === 'source' || requirement.kind === 'repository'
  )) return true

  const intent = contract.executionIntent
  if (!intent) return true
  return intent.kind === 'workspace-change' || intent.workspaceMutation === 'required'
}

function assertion(
  acceptanceCriterionId: string,
  index: number,
  input: Omit<SessionAcceptanceAssertion, 'id' | 'acceptanceCriterionId'>,
): SessionAcceptanceAssertion {
  return {
    id: `assertion-${stableDigest(`${acceptanceCriterionId}|${index}|${input.kind}|${input.description}`)}`,
    acceptanceCriterionId,
    ...input,
  }
}

function extractAcceptanceAssertions(
  contract: AgentRunContract | null,
): Map<string, SessionAcceptanceAssertion[]> {
  const byCriterion = new Map<string, SessionAcceptanceAssertion[]>()
  if (!contract) return byCriterion

  for (const criterion of contract.acceptanceCriteria) {
    const assertions: SessionAcceptanceAssertion[] = []
    const text = criterion.text
    const sourceStructuralEvidenceAllowed = supportsRepositoryStructuralAssertions(contract, text)
    let index = 0
    const add = (input: Omit<SessionAcceptanceAssertion, 'id' | 'acceptanceCriterionId'>) => {
      index += 1
      assertions.push(assertion(criterion.id, index, input))
    }

    for (const path of extractPathTokens(text)) {
      add({
        tier: 'structural',
        kind: 'file_exists',
        description: `${path} exists in the project.`,
        fileHint: path,
        confidence: 0.86,
      })
    }

    if (sourceStructuralEvidenceAllowed) {
      for (const match of text.matchAll(/\b([A-Z][A-Z0-9_]{2,})\s*(?:=|:)\s*([A-Za-z0-9_.-]+)\b/g)) {
        const name = match[1] ?? ''
        const value = match[2] ?? ''
        add({
          tier: 'constant',
          kind: 'text_match',
          description: `${name} is set to ${value}.`,
          pattern: `${escapeRegExp(name)}\\s*(?:=|:)\\s*["']?${escapeRegExp(value)}["']?`,
          expectedValue: value,
          confidence: 0.78,
        })
      }
    }

    if (sourceStructuralEvidenceAllowed) {
      for (const match of text.matchAll(/\b(?:interface|class|function|type|enum|schema|component)\s+([A-Za-z_$][\w$.-]*)/gi)) {
        const name = match[1] ?? ''
        add({
          tier: 'structural',
          kind: 'symbol_exists',
          description: `${name} is present in source.`,
          pattern: `\\b${escapeRegExp(name)}\\b`,
          expectedValue: name,
          confidence: 0.72,
        })
      }
    }

    if (sourceStructuralEvidenceAllowed) {
      for (const match of text.matchAll(/\/api\/[A-Za-z0-9_./:-]+/g)) {
        const route = match[0]
        add({
          tier: 'structural',
          kind: 'text_match',
          description: `${route} route text is present in source.`,
          pattern: escapeRegExp(route),
          expectedValue: route,
          confidence: 0.7,
        })
      }

      for (const token of extractBacktickTokens(text)) {
        const normalized = normalizeAssertionPath(token)
        if (normalized && isSourceLikePath(normalized)) continue
        if (token.startsWith('/')) {
          add({
            tier: 'structural',
            kind: 'text_match',
            description: `${token} appears in source.`,
            pattern: escapeRegExp(token),
            expectedValue: token,
            confidence: 0.68,
          })
        } else if (isLikelyIdentifier(token)) {
          add({
            tier: 'structural',
            kind: 'symbol_exists',
            description: `${token} is present in source.`,
            pattern: `\\b${escapeRegExp(token)}\\b`,
            expectedValue: token,
            confidence: 0.68,
          })
        }
      }
    }

    if (assertions.length === 0 && isSubjectiveCriterion(text)) {
      add({
        tier: 'subjective',
        kind: 'human_review',
        description: 'Criterion requires human judgment instead of deterministic source scanning.',
        confidence: 0.62,
      })
    }

    if (assertions.length === 0 || isBehavioralCriterion(text)) {
      add({
        tier: 'behavioral',
        kind: 'validation_required',
        description: 'Criterion requires successful validation evidence.',
        confidence: 0.66,
      })
    }

    byCriterion.set(criterion.id, assertions)
  }

  return byCriterion
}

function artifactEvidenceForPath(
  artifactBundle: SessionEvaluationArtifactBundle,
  path: string,
): string[] {
  const normalized = normalizeArtifactPath(path)
  return artifactBundle.files
    .filter((file) => normalizeArtifactPath(file.path) === normalized)
    .map((file) => file.artifactId)
}

function scopedValidationArtifacts(
  evidenceManifest: SessionEvidenceManifest,
  currentRevisionOnly: boolean,
): SessionEvidenceArtifact[] {
  return evidenceManifest.artifacts.filter((artifact) =>
    artifact.kind === 'validation'
    && (
      !currentRevisionOnly
      || artifact.currentRevision === true
    )
  )
}

function criterionScopedValidationArtifacts(
  assertion: SessionAcceptanceAssertion,
  evidenceManifest: SessionEvidenceManifest,
  currentRevisionOnly = false,
): SessionEvidenceArtifact[] {
  return scopedValidationArtifacts(evidenceManifest, currentRevisionOnly)
    .filter((artifact) => artifact.relatedAcceptanceCriteriaIds
      ?.includes(assertion.acceptanceCriterionId) === true)
}

function successfulValidationEvidenceIds(
  evidenceManifest: SessionEvidenceManifest,
  currentRevisionOnly = false,
): string[] {
  return scopedValidationArtifacts(evidenceManifest, currentRevisionOnly)
    .filter((artifact) => artifact.status === 'success')
    .map((artifact) => artifact.id)
}

function failedValidationEvidenceIds(
  evidenceManifest: SessionEvidenceManifest,
  currentRevisionOnly = false,
): string[] {
  return scopedValidationArtifacts(evidenceManifest, currentRevisionOnly)
    .filter((artifact) => artifact.status === 'error')
    .map((artifact) => artifact.id)
}

function executionEvidenceIds(
  assertion: SessionAcceptanceAssertion,
  evidenceManifest: SessionEvidenceManifest,
): string[] {
  return evidenceManifest.artifacts
    .filter((artifact) =>
      (artifact.kind === 'observation' || artifact.kind === 'action_receipt')
      && artifact.relatedAcceptanceCriteriaIds?.includes(assertion.acceptanceCriterionId)
    )
    .map((artifact) => artifact.id)
}

function candidateFilesForAssertion(
  assertion: SessionAcceptanceAssertion,
  session: SessionMeta,
  artifactBundle: SessionEvaluationArtifactBundle,
): string[] {
  const paths = new Set<string>()
  for (const file of artifactBundle.files) {
    if (isSourceLikePath(file.path)) paths.add(normalizeArtifactPath(file.path))
  }

  const hint = assertion.fileHint?.trim()
  if (hint) {
    const normalizedHint = normalizeAssertionPath(hint)
    if (normalizedHint && !hasGlobMagic(normalizedHint)) {
      paths.add(normalizedHint)
    } else if (normalizedHint && session.cwd) {
      for (const path of collectProjectSourceFiles(session.cwd)) {
        if (matchesFileHint(path, normalizedHint)) paths.add(path)
      }
    }
  }

  if (
    paths.size === 0
    && session.cwd
    && assertion.kind !== 'file_exists'
    && assertion.kind !== 'validation_required'
    && assertion.kind !== 'human_review'
  ) {
    for (const path of collectProjectSourceFiles(session.cwd)) {
      paths.add(path)
    }
  }

  return [...paths].sort((left, right) => left.localeCompare(right)).slice(0, ASSERTION_SCAN_MAX_FILES)
}

function compileAssertionPattern(assertion: SessionAcceptanceAssertion): RegExp | null {
  const source = assertion.pattern
    ?? (assertion.expectedValue ? escapeRegExp(assertion.expectedValue) : '')
  if (!source || source.length > ASSERTION_PATTERN_MAX_LENGTH) return null
  try {
    return new RegExp(source, 'i')
  } catch {
    return null
  }
}

function verifyFileAssertion(
  assertion: SessionAcceptanceAssertion,
  session: SessionMeta,
  artifactBundle: SessionEvaluationArtifactBundle,
): SessionAcceptanceAssertionResult {
  const path = assertion.fileHint ? normalizeAssertionPath(assertion.fileHint) : null
  if (!path) {
    return {
      assertion,
      status: 'unverified',
      detail: 'The file assertion did not include a safe project-relative path.',
      evidenceIds: [],
    }
  }

  const evidenceIds = artifactEvidenceForPath(artifactBundle, path)
  if (evidenceIds.length > 0) {
    return {
      assertion,
      status: 'verified',
      detail: `File-change evidence includes ${path}.`,
      evidenceIds,
      path,
    }
  }

  const exists = projectFileExists(session.cwd, path)
  if (exists === true) {
    return {
      assertion,
      status: 'verified',
      detail: `${path} exists on disk.`,
      evidenceIds: [],
      path,
    }
  }
  if (exists === false) {
    return {
      assertion,
      status: 'failed',
      detail: `${path} was not found on disk.`,
      evidenceIds: [],
      path,
    }
  }
  return {
    assertion,
    status: 'unverified',
    detail: 'The session has no cwd, so the file assertion cannot be checked.',
    evidenceIds,
    path,
  }
}

function verifyTextAssertion(
  assertion: SessionAcceptanceAssertion,
  session: SessionMeta,
  artifactBundle: SessionEvaluationArtifactBundle,
): SessionAcceptanceAssertionResult {
  const pattern = compileAssertionPattern(assertion)
  if (!pattern) {
    return {
      assertion,
      status: 'unverified',
      detail: 'The generated assertion pattern was empty or unsafe to compile.',
      evidenceIds: [],
    }
  }

  const candidates = candidateFilesForAssertion(assertion, session, artifactBundle)
  if (candidates.length === 0) {
    return {
      assertion,
      status: 'unverified',
      detail: 'No bounded source-file candidates were available for this assertion.',
      evidenceIds: [],
    }
  }

  let readableFiles = 0
  for (const path of candidates) {
    const basename = path.split('/').at(-1) ?? path
    const nameMatches = assertion.kind === 'symbol_exists' && pattern.test(basename)
    const content = readAssertionCandidateFile(session.cwd, path)
    if (content !== null) readableFiles += 1
    const contentMatch = content?.match(pattern)
    if (nameMatches || contentMatch) {
      return {
        assertion,
        status: 'verified',
        detail: nameMatches
          ? `Matched ${assertion.expectedValue ?? assertion.description} in filename ${basename}.`
          : `Matched ${assertion.expectedValue ?? assertion.description} in ${path}.`,
        evidenceIds: artifactEvidenceForPath(artifactBundle, path),
        path,
        actualValue: contentMatch?.[0] ?? basename,
      }
    }
  }

  if (!session.cwd || readableFiles === 0) {
    return {
      assertion,
      status: 'unverified',
      detail: 'Candidate files were metadata-only or unreadable, so the assertion could not be checked.',
      evidenceIds: candidates.flatMap((path) => artifactEvidenceForPath(artifactBundle, path)),
    }
  }

  return {
    assertion,
    status: 'failed',
    detail: `No candidate source file matched ${assertion.expectedValue ?? assertion.description}.`,
    evidenceIds: [],
  }
}

function verifyValidationAssertion(
  assertion: SessionAcceptanceAssertion,
  evidenceManifest: SessionEvidenceManifest,
  currentRevisionOnly = false,
): SessionAcceptanceAssertionResult {
  const scoped = criterionScopedValidationArtifacts(
    assertion,
    evidenceManifest,
    currentRevisionOnly,
  )
  const successful = scoped
    .filter((artifact) => artifact.status === 'success')
    .map((artifact) => artifact.id)
  if (successful.length > 0) {
    return {
      assertion,
      status: 'verified',
      detail: 'Successful validation evidence supports this behavioral criterion.',
      evidenceIds: successful,
    }
  }

  const failed = scoped
    .filter((artifact) => artifact.status === 'error')
    .map((artifact) => artifact.id)
  if (failed.length > 0) {
    return {
      assertion,
      status: 'failed',
      detail: 'Validation evidence failed for this behavioral criterion.',
      evidenceIds: failed,
    }
  }

  const criterionEvidence = evidenceManifest.acceptanceCriteria?.find(
    (criterion) => criterion.id === assertion.acceptanceCriterionId,
  )
  if (criterionEvidence?.status === 'supported') {
    return {
      assertion,
      status: 'verified',
      detail: 'Exact successful executor evidence was linked to the accepted terminal criterion verdict.',
      evidenceIds: criterionEvidence.evidenceIds,
    }
  }
  if (criterionEvidence?.status === 'failed') {
    return {
      assertion,
      status: 'failed',
      detail: 'The accepted terminal completion diagnostics explicitly marked this criterion unmet.',
      evidenceIds: criterionEvidence.evidenceIds,
    }
  }

  const executionEvidence = executionEvidenceIds(assertion, evidenceManifest)
  if (executionEvidence.length > 0) {
    return {
      assertion,
      status: 'unverified',
      detail: 'Read-only observation or external-action receipt evidence exists, but no criterion-level validation verdict was recorded.',
      evidenceIds: executionEvidence,
    }
  }

  return {
    assertion,
    status: 'unverified',
    detail: 'No successful validation evidence was recorded for this behavioral criterion.',
    evidenceIds: [],
  }
}

function verifyAcceptanceAssertion(
  assertion: SessionAcceptanceAssertion,
  session: SessionMeta,
  evidenceManifest: SessionEvidenceManifest,
  artifactBundle: SessionEvaluationArtifactBundle,
  currentRevisionOnly = false,
): SessionAcceptanceAssertionResult {
  switch (assertion.kind) {
    case 'file_exists':
      return verifyFileAssertion(assertion, session, artifactBundle)
    case 'symbol_exists':
    case 'text_match':
      return verifyTextAssertion(assertion, session, artifactBundle)
    case 'validation_required':
      return verifyValidationAssertion(assertion, evidenceManifest, currentRevisionOnly)
    case 'human_review':
      return {
        assertion,
        status: 'skipped',
        detail: 'This criterion is subjective and requires human review.',
        evidenceIds: [],
      }
  }
}

function reportStatus(results: SessionAcceptanceAssertionResult[]): SessionAcceptanceAssertionStatus {
  if (results.some((result) => result.status === 'failed')) return 'failed'
  if (results.some((result) => result.status === 'unverified')) return 'unverified'
  if (results.some((result) => result.status === 'verified')) return 'verified'
  return 'skipped'
}

function reportReason(status: SessionAcceptanceAssertionStatus, results: SessionAcceptanceAssertionResult[]): string {
  switch (status) {
    case 'failed':
      return `${results.filter((result) => result.status === 'failed').length} assertion(s) failed.`
    case 'unverified':
      return `${results.filter((result) => result.status === 'unverified').length} assertion(s) could not be verified.`
    case 'verified':
      return `${results.filter((result) => result.status === 'verified').length} assertion(s) verified.`
    case 'skipped':
      return 'Only subjective assertions were extracted; human review is required.'
  }
}

function buildSessionAcceptanceVerification(
  session: SessionMeta,
  events: SessionEvent[],
  evidenceManifest: SessionEvidenceManifest,
  artifactBundle: SessionEvaluationArtifactBundle,
): SessionAcceptanceVerification {
  const contract = latestRunContractEvent(events)?.contract ?? null
  const assertionsByCriterion = extractAcceptanceAssertions(contract)
  const reports: SessionAcceptanceVerificationReport[] = []

  for (const criterion of contract?.acceptanceCriteria ?? []) {
    const assertions = assertionsByCriterion.get(criterion.id) ?? []
    const results = assertions.map((item) =>
      verifyAcceptanceAssertion(
        item,
        session,
        evidenceManifest,
        artifactBundle,
        contract !== null,
      )
    )
    const status = reportStatus(results)
    reports.push({
      acceptanceCriterionId: criterion.id,
      acceptanceCriterionText: criterion.text,
      status,
      results,
      evidenceIds: [...new Set(results.flatMap((result) => result.evidenceIds))],
      reason: reportReason(status, results),
    })
  }

  const results = reports.flatMap((report) => report.results)
  const verifiedAssertions = results.filter((result) => result.status === 'verified').length
  const failedAssertions = results.filter((result) => result.status === 'failed').length
  const unverifiedAssertions = results.filter((result) => result.status === 'unverified').length
  const skippedAssertions = results.filter((result) => result.status === 'skipped').length
  const status: SessionAcceptanceVerificationStatus = reports.length === 0
    ? 'empty'
    : failedAssertions > 0
      ? 'failed'
      : unverifiedAssertions > 0
        ? 'unverified'
        : verifiedAssertions > 0
          ? 'passed'
          : 'skipped'
  const lastEventAt = latestTimestamp(session, events)
  const fingerprintPayload = {
    sessionId: session.id,
    eventCount: events.length,
    contractAcceptanceCriteria: contract?.acceptanceCriteria.map((criterion) => criterion.id) ?? [],
    artifactBundleRevision: artifactBundle.revision,
    reportStatuses: reports.map((report) => `${report.acceptanceCriterionId}:${report.status}`),
    resultStatuses: results.map((result) => `${result.assertion.id}:${result.status}:${result.path ?? ''}`),
  }
  const fingerprint = stableFingerprint(fingerprintPayload)

  return {
    schemaVersion: 1,
    status,
    revision: `acceptance-verification-${events.length}-${fingerprint.slice(0, 8)}`,
    fingerprint,
    eventCount: events.length,
    summary: {
      acceptanceCriteria: reports.length,
      assertions: results.length,
      verifiedAssertions,
      failedAssertions,
      unverifiedAssertions,
      skippedAssertions,
      constantAssertions: results.filter((result) => result.assertion.tier === 'constant').length,
      structuralAssertions: results.filter((result) => result.assertion.tier === 'structural').length,
      behavioralAssertions: results.filter((result) => result.assertion.tier === 'behavioral').length,
      subjectiveAssertions: results.filter((result) => result.assertion.tier === 'subjective').length,
    },
    reports,
    lastUpdatedAt: lastEventAt,
  }
}

function firstRiskEvidence(
  evidenceManifest: SessionEvidenceManifest,
  code: SessionEvidenceRisk['code'],
): SessionEvidenceRisk | undefined {
  return evidenceManifest.risks.find((risk) => risk.code === code)
}

function buildSessionConsensusTriggerMatrix({
  evidenceManifest,
  artifactBundle,
  mechanical,
  semantic,
  runContractPresent,
  assistantMarkedUnverified,
}: {
  evidenceManifest: SessionEvidenceManifest
  artifactBundle: SessionEvaluationArtifactBundle
  mechanical: SessionEvaluationStage
  semantic: SessionEvaluationStage
  runContractPresent: boolean
  assistantMarkedUnverified: boolean
}): SessionConsensusTriggerMatrix {
  const validationFailedRisk = firstRiskEvidence(evidenceManifest, 'validation_failed')
  const postEditRisk = firstRiskEvidence(evidenceManifest, 'post_edit_findings')
  const assistantUnverifiedRisk = firstRiskEvidence(evidenceManifest, 'assistant_marked_unverified')
  const failedValidationEvidenceIds = evidenceManifest.artifacts
    .filter((artifact) => artifact.kind === 'validation' && artifact.status === 'error')
    .map((artifact) => artifact.id)
  const fileChangeEvidenceIds = evidenceManifest.artifacts
    .filter((artifact) => artifact.kind === 'file_change')
    .map((artifact) => artifact.id)
  const changeSetSize = new Set([
    ...artifactBundle.files.map((file) => file.path),
    ...artifactBundle.skipped
      .filter((item) => item.reason !== 'duplicate_path')
      .map((item) => item.path ?? item.artifactId ?? item.reason),
  ]).size
  const artifactBundleNeedsReview = artifactBundle.status === 'partial'
    || (artifactBundle.status === 'metadata_only' && artifactBundle.summary.files > 0)
  const triggers: SessionConsensusTrigger[] = [
    {
      code: 'validation_failed',
      priority: 1,
      fired: Boolean(validationFailedRisk) || evidenceManifest.summary.validationFailures > 0,
      severity: 'warning',
      message: validationFailedRisk?.message
        ?? `${evidenceManifest.summary.validationFailures} validation run(s) failed.`,
      evidenceIds: validationFailedRisk?.evidenceIds ?? failedValidationEvidenceIds,
    },
    {
      code: 'post_edit_findings',
      priority: 2,
      fired: Boolean(postEditRisk),
      severity: 'warning',
      message: postEditRisk?.message
        ?? 'Post-edit analysis recorded diagnostics that need review.',
      evidenceIds: postEditRisk?.evidenceIds,
    },
    {
      code: 'assistant_marked_unverified',
      priority: 3,
      fired: assistantMarkedUnverified || Boolean(assistantUnverifiedRisk),
      severity: 'warning',
      message: assistantUnverifiedRisk?.message
        ?? 'The assistant explicitly marked the result as unverified or incomplete.',
      evidenceIds: assistantUnverifiedRisk?.evidenceIds,
    },
    {
      code: 'mechanical_validation_missing',
      priority: 4,
      fired: mechanical.status === 'unverified',
      severity: 'warning',
      message: mechanical.status === 'unverified'
        ? mechanical.summary
        : 'Mechanical validation evidence is present or not required.',
      evidenceIds: mechanical.evidenceIds,
    },
    {
      code: 'semantic_acceptance_unverified',
      priority: 5,
      fired: semantic.status === 'unverified',
      severity: 'warning',
      message: semantic.status === 'unverified'
        ? semantic.summary
        : 'Acceptance criteria are verified, blocked, failed, or not required.',
      evidenceIds: semantic.evidenceIds,
    },
    {
      code: 'artifact_bundle_partial',
      priority: 6,
      fired: artifactBundleNeedsReview,
      severity: 'warning',
      message: artifactBundleNeedsReview
        ? `${artifactBundle.summary.skippedFiles} file(s) were skipped and ${artifactBundle.summary.metadataOnlyFiles} file(s) are metadata-only.`
        : 'Evaluation artifact bundle is complete enough for its current evidence.',
      evidenceIds: artifactBundle.files.map((file) => file.artifactId),
    },
    {
      code: 'large_change_set',
      priority: 7,
      fired: changeSetSize >= 10,
      severity: 'warning',
      message: changeSetSize >= 10
        ? `${changeSetSize} changed file artifact(s) should receive independent review.`
        : 'Change-set size does not require consensus review.',
      evidenceIds: fileChangeEvidenceIds,
    },
    {
      code: 'missing_run_contract',
      priority: 8,
      fired: !runContractPresent && evidenceManifest.summary.filesChanged > 0,
      severity: 'warning',
      message: !runContractPresent && evidenceManifest.summary.filesChanged > 0
        ? 'File changes were recorded without a run contract or acceptance criteria.'
        : 'Run contract coverage is present or no changed files were recorded.',
      evidenceIds: fileChangeEvidenceIds,
    },
  ]
  const primaryTrigger = triggers.find((trigger) => trigger.fired)
  return {
    required: triggers.some((trigger) => trigger.fired && trigger.severity === 'warning'),
    primaryTrigger,
    triggers,
  }
}

export function buildSessionEvaluationGate(
  session: SessionMeta,
  events: SessionEvent[],
  evidenceManifest: SessionEvidenceManifest = buildSessionEvidenceManifest(session, events),
): SessionEvaluationGate {
  const contractEvent = latestRunContractEvent(events)
  const contractIndexes = eventIndexById(events)
  const contractIndex = contractEvent ? contractIndexes.get(contractEvent.id) ?? -1 : -1
  const documentReadBackEvidence = currentDocumentArtifactReadBackEvidence({
    contract: contractEvent?.contract ?? null,
    contractIndex,
    indexes: contractIndexes,
    fileChanges: evidenceManifest.artifacts.filter((artifact) => artifact.kind === 'file_change'),
    artifactReadBacks: evidenceManifest.artifacts.filter((artifact) => artifact.kind === 'artifact_readback'),
  })
  const latestAssistant = latestAssistantMessage(events)
  const assistantMarkedIncomplete = latestAssistant
    ? /(?:^|\r?\n)\s*INCOMPLETE\s*:/i.test(latestAssistant.content)
    : false
  const assistantMarkedUnverified = latestAssistant
    ? assistantContentMarkedUnverified(latestAssistant.content)
    : false
  const assistantMarkedVerified = latestAssistant
    ? /\bVERIFIED\b/i.test(latestAssistant.content) && !assistantMarkedUnverified
    : false
  const executionComplete = events.some((event) => event.type === 'session_end')
  const runContractPresent = evidenceManifest.summary.acceptanceCriteria > 0
  const pendingToolResults = countPendingToolResults(events)
  const currentValidationArtifacts = scopedValidationArtifacts(
    evidenceManifest,
    runContractPresent,
  )
  const validationEvidenceIds = currentValidationArtifacts
    .map((artifact) => artifact.id)
  const successfulValidationIds = currentValidationArtifacts
    .filter((artifact) => artifact.status === 'success')
    .map((artifact) => artifact.id)
  const failedValidationEvidenceIds = currentValidationArtifacts
    .filter((artifact) => artifact.status === 'error')
    .map((artifact) => artifact.id)
  const acceptanceEvidenceIds = evidenceManifest.acceptanceCriteria
    .flatMap((criterion) => criterion.evidenceIds)
  const artifactBundle = buildSessionEvaluationArtifactBundle(session, events, evidenceManifest)
  const acceptanceVerification = buildSessionAcceptanceVerification(
    session,
    events,
    evidenceManifest,
    artifactBundle,
  )
  const acceptanceAssertionEvidenceIds = acceptanceVerification.reports
    .flatMap((report) => report.evidenceIds)

  const mechanical: SessionEvaluationStage = pendingToolResults > 0
    ? {
        status: 'running',
        summary: `${pendingToolResults} tool call(s) are still waiting for results.`,
        evidenceIds: [],
      }
    : evidenceManifest.summary.validationFailures > 0
      ? {
          status: 'failed',
          summary: `${evidenceManifest.summary.validationFailures} mechanical validation run(s) failed.`,
          evidenceIds: failedValidationEvidenceIds,
        }
      : successfulValidationIds.length > 0
        ? {
            status: 'passed',
            summary: `${successfulValidationIds.length} mechanical validation run(s) passed for the current revision.`,
            evidenceIds: successfulValidationIds,
          }
        : evidenceManifest.summary.validationRuns > 0
          ? {
              status: 'unverified',
              summary: 'Validation commands ran, but their wrapper exit status was not reliable enough to verify the current revision.',
              evidenceIds: validationEvidenceIds,
            }
        : documentReadBackEvidence.length > 0
          ? {
              status: 'passed',
              summary: `${documentReadBackEvidence.length} required document artifact(s) were read back after their latest write; no explicit runtime validation was required.`,
              evidenceIds: documentReadBackEvidence.map((artifact) => artifact.id),
            }
        : evidenceManifest.summary.filesChanged > 0
          ? {
              status: 'unverified',
              summary: 'File changes were recorded without mechanical validation evidence.',
              evidenceIds: evidenceManifest.artifacts
                .filter((artifact) => artifact.kind === 'file_change')
                .map((artifact) => artifact.id),
            }
          : {
              status: 'skipped',
              summary: 'No file-changing work required mechanical validation.',
              evidenceIds: [],
            }

  const semantic: SessionEvaluationStage = !runContractPresent
    ? {
        status: events.length === 0 ? 'not_started' : 'skipped',
        summary: events.length === 0
          ? 'No session events are available for semantic evaluation.'
          : 'No run contract or acceptance criteria were recorded.',
        evidenceIds: [],
      }
    : evidenceManifest.summary.failedAcceptanceCriteria > 0
      ? {
          status: 'failed',
          summary: `${evidenceManifest.summary.failedAcceptanceCriteria} acceptance criterion/criteria failed.`,
          evidenceIds: [...new Set([...acceptanceEvidenceIds, ...acceptanceAssertionEvidenceIds])],
        }
      : acceptanceVerification.summary.failedAssertions > 0
        ? {
            status: 'failed',
            summary: `${acceptanceVerification.summary.failedAssertions} concrete acceptance assertion(s) failed.`,
            evidenceIds: [...new Set(acceptanceAssertionEvidenceIds)],
          }
      : evidenceManifest.summary.blockedAcceptanceCriteria > 0
        ? {
            status: 'blocked',
            summary: `${evidenceManifest.summary.blockedAcceptanceCriteria} acceptance criterion/criteria are blocked.`,
            evidenceIds: acceptanceEvidenceIds,
          }
        : evidenceManifest.summary.unverifiedAcceptanceCriteria > 0
          || acceptanceVerification.summary.unverifiedAssertions > 0
          || assistantMarkedUnverified
          ? {
              status: 'unverified',
              summary: assistantMarkedUnverified
                ? assistantMarkedIncomplete
                  ? 'The latest assistant response explicitly marked the result as incomplete.'
                  : 'The latest assistant response explicitly marked the result as unverified.'
                : acceptanceVerification.summary.unverifiedAssertions > 0
                  ? `${acceptanceVerification.summary.unverifiedAssertions} concrete acceptance assertion(s) could not be verified.`
                  : `${evidenceManifest.summary.unverifiedAcceptanceCriteria} acceptance criterion/criteria lack supporting evidence.`,
              evidenceIds: [...new Set([...acceptanceEvidenceIds, ...acceptanceAssertionEvidenceIds])],
            }
          : {
              status: 'passed',
              summary: `${evidenceManifest.summary.supportedAcceptanceCriteria}/${evidenceManifest.summary.acceptanceCriteria} acceptance criterion/criteria are supported by evidence; ${acceptanceVerification.summary.verifiedAssertions} concrete assertion(s) verified.`,
              evidenceIds: [...new Set([...acceptanceEvidenceIds, ...acceptanceAssertionEvidenceIds])],
            }

  const consensusTriggers = buildSessionConsensusTriggerMatrix({
    evidenceManifest,
    artifactBundle,
    mechanical,
    semantic,
    runContractPresent,
    assistantMarkedUnverified,
  })
  const consensusRequired = consensusTriggers.required
  const consensusEvidenceIds = [
    ...new Set(consensusTriggers.triggers
      .filter((trigger) => trigger.fired)
      .flatMap((trigger) => trigger.evidenceIds ?? [])),
  ]
  const consensus: SessionEvaluationStage = consensusRequired
    ? {
        status: 'unverified',
        summary: consensusTriggers.primaryTrigger
          ? `Consensus review is required: ${consensusTriggers.primaryTrigger.message}`
          : 'Independent consensus review is required before approving this session.',
        evidenceIds: consensusEvidenceIds,
      }
    : {
        status: 'skipped',
        summary: 'No consensus trigger requires independent review.',
        evidenceIds: [],
      }

  const risks: SessionEvaluationGateRisk[] = []
  if (executionComplete && mechanical.status !== 'passed' && evidenceManifest.summary.filesChanged > 0) {
    risks.push({
      code: 'execution_complete_without_evaluation',
      severity: 'warning',
      message: 'The session is complete, but mechanical validation did not pass.',
      evidenceIds: mechanical.evidenceIds,
    })
  }
  if (mechanical.status === 'unverified') {
    risks.push({
      code: 'mechanical_validation_missing',
      severity: 'warning',
      message: 'Mechanical validation is missing for recorded file changes.',
      evidenceIds: mechanical.evidenceIds,
    })
  } else if (mechanical.status === 'failed') {
    risks.push({
      code: 'mechanical_validation_failed',
      severity: 'warning',
      message: 'Mechanical validation failed.',
      evidenceIds: mechanical.evidenceIds,
    })
  }
  if (semantic.status === 'failed') {
    risks.push({
      code: 'semantic_acceptance_failed',
      severity: 'warning',
      message: 'At least one acceptance criterion failed.',
      evidenceIds: semantic.evidenceIds,
    })
  } else if (semantic.status === 'unverified') {
    risks.push({
      code: 'semantic_acceptance_unverified',
      severity: 'warning',
      message: 'Acceptance criteria are not fully verified.',
      evidenceIds: semantic.evidenceIds,
    })
  }
  if (pendingToolResults > 0) {
    risks.push({
      code: 'pending_tool_results',
      severity: 'info',
      message: `${pendingToolResults} tool call(s) are still pending.`,
    })
  }
  if (consensusRequired) {
    risks.push({
      code: 'consensus_required',
      severity: 'info',
      message: consensusTriggers.primaryTrigger
        ? `Independent review is required because ${consensusTriggers.primaryTrigger.message}`
        : 'Independent review is required before treating the result as approved.',
      evidenceIds: consensus.evidenceIds,
    })
  }

  const status: SessionEvaluationGateStatus = events.length === 0
    ? 'empty'
    : pendingToolResults > 0
      ? 'running'
      : mechanical.status === 'failed' || semantic.status === 'failed'
        ? 'failed'
        : mechanical.status === 'blocked' || semantic.status === 'blocked'
          ? 'blocked'
          : mechanical.status === 'unverified' || semantic.status === 'unverified' || consensus.status === 'unverified'
            ? 'unverified'
            : mechanical.status === 'not_started' && semantic.status === 'not_started'
              ? 'not_started'
              : 'passed'

  const approved = status === 'passed'
  const reason = approved
    ? 'Mechanical and semantic gates are satisfied; consensus was not required.'
    : status === 'failed'
      ? 'A mechanical or semantic evaluation stage failed.'
      : status === 'running'
        ? 'Evaluation is waiting on running tool work.'
        : status === 'blocked'
          ? 'Evaluation is blocked by unresolved criteria or runtime work.'
          : status === 'empty'
            ? 'No session evidence is available.'
            : 'Evaluation evidence is incomplete.'

  const lastEvent = events.at(-1)
  const lastEventAt = latestTimestamp(session, events)
  const fingerprintPayload = {
    sessionId: session.id,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt,
    evidenceRevision: evidenceManifest.revision,
    status,
    mechanical: mechanical.status,
    semantic: semantic.status,
    consensus: consensus.status,
    artifactBundleRevision: artifactBundle.revision,
    acceptanceVerificationRevision: acceptanceVerification.revision,
    consensusTriggers: consensusTriggers.triggers
      .filter((trigger) => trigger.fired)
      .map((trigger) => trigger.code),
    risks: risks.map((risk) => risk.code),
  }
  const fingerprint = stableFingerprint(fingerprintPayload)

  return {
    schemaVersion: 1,
    status,
    revision: `eval-${events.length}-${fingerprint.slice(0, 8)}`,
    fingerprint,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt,
    stages: {
      mechanical,
      semantic,
      consensus,
    },
    artifactBundle,
    acceptanceVerification,
    consensusTriggers,
    signals: {
      executionComplete,
      runContractPresent,
      fileChanges: evidenceManifest.summary.filesChanged,
      validationRuns: evidenceManifest.summary.validationRuns,
      validationFailures: evidenceManifest.summary.validationFailures,
      acceptanceCriteria: evidenceManifest.summary.acceptanceCriteria,
      supportedAcceptanceCriteria: evidenceManifest.summary.supportedAcceptanceCriteria,
      failedAcceptanceCriteria: evidenceManifest.summary.failedAcceptanceCriteria,
      blockedAcceptanceCriteria: evidenceManifest.summary.blockedAcceptanceCriteria,
      unverifiedAcceptanceCriteria: evidenceManifest.summary.unverifiedAcceptanceCriteria,
      pendingToolResults,
      assistantMarkedVerified,
      assistantMarkedUnverified,
      consensusRequired,
    },
    verdict: {
      approved,
      reason,
    },
    risks,
    lastUpdatedAt: lastEventAt,
  }
}

export function buildSessionContextEngine(
  session: SessionMeta,
  events: SessionEvent[],
  pendingQuestions: PendingQuestionLike[] = [],
): SessionContextEngine {
  const traceMetrics = buildSessionTraceMetrics(session, events)
  const workingMemory = buildSessionWorkingMemory(session, events, pendingQuestions)
  const runContract = buildSessionRunContract(events)
  const lastEvent = events.at(-1)

  const hasVersionedContext = traceMetrics.memoryContextEvents > 0
    || traceMetrics.contextCompactions > 0
    || traceMetrics.memorySummaryEvents > 0
    || runContract !== null
    || workingMemory.keyDecisions.length > 0
    || workingMemory.fileChanges.length > 0
  const status: SessionContextEngineStatus = events.length === 0
    ? 'empty'
    : hasVersionedContext
      ? 'versioned'
      : 'warming'

  const fingerprintPayload = {
    sessionId: session.id,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt: traceMetrics.lastEventAt,
    memoryContextEvents: traceMetrics.memoryContextEvents,
    memoryContextItems: traceMetrics.memoryContextItems,
    contextCompactions: traceMetrics.contextCompactions,
    contextCompactionTokensSaved: traceMetrics.contextCompactionTokensSaved,
    memorySummaryEvents: traceMetrics.memorySummaryEvents,
    memorySummarySemanticExtractions: traceMetrics.memorySummarySemanticExtractions,
    memorySummaryRagPromotions: traceMetrics.memorySummaryRagPromotions,
    workingMemoryDecisions: workingMemory.keyDecisions.length,
    workingMemoryFileChanges: workingMemory.fileChanges.length,
    openQuestions: workingMemory.openQuestions.length,
    runContractSource: runContract?.source,
    runContractAcceptanceCriteria: runContract?.acceptanceCriteria.length ?? 0,
  }
  const fingerprint = stableFingerprint(fingerprintPayload)

  return {
    schemaVersion: 1,
    status,
    revision: `ctx-${events.length}-${fingerprint.slice(0, 8)}`,
    fingerprint,
    eventCount: events.length,
    lastEventId: lastEvent?.id,
    lastEventAt: traceMetrics.lastEventAt,
    sources: {
      memoryContext: {
        events: traceMetrics.memoryContextEvents,
        items: traceMetrics.memoryContextItems,
        lastAt: events.findLast((event) => event.type === 'memory_context')?.timestamp,
      },
      compaction: {
        events: traceMetrics.contextCompactions,
        tokensSaved: traceMetrics.contextCompactionTokensSaved,
        lastAt: traceMetrics.lastContextCompactedAt,
      },
      memorySummary: {
        events: traceMetrics.memorySummaryEvents,
        semanticExtractions: traceMetrics.memorySummarySemanticExtractions,
        ragPromotions: traceMetrics.memorySummaryRagPromotions,
        lastAt: traceMetrics.lastMemorySummarizedAt,
      },
      workingMemory: {
        decisions: workingMemory.keyDecisions.length,
        fileChanges: workingMemory.fileChanges.length,
        openQuestions: workingMemory.openQuestions.length,
        lastUpdatedAt: workingMemory.lastUpdatedAt,
      },
      runContract: {
        present: runContract !== null,
        source: runContract?.source,
        acceptanceCriteria: runContract?.acceptanceCriteria.length ?? 0,
      },
    },
  }
}

export type SessionTraceMetrics = ReturnType<typeof buildSessionTraceMetrics>
export type SessionHistoryManagement = ReturnType<typeof buildSessionHistoryManagement>
export type SessionCompletionChecklist = ReturnType<typeof buildSessionCompletionChecklist>
export type SessionWorkingMemory = ReturnType<typeof buildSessionWorkingMemory>
export type SessionDebateRounds = ReturnType<typeof buildSessionDebateRounds>
export type SessionPlannerWorkingMemory = ReturnType<typeof buildSessionPlannerWorkingMemory>
export type SessionRunContract = ReturnType<typeof buildSessionRunContract>

export const __testables = {
  artifactEvidenceForPath,
  artifactOperation,
  artifactSkip,
  assistantContentMarkedUnverified,
  buildSessionAcceptanceVerification,
  buildSessionConsensusTriggerMatrix,
  buildSessionEvaluationArtifactBundle,
  candidateFilesForAssertion,
  collectProjectSourceFiles,
  commandFromToolInput,
  commandArgvFromToolInput,
  compileAssertionPattern,
  detectContractLedgerBlockers,
  extractAcceptanceAssertions,
  extractBacktickTokens,
  extractPathTokens,
  firstUserMessage,
  globPatternToRegExp,
  hasGlobMagic,
  isBehavioralCriterion,
  isGeneratedOrVendorPath,
  isValidationCommand,
  isLikelyIdentifier,
  isReadOnlyObservationResult,
  isSourceLikePath,
  isSubjectiveCriterion,
  latestAssistantMessage,
  matchesFileHint,
  normalizeArtifactPath,
  normalizeAssertionPath,
  executionEvidenceIds,
  projectFileExists,
  readAssertionCandidateFile,
  reportReason,
  reportStatus,
  safeEvaluationArtifactPath,
  sectionStatus,
  sectionSummary,
  supportsRepositoryStructuralAssertions,
  successfulValidationEvidenceIds,
  failedValidationEvidenceIds,
  verifyFileAssertion,
  verifyTextAssertion,
  verifyValidationAssertion,
}
