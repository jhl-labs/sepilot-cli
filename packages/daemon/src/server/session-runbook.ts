import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { AgentRunContract, SessionMeta } from '@sepilotd/core'
import type {
  SessionCompletionChecklist,
  SessionContractLedger,
  SessionEvidenceArtifact,
  SessionEvidenceManifest,
  SessionEvaluationGate,
  SessionHistoryManagement,
} from './session-contracts.js'

export type SessionRunbookStatus =
  | 'ready'
  | 'needs_attention'
  | 'blocked'
  | 'recoverable'

export type SessionRunbookSeverity = 'info' | 'warning' | 'critical'
export type SessionRunbookActionPriority = 'now' | 'next' | 'optional'
export type SessionRunbookSignalSource =
  | 'evidence'
  | 'runtime'
  | 'history'
  | 'checklist'
  | 'session'
export type SessionRunbookMechanicalValidationStatus =
  | 'not_needed'
  | 'already_validated'
  | 'ready'
  | 'missing_project_context'
  | 'unsupported_project'
export type SessionRunbookMechanicalValidationToolchain =
  | 'node-pnpm-turbo'
  | 'node-pnpm'
  | 'node-npm'
  | 'node-yarn'
  | 'node-bun'
  | 'python-uv'
  | 'python'
  | 'rust'
  | 'go'
  | 'zig'
export type SessionRunbookMechanicalValidationCommandKind =
  | 'lint'
  | 'typecheck'
  | 'build'
  | 'test'
  | 'coverage'
  | 'static'

export interface SessionRunbookSignal {
  id: string
  severity: SessionRunbookSeverity
  title: string
  detail: string
  source: SessionRunbookSignalSource
  evidenceIds?: string[]
}

export interface SessionRunbookAction {
  id: string
  priority: SessionRunbookActionPriority
  title: string
  command: string
  description: string
  destructive: boolean
  requiresReview: boolean
}

export interface SessionRunbookMechanicalValidationCommand {
  kind: SessionRunbookMechanicalValidationCommandKind
  label: string
  command: string
  reason: string
}

export interface SessionRunbookMechanicalValidationPlan {
  status: SessionRunbookMechanicalValidationStatus
  reason: string
  cwd?: string
  toolchain?: SessionRunbookMechanicalValidationToolchain
  commands: SessionRunbookMechanicalValidationCommand[]
}

export interface SessionRunbookEvidenceSummary {
  manifestRevision?: string
  manifestStatus?: SessionEvidenceManifest['status']
  acceptanceCriteria: SessionEvidenceManifest['acceptanceCriteria']
  risks: SessionEvidenceManifest['risks']
  artifacts: SessionEvidenceArtifact[]
}

export interface SessionRunbook {
  schemaVersion: 1
  generatedAt: string
  status: SessionRunbookStatus
  headline: string
  session: {
    id: string
    title: string
    status: SessionMeta['status']
    provider: string
    model: string
    createdAt: string
    updatedAt: string
    eventCount: number
  }
  summary: {
    signals: number
    criticalSignals: number
    warningSignals: number
    pendingApprovals: number
    pendingQuestions: number
    evidenceRisks: number
    validationRuns: number
    validationFailures: number
    filesChanged: number
    acceptanceCriteria: number
    supportedAcceptanceCriteria: number
    failedAcceptanceCriteria: number
    blockedAcceptanceCriteria: number
    unverifiedAcceptanceCriteria: number
    validationSuggestions: number
  }
  signals: SessionRunbookSignal[]
  actions: SessionRunbookAction[]
  evidence: SessionRunbookEvidenceSummary
  mechanicalValidation: SessionRunbookMechanicalValidationPlan
  supportBundle: {
    recommended: boolean
    command: string
    reason: string
  }
}

export interface SessionRunbookInput extends SessionMeta {
  events?: unknown[]
  pendingApprovals?: Array<{ requestId: string; state?: string; resumeAvailable?: boolean }>
  pendingQuestions?: Array<{ id: string; prompt: string }>
  completionChecklist?: SessionCompletionChecklist
  runContract?: AgentRunContract | null
  contractLedger?: SessionContractLedger
  evidenceManifest?: SessionEvidenceManifest
  evaluationGate?: SessionEvaluationGate
  historyManagement?: SessionHistoryManagement
  resumableRun?: {
    mode: 'exact' | 'replay-safe' | 'replay-risky'
    forceRequired: boolean
    currentTool?: string
    currentToolCount?: number
    currentTools?: string[]
    journaledResultAvailable?: boolean
    recoveryProbeAvailable?: boolean
  }
  resumableRunIssue?: {
    status: 'corrupt' | 'unreadable'
    message: string
  }
  delegation?: {
    claimHealth: 'healthy' | 'degraded' | 'lost'
    targetDevice: string
    lastError?: string
    leaseLossSource?: 'gateway' | 'comments' | 'transport'
  }
  traceMetrics?: {
    toolFailures?: number
    providerAttemptFailures?: number
    providerFallbacks?: number
    finalProvider?: string
    finalModel?: string
  }
}

export interface SessionRunbookBuildOptions {
  generatedAt?: string
  projectFileExists?: (cwd: string, relativePath: string) => boolean
  readProjectJson?: (cwd: string, relativePath: string) => unknown
}

interface ValidationCommandCandidate {
  kind: SessionRunbookMechanicalValidationCommandKind
  scripts: string[]
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9._:/@+-]+$/.test(value)
    ? value
    : `'${value.replaceAll('\'', '\'\\\'\'')}'`
}

function commandInCwd(cwd: string | undefined, command: string): string {
  return cwd ? `cd ${shellQuote(cwd)} && ${command}` : command
}

function commandSessionId(sessionId: string): string {
  return shellQuote(sessionId)
}

function safeFileStem(value: string): string {
  const stem = value.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '')
  return (stem || 'session').slice(0, 48)
}

function signalTitle(code: string): string {
  return code
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function signalSeverityRank(severity: SessionRunbookSeverity): number {
  switch (severity) {
    case 'critical':
      return 0
    case 'warning':
      return 1
    case 'info':
      return 2
  }
}

function actionPriorityRank(priority: SessionRunbookActionPriority): number {
  switch (priority) {
    case 'now':
      return 0
    case 'next':
      return 1
    case 'optional':
      return 2
  }
}

function addSignal(
  signals: SessionRunbookSignal[],
  signal: SessionRunbookSignal,
): void {
  if (signals.some((item) => item.id === signal.id)) return
  signals.push(signal)
}

function addAction(
  actions: Map<string, SessionRunbookAction>,
  action: SessionRunbookAction,
): void {
  const existing = actions.get(action.id)
  if (!existing || actionPriorityRank(action.priority) < actionPriorityRank(existing.priority)) {
    actions.set(action.id, action)
  }
}

function selectRunbookArtifacts(
  manifest: SessionEvidenceManifest | undefined,
): SessionEvidenceArtifact[] {
  if (!manifest) return []
  const important = manifest.artifacts.filter((artifact) =>
    artifact.status === 'error'
    || artifact.status === 'warning'
    || artifact.kind === 'validation'
    || artifact.kind === 'post_edit_findings'
    || artifact.kind === 'file_change'
  )
  const latest = manifest.artifacts.slice(-5)
  const selected = new Map<string, SessionEvidenceArtifact>()
  for (const artifact of [...important, ...latest]) {
    selected.set(artifact.id, artifact)
  }
  return [...selected.values()]
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id))
    .slice(-12)
}

function defaultProjectFileExists(cwd: string, relativePath: string): boolean {
  try {
    return existsSync(join(cwd, relativePath))
  } catch {
    return false
  }
}

function defaultReadProjectJson(cwd: string, relativePath: string): unknown {
  try {
    return JSON.parse(readFileSync(join(cwd, relativePath), 'utf8')) as unknown
  } catch {
    return undefined
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function readPackageJson(
  cwd: string,
  options: SessionRunbookBuildOptions,
): Record<string, unknown> | undefined {
  return asRecord((options.readProjectJson ?? defaultReadProjectJson)(cwd, 'package.json'))
}

function packageScripts(packageJson: Record<string, unknown> | undefined): Record<string, string> {
  const scripts = asRecord(packageJson?.scripts)
  if (!scripts) return {}
  return Object.fromEntries(
    Object.entries(scripts)
      .filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
  )
}

function packageManagerName(
  cwd: string,
  packageJson: Record<string, unknown> | undefined,
  fileExists: (cwd: string, relativePath: string) => boolean,
): 'pnpm' | 'npm' | 'yarn' | 'bun' {
  const declared = typeof packageJson?.packageManager === 'string'
    ? packageJson.packageManager
    : ''
  if (declared.startsWith('pnpm@') || fileExists(cwd, 'pnpm-lock.yaml')) return 'pnpm'
  if (declared.startsWith('bun@') || fileExists(cwd, 'bun.lockb') || fileExists(cwd, 'bun.lock')) return 'bun'
  if (declared.startsWith('yarn@') || fileExists(cwd, 'yarn.lock')) return 'yarn'
  return 'npm'
}

function runScriptCommand(packageManager: 'pnpm' | 'npm' | 'yarn' | 'bun', script: string): string {
  switch (packageManager) {
    case 'pnpm':
      return `pnpm run ${script}`
    case 'yarn':
      return `yarn ${script}`
    case 'bun':
      return `bun run ${script}`
    case 'npm':
      return `npm run ${script}`
  }
}

function addValidationCommand(
  commands: SessionRunbookMechanicalValidationCommand[],
  cwd: string,
  command: Omit<SessionRunbookMechanicalValidationCommand, 'command'> & { command: string },
): void {
  if (commands.some((item) => item.command === command.command)) return
  commands.push({
    ...command,
    command: commandInCwd(cwd, command.command),
  })
}

function validationCommandReason(input: SessionRunbookInput): string {
  const summary = input.evidenceManifest?.summary
  if ((summary?.validationFailures ?? 0) > 0) {
    return 'Re-run after repair because the prior validation evidence failed.'
  }
  if ((summary?.failedAcceptanceCriteria ?? 0) > 0) {
    return 'Confirm the repaired acceptance criteria mechanically before trusting the result.'
  }
  if ((summary?.blockedAcceptanceCriteria ?? 0) > 0 || (summary?.unverifiedAcceptanceCriteria ?? 0) > 0) {
    return 'Close incomplete acceptance evidence with a deterministic validation command.'
  }
  const assertions = input.evaluationGate?.acceptanceVerification.summary
  if ((assertions?.failedAssertions ?? 0) > 0 || (assertions?.unverifiedAssertions ?? 0) > 0) {
    return 'Close failed or unverified acceptance assertions with deterministic validation evidence.'
  }
  return 'Validate changed workspace files before handing off the session.'
}

function detectNodeValidationCommands(
  cwd: string,
  input: SessionRunbookInput,
  options: SessionRunbookBuildOptions,
): SessionRunbookMechanicalValidationPlan | undefined {
  const fileExists = options.projectFileExists ?? defaultProjectFileExists
  if (!fileExists(cwd, 'package.json')) return undefined

  const packageJson = readPackageJson(cwd, options)
  const scripts = packageScripts(packageJson)
  const packageManager = packageManagerName(cwd, packageJson, fileExists)
  const toolchain = packageManager === 'pnpm' && (fileExists(cwd, 'turbo.json') || fileExists(cwd, 'pnpm-workspace.yaml'))
    ? 'node-pnpm-turbo'
    : (`node-${packageManager}` as SessionRunbookMechanicalValidationToolchain)
  const reason = validationCommandReason(input)
  const commands: SessionRunbookMechanicalValidationCommand[] = []
  const candidates: ValidationCommandCandidate[] = [
    { kind: 'lint', scripts: ['lint'] },
    { kind: 'typecheck', scripts: ['typecheck', 'check:types'] },
    { kind: 'test', scripts: ['test'] },
    { kind: 'build', scripts: ['build'] },
    { kind: 'coverage', scripts: ['coverage', 'test:coverage'] },
  ]

  for (const candidate of candidates) {
    const script = candidate.scripts.find((name) => scripts[name])
    if (!script) continue
    addValidationCommand(commands, cwd, {
      kind: candidate.kind,
      label: script,
      command: runScriptCommand(packageManager, script),
      reason,
    })
  }

  return {
    status: commands.length > 0 ? 'ready' : 'unsupported_project',
    reason: commands.length > 0
      ? 'Detected a Node.js project and generated validation commands from package scripts.'
      : 'Detected package.json, but no validation-oriented package scripts were found.',
    cwd,
    toolchain,
    commands,
  }
}

function detectPythonValidationCommands(
  cwd: string,
  input: SessionRunbookInput,
  options: SessionRunbookBuildOptions,
): SessionRunbookMechanicalValidationPlan | undefined {
  const fileExists = options.projectFileExists ?? defaultProjectFileExists
  if (!fileExists(cwd, 'pyproject.toml') && !fileExists(cwd, 'requirements.txt')) return undefined

  const useUv = fileExists(cwd, 'uv.lock')
  const prefix = useUv ? 'uv run ' : ''
  const reason = validationCommandReason(input)
  const commands: SessionRunbookMechanicalValidationCommand[] = []
  if (
    fileExists(cwd, 'ruff.toml')
    || fileExists(cwd, '.ruff.toml')
    || fileExists(cwd, 'pyproject.toml')
  ) {
    addValidationCommand(commands, cwd, {
      kind: 'lint',
      label: 'ruff',
      command: `${prefix}ruff check .`,
      reason,
    })
  }
  if (
    fileExists(cwd, 'mypy.ini')
    || fileExists(cwd, 'pyrightconfig.json')
    || fileExists(cwd, 'pyproject.toml')
  ) {
    const usePyright = fileExists(cwd, 'pyrightconfig.json')
    addValidationCommand(commands, cwd, {
      kind: 'typecheck',
      label: usePyright ? 'pyright' : 'mypy',
      command: usePyright ? `${prefix}pyright` : `${prefix}mypy .`,
      reason,
    })
  }
  addValidationCommand(commands, cwd, {
    kind: 'test',
    label: 'pytest',
    command: `${prefix}pytest`,
    reason,
  })

  return {
    status: 'ready',
    reason: `Detected a Python project and generated ${useUv ? 'uv-backed ' : ''}validation commands.`,
    cwd,
    toolchain: useUv ? 'python-uv' : 'python',
    commands,
  }
}

function detectValidationCommands(
  cwd: string,
  input: SessionRunbookInput,
  options: SessionRunbookBuildOptions,
): SessionRunbookMechanicalValidationPlan {
  const fileExists = options.projectFileExists ?? defaultProjectFileExists
  const node = detectNodeValidationCommands(cwd, input, options)
  if (node) return node
  const python = detectPythonValidationCommands(cwd, input, options)
  if (python) return python

  const reason = validationCommandReason(input)
  if (fileExists(cwd, 'Cargo.toml')) {
    return {
      status: 'ready',
      reason: 'Detected a Rust project and generated cargo validation commands.',
      cwd,
      toolchain: 'rust',
      commands: [
        {
          kind: 'test',
          label: 'cargo test',
          command: commandInCwd(cwd, 'cargo test'),
          reason,
        },
        {
          kind: 'static',
          label: 'cargo clippy',
          command: commandInCwd(cwd, 'cargo clippy --all-targets --all-features'),
          reason,
        },
      ],
    }
  }
  if (fileExists(cwd, 'go.mod')) {
    return {
      status: 'ready',
      reason: 'Detected a Go project and generated go validation commands.',
      cwd,
      toolchain: 'go',
      commands: [
        {
          kind: 'test',
          label: 'go test',
          command: commandInCwd(cwd, 'go test ./...'),
          reason,
        },
        {
          kind: 'static',
          label: 'go vet',
          command: commandInCwd(cwd, 'go vet ./...'),
          reason,
        },
      ],
    }
  }
  if (fileExists(cwd, 'build.zig')) {
    return {
      status: 'ready',
      reason: 'Detected a Zig project and generated zig validation commands.',
      cwd,
      toolchain: 'zig',
      commands: [
        {
          kind: 'test',
          label: 'zig build test',
          command: commandInCwd(cwd, 'zig build test'),
          reason,
        },
        {
          kind: 'build',
          label: 'zig build',
          command: commandInCwd(cwd, 'zig build'),
          reason,
        },
      ],
    }
  }

  return {
    status: 'unsupported_project',
    reason: 'Could not infer validation commands from known project files.',
    cwd,
    commands: [],
  }
}

function buildMechanicalValidationPlan(
  input: SessionRunbookInput,
  options: SessionRunbookBuildOptions,
): SessionRunbookMechanicalValidationPlan {
  const summary = input.evidenceManifest?.summary
  const validationRuns = summary?.validationRuns ?? 0
  const validationFailures = summary?.validationFailures ?? 0
  const changedWithoutValidation = (summary?.filesChanged ?? 0) > 0 && validationRuns === 0
  const incompleteAcceptance =
    (summary?.failedAcceptanceCriteria ?? 0) > 0
    || (summary?.blockedAcceptanceCriteria ?? 0) > 0
    || (summary?.unverifiedAcceptanceCriteria ?? 0) > 0
  const assertionSummary = input.evaluationGate?.acceptanceVerification.summary
  const incompleteAssertions =
    (assertionSummary?.failedAssertions ?? 0) > 0
    || (assertionSummary?.unverifiedAssertions ?? 0) > 0
  const needsValidation = validationFailures > 0
    || changedWithoutValidation
    || incompleteAcceptance
    || incompleteAssertions

  if (!needsValidation) {
    if (validationRuns > 0 && validationFailures === 0) {
      return {
        status: 'already_validated',
        reason: 'Existing evidence includes successful mechanical validation.',
        commands: [],
      }
    }
    return {
      status: 'not_needed',
      reason: 'No changed files, failed validation, or incomplete acceptance evidence require mechanical validation.',
      commands: [],
    }
  }

  if (!input.cwd) {
    return {
      status: 'missing_project_context',
      reason: 'The session does not record a cwd, so sepilotd cannot infer project validation commands.',
      commands: [],
    }
  }

  return detectValidationCommands(input.cwd, input, options)
}

function buildEvidenceSignals(
  input: SessionRunbookInput,
  signals: SessionRunbookSignal[],
): void {
  const manifest = input.evidenceManifest
  if (!manifest) {
    addSignal(signals, {
      id: 'evidence-missing',
      severity: 'info',
      title: 'Evidence manifest missing',
      detail: 'No evidence manifest is available for this session yet.',
      source: 'evidence',
    })
    return
  }

  for (const risk of manifest.risks) {
    addSignal(signals, {
      id: `evidence-${risk.code}`,
      severity: risk.severity === 'warning' ? 'warning' : 'info',
      title: signalTitle(risk.code),
      detail: risk.message,
      source: 'evidence',
      evidenceIds: risk.evidenceIds,
    })
  }

  const failedCriteria = manifest.acceptanceCriteria.filter((criterion) =>
    criterion.status === 'failed'
  )
  if (failedCriteria.length > 0) {
    addSignal(signals, {
      id: 'acceptance-failed',
      severity: 'critical',
      title: 'Acceptance criteria failed',
      detail: `${failedCriteria.length} acceptance criterion/criteria are backed by failed validation evidence.`,
      source: 'evidence',
      evidenceIds: [...new Set(failedCriteria.flatMap((criterion) => criterion.evidenceIds))],
    })
  }
}

function buildEvaluationGateSignals(
  input: SessionRunbookInput,
  signals: SessionRunbookSignal[],
): void {
  const verification = input.evaluationGate?.acceptanceVerification
  if (!verification) return

  if (verification.summary.failedAssertions > 0) {
    addSignal(signals, {
      id: 'acceptance-assertions-failed',
      severity: 'critical',
      title: 'Acceptance assertions failed',
      detail: `${verification.summary.failedAssertions} concrete acceptance assertion(s) failed independent verification.`,
      source: 'evidence',
      evidenceIds: [...new Set(verification.reports.flatMap((report) => report.evidenceIds))],
    })
  } else if (verification.summary.unverifiedAssertions > 0) {
    addSignal(signals, {
      id: 'acceptance-assertions-unverified',
      severity: 'warning',
      title: 'Acceptance assertions unverified',
      detail: `${verification.summary.unverifiedAssertions} concrete acceptance assertion(s) could not be verified.`,
      source: 'evidence',
      evidenceIds: [...new Set(verification.reports.flatMap((report) => report.evidenceIds))],
    })
  }
}

function hasDeclaredRunContract(input: SessionRunbookInput): boolean {
  if (input.runContract) return true
  return input.contractLedger?.sections.some((section) =>
    section.entries.some((entry) => entry.source === 'run_contract')
  ) ?? false
}

function buildContractLedgerSignals(
  input: SessionRunbookInput,
  signals: SessionRunbookSignal[],
): void {
  const ledger = input.contractLedger
  if (!ledger) return

  for (const blocker of ledger.blockers) {
    addSignal(signals, {
      id: `contract-${blocker.code}`,
      severity: blocker.severity === 'blocker' ? 'critical' : 'warning',
      title: signalTitle(blocker.code),
      detail: blocker.message,
      source: 'session',
    })
  }

  // Lightweight operational graphs intentionally omit a run contract. Their
  // conservative ledger defaults are diagnostic context, not missing user
  // requirements. Authority blockers above remain actionable either way.
  if (!hasDeclaredRunContract(input)) return

  const missingSections = ledger.sections.filter((section) =>
    section.status === 'missing' || section.status === 'weak'
  )
  if (missingSections.length > 0) {
    addSignal(signals, {
      id: 'contract-ledger-incomplete',
      severity: 'warning',
      title: 'Contract ledger incomplete',
      detail: `${missingSections.length} contract section(s) need stronger evidence: ${missingSections.map((section) => section.name).join(', ')}.`,
      source: 'session',
    })
  }
}

function buildRuntimeSignals(
  input: SessionRunbookInput,
  signals: SessionRunbookSignal[],
): void {
  if (input.pendingApprovals?.length) {
    addSignal(signals, {
      id: 'pending-approvals',
      severity: 'warning',
      title: 'Pending approvals',
      detail: `${input.pendingApprovals.length} approval request(s) are waiting for an operator decision.`,
      source: 'runtime',
    })
  }

  if (input.pendingQuestions?.length) {
    addSignal(signals, {
      id: 'pending-questions',
      severity: 'warning',
      title: 'Pending questions',
      detail: `${input.pendingQuestions.length} user question(s) are blocking progress.`,
      source: 'runtime',
    })
  }

  if (input.resumableRunIssue) {
    addSignal(signals, {
      id: 'resume-checkpoint-unavailable',
      severity: 'critical',
      title: 'Resume checkpoint unavailable',
      detail: input.resumableRunIssue.message,
      source: 'runtime',
    })
  } else if (input.resumableRun) {
    const force = input.resumableRun.forceRequired ? ' Force replay is required.' : ''
    const tool = input.resumableRun.currentTool
      ? ` Current tool: ${input.resumableRun.currentTool}.`
      : ''
    addSignal(signals, {
      id: 'resumable-run',
      severity: 'warning',
      title: 'Interrupted run can be resumed',
      detail: `A ${input.resumableRun.mode} checkpoint is available.${tool}${force}`,
      source: 'runtime',
    })
  }

  if (input.delegation?.claimHealth === 'lost') {
    addSignal(signals, {
      id: 'delegation-lost',
      severity: 'critical',
      title: 'Delegation lease lost',
      detail: `Delegation to ${input.delegation.targetDevice} is lost${input.delegation.lastError ? `: ${input.delegation.lastError}` : '.'}`,
      source: 'runtime',
    })
  } else if (input.delegation?.claimHealth === 'degraded') {
    addSignal(signals, {
      id: 'delegation-degraded',
      severity: 'warning',
      title: 'Delegation degraded',
      detail: `Delegation to ${input.delegation.targetDevice} is degraded${input.delegation.lastError ? `: ${input.delegation.lastError}` : '.'}`,
      source: 'runtime',
    })
  }

  if ((input.traceMetrics?.toolFailures ?? 0) > 0) {
    addSignal(signals, {
      id: 'tool-failures',
      severity: 'warning',
      title: 'Tool failures recorded',
      detail: `${input.traceMetrics?.toolFailures ?? 0} tool result(s) failed during the session.`,
      source: 'runtime',
    })
  }

  if ((input.traceMetrics?.providerAttemptFailures ?? 0) > 0) {
    addSignal(signals, {
      id: 'provider-attempt-failures',
      severity: 'warning',
      title: 'Provider attempt failures',
      detail: `${input.traceMetrics?.providerAttemptFailures ?? 0} provider attempt(s) failed before final model ${input.traceMetrics?.finalProvider ?? 'unknown'}/${input.traceMetrics?.finalModel ?? 'unknown'}.`,
      source: 'runtime',
    })
  }
}

function buildChecklistSignals(
  input: SessionRunbookInput,
  signals: SessionRunbookSignal[],
): void {
  const checklist = input.completionChecklist
  if (!checklist) return
  if (checklist.status === 'blocked') {
    const blocked = checklist.items.filter((item) => item.status === 'blocked')
    addSignal(signals, {
      id: 'completion-blocked',
      severity: 'warning',
      title: 'Completion checklist blocked',
      detail: blocked.length > 0
        ? `${blocked.length} checklist item(s) are blocked.`
        : 'The session completion checklist is blocked.',
      source: 'checklist',
    })
  }
}

function buildHistorySignals(
  input: SessionRunbookInput,
  signals: SessionRunbookSignal[],
): void {
  const history = input.historyManagement
  if (!history) return
  for (const risk of history.risks) {
    addSignal(signals, {
      id: `history-${risk.code}`,
      severity: risk.severity === 'warning' ? 'warning' : 'info',
      title: signalTitle(risk.code),
      detail: risk.message,
      source: 'history',
    })
  }
}

function deriveStatus(
  input: SessionRunbookInput,
  signals: SessionRunbookSignal[],
): SessionRunbookStatus {
  const hasOperatorBlock = Boolean(
    input.pendingApprovals?.length
    || input.pendingQuestions?.length
    || input.completionChecklist?.status === 'blocked'
    || input.evidenceManifest?.risks.some((risk) => risk.code === 'pending_tool_results'),
  )
  const hasCritical = signals.some((signal) => signal.severity === 'critical')
  if (hasOperatorBlock || hasCritical) {
    return 'blocked'
  }
  if (input.resumableRun && !input.resumableRunIssue) {
    return 'recoverable'
  }
  const hasWarning = signals.some((signal) => signal.severity === 'warning')
  return hasWarning ? 'needs_attention' : 'ready'
}

function headlineForStatus(status: SessionRunbookStatus): string {
  switch (status) {
    case 'blocked':
      return 'Session needs operator action before it can be trusted.'
    case 'recoverable':
      return 'Session has an interrupted run checkpoint that can be resumed.'
    case 'needs_attention':
      return 'Session completed with evidence that needs review.'
    case 'ready':
      return 'Session evidence is ready for review.'
  }
}

function buildActions(
  input: SessionRunbookInput,
  status: SessionRunbookStatus,
  mechanicalValidation: SessionRunbookMechanicalValidationPlan,
): SessionRunbookAction[] {
  const id = commandSessionId(input.id)
  const outputPath = shellQuote(`sepilot-session-${safeFileStem(input.id)}.md`)
  const actions = new Map<string, SessionRunbookAction>()
  const hasAttention = status !== 'ready'

  const contractNeedsClarification = hasDeclaredRunContract(input)
    && (input.contractLedger?.status === 'blocked'
      || input.contractLedger?.status === 'needs_attention')
  if (input.pendingApprovals?.length || input.pendingQuestions?.length || contractNeedsClarification) {
    addAction(actions, {
      id: 'open-chat',
      priority: 'now',
      title: 'Open the session in chat',
      command: `sepilot chat --session ${id}`,
      description: contractNeedsClarification
        ? 'Clarify missing contract sections or authority blockers before treating the session as complete.'
        : 'Answer pending questions or approve/deny pending tool requests from the interactive shell.',
      destructive: false,
      requiresReview: true,
    })
  }

  if (input.resumableRun) {
    addAction(actions, {
      id: 'resume-run',
      priority: 'now',
      title: input.resumableRun.forceRequired ? 'Resume with force review' : 'Resume interrupted run',
      command: `sepilot chat --session ${id}`,
      description: input.resumableRun.forceRequired
        ? 'Open the session and run /resume --force after reviewing replay risk.'
        : 'Open the session and run /resume to continue from the checkpoint.',
      destructive: false,
      requiresReview: input.resumableRun.forceRequired,
    })
  }

  if (mechanicalValidation.status === 'ready') {
    for (const command of mechanicalValidation.commands) {
      addAction(actions, {
        id: `validate-${command.kind}`,
        priority: 'now',
        title: `Run ${command.label.toLowerCase()} validation`,
        command: command.command,
        description: `${command.reason} Inferred from ${mechanicalValidation.toolchain ?? 'project'} markers; the command is not executed automatically.`,
        destructive: false,
        requiresReview: false,
      })
    }
  }

  if (hasAttention) {
    addAction(actions, {
      id: 'branch-session',
      priority: 'next',
      title: 'Branch before repair',
      command: `sepilot sessions branch ${id}`,
      description: 'Create a fork before retrying validation or repair work, preserving the original evidence trail.',
      destructive: false,
      requiresReview: false,
    })
  }

  addAction(actions, {
    id: 'show-session',
    priority: hasAttention ? 'next' : 'optional',
    title: 'Inspect session detail',
    command: `sepilot sessions show ${id}`,
    description: 'Review the event tail, evidence manifest, pending approvals, and working memory.',
    destructive: false,
    requiresReview: false,
  })

  addAction(actions, {
    id: 'export-session',
    priority: hasAttention ? 'next' : 'optional',
    title: 'Export sanitized transcript',
    command: `sepilot sessions export ${id} --sanitize --output ${outputPath}`,
    description: 'Write a sanitized Markdown transcript for local review or issue handoff.',
    destructive: false,
    requiresReview: false,
  })

  addAction(actions, {
    id: 'support-bundle',
    priority: hasAttention ? 'next' : 'optional',
    title: 'Create diagnostic support bundle',
    command: 'sepilot diagnostics bundle --range 7d --limit 100',
    description: 'Collect redacted observability, feedback, crash, and health context around the session.',
    destructive: false,
    requiresReview: false,
  })

  return [...actions.values()].sort((a, b) =>
    actionPriorityRank(a.priority) - actionPriorityRank(b.priority)
    || a.id.localeCompare(b.id)
  )
}

export function buildSessionRunbook(
  input: SessionRunbookInput,
  options: SessionRunbookBuildOptions = {},
): SessionRunbook {
  const generatedAt = options.generatedAt ?? new Date().toISOString()
  const mechanicalValidation = buildMechanicalValidationPlan(input, options)
  const signals: SessionRunbookSignal[] = []
  buildEvidenceSignals(input, signals)
  buildEvaluationGateSignals(input, signals)
  buildContractLedgerSignals(input, signals)
  buildRuntimeSignals(input, signals)
  buildChecklistSignals(input, signals)
  buildHistorySignals(input, signals)
  signals.sort((a, b) =>
    signalSeverityRank(a.severity) - signalSeverityRank(b.severity)
    || a.id.localeCompare(b.id)
  )

  const status = deriveStatus(input, signals)
  const evidenceSummary = input.evidenceManifest?.summary
  const warningSignals = signals.filter((signal) => signal.severity === 'warning').length
  const criticalSignals = signals.filter((signal) => signal.severity === 'critical').length

  return {
    schemaVersion: 1,
    generatedAt,
    status,
    headline: headlineForStatus(status),
    session: {
      id: input.id,
      title: input.title,
      status: input.status,
      provider: input.provider,
      model: input.model,
      createdAt: input.createdAt,
      updatedAt: input.updatedAt,
      eventCount: input.events?.length ?? input.evidenceManifest?.eventCount ?? 0,
    },
    summary: {
      signals: signals.length,
      criticalSignals,
      warningSignals,
      pendingApprovals: input.pendingApprovals?.length ?? 0,
      pendingQuestions: input.pendingQuestions?.length ?? 0,
      evidenceRisks: input.evidenceManifest?.risks.length ?? 0,
      validationRuns: evidenceSummary?.validationRuns ?? 0,
      validationFailures: evidenceSummary?.validationFailures ?? 0,
      filesChanged: evidenceSummary?.filesChanged ?? 0,
      acceptanceCriteria: evidenceSummary?.acceptanceCriteria ?? 0,
      supportedAcceptanceCriteria: evidenceSummary?.supportedAcceptanceCriteria ?? 0,
      failedAcceptanceCriteria: evidenceSummary?.failedAcceptanceCriteria ?? 0,
      blockedAcceptanceCriteria: evidenceSummary?.blockedAcceptanceCriteria ?? 0,
      unverifiedAcceptanceCriteria: evidenceSummary?.unverifiedAcceptanceCriteria ?? 0,
      validationSuggestions: mechanicalValidation.commands.length,
    },
    signals,
    actions: buildActions(input, status, mechanicalValidation),
    evidence: {
      manifestRevision: input.evidenceManifest?.revision,
      manifestStatus: input.evidenceManifest?.status,
      acceptanceCriteria: input.evidenceManifest?.acceptanceCriteria ?? [],
      risks: input.evidenceManifest?.risks ?? [],
      artifacts: selectRunbookArtifacts(input.evidenceManifest),
    },
    mechanicalValidation,
    supportBundle: {
      recommended: status !== 'ready',
      command: 'sepilot diagnostics bundle --range 7d --limit 100',
      reason: status === 'ready'
        ? 'Optional for archival handoff.'
        : 'Recommended because this session has warnings, blockers, or recovery state.',
    },
  }
}
