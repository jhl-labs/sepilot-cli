import type {
  AgentCriterionVerdictSnapshot,
  AgentEvidenceLedgerEntry,
  AgentState,
} from './types.js'
import {
  collectCriterionReferenceableObservations,
  criterionEvidenceEpisode,
  documentArtifactCriterionReviewEligible,
} from '../criterion-evidence-review.js'
import { isExecutorConfirmedExternalActionReceipt } from '../external-action-receipt-evidence.js'
import {
  browserInteractionTargetStateIsSpecific,
  compareEvidenceOrder,
  evidenceEntryOccursAfter,
  evaluateContractEvidenceGaps,
  toolCallRepresentsProductMutation,
  type EvidenceOrderMarker,
} from './evidence-ledger.js'
import {
  contractRequiresRenderedUiValidation,
  inputRequiresRenderedUiInteractionSmoke,
  renderedUiRequestedScope,
  type RenderedUiRequestedScope,
} from '../task-contract.js'
import { parseTodoItems } from '../../tools/todo.js'
import { hasAcceptedProvidedContextReview } from '../provided-context-review.js'

/**
 * Completion gate: before a run with a contract that declares acceptance
 * criteria is allowed to finish, the final answer must be backed by at least
 * one *verified* evidence ledger entry (validation pass or artifact read-back).
 * Structured `CRITERION <id>: MET|UNMET` lines remain supported diagnostics:
 * once a model starts that protocol it must close every criterion, and an
 * explicit UNMET always blocks. A normal user-facing answer is not rejected
 * solely for omitting internal protocol text. Runs without acceptance criteria
 * (lightweight react turns) are never gated. Blocking is bounded so the gate
 * can never loop forever: past the block budget the run passes and reports
 * honestly.
 */
export const MAX_COMPLETION_GATE_BLOCKS = 2

/**
 * Operator knob for the block budget: `SEPILOTD_COMPLETION_GATE_MAX_BLOCKS`
 * (default MAX_COMPLETION_GATE_BLOCKS, clamped to 1..5). A general option for
 * the shared daemon — never a per-model or per-dataset branch.
 */
export function resolveMaxCompletionGateBlocks(): number {
  const raw = process.env.SEPILOTD_COMPLETION_GATE_MAX_BLOCKS
  if (!raw) return MAX_COMPLETION_GATE_BLOCKS
  const n = Number(raw)
  if (!Number.isFinite(n)) return MAX_COMPLETION_GATE_BLOCKS
  return Math.min(5, Math.max(1, Math.floor(n)))
}

// Line parser for the gate's own structured output protocol (same family as
// the existing ANSWER:/INCOMPLETE: stem parsers) — not content matching.
const CRITERION_LINE = /^CRITERION\s+(\S+):\s*(MET|UNMET)(?:\s+EVIDENCE\s+([A-Za-z0-9][A-Za-z0-9_.:-]{0,127}(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9_.:-]{0,127})*))?\s*$/i
const HONEST_BLOCKER_STEM = /^\s*(?:UNVERIFIED|INCOMPLETE):/im
const MAX_CRITERION_EVIDENCE_REFERENCES = 16

interface ParsedCriterionVerdict {
  verdict: 'MET' | 'UNMET'
  evidenceToolCallIds: string[]
}

export interface CompletionGateResult {
  decision: 'pass' | 'block'
  unmet: string[]
  reason?: string
  /** Criterion diagnostics retained for this exact tool-result episode. */
  criterionVerdictSnapshot?: AgentCriterionVerdictSnapshot
  /** True only when the retry budget is exhausted and the caller must report INCOMPLETE. */
  budgetExhausted?: boolean
  /** Structural cause of a block, so retry guidance never parses `reason`. */
  cause?: 'criteria' | 'observation_evidence' | 'evidence' | 'todo' | 'ui_audit'
}

export function evaluateCompletionGate(
  state: Pick<
    AgentState,
    'seedContract'
      | 'evidenceLedger'
      | 'completionGateBlocks'
      | 'todoList'
      | 'input'
      | 'visualAttachmentsDisabled'
      | 'toolCallHistory'
      | 'completionDiagnostics'
      | 'stuckRepeatForcedFinal'
      | 'forcedFinalSynthesisReason'
  > & Partial<Pick<AgentState, 'messages'>>,
  finalText: string,
): CompletionGateResult {
  // Operator kill switch (`SEPILOTD_COMPLETION_GATE=off`) for lightweight or
  // experimental runs. The gate is default-on: validation enforcement is a
  // general capability improvement, not a tuning branch.
  if (process.env.SEPILOTD_COMPLETION_GATE === 'off') {
    return { decision: 'pass', unmet: [] }
  }
  const criteria = state.seedContract?.acceptanceCriteria ?? []
  if (criteria.length === 0) return { decision: 'pass', unmet: [] }

  const currentVerdicts = criterionVerdicts(finalText)
  const criterionVerdictSnapshot = buildCriterionVerdictSnapshot(
    state,
    criteria,
    currentVerdicts,
  )
  const effectiveVerdicts = criterionVerdictsFromSnapshot(criterionVerdictSnapshot)
  const referenceableExecutionIds = criterionReferenceableExecutionIds(state)
  // Preserve ordinary bare MET protocol for contracts whose completion is
  // proven by generic validation, while applying evidence precedence when a
  // semantic review already established an exact criterion link. An explicit
  // UNMET remains authoritative. A new MET is authoritative when all of its
  // references are executor-confirmed. A bare/invalid MET is weaker than an
  // existing exact MET on the same evidence episode and cannot erase it when
  // a child graph hands its original candidate to a parent reporter.
  for (const [id, verdict] of currentVerdicts) {
    const retained = effectiveVerdicts.get(id)
    const retainedExactMet = retained?.verdict === 'MET'
      && retained.evidenceToolCallIds.length > 0
    const currentExactMet = verdict.verdict === 'MET'
      && verdict.evidenceToolCallIds.length > 0
      && verdict.evidenceToolCallIds.every((toolCallId) => (
        referenceableExecutionIds.has(toolCallId)
      ))
    if (verdict.verdict === 'UNMET' || currentExactMet || !retainedExactMet) {
      effectiveVerdicts.set(id, verdict)
    }
  }
  const withCriterionSnapshot = (
    result: CompletionGateResult,
  ): CompletionGateResult => ({
    ...result,
    criterionVerdictSnapshot,
  })

  const evaluated = evaluateWithoutBudget(state, finalText, criteria, effectiveVerdicts)
  const imageInputScope = renderedUiRequestedScope(state.input ?? '')
  const isBoundedCaptureOnly = imageInputScope.bounded
    && state.seedContract?.executionIntent?.workspaceMutation === 'forbidden'
    && (
      imageInputScope.excludesSemanticVisualInspection
      || !contractRequiresRenderedUiVisualSemanticInspection(state.seedContract)
    )
  const isRenderedUiImageInputBlocker =
    state.visualAttachmentsDisabled === true
    && contractRequiresRenderedUiValidation(state.seedContract)
    && !isBoundedCaptureOnly
    && reportsVisualImageInputBlocker(finalText)
  const missingImageBlockerVerdicts = isRenderedUiImageInputBlocker
    ? missingCriterionVerdictIds(criteria, effectiveVerdicts)
    : []
  // The image-input exception is itself a structured partial-completion path:
  // keep requiring explicit scope boundaries there so a visual blocker cannot
  // silently excuse unrelated criteria. Ordinary successful completions remain
  // free to use natural user-facing prose.
  const raw: CompletionGateResult = missingImageBlockerVerdicts.length > 0
    ? { decision: 'block', unmet: missingImageBlockerVerdicts, cause: 'criteria' }
    : evaluated

  const canUseRenderedUiImageInputException =
    isRenderedUiImageInputBlocker
    && raw.cause !== 'ui_audit'
    && hasFreshVerifiedEvidence(state.evidenceLedger)
    && missingCriterionVerdictIds(criteria, effectiveVerdicts).length === 0
    && unmetCriteriaAreOnlyVisualImageValidation(criteria, effectiveVerdicts)
    && openTodosAreOnlyVisualImageValidationBlockers(state.todoList)

  if (canUseRenderedUiImageInputException) {
    const nonVisualEvidenceGaps = renderedUiNonVisualEvidenceGaps(state, finalText)
    if (nonVisualEvidenceGaps.length > 0) {
      return withCriterionSnapshot({
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: `rendered UI image-input blocker must still summarize non-visual validation evidence: missing ${nonVisualEvidenceGaps.slice(0, 4).join(', ')}`,
      })
    }
    return withCriterionSnapshot({
      decision: 'pass',
      unmet: [],
      reason: 'rendered UI visual inspection is honestly reported as blocked by image-input support after verified non-visual evidence',
    })
  }
  if ((state.completionGateBlocks ?? 0) >= resolveMaxCompletionGateBlocks()) {
    // Block budget exhausted: pass honestly instead of looping forever, but
    // keep the unresolved verdict visible so the caller can report it.
    if (raw.decision === 'block') {
      if (canAcceptStructurallyVerifiedArtifactDraft(state, finalText, raw)) {
        return withCriterionSnapshot({
          decision: 'pass',
          unmet: [],
          reason: 'natural-language completion accepted after structurally verified simple artifact work',
        })
      }
      return withCriterionSnapshot({
        decision: 'pass',
        unmet: raw.unmet,
        reason: 'completion gate block budget exhausted; acceptance criteria remain unproven',
        budgetExhausted: true,
        cause: raw.cause,
      })
    }
    return withCriterionSnapshot({ decision: 'pass', unmet: [] })
  }
  return withCriterionSnapshot(raw)
}

function canAcceptStructurallyVerifiedArtifactDraft(
  state: Pick<
    AgentState,
    'seedContract'
      | 'evidenceLedger'
      | 'completionGateBlocks'
      | 'todoList'
      | 'input'
      | 'visualAttachmentsDisabled'
      | 'toolCallHistory'
  >,
  finalText: string,
  result: CompletionGateResult,
): boolean {
  if (result.cause !== 'criteria' || HONEST_BLOCKER_STEM.test(finalText)) return false
  if (finalText.trim().length < 12 || incompleteTodos(state.todoList).length > 0) return false

  const contract = state.seedContract
  const artifacts = contract?.requiredArtifacts ?? []
  if (
    !contract
    || artifacts.length === 0
    || artifacts.some((artifact) => artifact.kind === 'directory')
    || (contract.artifactSections?.length ?? 0) > 0
    || contractRequiresRenderedUiValidation(contract)
  ) {
    return false
  }

  if ((contract.evidenceRequirements ?? []).some((requirement) =>
    requirement.kind === 'source'
    || requirement.kind === 'repository'
    || requirement.kind === 'validation'
    || requirement.requiresArtifactEvidenceMap === true
    || requirement.requiresArtifactSelfReview === true
  )) {
    return false
  }

  if (evaluateContractEvidenceGaps(state as AgentState).length > 0) return false

  const verified = verifiedEvidenceEntries(state.evidenceLedger)
    .filter((entry) => entry.verified === true)
    .sort(compareEvidenceOrder)
    .at(-1)
  if (!verified) return false
  const latestError = [...(state.evidenceLedger?.errors ?? [])]
    .sort(compareEvidenceOrder)
    .at(-1)
  return !latestError || compareEvidenceOrder(latestError, verified) < 0
}

export function buildCompletionGateBudgetExhaustedOutput(
  result: CompletionGateResult,
  rejectedDraft?: string,
): string {
  const unresolved = result.unmet.length > 0
    ? `acceptance criteria not closed as MET: ${result.unmet.join(', ')}`
    : result.reason ?? 'completion gate block budget exhausted; acceptance criteria remain unproven'
  const sentence = unresolved.endsWith('.') ? unresolved : `${unresolved}.`
  const draft = rejectedDraft?.trim()
  return [
    `INCOMPLETE: ${sentence}`,
    draft
      // Real tool work happened and the model produced a substantive draft;
      // discarding it into a bare "no final answer" wastes the run. Surface
      // it honestly marked as unverified instead of pretending it is done.
      ? 'The draft below did not pass completion verification — treat it as unverified.'
      : 'The previous completion draft was rejected by the completion gate and is not being reported as a finished result.',
    `Next required action: ${completionGateRecoveryHint(result, { includeProtocolStems: false })}`,
    ...(draft ? ['', '--- Unverified draft ---', draft] : []),
  ].join('\n')
}

function completionGateRecoveryHint(
  result: CompletionGateResult,
  options: { includeProtocolStems?: boolean; requireExecutableAction?: boolean } = {},
): string {
  if (result.cause === 'evidence' && options.requireExecutableAction === true) {
    return 'the next response must issue a relevant advertised tool call using the active tool-call format so fresh evidence is actually executed. Do not return planning prose, a bare tool name, or a JSON payload without an executable tool call. If no advertised tool can run, reply with `INCOMPLETE:` and the concrete capability or policy blocker.'
  }
  if (result.cause === 'observation_evidence') {
    if (options.includeProtocolStems === false) {
      return 'continue working on the unmet criteria, then close every acceptance criterion with an explicit verdict and provide the required final answer or incomplete-blocker response.'
    }
    return 'for each listed criterion, emit `CRITERION <id>: MET EVIDENCE <exact successful tool-call-id,...>` using only criterion-referenceable ids from the evidence ledger, then put the concise user-facing result after `ANSWER:`. Use `CRITERION <id>: UNMET` and an `INCOMPLETE:` response if the evidence does not satisfy it.'
  }
  if (result.unmet.length > 0) {
    if (options.includeProtocolStems === false) {
      return 'continue working on the unmet criteria, then close every acceptance criterion with an explicit verdict and provide the required final answer or incomplete-blocker response.'
    }
    return 'continue working on the unmet criteria, then provide a concise final answer or an INCOMPLETE: response naming the concrete blocker.'
  }
  if (result.cause === 'todo') {
    return 'finish the remaining todo items, or mark items that no longer apply as completed with todowrite, then provide the final answer.'
  }
  if (result.cause === 'ui_audit') {
    return 'continue only with the rendered UI validation actions and viewport/interaction scope stated in the run contract, resolve any reported defect when mutation is authorized, then summarize the evidence requested by that contract without adding excluded screenshots, interactions, tools, or QA bookkeeping.'
  }
  return 'run a validation/check step or read back the required artifact so the evidence ledger records verified evidence, then provide the final answer.'
}

function verifiedEvidenceEntries(
  stateLedger: Pick<NonNullable<AgentState['evidenceLedger']>, 'validationRuns' | 'artifactReadBacks'> | undefined,
): AgentEvidenceLedgerEntry[] {
  return [
    ...(stateLedger?.validationRuns ?? []),
    ...(stateLedger?.artifactReadBacks ?? []),
  ].filter((entry) => entry.verified)
}

/**
 * Whether the contract promised something to be produced. Such a run still owes
 * verification of what it produced, so observation evidence must not stand in
 * for it. Contracts often carry no executionIntent at all, so this asks what
 * was promised rather than reading a mutation flag that is frequently absent.
 */
function contractOwesArtifactWork(
  contract: AgentState['seedContract'] | undefined,
): boolean {
  if (!contract) return false
  return (contract.requiredArtifacts?.length ?? 0) > 0
    || (contract.artifactSections?.length ?? 0) > 0
    || contract.executionIntent?.workspaceMutation === 'required'
}

/**
 * Successful read-only observation evidence. A run that produces nothing has
 * no validation run and no artifact read-back to offer, so this is the only
 * evidence it can carry.
 */
function hasSuccessfulObservationEvidence(
  stateLedger: Pick<
    NonNullable<AgentState['evidenceLedger']>,
    'sourceReads' | 'sourceSearches'
  > | undefined,
): boolean {
  return [
    ...(stateLedger?.sourceReads ?? []),
    ...(stateLedger?.sourceSearches ?? []),
  ].some((entry) => entry.status === 'success')
}

function hasFreshVerifiedEvidence(
  stateLedger: Pick<
    NonNullable<AgentState['evidenceLedger']>,
    'validationRuns' | 'artifactReadBacks' | 'artifactWrites'
  > | undefined,
): boolean {
  const verified = verifiedEvidenceEntries(stateLedger)
  if (verified.length === 0) return false
  const latestWrite = latestSuccessfulArtifactWriteOrder(stateLedger)
  if (!latestWrite) return true
  return verified.some((entry) => evidenceEntryOccursAfter(entry, latestWrite))
}

const BROWSER_UI_AUDIT_TOOLS = new Set(['browser.screenshot', 'browser.click', 'browser.evaluate'])
const RENDERED_UI_FIX_PATH_EXTENSIONS = new Set([
  '.astro',
  '.avif',
  '.bmp',
  '.canvas',
  '.css',
  '.gif',
  '.htm',
  '.html',
  '.ico',
  '.jpeg',
  '.jpg',
  '.js',
  '.jsx',
  '.less',
  '.mjs',
  '.mdx',
  '.mp4',
  '.png',
  '.sass',
  '.scss',
  '.svelte',
  '.svg',
  '.ts',
  '.tsx',
  '.vue',
  '.webm',
  '.webp',
  '.woff',
  '.woff2',
])
const RENDERED_UI_FIX_CONFIG_BASENAMES = new Set([
  'next.config.js',
  'next.config.mjs',
  'next.config.ts',
  'nuxt.config.js',
  'nuxt.config.mjs',
  'nuxt.config.ts',
  'postcss.config.js',
  'tailwind.config.js',
  'tailwind.config.ts',
  'vite.config.js',
  'vite.config.mjs',
  'vite.config.ts',
])
const BROWSER_INTERACTION_SMOKE_TOOLS = new Set(['browser.click', 'browser.evaluate'])

type RenderedUiAuditViewport = 'desktop' | 'mobile'

function latestSuccessfulArtifactWriteOrder(
  ledger: Pick<NonNullable<AgentState['evidenceLedger']>, 'artifactWrites'> | undefined,
): EvidenceOrderMarker | null {
  let latest: AgentEvidenceLedgerEntry | null = null
  for (const entry of ledger?.artifactWrites ?? []) {
    if (entry.status !== 'success') continue
    if (!latest || compareEvidenceOrder(entry, latest) > 0) {
      latest = entry
    }
  }
  return latest ? { ts: latest.ts, order: latest.order } : null
}

function browserUiAuditEntriesAfterLatestWrite(
  ledger: AgentState['evidenceLedger'] | undefined,
): AgentEvidenceLedgerEntry[] {
  if (!ledger) return []
  const latestWrite = latestSuccessfulArtifactWriteOrder(ledger)
  const latestByKey = new Map<string, AgentEvidenceLedgerEntry>()
  for (const entry of [...ledger.sourceReads, ...ledger.sourceSearches]
    .filter((entry) =>
      entry.status === 'success'
      && BROWSER_UI_AUDIT_TOOLS.has(entry.tool)
      && (!latestWrite || evidenceEntryOccursAfter(entry, latestWrite))
    )) {
    const key = entry.uiAuditKey ?? entry.path ?? entry.query ?? entry.tool
    const existing = latestByKey.get(key)
    if (!existing || compareEvidenceOrder(entry, existing) >= 0) {
      latestByKey.set(key, entry)
    }
  }
  return [...latestByKey.values()]
}

function supersededBrowserUiAuditDefects(
  ledger: AgentState['evidenceLedger'] | undefined,
): string[] {
  if (!ledger) return []
  const grouped = new Map<string, AgentEvidenceLedgerEntry[]>()
  for (const entry of [...ledger.sourceReads, ...ledger.sourceSearches]
    .filter((candidate) =>
      candidate.status === 'success'
      && BROWSER_UI_AUDIT_TOOLS.has(candidate.tool)
    )) {
    const key = entry.uiAuditKey ?? entry.path ?? entry.query ?? entry.tool
    const entries = grouped.get(key) ?? []
    entries.push(entry)
    grouped.set(key, entries)
  }

  const defects: string[] = []
  for (const entries of grouped.values()) {
    const ordered = [...entries].sort(compareEvidenceOrder)
    const latest = ordered.at(-1)
    if (!latest || (latest.defects?.length ?? 0) > 0) continue
    for (const entry of ordered.slice(0, -1)) {
      defects.push(...(entry.defects ?? []))
    }
  }
  return defects
}

function browserUiAuditDefectsMissingFixMutation(
  ledger: AgentState['evidenceLedger'] | undefined,
): string[] {
  if (!ledger) return []
  const writes = ledger.artifactWrites.filter((entry) =>
    entry.status === 'success' && isRenderedUiFixMutationEntry(entry)
  )
  const grouped = new Map<string, AgentEvidenceLedgerEntry[]>()
  for (const entry of [...ledger.sourceReads, ...ledger.sourceSearches]
    .filter((candidate) =>
      candidate.status === 'success'
      && BROWSER_UI_AUDIT_TOOLS.has(candidate.tool)
    )) {
    const key = entry.uiAuditKey ?? entry.path ?? entry.query ?? entry.tool
    const entries = grouped.get(key) ?? []
    entries.push(entry)
    grouped.set(key, entries)
  }

  const defects: string[] = []
  for (const entries of grouped.values()) {
    const ordered = [...entries].sort(compareEvidenceOrder)
    const latest = ordered.at(-1)
    if (!latest || (latest.defects?.length ?? 0) > 0) continue
    for (const entry of ordered.slice(0, -1)) {
      if (!entry.defects?.length) continue
      const hasFixMutationBeforeCleanAudit = writes.some((write) =>
        compareEvidenceOrder(write, entry) > 0 && compareEvidenceOrder(latest, write) > 0
      )
      if (!hasFixMutationBeforeCleanAudit) {
        defects.push(...entry.defects)
      }
    }
  }
  return defects
}

function isRenderedUiFixMutationEntry(entry: AgentEvidenceLedgerEntry): boolean {
  const path = entry.path?.trim().toLowerCase()
  if (!path) return false
  const normalizedPath = path.split(/[?#]/, 1)[0]?.replace(/\\/g, '/') ?? ''
  const basename = normalizedPath.split('/').pop() ?? ''
  if (RENDERED_UI_FIX_CONFIG_BASENAMES.has(basename)) return true
  const extensionMatch = basename.match(/(\.[a-z0-9]+)$/i)
  return extensionMatch ? RENDERED_UI_FIX_PATH_EXTENSIONS.has(extensionMatch[1]!) : false
}

function browserInteractionSmokeEntriesAfterLatestWrite(
  ledger: AgentState['evidenceLedger'] | undefined,
): AgentEvidenceLedgerEntry[] {
  if (!ledger) return []
  const latestWrite = latestSuccessfulArtifactWriteOrder(ledger)
  return [...ledger.sourceReads, ...ledger.sourceSearches].filter((entry) =>
    entry.status === 'success'
    && BROWSER_INTERACTION_SMOKE_TOOLS.has(entry.tool)
    && entry.uiInteractionSmoke
    && (!latestWrite || evidenceEntryOccursAfter(entry, latestWrite))
  )
}

function renderedUiRequiresInteractionSmoke(
  state: Pick<AgentState, 'seedContract' | 'input'>,
): boolean {
  return inputRequiresRenderedUiInteractionSmoke([
    state.input,
    state.seedContract?.summary,
  ].filter(Boolean).join('\n'))
}

function renderedUiRequiresDynamicStateEvidence(
  state: Pick<AgentState, 'seedContract' | 'input'>,
): boolean {
  const text = [
    state.input,
    state.seedContract?.summary,
    ...(state.seedContract?.acceptanceCriteria ?? []).map((criterion) => criterion.text),
  ].filter(Boolean).join('\n')
  return /\b(?:games?|playable|animated|animation|canvas\s+games?|browser\s+games?|three\.?js|webgl|requestanimationframe)\b|(?:게임|플레이어블|애니메이션|캔버스\s*게임)/iu.test(text)
}

function classifyRenderedUiAuditViewport(entry: AgentEvidenceLedgerEntry): RenderedUiAuditViewport | null {
  const viewportMatch = entry.uiAuditKey?.match(/\bviewport:(\d+|auto)x(\d+|auto)\b/i)
  if (viewportMatch) {
    const width = viewportMatch[1] === 'auto' ? Number.NaN : Number(viewportMatch[1])
    if (Number.isFinite(width)) {
      if (width <= 767) return 'mobile'
      if (width >= 900) return 'desktop'
    }
  }

  const fallback = `${entry.uiAuditKey ?? ''} ${entry.path ?? ''}`.toLowerCase()
  if (/\b(?:mobile|phone|handset|small)\b/.test(fallback)) return 'mobile'
  if (/\b(?:desktop|wide|large)\b/.test(fallback)) return 'desktop'
  return null
}

function hasAttachedBrowserScreenshotEvidence(entry: AgentEvidenceLedgerEntry): boolean {
  return typeof entry.path === 'string'
    && entry.path.trim().length > 0
    && entry.uiAuditHasImageAttachment === true
}

function missingRenderedUiAuditViewports(
  audits: AgentEvidenceLedgerEntry[],
  required: readonly RenderedUiAuditViewport[] = ['desktop', 'mobile'],
): RenderedUiAuditViewport[] {
  const observed = new Set<RenderedUiAuditViewport>()
  for (const audit of audits) {
    const viewport = classifyRenderedUiAuditViewport(audit)
    if (viewport) observed.add(viewport)
  }
  return required.filter((viewport) => !observed.has(viewport))
}

function renderedUiAuditViewportDimension(entry: AgentEvidenceLedgerEntry): string | null {
  const match = entry.uiAuditKey?.match(/\bviewport:(\d{3,5})x(\d{3,5})\b/i)
  return match ? `${match[1]}×${match[2]}` : null
}

function missingRenderedUiAuditDimensions(
  audits: AgentEvidenceLedgerEntry[],
  required: readonly string[],
): string[] {
  if (required.length === 0) return []
  const observed = new Set(
    audits
      .map(renderedUiAuditViewportDimension)
      .filter((dimension): dimension is string => dimension !== null),
  )
  return required.filter((dimension) => !observed.has(dimension))
}

function requiredRenderedUiAuditViewports(
  scope: RenderedUiRequestedScope,
): RenderedUiAuditViewport[] {
  if (!scope.bounded) return ['desktop', 'mobile']
  if (scope.excludesMobile) return ['desktop']
  if (scope.excludesDesktop) return ['mobile']
  const inferred = scope.viewportDimensions
    .map((dimension) => Number.parseInt(dimension.split('×')[0] ?? '', 10))
    .map((width): RenderedUiAuditViewport | null => {
      if (width <= 767) return 'mobile'
      if (width >= 900) return 'desktop'
      return null
    })
    .filter((viewport): viewport is RenderedUiAuditViewport => viewport !== null)
  return [...new Set(inferred)]
}

function renderedUiAuditHasRequestedCoverage(
  audits: AgentEvidenceLedgerEntry[],
  scope: RenderedUiRequestedScope,
): boolean {
  const viewports = requiredRenderedUiAuditViewports(scope)
  return missingRenderedUiAuditViewports(audits, viewports).length === 0
    && missingRenderedUiAuditDimensions(
      audits,
      scope.bounded ? scope.viewportDimensions : [],
    ).length === 0
}

function normalizeRenderedUiAuditUrl(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) return ''
  try {
    const url = new URL(trimmed)
    const pathname = url.pathname === '/' ? '/' : url.pathname.replace(/\/+$/g, '')
    const hash = url.hash ? url.hash.replace(/\/+$/g, '') : ''
    return `${url.protocol}//${url.host}${pathname}${url.search}${hash}`.toLowerCase()
  } catch {
    return trimmed.replace(/\/+$/g, '').toLowerCase()
  }
}

function renderedUiAuditTargetUrl(entry: AgentEvidenceLedgerEntry): string | null {
  const rawUrl = entry.uiAuditKey?.split('|')[1]?.trim()
  if (!rawUrl || rawUrl === 'unknown-url') return null
  const normalized = normalizeRenderedUiAuditUrl(rawUrl)
  return normalized || null
}

function renderedUiAuditTargetWithRequestedCoverage(
  audits: AgentEvidenceLedgerEntry[],
  scope: RenderedUiRequestedScope,
): string | null | undefined {
  const grouped = new Map<string, AgentEvidenceLedgerEntry[]>()
  for (const audit of audits) {
    const target = renderedUiAuditTargetUrl(audit)
    if (!target) continue
    const entries = grouped.get(target) ?? []
    entries.push(audit)
    grouped.set(target, entries)
  }
  if (grouped.size === 0) {
    return renderedUiAuditHasRequestedCoverage(audits, scope) ? null : undefined
  }
  for (const [target, entries] of grouped) {
    if (renderedUiAuditHasRequestedCoverage(entries, scope)) return target
  }
  return undefined
}

function renderedUiAuditInteractionTarget(entry: AgentEvidenceLedgerEntry): string | null {
  const parts = entry.uiAuditKey?.split('|')
  if (!parts || parts.length < 4) return null
  const tool = parts[0]?.trim().toLowerCase()
  const rawUrl = parts[1]?.trim()
  const state = parts[3]?.trim().toLowerCase().replace(/\s+/g, ' ')
  if (!tool || !rawUrl || !state) return null
  const url = rawUrl === 'unknown-url' ? rawUrl : normalizeRenderedUiAuditUrl(rawUrl)
  if (!url) return null
  return `${url}|${tool}|${state}`
}

function renderedUiAuditHasSpecificInteractionTarget(entry: AgentEvidenceLedgerEntry): boolean {
  const parts = entry.uiAuditKey?.split('|')
  if (!parts || parts.length < 4) return false
  const tool = parts[0]?.trim().toLowerCase()
  const state = parts[3]?.trim().toLowerCase().replace(/\s+/g, ' ')
  return browserInteractionTargetStateIsSpecific(tool ?? '', state)
}

function renderedUiInteractionTargetWithAllViewports(
  audits: AgentEvidenceLedgerEntry[],
): string | null | undefined {
  const grouped = new Map<string, AgentEvidenceLedgerEntry[]>()
  for (const audit of audits) {
    const target = renderedUiAuditInteractionTarget(audit)
    if (!target) continue
    const entries = grouped.get(target) ?? []
    entries.push(audit)
    grouped.set(target, entries)
  }
  if (grouped.size === 0) return null
  for (const [target, entries] of grouped) {
    if (missingRenderedUiAuditViewports(entries).length === 0) {
      return target
    }
  }
  return undefined
}

function finalTextReportsFixAndRescreenshotPass(normalizedFinalText: string): boolean {
  return /(?:fix(?:ed|es)?|resolved?|수정|고침|해결)/i.test(normalizedFinalText)
    && /(?:re-?screenshot(?:ed)?|fresh\s+(?:screenshot|audit)|follow-?up\s+(?:screenshot|audit)|clean\s+(?:follow-?up|fresh)\s+(?:screenshot|audit)|captured\s+fresh|재촬영|다시\s*(?:캡처|촬영|검증)|새\s*스크린샷)/i.test(normalizedFinalText)
}

function finalTextReportsVisualQaIssueDetail(normalizedFinalText: string): boolean {
  return /(?:overlap|colli(?:de|sion)|contrast|overflow|cut\s*off|cutoff|clipp|blank|spacing|wrap|low-contrast|low-detail|visual\s+density|touch-target|too\s+small|too\s+tight|cramped|hard\s+to\s+read|unpolished|unfinished|broken|canvas\s+band|top\s+band|score\s+label|restart\s+button)/i
    .test(normalizedFinalText)
    || /(?:겹|충돌|대비|저대비|오버플로|잘림|공백|간격|깨져|삐져|터치\s*타깃|너무\s*작|읽기\s*어려|미완성|어색|캔버스|상단\s*띠|점수\s*라벨|재시작\s*버튼)/i
      .test(normalizedFinalText)
}

const GENERIC_INTERACTION_TARGET_TOKENS = new Set([
  'aria',
  'button',
  'click',
  'data',
  'dispatch',
  'event',
  'evaluate',
  'getcontext',
  'input',
  'label',
  'mouseevent',
  'new',
  'script',
  'selector',
  'test',
  'testid',
  'window',
])

function interactionTargetTokens(target: string | null | undefined): string[] {
  const state = target?.split('|')[2]?.toLowerCase() ?? ''
  if (!state) return []
  return [...new Set(
    state
      .replace(/^evaluate:/, '')
      .replace(/^selector:/, '')
      .replace(/^id:/, '')
      .replace(/[^a-z0-9가-힣]+/g, ' ')
      .split(/\s+/)
      .filter((token) =>
        token.length >= 3
        && !GENERIC_INTERACTION_TARGET_TOKENS.has(token)
      ),
  )]
}

function renderedUiValidatedInteractionTarget(
  state: Pick<AgentState, 'evidenceLedger' | 'visualAttachmentsDisabled'>,
): string | null | undefined {
  const interactionSmokes = browserInteractionSmokeEntriesAfterLatestWrite(state.evidenceLedger)
    .filter(renderedUiAuditHasSpecificInteractionTarget)
  const activeEvidenceSmokes = state.visualAttachmentsDisabled === true
    ? interactionSmokes
    : interactionSmokes.filter(hasAttachedBrowserScreenshotEvidence)
  return renderedUiInteractionTargetWithAllViewports(
    activeEvidenceSmokes.filter((entry) => entry.uiAuditHasLayout),
  )
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function finalTextContainsInteractionTargetToken(normalizedFinalText: string, token: string): boolean {
  return new RegExp(`(?:^|[^a-z0-9가-힣])${escapeRegExp(token)}(?:$|[^a-z0-9가-힣])`, 'i')
    .test(normalizedFinalText)
}

function finalTextReportsInteractionTarget(normalizedFinalText: string, target: string | null | undefined): boolean {
  const tokens = interactionTargetTokens(target)
  return tokens.length === 0
    || tokens.some((token) => finalTextContainsInteractionTargetToken(normalizedFinalText, token))
}

function browserEvaluateReportsDynamicStateEvidence(entry: AgentEvidenceLedgerEntry): boolean {
  if (entry.tool !== 'browser.evaluate') return false
  const text = `${entry.summary ?? ''} ${entry.uiAuditKey ?? ''}`
    .toLowerCase()
    .replace(/\s+/g, ' ')
  if (!text) return false
  if (/"(?:canvas|frame|pixel|image|position|sprite|animation|motion|game|state|tick|time|elapsed)[a-z0-9_-]*(?:changed|advanced|moved|delta|progress(?:ed)?|running|animated|moving)"?\s*:\s*(?:true|[1-9]\d*(?:\.\d+)?)/i.test(text)) {
    return true
  }
  return /\b(?:canvas|frame|pixel|image|position|sprite|animation|motion|game\s*state|tick|elapsed|time)\b.{0,80}\b(?:changed|different|moved|advanced|delta|progressed|running|animated|moving)\b/i.test(text)
}

function finalTextReportsDynamicStateEvidence(normalizedFinalText: string): boolean {
  return /\b(?:frame|pixel|canvas|animation|motion|movement|game\s*state|tick|elapsed|time)\b.{0,120}\b(?:changed|different|moved|advanced|delta|progressed|running|animated|moving|verified|validated)\b/i
    .test(normalizedFinalText)
    || /(?:프레임|픽셀|캔버스|애니메이션|움직임|동작|게임\s*상태|시간).{0,80}(?:변화|달라|진행|이동|움직|검증|확인)/i
      .test(normalizedFinalText)
}

function renderedUiFinalAnswerGaps(
  state: Pick<
    AgentState,
    'seedContract' | 'input' | 'visualAttachmentsDisabled' | 'evidenceLedger' | 'toolCallHistory'
  >,
  finalText: string,
): string[] {
  if (state.visualAttachmentsDisabled === true && reportsVisualImageInputBlocker(finalText)) return []
  const normalized = finalText.toLowerCase()
  const gaps: string[] = []
  if (!/(?:design|ux|ui)\s+(?:plan|brief|direction)|target\s+user|primary\s+screens?|non-empty\s+states?|responsive\s+layout|visual\s+style|디자인\s*(?:계획|브리프|방향)|대상\s*사용자|주요\s*화면|빈\s*상태|반응형\s*레이아웃|시각\s*스타일/i.test(normalized)) {
    gaps.push('design plan summary')
  }
  if (!/(?:desktop|데스크톱|wide viewport|1440x|1280x|viewport[^.\n]*(?:desktop|1440|1280))/i.test(normalized)) {
    gaps.push('desktop viewport inspected')
  }
  if (!/(?:mobile|모바일|phone|전화|390x|375x|414x|viewport[^.\n]*(?:mobile|390|375|414))/i.test(normalized)) {
    gaps.push('mobile viewport inspected')
  }
  if (
    state.visualAttachmentsDisabled !== true
    && !/(?:screenshot\s+image\s+attachment:\s*attached|screenshot(?:s| images?)?\s+(?:were\s+)?attached|attached\s+screenshot(?:s| images?)?|스크린샷.{0,24}첨부|이미지.{0,24}첨부)/i.test(normalized)
  ) {
    gaps.push('attached screenshot image evidence')
  }
  if (!/(?:visual qa|visual review|visual inspection|시각\s*(?:검수|리뷰)|layout issues?|visual issues?|issues? (?:found|none)|none found|no (?:visible|visual|layout) (?:issues?|defects?))/i.test(normalized)) {
    gaps.push('visual QA issues found or explicitly none')
  }
  const supersededDefects = supersededBrowserUiAuditDefects(state.evidenceLedger)
  if (
    supersededDefects.length > 0
    && !finalTextReportsFixAndRescreenshotPass(normalized)
  ) {
    gaps.push('fix-and-rescreenshot pass for prior browser audit defects')
  }
  const visualQaIssueCount = completedRenderedUiVisualQaIssueIndices(state.toolCallHistory).length
  if (
    visualQaIssueCount > 0
    && !finalTextReportsFixAndRescreenshotPass(normalized)
  ) {
    gaps.push('fix-and-rescreenshot pass for visual QA issues')
  }
  if (
    visualQaIssueCount > 0
    && !finalTextReportsVisualQaIssueDetail(normalized)
  ) {
    gaps.push('visual QA issue detail')
  }
  if (!/(?:console|page\s*error|pageerror|콘솔|페이지\s*오류)/i.test(normalized)) {
    gaps.push('console/page-error result')
  }
  if (
    renderedUiRequiresInteractionSmoke(state)
    && !/(?:interaction|interactive|active state|browser\.click|browser\.evaluate|click smoke|evaluate smoke|상호작용|인터랙션|클릭)/i.test(normalized)
  ) {
    gaps.push('interaction smoke result')
  }
  if (
    renderedUiRequiresInteractionSmoke(state)
    && !finalTextReportsInteractionTarget(
      normalized,
      renderedUiValidatedInteractionTarget(state),
    )
  ) {
    gaps.push('interaction smoke target')
  }
  if (
    renderedUiRequiresDynamicStateEvidence(state)
    && !finalTextReportsDynamicStateEvidence(normalized)
  ) {
    gaps.push('dynamic canvas/game state evidence')
  }
  return gaps
}

function renderedUiNonVisualEvidenceGaps(
  state: Pick<AgentState, 'seedContract' | 'input'>,
  finalText: string,
): string[] {
  const normalized = finalText.toLowerCase()
  const gaps: string[] = []
  if (!/(?:design|ux|ui)\s+(?:plan|brief|direction)|target\s+user|primary\s+screens?|non-empty\s+states?|responsive\s+layout|visual\s+style|디자인\s*(?:계획|브리프|방향)|대상\s*사용자|주요\s*화면|빈\s*상태|반응형\s*레이아웃|시각\s*스타일/i.test(normalized)) {
    gaps.push('design plan summary')
  }
  if (!/(?:desktop|데스크톱|wide viewport|1440x|1280x|viewport[^.\n]*(?:desktop|1440|1280))/i.test(normalized)) {
    gaps.push('desktop viewport inspected')
  }
  if (!/(?:mobile|모바일|phone|전화|390x|375x|414x|viewport[^.\n]*(?:mobile|390|375|414))/i.test(normalized)) {
    gaps.push('mobile viewport inspected')
  }
  if (!/(?:layout\s+audits?|dom\/canvas|browser\s+layout|레이아웃\s*(?:감사|검사|확인|검증))/i.test(normalized)) {
    gaps.push('layout audit result')
  }
  if (!/(?:console|page\s*error|pageerror|콘솔|페이지\s*오류)/i.test(normalized)) {
    gaps.push('console/page-error result')
  }
  if (
    renderedUiRequiresInteractionSmoke(state)
    && !/(?:interaction|interactive|active state|browser\.click|browser\.evaluate|click smoke|evaluate smoke|상호작용|인터랙션|클릭)/i.test(normalized)
  ) {
    gaps.push('interaction smoke result')
  }
  if (
    renderedUiRequiresDynamicStateEvidence(state)
    && !finalTextReportsDynamicStateEvidence(normalized)
  ) {
    gaps.push('dynamic canvas/game state evidence')
  }
  return gaps
}

const GENERIC_DESIGN_BRIEF_WORDS = new Set([
  'active',
  'asset',
  'assets',
  'brief',
  'button',
  'buttons',
  'click',
  'color',
  'colors',
  'control',
  'controls',
  'desktop',
  'design',
  'direction',
  'empty',
  'error',
  'image',
  'images',
  'interaction',
  'interactions',
  'layout',
  'media',
  'mobile',
  'non',
  'plan',
  'primary',
  'required',
  'responsive',
  'screen',
  'screens',
  'state',
  'states',
  'style',
  'target',
  'typography',
  'user',
  'users',
  'viewport',
  'viewports',
  'visual',
  'workflow',
  '계획',
  '방향',
  '대상',
  '데스크톱',
  '디자인',
  '레이아웃',
  '모바일',
  '미디어',
  '버튼',
  '브리프',
  '반응형',
  '사용자',
  '상태',
  '색상',
  '스타일',
  '시각',
  '상호작용',
  '에셋',
  '이미지',
  '인터랙션',
  '주요',
  '필요한',
  '컨트롤',
  '클릭',
  '타이포',
  '화면',
  '활성',
  '빈',
  '흐름',
])

function concreteDesignBriefTokenCount(content: string): number {
  const tokens = content
    .toLowerCase()
    .replace(/[^a-z0-9가-힣]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
  return tokens.filter((token) => (
    token.length >= 3
    && !GENERIC_DESIGN_BRIEF_WORDS.has(token)
    && !/^\d+$/.test(token)
  )).length
}

function renderedUiDesignBriefFacetCount(content: string): number {
  const text = content.toLowerCase()
  return [
    /target\s+(?:user|workflow)|user\s+workflow|대상\s*사용자|사용자\s*흐름|workflow/i,
    /primary\s+(?:screens?|views?)|non-empty|empty\s+state|active\s+state|error\s+state|주요\s*화면|빈\s*상태|활성\s*상태/i,
    /responsive|mobile|desktop|viewport|반응형|모바일|데스크톱/i,
    /visual\s+style|style\s+direction|color|typography|polish|시각\s*스타일|스타일|색상|타이포/i,
    /assets?|media|images?|sprites?|required\s+(?:assets|media)|이미지|미디어|에셋/i,
    /interactions?|controls?|buttons?|click|active\s+state|상호작용|인터랙션|컨트롤|버튼|클릭/i,
  ].filter((pattern) => pattern.test(text)).length
}

const REQUIRED_RENDERED_UI_DESIGN_BRIEF_FACETS = 6
const MIN_CONCRETE_DESIGN_BRIEF_TOKENS = 8

type RenderedUiDesignBriefQuality = 'ok' | 'missing' | 'placeholder' | 'incomplete' | 'late'
type RenderedUiVisualQaQuality = 'ok' | 'missing' | 'placeholder' | 'issue'

function todoContentHasRenderedUiDesignBriefShape(content: string): boolean {
  const text = content.toLowerCase()
  const namesBrief = /(?:design|ux|ui)\s*(?:brief|plan|direction)|디자인\s*(?:브리프|계획|방향)/i.test(text)
  const facets = renderedUiDesignBriefFacetCount(content)
  return (namesBrief && facets >= 3) || facets >= 4
}

function renderedUiDesignBriefQuality(content: string): RenderedUiDesignBriefQuality {
  if (!todoContentHasRenderedUiDesignBriefShape(content)) return 'missing'

  const facetCount = renderedUiDesignBriefFacetCount(content)
  if (facetCount < REQUIRED_RENDERED_UI_DESIGN_BRIEF_FACETS) return 'incomplete'

  const concreteTokenCount = concreteDesignBriefTokenCount(content)
  if (concreteTokenCount < MIN_CONCRETE_DESIGN_BRIEF_TOKENS) return 'placeholder'

  return 'ok'
}

function completedRenderedUiDesignBriefTodoStatusBeforeIndex(
  history: NonNullable<AgentState['toolCallHistory']>,
  index: number,
): Exclude<RenderedUiDesignBriefQuality, 'late'> {
  let sawPlaceholderBrief = false
  let sawIncompleteBrief = false
  for (let i = 0; i < index; i++) {
    const entry = history[i]
    if (!entry || entry.status !== 'success' || entry.tool !== 'todowrite') continue
    const todoItems = parseTodoItems((entry.input as { items?: unknown }).items)
    for (const item of todoItems ?? []) {
      if (item.status !== 'completed') continue
      const quality = renderedUiDesignBriefQuality(item.content)
      if (quality === 'ok') return 'ok'
      if (quality === 'incomplete') sawIncompleteBrief = true
      if (quality === 'placeholder') sawPlaceholderBrief = true
    }
  }
  if (sawIncompleteBrief) return 'incomplete'
  return sawPlaceholderBrief ? 'placeholder' : 'missing'
}

function completedRenderedUiDesignBriefTodoStatusBeforeImplementationMutation(
  history: AgentState['toolCallHistory'] | undefined,
  ledger: AgentState['evidenceLedger'] | undefined,
): RenderedUiDesignBriefQuality {
  if (!history || history.length === 0) {
    return latestSuccessfulArtifactWriteOrder(ledger) ? 'missing' : 'ok'
  }

  let firstMutationIndex = -1
  let latestMutationIndex = -1
  for (let i = 0; i < history.length; i++) {
    const entry = history[i]
    if (entry?.status !== 'success' || !toolCallRepresentsProductMutation(entry.tool, entry.input)) continue
    if (firstMutationIndex < 0) firstMutationIndex = i
    latestMutationIndex = i
  }
  if (firstMutationIndex < 0) return 'ok'

  const beforeFirst = completedRenderedUiDesignBriefTodoStatusBeforeIndex(
    history,
    firstMutationIndex,
  )
  if (beforeFirst !== 'missing') return beforeFirst

  const beforeLatest = completedRenderedUiDesignBriefTodoStatusBeforeIndex(
    history,
    latestMutationIndex,
  )
  return beforeLatest === 'ok' ? 'late' : beforeLatest
}

function finalTextReportsLateDesignBriefRecovery(finalText: string): boolean {
  const text = finalText.toLowerCase()
  const admitsLatePlanning =
    /(?:initial|first|early|unplanned|late|post-?hoc|after\s+(?:starting|the first|initial)|started\s+before|before\s+(?:the\s+)?design|초기|처음|사후|늦게|계획\s*없이|시작\s*후)/i.test(text)
  const mentionsDesignBrief =
    /(?:design|ux|ui)\s*(?:brief|plan|direction)|디자인\s*(?:브리프|계획|방향)/i.test(text)
  const reportsRework =
    /(?:rework(?:ed)?|revis(?:ed|ion)|redesign(?:ed)?|rebuilt|rewrote|refactor(?:ed)?|correct(?:ed)?|from\s+the\s+(?:completed\s+)?(?:design\s+)?(?:brief|plan)|재작업|다시\s*(?:수정|작성|구성)|수정|개선|고침|디자인\s*(?:브리프|계획).*(?:기준|따라))/i.test(text)
  return admitsLatePlanning && mentionsDesignBrief && reportsRework
}

const REQUIRED_RENDERED_UI_VISUAL_QA_ASPECTS = 3

function renderedUiVisualQaInspectionAspectCount(content: string): number {
  const text = content.toLowerCase()
  return [
    /(?:layout|spacing|alignment|composition|hierarchy|density|grid|gutter|padding|margin|레이아웃|간격|정렬|구도|위계|밀도|여백)/i,
    /(?:text|typography|copy|label|readab|wrapp|overflow|cut\s*off|cutoff|clipp|truncat|텍스트|타이포|문구|라벨|가독|줄바꿈|오버플로|잘림|삐져)/i,
    /(?:contrast|low-contrast|color|foreground|background|legib|대비|저대비|색상|전경|배경|선명)/i,
    /(?:overlap|colli(?:de|sion)|occlud|stacking|z-index|겹|충돌|가려)/i,
    /(?:control|button|input|touch[-\s]?target|hit\s*area|tap|click|keyboard|swipe|컨트롤|버튼|입력|터치\s*타깃|터치|클릭|키보드|스와이프)/i,
    /(?:assets?|media|image|icon|sprite|canvas|blank|placeholder|empty|unfinished|polish|completeness|에셋|미디어|이미지|아이콘|스프라이트|캔버스|빈|공백|플레이스홀더|미완성|완성도|다듬)/i,
  ].filter((pattern) => pattern.test(text)).length
}

function renderedUiVisualQaQuality(content: string): RenderedUiVisualQaQuality {
  const text = content.toLowerCase()
  const namesVisualQa =
    /(?:visual\s*(?:qa|review|inspection)|screenshot\s*(?:comparison|review)|시각\s*(?:검수|리뷰|검사)|스크린샷\s*(?:비교|검수|리뷰))/i.test(text)
  if (!namesVisualQa) return 'missing'

  const mentionsDesktop =
    /(?:desktop|데스크톱|wide viewport|1440x|1280x|viewport[^.\n]*(?:desktop|1440|1280))/i.test(text)
  const mentionsMobile =
    /(?:mobile|모바일|phone|전화|390x|375x|414x|viewport[^.\n]*(?:mobile|390|375|414))/i.test(text)
  const referencesDesignBasis =
    /(?:design|brief|plan|target user|workflow|디자인|브리프|계획|대상\s*사용자|흐름)/i.test(text)
  const reportsNoIssues =
    /(?:issues?\s*(?:found\s*)?(?:none|no\b)|none\s*found|no\s+(?:remaining\s+)?(?:visible|visual|layout)\s+(?:issues?|defects?)|문제(?:가)?\s*(?:없|없음|없었다)|이슈(?:가)?\s*(?:없|없음|없었다)|결함(?:이)?\s*(?:없|없음|없었다)|없음|없었다)/i.test(text)
  if (renderedUiVisualQaReportsIssueNeedingFollowUp(content)) return 'issue'
  const aspectCount = renderedUiVisualQaInspectionAspectCount(content)

  return mentionsDesktop
    && mentionsMobile
    && referencesDesignBasis
    && reportsNoIssues
    && aspectCount >= REQUIRED_RENDERED_UI_VISUAL_QA_ASPECTS
    ? 'ok'
    : 'placeholder'
}

function renderedUiVisualQaReportsIssueNeedingFollowUp(content: string): boolean {
  const text = content.toLowerCase()
  return /(?:issues?|defects?|problems?)\s+(?:found|detected|identified)\s*[:：-]?\s*(?!none\b|no\b|없(?:음|었다)?\b)(?:\w|[가-힣]|#|\.)/i.test(text)
    || /(?:issues?|defects?|problems?)\s*[:：-]\s*(?!none\b|no\b|없(?:음|었다)?\b)(?:\w|[가-힣]|#|\.)/i.test(text)
    || /(?:except|other\s+than|apart\s+from|but|however|though|still|remaining|minor|small)\b.{0,120}(?:overlap|contrast|overflow|cut\s*off|cutoff|clipp|blank|spacing|wrap|low-contrast|broken|touch-target|too\s+small|too\s+tight|cramped|hard\s+to\s+read|unpolished|unfinished)/i.test(text)
    || /(?:overlap|contrast|overflow|cut\s*off|cutoff|clipp|blank|spacing|wrap|low-contrast|broken|touch-target|too\s+small|too\s+tight|cramped|hard\s+to\s+read|unpolished|unfinished).{0,120}(?:but|however|though|still|remaining|minor|small|acceptable|ok)\b/i.test(text)
    || /(?:found|detected|identified)\s+(?:an?\s+)?(?:overlap|contrast|overflow|cutoff|blank|spacing|wrap|clipp|low-contrast|broken|small|touch-target)/i.test(text)
    || /(?:overlap|contrast|overflow|cutoff|blank|spacing|wrap|clipp|low-contrast|broken|small\s+mobile|touch-target|controls?)\s+(?:issue|defect|problem|fixed|resolved)/i.test(text)
    || /(?:fixed|resolved|corrected|reworked|patched)\s+(?:overlap|contrast|overflow|cutoff|blank|spacing|wrap|clipp|layout|visual|control|text|touch-target)/i.test(text)
    || /(?:문제|이슈|결함)\s*(?:발견|검출|확인|발생|있었|있음|있다)/i.test(text)
    || /(?:다만|하지만|그러나|없지만|없으나|없는데|제외|여전히|아직|약간|사소).{0,80}(?:겹|대비|저대비|오버플로|잘림|공백|간격|깨져|삐져|너무\s*작|읽기\s*어려|미완성|어색)/i.test(text)
    || /(?:겹|대비|저대비|오버플로|잘림|공백|간격|깨져|삐져|너무\s*작|읽기\s*어려|미완성|어색).{0,80}(?:다만|하지만|그러나|없지만|없으나|없는데|여전히|아직|약간|사소|허용|괜찮)/i.test(text)
    || /(?:겹침|대비|오버플로|잘림|공백|간격|저대비|터치\s*타깃).{0,24}(?:문제|이슈|결함|수정|해결)/i.test(text)
    || /(?:수정|해결|고침|재작업).{0,24}(?:겹침|대비|오버플로|잘림|공백|간격|레이아웃|시각|컨트롤|텍스트)/i.test(text)
}

function completedRenderedUiVisualQaTodoStatusAfterLatestBrowserAudit(
  history: AgentState['toolCallHistory'] | undefined,
  ledger: AgentState['evidenceLedger'] | undefined,
): RenderedUiVisualQaQuality {
  if (!history || history.length === 0) {
    return browserUiAuditEntriesAfterLatestWrite(ledger).length > 0 ? 'missing' : 'ok'
  }

  let latestBrowserAuditIndex = -1
  for (let i = history.length - 1; i >= 0; i--) {
    const entry = history[i]
    if (entry?.status === 'success' && BROWSER_UI_AUDIT_TOOLS.has(entry.tool)) {
      latestBrowserAuditIndex = i
      break
    }
  }
  if (latestBrowserAuditIndex < 0) {
    return browserUiAuditEntriesAfterLatestWrite(ledger).length > 0 ? 'missing' : 'ok'
  }

  let sawOkQa = false
  let sawPlaceholderQa = false
  let sawIssueQa = false
  for (let i = latestBrowserAuditIndex + 1; i < history.length; i++) {
    const entry = history[i]
    if (!entry || entry.status !== 'success' || entry.tool !== 'todowrite') continue
    const todoItems = parseTodoItems((entry.input as { items?: unknown }).items)
    for (const item of todoItems ?? []) {
      if (item.status !== 'completed') continue
      const quality = renderedUiVisualQaQuality(item.content)
      if (quality === 'issue') sawIssueQa = true
      if (quality === 'ok') sawOkQa = true
      if (quality === 'placeholder') sawPlaceholderQa = true
    }
  }
  if (sawIssueQa) return 'issue'
  if (sawOkQa) return 'ok'
  return sawPlaceholderQa ? 'placeholder' : 'missing'
}

function completedRenderedUiVisualQaIssueIndices(
  history: AgentState['toolCallHistory'] | undefined,
): number[] {
  if (!history || history.length === 0) return []

  const indices: number[] = []
  let sawBrowserAudit = false
  for (let i = 0; i < history.length; i++) {
    const entry = history[i]
    if (!entry || entry.status !== 'success') continue
    if (BROWSER_UI_AUDIT_TOOLS.has(entry.tool)) {
      sawBrowserAudit = true
      continue
    }
    if (!sawBrowserAudit || entry.tool !== 'todowrite') continue

    const todoItems = parseTodoItems((entry.input as { items?: unknown }).items)
    if ((todoItems ?? []).some((item) =>
      item.status === 'completed'
      && renderedUiVisualQaQuality(item.content) === 'issue'
    )) {
      indices.push(i)
    }
  }
  return indices
}

function toolHistoryEntryIsRenderedUiFixMutation(
  entry: NonNullable<AgentState['toolCallHistory']>[number] | undefined,
  ledger: AgentState['evidenceLedger'] | undefined,
): boolean {
  if (!entry || entry.status !== 'success' || !toolCallRepresentsProductMutation(entry.tool, entry.input)) {
    return false
  }
  const path = typeof entry.input.path === 'string' ? entry.input.path : undefined
  if (path) {
    return isRenderedUiFixMutationEntry({
      tool: entry.tool,
      status: entry.status,
      ts: entry.ts,
      path,
    })
  }
  return (ledger?.artifactWrites ?? []).some((write) =>
    write.tool === entry.tool
    && write.status === 'success'
    && write.ts === entry.ts
    && isRenderedUiFixMutationEntry(write)
  )
}

function renderedUiVisualQaIssueFollowUpAuditMissingFixMutation(
  history: AgentState['toolCallHistory'] | undefined,
  ledger: AgentState['evidenceLedger'] | undefined,
): boolean {
  if (!history || history.length === 0) return false

  for (const issueIndex of completedRenderedUiVisualQaIssueIndices(history)) {
    const followUpAuditIndex = history.findIndex((entry, index) =>
      index > issueIndex
      && entry?.status === 'success'
      && BROWSER_UI_AUDIT_TOOLS.has(entry.tool)
    )
    if (followUpAuditIndex < 0) continue

    const hasFixMutationBeforeFollowUpAudit = history
      .slice(issueIndex + 1, followUpAuditIndex)
      .some((entry) => toolHistoryEntryIsRenderedUiFixMutation(entry, ledger))
    if (!hasFixMutationBeforeFollowUpAudit) return true
  }
  return false
}

function evaluateRenderedUiAuditGate(
  state: Pick<
    AgentState,
    'seedContract' | 'evidenceLedger' | 'input' | 'visualAttachmentsDisabled' | 'toolCallHistory'
  >,
  finalText: string,
): CompletionGateResult | null {
  if (!state.seedContract || !contractRequiresRenderedUiValidation(state.seedContract)) {
    return null
  }
  const requestedScope = renderedUiRequestedScope(state.input ?? '')
  const boundedValidationOnly = requestedScope.bounded
    && state.seedContract.executionIntent?.workspaceMutation === 'forbidden'
  const requiresVisualSemanticInspection = !requestedScope.excludesSemanticVisualInspection
    && (
      !boundedValidationOnly
      || contractRequiresRenderedUiVisualSemanticInspection(state.seedContract)
    )
  const designBriefStatus = completedRenderedUiDesignBriefTodoStatusBeforeImplementationMutation(
    state.toolCallHistory,
    state.evidenceLedger,
  )
  if (
    designBriefStatus !== 'ok'
    && !(designBriefStatus === 'late' && finalTextReportsLateDesignBriefRecovery(finalText))
  ) {
    const reason = {
      incomplete:
        'rendered UI implementation requires a completed design brief todowrite before the first implementation file change that covers all required facets: target user/workflow, primary screens/states, responsive strategy, visual style, assets/media, and interactions/controls',
      placeholder:
        'rendered UI implementation requires a concrete completed design brief todowrite before the first implementation file change; placeholder labels like target user/screens/responsive/style/assets are not enough',
      missing:
        'rendered UI implementation requires a completed design brief todowrite before the first implementation file change',
      late:
        'rendered UI implementation started before a concrete design brief; after recording the design brief and revising the implementation from it, the final answer must disclose that late design-brief recovery instead of presenting the work as planned upfront',
    }[designBriefStatus]
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason,
    }
  }
  const audits = browserUiAuditEntriesAfterLatestWrite(state.evidenceLedger)
  if (audits.length === 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: 'rendered UI validation requires a browser screenshot/click/evaluate audit after the latest file change',
    }
  }
  const defects = audits.flatMap((entry) => entry.defects ?? [])
  if (defects.length > 0) {
    const preview = defects.slice(0, 3).join('; ')
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: `rendered UI browser layout audit still reports ${defects.length} defect(s): ${preview}`,
    }
  }
  const defectsMissingFixMutation = browserUiAuditDefectsMissingFixMutation(state.evidenceLedger)
  if (defectsMissingFixMutation.length > 0) {
    const preview = defectsMissingFixMutation.slice(0, 3).join('; ')
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: `rendered UI browser audit defect(s) need an actual file change before the clean follow-up screenshot: ${preview}`,
    }
  }
  if (renderedUiVisualQaIssueFollowUpAuditMissingFixMutation(state.toolCallHistory, state.evidenceLedger)) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: 'rendered UI visual QA issue requires an actual UI file change after the visual QA issue and before the follow-up browser audit; re-screenshoting alone is not evidence that the reported issue was fixed',
    }
  }
  const layoutAudits = audits.filter((entry) => entry.uiAuditHasLayout)
  if (layoutAudits.length === 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: 'rendered UI validation requires a browser layout audit after the latest file change',
    }
  }
  const visualLayoutAudits = state.visualAttachmentsDisabled === true
    ? layoutAudits
    : layoutAudits.filter(hasAttachedBrowserScreenshotEvidence)
  if (visualLayoutAudits.length === 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: 'rendered UI visual validation requires saved screenshot evidence with an attached image after the latest file change; use browser.screenshot or browser.click/browser.evaluate with a PNG path that can be attached, not layout-audit text or an oversized/unreadable screenshot alone',
    }
  }
  const requiredViewports = requiredRenderedUiAuditViewports(requestedScope)
  const missingViewports = missingRenderedUiAuditViewports(visualLayoutAudits, requiredViewports)
  if (missingViewports.length > 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: `rendered UI validation requires clean saved screenshot/layout audits at the requested viewports after the latest file change; missing ${missingViewports.join(' and ')}`,
    }
  }
  const missingDimensions = missingRenderedUiAuditDimensions(
    visualLayoutAudits,
    requestedScope.bounded ? requestedScope.viewportDimensions : [],
  )
  if (missingDimensions.length > 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: `rendered UI validation requires clean saved screenshot/layout audits at the explicitly requested viewport dimensions after the latest file change; missing ${missingDimensions.join(' and ')}`,
    }
  }
  const validatedUiTarget = renderedUiAuditTargetWithRequestedCoverage(
    visualLayoutAudits,
    requestedScope,
  )
  if (validatedUiTarget === undefined) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: 'rendered UI validation requires saved screenshot/layout audits covering the requested viewport scope for the same rendered UI URL after the latest file change',
    }
  }
  const consoleAudits = visualLayoutAudits.filter((entry) =>
    entry.uiAuditHasConsole
    && (validatedUiTarget === null || renderedUiAuditTargetUrl(entry) === validatedUiTarget)
  )
  const missingConsoleViewports = missingRenderedUiAuditViewports(consoleAudits, requiredViewports)
  if (missingConsoleViewports.length > 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: `rendered UI validation requires browser console/page audits at the requested viewports after the latest file change; missing ${missingConsoleViewports.join(' and ')}`,
    }
  }
  const missingConsoleDimensions = missingRenderedUiAuditDimensions(
    consoleAudits,
    requestedScope.bounded ? requestedScope.viewportDimensions : [],
  )
  if (missingConsoleDimensions.length > 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: `rendered UI validation requires browser console/page audits at the explicitly requested viewport dimensions after the latest file change; missing ${missingConsoleDimensions.join(' and ')}`,
    }
  }
  if (renderedUiRequiresInteractionSmoke(state)) {
    const interactionSmokes = browserInteractionSmokeEntriesAfterLatestWrite(state.evidenceLedger)
      .filter((entry) => validatedUiTarget === null || renderedUiAuditTargetUrl(entry) === validatedUiTarget)
    const specificInteractionSmokes = interactionSmokes.filter(renderedUiAuditHasSpecificInteractionTarget)
    if (specificInteractionSmokes.length === 0) {
      return {
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: validatedUiTarget
          ? 'interactive rendered UI validation requires a representative browser.click or browser.evaluate smoke check against the same rendered UI URL after the latest file change with a specific interaction target; use selectors such as #start, .start-button, [data-testid="start"], or button[aria-label="Start"], or a concrete evaluate script fingerprint for keyboard/canvas flows, not broad selectors like button or #root button'
          : 'interactive rendered UI validation requires a representative browser.click or browser.evaluate smoke check after the latest file change with a specific interaction target; use selectors such as #start, .start-button, [data-testid="start"], or button[aria-label="Start"], or a concrete evaluate script fingerprint for keyboard/canvas flows, not broad selectors like button or #root button',
      }
    }
    const activeEvidenceSmokes = state.visualAttachmentsDisabled === true
      ? specificInteractionSmokes
      : specificInteractionSmokes.filter(hasAttachedBrowserScreenshotEvidence)
    if (activeEvidenceSmokes.length === 0) {
      return {
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: 'interactive rendered UI validation requires saved active-state screenshot evidence with attached images from browser.click/browser.evaluate at desktop and mobile viewports; pass PNG screenshot paths that can be attached so the active state can be visually inspected',
      }
    }
    if (!activeEvidenceSmokes.some((entry) => entry.uiAuditHasLayout)) {
      return {
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: 'interactive rendered UI validation requires the representative browser.click or browser.evaluate smoke check to capture an active-state screenshot/layout audit after the latest file change',
      }
    }
    const activeLayoutAudits = activeEvidenceSmokes.filter((entry) => entry.uiAuditHasLayout)
    const missingActiveViewports = missingRenderedUiAuditViewports(activeLayoutAudits)
    if (missingActiveViewports.length > 0) {
      return {
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: `interactive rendered UI validation requires active-state browser.click/browser.evaluate layout audits at desktop and mobile viewports after the latest file change; missing ${missingActiveViewports.join(' and ')}`,
      }
    }
    const validatedInteractionTarget = renderedUiInteractionTargetWithAllViewports(activeLayoutAudits)
    if (validatedInteractionTarget === undefined) {
      return {
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: 'interactive rendered UI validation requires desktop and mobile active-state browser.click/browser.evaluate audits for the same interaction target after the latest file change',
      }
    }
    const activeLayoutAuditsForValidatedInteraction = validatedInteractionTarget === null
      ? activeLayoutAudits
      : activeLayoutAudits.filter((entry) =>
        renderedUiAuditInteractionTarget(entry) === validatedInteractionTarget
      )
    const activeConsoleAudits = activeLayoutAuditsForValidatedInteraction.filter((entry) => entry.uiAuditHasConsole)
    const missingActiveConsoleViewports = missingRenderedUiAuditViewports(activeConsoleAudits)
    if (missingActiveConsoleViewports.length > 0) {
      return {
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: `interactive rendered UI validation requires active-state browser.click/browser.evaluate console/page audit evidence at desktop and mobile viewports after the latest file change; missing ${missingActiveConsoleViewports.join(' and ')}`,
      }
    }
    if (renderedUiRequiresDynamicStateEvidence(state)) {
      const dynamicStateAudits = activeLayoutAuditsForValidatedInteraction.filter(browserEvaluateReportsDynamicStateEvidence)
      const missingDynamicViewports = missingRenderedUiAuditViewports(dynamicStateAudits)
      if (missingDynamicViewports.length > 0) {
        return {
          decision: 'block',
          unmet: [],
          cause: 'ui_audit',
          reason: `dynamic rendered UI validation for games/animated canvas work requires browser.evaluate evidence that frame, pixel, position, or game state changes over time at desktop and mobile viewports after the latest file change; missing ${missingDynamicViewports.join(' and ')}`,
        }
      }
    }
  }
  if (
    state.visualAttachmentsDisabled === true
    && requiresVisualSemanticInspection
    && !reportsVisualImageInputBlocker(finalText)
  ) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: 'rendered UI visual inspection requires an honest UNVERIFIED/INCOMPLETE image-input blocker because visual attachments are disabled',
    }
  }
  if (
    state.visualAttachmentsDisabled !== true
    && requiresVisualSemanticInspection
    && reportsVisualImageInputBlocker(finalText)
  ) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: 'rendered UI visual inspection cannot be reported as blocked by image-input support when visual attachments are enabled',
    }
  }
  if (state.visualAttachmentsDisabled !== true && !boundedValidationOnly) {
    const visualQaStatus = completedRenderedUiVisualQaTodoStatusAfterLatestBrowserAudit(
      state.toolCallHistory,
      state.evidenceLedger,
    )
    if (visualQaStatus !== 'ok') {
      return {
        decision: 'block',
        unmet: [],
        cause: 'ui_audit',
        reason: visualQaStatus === 'issue'
          ? 'rendered UI visual QA found an issue after the latest browser audit; apply an actual UI fix, capture fresh desktop and mobile browser screenshots after the fix, then record a follow-up visual QA todowrite with no remaining issues'
          : visualQaStatus === 'placeholder'
            ? 'rendered UI visual QA requires a concrete completed todowrite after the latest browser screenshot/click/evaluate audit that compares desktop and mobile screenshots against the design plan, checks concrete visual quality aspects such as layout/spacing, text wrapping/overflow, contrast/readability, overlap, controls/touch targets, and assets/completeness, and states issues found or explicitly none'
            : 'rendered UI visual QA requires a completed todowrite after the latest browser screenshot/click/evaluate audit',
      }
    }
  }
  const finalAnswerGaps = boundedValidationOnly
    ? []
    : renderedUiFinalAnswerGaps(state, finalText)
  if (finalAnswerGaps.length > 0) {
    return {
      decision: 'block',
      unmet: [],
      cause: 'ui_audit',
      reason: `rendered UI final answer must summarize validation evidence: missing ${finalAnswerGaps.slice(0, 4).join(', ')}`,
    }
  }
  return null
}

function reportsVisualImageInputBlocker(finalText: string): boolean {
  if (!HONEST_BLOCKER_STEM.test(finalText)) {
    return false
  }
  const normalized = finalText.toLowerCase()
  const namesVisualCheck =
    /visual(?:ly)?|screenshot|image/.test(normalized)
    || /시각|스크린샷|이미지/.test(normalized)
  const namesImageInputBlocker =
    normalized.includes('image input')
    || normalized.includes('cannot receive image')
    || normalized.includes('could not be visually inspected')
    || normalized.includes('cannot visually inspect')
    || normalized.includes('does not advertise image')
    || /이미지\s*(?:입력|첨부|전달|수신|처리)/.test(normalized)
    || /이미지를\s*(?:받을|볼|읽을|처리할)\s*수\s*없/.test(normalized)
    || /시각(?:적)?(?:으로)?\s*(?:검수|확인|검증|검토|리뷰)(?:할)?\s*수\s*없/.test(normalized)
    || /스크린샷(?:을)?\s*(?:볼|읽을|검수|확인|검증|검토|리뷰)(?:할)?\s*수\s*없/.test(normalized)
  return namesVisualCheck && namesImageInputBlocker
}

function contractRequiresRenderedUiVisualSemanticInspection(
  contract: AgentState['seedContract'] | undefined,
): boolean {
  const text = [
    ...(contract?.acceptanceCriteria ?? []).map((criterion) => criterion.text),
    ...(contract?.evidenceRequirements ?? []).map((requirement) => requirement.description),
  ].join('\n')
  return /\b(?:visual\s+(?:qa|review|inspection)|aesthetic|polish|compare[sd]?\s+(?:the\s+)?screenshots?|contrast|readability|typography|color\s+quality)\b|(?:시각\s*(?:검수|리뷰|확인|평가)|미적|디자인\s*품질|스크린샷\s*비교|대비|가독성|타이포그래피|색상\s*품질)/iu
    .test(text)
}

function criterionVerdicts(finalText: string): Map<string, ParsedCriterionVerdict> {
  const verdicts = new Map<string, ParsedCriterionVerdict>()
  for (const line of finalText.split('\n')) {
    const m = CRITERION_LINE.exec(line.trim())
    if (m) {
      verdicts.set(m[1].toUpperCase(), {
        verdict: m[2]!.toUpperCase() as ParsedCriterionVerdict['verdict'],
        evidenceToolCallIds: [...new Set(
          (m[3] ?? '')
            .split(',')
            .map((candidate) => candidate.trim())
            .filter(Boolean),
        )].slice(0, MAX_CRITERION_EVIDENCE_REFERENCES),
      })
    }
  }
  return verdicts
}

function exactFallbackActionReceiptIds(
  state: Pick<
    AgentState,
    | 'seedContract'
    | 'toolCallHistory'
    | 'stuckRepeatForcedFinal'
    | 'forcedFinalSynthesisReason'
  >,
): Set<string> {
  const ids = new Set<string>()
  const contract = state.seedContract
  const intent = contract?.executionIntent
  if (
    state.stuckRepeatForcedFinal !== true
    || state.forcedFinalSynthesisReason !== 'exact-tool-budget'
    || contract?.source !== 'fallback'
    || intent?.capabilityPolicy !== 'closed'
    || intent.retryPolicy !== 'forbidden'
    || intent.workspaceMutation !== 'forbidden'
    || !intent.allowedTools?.length
  ) {
    return ids
  }
  const receipts = (state.toolCallHistory ?? []).filter((entry) => (
    Boolean(entry.toolCallId)
    && intent.allowedTools!.includes(entry.tool)
    && isExecutorConfirmedExternalActionReceipt({
      status: entry.status,
      output: entry.output,
      securityEffect: entry.securityEffect,
      executionObserved: entry.executionObserved,
    })
  ))
  const observations = collectCriterionReferenceableObservations(state)
  if (!intent.allowedTools.every((toolName) => (
    receipts.some((entry) => entry.tool === toolName)
    || observations.some((entry) => entry.tool === toolName)
  ))) {
    return ids
  }
  for (const receipt of receipts) ids.add(receipt.toolCallId!)
  return ids
}

function criterionReferenceableExecutionIds(
  state: Pick<
    AgentState,
    | 'seedContract'
    | 'toolCallHistory'
    | 'stuckRepeatForcedFinal'
    | 'forcedFinalSynthesisReason'
  >,
): Set<string> {
  const ids = new Set(
    collectCriterionReferenceableObservations(state).map((entry) => entry.toolCallId),
  )
  for (const receiptId of exactFallbackActionReceiptIds(state)) ids.add(receiptId)
  return ids
}

function buildCriterionVerdictSnapshot(
  state: Pick<
    AgentState,
    | 'completionDiagnostics'
    | 'seedContract'
    | 'toolCallHistory'
    | 'stuckRepeatForcedFinal'
    | 'forcedFinalSynthesisReason'
  >,
  criteria: Array<{ id: string }>,
  currentVerdicts: Map<string, ParsedCriterionVerdict>,
): AgentCriterionVerdictSnapshot {
  const {
    toolResultCount,
    toolResultFingerprint: currentToolResultFingerprint,
  } = criterionEvidenceEpisode(state)
  const canonicalIds = new Map(criteria.map((criterion) => [
    criterion.id.toUpperCase(),
    criterion.id,
  ]))
  const previous = state.completionDiagnostics?.criterionVerdictSnapshot
  const latest = new Map<string, AgentCriterionVerdictSnapshot['criterionVerdicts'][number]>()
  if (
    previous?.toolResultCount === toolResultCount
    && previous.toolResultFingerprint === currentToolResultFingerprint
  ) {
    for (const candidate of previous.criterionVerdicts) {
      const id = canonicalIds.get(candidate.id.toUpperCase())
      if (!id) continue
      latest.set(id, {
        id,
        verdict: candidate.verdict,
        ...(candidate.evidenceToolCallIds?.length
          ? { evidenceToolCallIds: [...candidate.evidenceToolCallIds] }
          : {}),
      })
    }
  }

  const referenceableIds = criterionReferenceableExecutionIds(state)
  for (const [candidateId, current] of currentVerdicts) {
    const id = canonicalIds.get(candidateId)
    if (!id) continue
    if (current.verdict === 'UNMET') {
      // An explicit current UNMET is a material state transition and must
      // invalidate a previously verified MET on the same evidence snapshot.
      latest.set(id, { id, verdict: 'unmet' })
      continue
    }
    if (
      current.evidenceToolCallIds.length > 0
      && current.evidenceToolCallIds.every((toolCallId) => referenceableIds.has(toolCallId))
    ) {
      latest.set(id, {
        id,
        verdict: 'met',
        evidenceToolCallIds: [...current.evidenceToolCallIds],
      })
    }
  }

  return {
    toolResultCount,
    toolResultFingerprint: currentToolResultFingerprint,
    criterionVerdicts: criteria
      .map((criterion) => latest.get(criterion.id))
      .filter((verdict): verdict is NonNullable<typeof verdict> => verdict !== undefined),
  }
}

function criterionVerdictsFromSnapshot(
  snapshot: AgentCriterionVerdictSnapshot,
): Map<string, ParsedCriterionVerdict> {
  return new Map(snapshot.criterionVerdicts.map((candidate) => [
    candidate.id.toUpperCase(),
    {
      verdict: candidate.verdict === 'met' ? 'MET' : 'UNMET',
      evidenceToolCallIds: [...(candidate.evidenceToolCallIds ?? [])],
    },
  ]))
}

function missingCriterionVerdictIds(
  criteria: Array<{ id: string }>,
  verdicts: Map<string, ParsedCriterionVerdict>,
): string[] {
  return criteria
    .map((c) => c.id.toUpperCase())
    .filter((id) => !verdicts.has(id))
}

function namesVisualImageValidation(text: string | undefined): boolean {
  const normalized = (text ?? '').toLowerCase()
  const namesVisualSurface =
    /\b(?:visual|visually|screenshot|image|pixel|layout|viewport|rendered ui|browser)\b/.test(normalized)
    || /(?:시각|스크린샷|이미지|화면|레이아웃|브라우저)/.test(normalized)
  const namesValidation =
    /\b(?:inspect|inspection|qa|review|compare|comparison|audit|evidence|validation|verify|verified)\b/.test(normalized)
    || /(?:검수|리뷰|비교|검증|확인|증거)/.test(normalized)
  return namesVisualSurface && namesValidation
}

function unmetCriteriaAreOnlyVisualImageValidation(
  criteria: Array<{ id: string; text?: string }>,
  verdicts: Map<string, ParsedCriterionVerdict>,
): boolean {
  return criteria.every((criterion) => {
    const verdict = verdicts.get(criterion.id.toUpperCase())?.verdict
    return verdict === 'MET' || namesVisualImageValidation(`${criterion.id} ${criterion.text ?? ''}`)
  })
}

function criterionScopedExecutorEvidenceGaps(
  state: Pick<
    AgentState,
    | 'seedContract'
    | 'toolCallHistory'
    | 'stuckRepeatForcedFinal'
    | 'forcedFinalSynthesisReason'
  >,
  criteria: Array<{ id: string }>,
  verdicts: Map<string, ParsedCriterionVerdict>,
): string[] | null {
  const readOnlyContract = state.seedContract?.executionIntent?.workspaceMutation === 'forbidden'
  const readBackCompleteDocumentContract = documentArtifactCriterionReviewEligible(state)
  if (!readOnlyContract && !readBackCompleteDocumentContract) return null
  const referenceableIds = criterionReferenceableExecutionIds(state)
  if (referenceableIds.size === 0) return null

  return criteria.flatMap((criterion) => {
    const record = verdicts.get(criterion.id.toUpperCase())
    if (!record || record.verdict !== 'MET' || record.evidenceToolCallIds.length === 0) {
      return [criterion.id.toUpperCase()]
    }
    return record.evidenceToolCallIds.every((toolCallId) => referenceableIds.has(toolCallId))
      ? []
      : [criterion.id.toUpperCase()]
  })
}

function incompleteTodos(
  todoList: AgentState['todoList'] | undefined,
): NonNullable<AgentState['todoList']> {
  return (todoList ?? []).filter(
    (item) => !(['completed', 'cancelled'] as string[]).includes(item.status),
  )
}

function openTodosAreOnlyVisualImageValidationBlockers(
  todoList: AgentState['todoList'] | undefined,
): boolean {
  return incompleteTodos(todoList).every((item) =>
    item.status === 'blocked' && namesVisualImageValidation(item.content)
  )
}

function evaluateWithoutBudget(
  state: Pick<
    AgentState,
    | 'seedContract'
    | 'evidenceLedger'
    | 'todoList'
    | 'input'
    | 'visualAttachmentsDisabled'
    | 'toolCallHistory'
    | 'stuckRepeatForcedFinal'
    | 'forcedFinalSynthesisReason'
    | 'completionDiagnostics'
  > & Partial<Pick<AgentState, 'messages'>>,
  finalText: string,
  criteria: Array<{ id: string; text?: string }>,
  verdicts: Map<string, ParsedCriterionVerdict>,
): CompletionGateResult {
  const renderedUiAuditBlock = evaluateRenderedUiAuditGate(state, finalText)
  if (renderedUiAuditBlock) {
    return renderedUiAuditBlock
  }

  // todoList signal integrated by P022-T3 (D6): a todo list the agent itself
  // declared and left incomplete is a structural not-done signal — same
  // bounded semantics as the criteria check (the block budget upstream in
  // evaluateCompletionGate applies, so this can never loop forever).
  const openTodos = incompleteTodos(state.todoList)
  if (openTodos.length > 0) {
    const preview = openTodos
      .slice(0, 3)
      .map((item) => `[${item.status}] ${item.content}`)
      .join('; ')
    return {
      decision: 'block',
      unmet: [],
      reason: `the session todo list still has ${openTodos.length} incomplete item(s): ${preview}`,
      cause: 'todo',
    }
  }

  if (verdicts.size > 0) {
    const unmet = criteria
      .map((c) => c.id.toUpperCase())
      .filter((id) => verdicts.get(id)?.verdict !== 'MET')
    if (unmet.length > 0) return { decision: 'block', unmet, cause: 'criteria' }
  }

  // Closed operational work can be verified criterion-by-criterion without
  // pretending that a source read, process probe, or external action receipt
  // is a validation run. Every MET line must cite exact successful current-run
  // executor evidence. External receipts are admitted only for the daemon's
  // closed fallback exactly-once contract and establish tool-boundary success,
  // not downstream provider finality.
  const executorEvidenceGaps = criterionScopedExecutorEvidenceGaps(
    state,
    criteria,
    verdicts,
  )
  const readBackCompleteDocumentContract = documentArtifactCriterionReviewEligible(state)
  const hasExternalActionReceipt = exactFallbackActionReceiptIds(state).size > 0
  if (executorEvidenceGaps?.length === 0) {
    return {
      decision: 'pass',
      unmet: [],
      reason: readBackCompleteDocumentContract
        ? 'every acceptance criterion cites successful executor-confirmed source observation or required-artifact read-back evidence'
        : hasExternalActionReceipt
          ? 'every acceptance criterion cites successful executor-confirmed closed-workflow evidence'
          : 'every acceptance criterion cites successful executor-confirmed read-only observation evidence',
    }
  }
  if (executorEvidenceGaps) {
    return {
      decision: 'block',
      unmet: executorEvidenceGaps,
      cause: 'observation_evidence',
      reason: readBackCompleteDocumentContract
        ? 'document completion must link every MET criterion to exact successful executor-confirmed source observation or required-artifact read-back evidence'
        : hasExternalActionReceipt
          ? 'closed operational completion must link every MET criterion to exact successful executor-confirmed evidence'
          : 'read-only operational completion must link every MET criterion to exact successful executor-confirmed observation evidence',
    }
  }

  // Self-confirmation guard: prose or structured claims are not enough — the
  // ledger must hold a verified entry after the latest successful write
  // (structural check, no criterion-to-evidence text matching).
  const ledger = state.evidenceLedger
  if (hasAcceptedProvidedContextReview(state, finalText)) {
    return { decision: 'pass', unmet: [], reason: 'Independent review verified this exact answer against user-provided premises; no external effect or observation is owed' }
  }
  if (!hasFreshVerifiedEvidence(ledger)) {
    const latestWrite = latestSuccessfulArtifactWriteOrder(ledger)
    // This guard asks "did you verify what you changed". A contract that
    // forbids workspace mutation changes nothing, so it can never produce a
    // validation run or an artifact read-back — the only evidence kinds the
    // guard counts. Requiring them made the gate unsatisfiable by construction
    // for read-only research: it blocked until the block budget ran out, and
    // the collected findings were discarded with it. Observation evidence is
    // what such a run has, and is what this gate already accepts
    // criterion-by-criterion above.
    if (
      !latestWrite
      && !contractOwesArtifactWork(state.seedContract)
      && hasSuccessfulObservationEvidence(ledger)
    ) {
      return {
        decision: 'pass',
        unmet: [],
        reason: 'read-only completion is backed by successful executor-confirmed observation evidence',
      }
    }
    return {
      decision: 'block',
      unmet: [],
      reason: latestWrite
        ? 'completion has no verified evidence after the latest file change'
        : 'completion has no verified evidence',
      cause: 'evidence',
    }
  }
  return { decision: 'pass', unmet: [] }
}

/**
 * Structured retry instruction pushed into the conversation when the gate
 * blocks a completion attempt.
 */
export function buildCompletionGateBlockMessage(
  result: CompletionGateResult,
  attempt: number,
  maxAttempts: number,
  options: { requireExecutableAction?: boolean } = {},
): string {
  const needsSpecificInteractionTarget = /interaction|browser\.(?:click|evaluate)|specific\s+interaction\s+target/i
    .test(result.reason ?? '')
  const uiAuditGuidance = /design brief/i.test(result.reason ?? '')
    ? 'Complete the concrete UI design brief required by the run contract before implementation edits whenever possible. If implementation already started, revise from that brief and disclose the late recovery. Then gather only the viewport, interaction, and screenshot evidence named in the run contract; do not expand a bounded validation request.'
    : [
        'Use the reported reason as the specific missing UI evidence. Continue only with browser actions, viewports, interactions, and QA bookkeeping that the run contract positively requires. Do not add an excluded viewport, tool, interaction, retry, or todowrite merely because broad UI defaults normally recommend it.',
        ...(needsSpecificInteractionTarget
          ? ['When interaction evidence is required, use a specific selector or target such as #start, .start-button, [data-testid="start"], or button[aria-label="Start"], or a concrete browser.evaluate script fingerprint, not broad selectors like button or #root button.']
          : []),
      ].join(' ')
  const guidance = result.unmet.length > 0
    ? sentenceCase(completionGateRecoveryHint(result, options))
    : result.cause === 'todo'
      ? sentenceCase(completionGateRecoveryHint(result, options))
      : result.cause === 'ui_audit'
        ? uiAuditGuidance
        : sentenceCase(completionGateRecoveryHint(result, options))
  const lines = [
    `[Completion gate ${attempt}/${maxAttempts}]`,
    result.unmet.length > 0
      ? `These acceptance criteria are not closed as MET yet: ${result.unmet.join(', ')}.`
      : result.reason ?? 'The completion claim is not backed by verified evidence.',
    guidance,
    'If a criterion genuinely cannot be met, answer with INCOMPLETE: naming the concrete blocker — do not claim completion without evidence.',
  ]
  return lines.join(' ')
}

function sentenceCase(value: string): string {
  if (!value) return value
  return value[0]!.toUpperCase() + value.slice(1)
}
