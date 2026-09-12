export type CompletionCriterionVerdict = 'met' | 'unmet' | 'not_applicable'

export interface RemovedCriterionVerdict {
  id: string
  verdict: CompletionCriterionVerdict
  evidenceToolCallIds?: string[]
}

export interface FinalAnswerPresentationDiagnostics {
  removedCriterionVerdicts: RemovedCriterionVerdict[]
  removedPhaseUsage: boolean
  removedProtocolLabels: string[]
}

export interface FinalAnswerPresentationResult {
  content: string
  diagnostics: FinalAnswerPresentationDiagnostics
}

export interface StructuredFinalReport {
  outcome: string
  details: string[]
  validation: string[]
  remainingRisks: string[]
}

interface SanitizeFinalAnswerOptions {
  criterionIds?: readonly string[]
}

const PHASE_USAGE_HEADING = /^_?\s*phase usage\s*:\s*_?$/i
const PHASE_USAGE_ITEM = /^\s*[-*+]\s+[^:\r\n]+:\s+\d[\d,]*\s+tokens?\b/i
const PROTOCOL_LABEL = /^\s*(ANSWER|INCOMPLETE|VERIFIED|UNVERIFIED)\s*:\s*(.*)$/i
const FENCE = /^\s*(`{3,}|~{3,})/

interface OpenFence {
  character: string
  length: number
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function criterionVerdictPattern(ids: readonly string[]): RegExp | null {
  const alternatives = [...new Set(ids.map((id) => id.trim()).filter(Boolean))]
    .sort((left, right) => right.length - left.length)
    .map(escapeRegExp)
  if (alternatives.length === 0) return null

  return new RegExp(
    `^\\s*(?:[-*+]\\s+|\\d+[.)]\\s+)?(?:\\|\\s*)?(?:CRITERION\\s+)?(${alternatives.join('|')})\\s*(?:\\|\\s*|[:=\\-–—]\\s*)?(?:\\|\\s*)?(MET|UNMET|N\\/?A|NOT\\s+APPLICABLE)\\b[^\\r\\n]*$`,
    'i',
  )
}

function normalizeVerdict(value: string): CompletionCriterionVerdict {
  const normalized = value.replace(/\s+/g, ' ').toUpperCase()
  if (normalized === 'MET') return 'met'
  if (normalized === 'UNMET') return 'unmet'
  return 'not_applicable'
}

const EVIDENCE_REFERENCE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/
const MAX_CRITERION_EVIDENCE_REFERENCES = 16

function extractCriterionEvidenceReferences(value: string | undefined): string[] {
  const match = value?.match(
    /\bEVIDENCE\s+([A-Za-z0-9_.:-]+(?:\s*,\s*[A-Za-z0-9_.:-]+)*)/i,
  )
  if (!match) return []
  return [...new Set(
    match[1]!
      .split(',')
      .map((candidate) => candidate.trim())
      .filter((candidate) => EVIDENCE_REFERENCE.test(candidate)),
  )].slice(0, MAX_CRITERION_EVIDENCE_REFERENCES)
}

function cleanupLines(lines: string[]): string {
  return lines
    .join('\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function hasCanonicalPhaseUsageItemAfter(lines: readonly string[], headingIndex: number): boolean {
  for (let index = headingIndex + 1; index < lines.length; index += 1) {
    const line = lines[index]!
    if (!line.trim()) continue
    return PHASE_USAGE_ITEM.test(line)
  }
  return false
}

/**
 * Removes daemon-owned completion protocols from text after those protocols
 * have already driven their internal gates. Matching is structural and is
 * limited to criterion ids from the active run contract. Fenced examples are
 * left untouched so a user asking about the protocol can still receive code or
 * documentation that contains the same tokens.
 */
export function sanitizeFinalAnswerPresentation(
  value: string,
  options: SanitizeFinalAnswerOptions = {},
): FinalAnswerPresentationResult {
  const lines = value.replace(/\r\n/g, '\n').split('\n')
  const criterionPattern = criterionVerdictPattern(options.criterionIds ?? [])
  const kept: string[] = []
  const diagnostics: FinalAnswerPresentationDiagnostics = {
    removedCriterionVerdicts: [],
    removedPhaseUsage: false,
    removedProtocolLabels: [],
  }
  let openFence: OpenFence | null = null
  let skippingPhaseUsage = false

  for (const [lineIndex, line] of lines.entries()) {
    const fence = line.match(FENCE)?.[1]
    if (fence) {
      if (!openFence) {
        openFence = { character: fence[0]!, length: fence.length }
      } else if (
        fence[0] === openFence.character
        && fence.length >= openFence.length
        && line.slice(line.indexOf(fence) + fence.length).trim().length === 0
      ) {
        openFence = null
      }
      kept.push(line)
      continue
    }

    if (openFence) {
      kept.push(line)
      continue
    }

    if (
      PHASE_USAGE_HEADING.test(line.trim())
      && hasCanonicalPhaseUsageItemAfter(lines, lineIndex)
    ) {
      diagnostics.removedPhaseUsage = true
      skippingPhaseUsage = true
      continue
    }
    if (skippingPhaseUsage) {
      if (!line.trim() || PHASE_USAGE_ITEM.test(line)) continue
      skippingPhaseUsage = false
    }

    const criterionMatch = criterionPattern?.exec(line)
    if (criterionMatch) {
      const evidenceToolCallIds = extractCriterionEvidenceReferences(criterionMatch[0])
      diagnostics.removedCriterionVerdicts.push({
        id: criterionMatch[1]!,
        verdict: normalizeVerdict(criterionMatch[2]!),
        ...(evidenceToolCallIds.length > 0
          ? { evidenceToolCallIds }
          : {}),
      })
      continue
    }

    const protocolMatch = line.match(PROTOCOL_LABEL)
    if (protocolMatch) {
      diagnostics.removedProtocolLabels.push(protocolMatch[1]!.toUpperCase())
      if (protocolMatch[2]!.trim()) kept.push(protocolMatch[2]!.trim())
      continue
    }

    kept.push(line)
  }

  return {
    content: cleanupLines(kept),
    diagnostics,
  }
}

function asShortString(value: unknown, maxLength = 1_200): string {
  if (typeof value !== 'string') return ''
  const trimmed = value.trim()
  if (!trimmed) return ''
  return trimmed.length <= maxLength ? trimmed : `${trimmed.slice(0, maxLength - 1).trimEnd()}…`
}

function asShortStringList(value: unknown, maxItems: number): string[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => asShortString(item, 500))
    .filter(Boolean)
    .slice(0, maxItems)
}

function extractJsonObject(value: string): Record<string, unknown> | null {
  const match = value.match(/\{[\s\S]*\}/)
  if (!match) return null
  try {
    const parsed = JSON.parse(match[0]) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null
  } catch {
    return null
  }
}

/** Parse the finalizer's bounded, user-facing JSON envelope. */
export function parseStructuredFinalReport(value: string): StructuredFinalReport | null {
  const parsed = extractJsonObject(value)
  if (!parsed) return null
  const outcome = asShortString(parsed.outcome)
  if (!outcome) return null
  return {
    outcome,
    details: asShortStringList(parsed.details, 2),
    // A completed task can require more than one distinct check (for example
    // formatting, tests, and a build). Keep this bounded, but do not silently
    // discard requested validation evidence merely because the reporter
    // returned more than one successful check.
    validation: asShortStringList(parsed.validation, 3),
    remainingRisks: asShortStringList(parsed.remainingRisks, 1),
  }
}

/**
 * Distinguishes a failed structured-reporter envelope from a legitimate
 * natural-language provider fallback. A malformed/truncated envelope must
 * never be rendered directly in chat.
 */
export function isStructuredFinalReportEnvelopeLike(value: string): boolean {
  const trimmed = value.trimStart()
  return trimmed.startsWith('{')
    || /^```(?:json)?\s*\{/i.test(trimmed)
    || /\{\s*"(?:outcome|details|validation|remainingRisks)"\s*:/i.test(trimmed)
}

const MAX_REPORTED_ARTIFACT_PATHS = 3

/**
 * Minimal locale-aware wording for a daemon-generated partial-result notice.
 * The rejected draft is deliberately not published here — it can claim success
 * the gate could not confirm — but it is retained with the session, so say
 * where the work went instead of leaving the user at a dead end.
 */
export function completionBudgetExhaustedMessage(
  userInput: string,
  /**
   * Files this run actually wrote, as recorded by the executor. These are
   * observed facts rather than model claims, so naming them adds no unverified
   * assertion — it only tells the user where the run's output already is.
   */
  writtenArtifactPaths: readonly string[] = [],
): string {
  const korean = /[\u3131-\u318e\uac00-\ud7a3]/u.test(userInput)
  const paths = writtenArtifactPaths.slice(0, MAX_REPORTED_ARTIFACT_PATHS)
  const extra = writtenArtifactPaths.length - paths.length
  const base = korean
    ? '결과는 생성됐지만 내부 검증에서 요청한 항목을 모두 확인하지 못했습니다.'
      + ' 검증되지 않은 초안은 이 세션에 보관되어 있으니, 세션을 이어서 실행하면 검증을 계속할 수 있습니다.'
    : 'A result was produced, but internal verification could not confirm every requested outcome.'
      + ' The unverified draft is kept with this session; resume the session to carry the verification forward.'
  if (paths.length === 0) return base
  const list = paths.join(', ') + (extra > 0 ? korean ? ` 외 ${extra}개` : `, and ${extra} more` : '')
  return korean
    ? `${base} 이번 실행이 실제로 작성한 파일: ${list} (파일 내용도 아직 검증되지 않았습니다).`
    : `${base} Files this run actually wrote: ${list} (their contents are unverified too).`
}

/**
 * Renders a compact answer without exposing the reporter's internal field
 * names. Each list entry must therefore be self-contained in the user's
 * language; the prompt enforces that contract.
 */
export function renderStructuredFinalReport(report: StructuredFinalReport): string {
  const items = [
    ...report.details,
    ...report.validation,
    ...report.remainingRisks,
  ].filter((item, index, all) => item !== report.outcome && all.indexOf(item) === index)

  return items.length > 0
    ? `${report.outcome}\n\n${items.map((item) => `- ${item}`).join('\n')}`
    : report.outcome
}
