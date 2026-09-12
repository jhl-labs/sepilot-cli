import { formatActiveUserInstructions } from './user-steering.js'
import { ThinkingLevel } from '@sepilotd/core'
import type { AgentRunContract, ChatRequest, Message, ToolCall } from '@sepilotd/core'
import type { AgentEvidenceLedger } from './graph/types.js'
import {
  evidenceRequirementMinSourceFiles,
  evidenceRequirementMinSourceObservations,
  evidenceRequirementMinSourceScopes,
  evidenceRequirementRequiresSearch,
  evidenceRequirementUsesRepositoryBreadth,
  sourceToolMatchesEvidenceRequirement,
} from './evidence-requirement-policy.js'
import { explicitlyForbidsToolUse } from './request-shape.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

export type RunOutcomeReviewStatus = 'complete' | 'needs_recovery'
export type RunOutcomeReviewRecoveryBasis =
  | 'missing-tool-execution'
  | 'insufficient-tool-evidence'
  | 'answer-synthesis'
  | 'user-input'

export interface RunOutcomeReview {
  status: RunOutcomeReviewStatus
  reason: string
  instruction?: string
  recoveryMode?: 'tool' | 'synthesis' | 'user-input'
  /**
   * Structured cause for a recovery request. Historical execution claims are
   * grounded against the trusted current-turn result ledger before they can
   * influence the agent loop; semantic evidence-quality judgments are not.
   */
  recoveryBasis?: RunOutcomeReviewRecoveryBasis
  /** Exact registry names claimed to lack a successful current-turn result. */
  missingToolNames?: string[]
  toolCalls?: Array<{
    name: string
    arguments: Record<string, unknown>
  }>
}

export const OUTCOME_REVIEW_MAX_TOKENS = 2048
const OUTCOME_REVIEW_CANDIDATE_MAX_CHARS = 12_000
export const MAX_OUTCOME_REVIEW_REPAIRS = 4
/**
 * Rewriting a final from an unchanged evidence snapshot can improve semantic
 * coverage, but repeated verifier/writer ping-pong cannot discover new facts.
 * Allow two comprehensive correction passes, then fail closed. Tool-backed
 * recovery keeps the broader MAX_OUTCOME_REVIEW_REPAIRS budget because each
 * executed result can materially change the evidence available to the judge.
 */
export const MAX_OUTCOME_REVIEW_SYNTHESIS_REPAIRS = 2
const OUTCOME_REVIEW_RECOVERY_MARKER = '[LLM outcome review]'
const GENERATED_OR_VENDOR_PATH_SEGMENTS = new Set([
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

function extractMessageText(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content
  }

  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function compact(value: string, max = 1200): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  if (normalized.length <= max) {
    return normalized
  }
  return `${normalized.slice(0, max - 3).trimEnd()}...`
}

function compactBoundaries(value: string, max: number): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  if (normalized.length <= max) return normalized
  const marker = ' ... '
  const available = Math.max(0, max - marker.length)
  const headLength = Math.ceil(available * 0.6)
  const tailLength = available - headLength
  return `${normalized.slice(0, headLength).trimEnd()}${marker}${normalized.slice(-tailLength).trimStart()}`
}

function formatCandidateAnswerForReview(value: string): string {
  const candidate = value.trim()
  if (!candidate) return 'Candidate transport: complete (0 chars).\n[empty]'
  if (candidate.length <= OUTCOME_REVIEW_CANDIDATE_MAX_CHARS) {
    return `Candidate transport: complete (${candidate.length} chars).\n${candidate}`
  }

  const marker = [
    '',
    '[outcome-review transport omitted bounded middle content; this marker is not part of the assistant answer]',
    '',
  ].join('\n')
  const available = OUTCOME_REVIEW_CANDIDATE_MAX_CHARS - marker.length
  const headLength = Math.ceil(available * 0.6)
  const tailLength = available - headLength
  return [
    `Candidate transport: bounded head/tail excerpt (${candidate.length} original chars).`,
    candidate.slice(0, headLength),
    marker,
    candidate.slice(-tailLength),
  ].join('\n')
}

function summarizeTextArgument(value: string): { chars: number; preview: string } {
  return {
    chars: value.length,
    preview: compact(value, 900),
  }
}

function summarizeToolArguments(name: string, args: Record<string, unknown>): string {
  const summary: Record<string, unknown> = { ...args }
  if (typeof summary.content === 'string') {
    summary.content = summarizeTextArgument(summary.content)
  }
  if (typeof summary.contents === 'string') {
    summary.contents = summarizeTextArgument(summary.contents)
  }
  if (typeof summary.patch === 'string') {
    summary.patch = summarizeTextArgument(summary.patch)
  }

  return compact(`${name} ${JSON.stringify(summary)}`, 700)
}

function formatToolCalls(message: Message): string {
  const calls = message.toolCalls ?? []
  if (calls.length === 0) {
    return ''
  }

  return [
    '[tool calls]',
    ...calls.map((toolCall) =>
      `- ${toolCall.id}: ${summarizeToolArguments(toolCall.name, toolCall.arguments)}`,
    ),
  ].join('\n')
}

interface WriteEvidence {
  tool: string
  path?: string
  status: 'success' | 'error' | 'unknown'
  chars?: number
  preview?: string
  result?: string
  resultMessageIndex?: number
}

interface EvidenceFloorResult {
  reason: string
  instruction: string
  toolCalls?: NonNullable<RunOutcomeReview['toolCalls']>
}

type ArtifactSection = NonNullable<AgentRunContract['artifactSections']>[number]

interface LedgerEvidenceStats {
  sourceObservationTools: string[]
  sourceReadPaths: string[]
  sourceScopes: string[]
  searchCount: number
  successfulWritePaths: string[]
  artifactReadBackPaths: string[]
  validationCount: number
}

function normalizeEvidencePath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^\.\//, '').replace(/\/+$/g, '')
}

function pathHasFileExtension(value: string): boolean {
  return /\.(?:[cm]?[jt]sx?|mjs|cjs|go|rs|py|java|kt|cs|rb|php|swift|md|mdx|json|ya?ml|toml|lock|sql|sh|bash|zsh|fish|html|css|scss|vue|svelte)$/i.test(value)
}

function pathLooksLikeGlob(value: string): boolean {
  return /[*?[\]{}]/.test(value)
}

function looksLikeMimeType(value: string): boolean {
  return /^(?:application|audio|example|font|image|message|model|multipart|text|video)\/[A-Za-z0-9.+-]+$/i.test(value)
}

function looksLikeExtensionOnlyLiteral(value: string): boolean {
  return /^\.[A-Za-z0-9][A-Za-z0-9_-]{0,15}$/.test(value)
}

function looksLikeKeyboardShortcutLiteral(value: string): boolean {
  return /^(?:ctrl|control|cmd|command|alt|option|shift|meta)(?:\+[a-z0-9_-]+(?:\/[a-z0-9_-]+)*)+$/i.test(value)
}

function hasExplicitRepositoryPathPrefix(value: string): boolean {
  return /^(?:\.{1,2}\/|\/|[A-Za-z]:\/)/.test(value)
}

function implicitRepositoryPathRootLooksPathLike(value: string): boolean {
  const normalized = normalizeEvidencePath(value)
    .replace(/^(?:\.{1,2}\/)+/, '')
  const first = normalized.split('/').filter(Boolean).at(0) ?? ''
  return /^[._a-z0-9-]/.test(first)
}

function singleSegmentFileLooksLikeRepositoryPath(value: string): boolean {
  if (!pathHasFileExtension(value) || value.includes('/')) {
    return false
  }
  if (implicitRepositoryPathRootLooksPathLike(value)) {
    return true
  }
  const basename = value.split('.').at(0) ?? ''
  return /^[A-Z0-9_-]+$/.test(basename)
}

function stripPathLineReferenceSuffix(value: string): string {
  return value.replace(
    /(?<=\.[A-Za-z0-9][A-Za-z0-9_-]{0,15})(?::\d+(?:-\d+)?(?::\d+(?:-\d+)?)?)$/u,
    '',
  )
}

function globStaticPrefix(value: string): string {
  const normalized = normalizeEvidencePath(value)
  const globIndex = normalized.search(/[*?[\]{}]/)
  return globIndex === -1 ? normalized : normalized.slice(0, globIndex)
}

function looksLikeRepositoryPath(value: string, source: 'code' | 'bare' = 'code', originalValue = value): boolean {
  const trimmed = value.trim()
  if (
    !trimmed
    // URI schemes may have opaque paths without // (module identifiers,
    // package URLs, resource URNs). Their slash does not imply a local file.
    // Keep drive-letter paths eligible for ordinary repository verification.
    || (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(trimmed)
      && !/^[A-Za-z]:[\\/]/.test(trimmed)
      && !hasExplicitRepositoryPathPrefix(originalValue.trim()))
    || trimmed.startsWith('~/')
    || trimmed.startsWith('$')
    || trimmed.includes('<')
    || trimmed.includes('>')
    || looksLikeMimeType(trimmed)
    || looksLikeExtensionOnlyLiteral(trimmed)
    || looksLikeKeyboardShortcutLiteral(trimmed)
    || /^@[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/.test(trimmed)
    || trimmed.includes('\n')
    || /\s/.test(trimmed)
  ) {
    return false
  }
  if (isGeneratedOrVendorClaimPath(trimmed)) {
    return false
  }
  if (!trimmed.includes('/') && pathHasFileExtension(trimmed) && !singleSegmentFileLooksLikeRepositoryPath(trimmed)) {
    return false
  }
  if (
    trimmed.includes('/')
    && !hasExplicitRepositoryPathPrefix(trimmed)
    && !pathLooksLikeGlob(trimmed)
    && !implicitRepositoryPathRootLooksPathLike(trimmed)
  ) {
    return false
  }
  if (source === 'bare') {
    if (trimmed.startsWith('/') && !pathHasFileExtension(trimmed)) {
      return false
    }
    return pathHasFileExtension(trimmed) || /^(?:\.{1,2}\/|\/|~\/)/.test(trimmed)
  }
  if (trimmed.startsWith('/') && !pathHasFileExtension(trimmed)) {
    return false
  }
  return (
    trimmed.includes('/')
    || pathHasFileExtension(trimmed)
  )
}

function isGeneratedOrVendorClaimPath(value: string): boolean {
  const segments = normalizeEvidencePath(value).split('/').filter(Boolean)
  if (segments.some((segment) => GENERATED_OR_VENDOR_PATH_SEGMENTS.has(segment))) {
    return true
  }
  const filename = segments.at(-1) ?? value
  return filename.endsWith('.map')
    || filename.endsWith('.min.css')
    || filename.endsWith('.min.js')
}

function normalizeClaimedPath(value: string): string {
  return normalizeEvidencePath(
    stripPathLineReferenceSuffix(
      value
        .trim()
        .replace(/^["'([]+|[)"'\].,;:]+$/g, ''),
    ),
  )
}

function extractClaimedRepositoryPaths(text: string): string[] {
  const paths = new Set<string>()
  const codeSpanPattern = /`([^`\r\n]{1,180})`/g
  let match: RegExpExecArray | null
  while ((match = codeSpanPattern.exec(text)) !== null) {
    const normalized = normalizeClaimedPath(match[1] ?? '')
    if (looksLikeRepositoryPath(normalized, 'code', match[1])) {
      paths.add(normalized)
    }
  }
  const barePathPattern =
    /(?:^|[\s("'[])(\.{0,2}\/?[A-Za-z0-9_.@-]+(?:\/[A-Za-z0-9_.@*{}[\]-]+)+(?:\.[A-Za-z0-9][A-Za-z0-9_-]{0,15})?\/?)(?=$|[\s"',.)\]])/g
  while ((match = barePathPattern.exec(text)) !== null) {
    const normalized = normalizeClaimedPath(match[1] ?? '')
    if (looksLikeRepositoryPath(normalized, 'bare')) {
      paths.add(normalized)
    }
  }
  return Array.from(paths).slice(0, 24)
}

function evidencePathsReferToSameFile(left: string, right: string): boolean {
  const normalizedLeft = normalizeEvidencePath(left)
  const normalizedRight = normalizeEvidencePath(right)
  return (
    normalizedLeft === normalizedRight
    || normalizedLeft.endsWith(`/${normalizedRight}`)
    || normalizedRight.endsWith(`/${normalizedLeft}`)
  )
}

function uniqueNormalizedPaths(paths: string[]): string[] {
  const seen = new Set<string>()
  const unique: string[] = []
  for (const path of paths) {
    const normalized = normalizeEvidencePath(path)
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    unique.push(normalized)
  }
  return unique
}

function directorySegmentsForEvidencePath(path: string): string[] {
  const segments = normalizeEvidencePath(path)
    .split('/')
    .filter(Boolean)
  if (segments.length === 0) {
    return []
  }
  const last = segments.at(-1) ?? ''
  return pathHasFileExtension(last) ? segments.slice(0, -1) : segments
}

function commonLeadingSegments(paths: string[][]): string[] {
  if (paths.length === 0) {
    return []
  }
  const [first, ...rest] = paths
  const common = [...first]
  for (const path of rest) {
    let index = 0
    while (index < common.length && index < path.length && common[index] === path[index]) {
      index += 1
    }
    common.length = index
    if (common.length === 0) {
      break
    }
  }
  return common
}

function sourceEvidenceScopes(paths: string[]): string[] {
  const sourcePaths = uniqueNormalizedPaths(paths.filter(isSourceCodeEvidencePath))
  const directories = sourcePaths
    .map(directorySegmentsForEvidencePath)
    .filter((segments) => segments.length > 0)
  if (directories.length === 0) {
    return []
  }
  const commonPrefix = commonLeadingSegments(directories)
  const scopes = new Set<string>()
  for (const segments of directories) {
    const scopeSegments = segments.length > commonPrefix.length
      ? segments.slice(0, commonPrefix.length + 1)
      : segments
    const scope = scopeSegments.join('/')
    if (scope) {
      scopes.add(scope)
    }
  }
  return Array.from(scopes)
}

function ledgerEvidenceStats(ledger: AgentEvidenceLedger | undefined): LedgerEvidenceStats {
  if (!ledger) {
    return {
      sourceObservationTools: [],
      sourceReadPaths: [],
      sourceScopes: [],
      searchCount: 0,
      successfulWritePaths: [],
      artifactReadBackPaths: [],
      validationCount: 0,
    }
  }
  const sourceReadPaths = uniqueNormalizedPaths(
    ledger.sourceReads
      .filter((entry) => entry.status === 'success' && entry.tool === 'fs.read' && entry.path)
      .map((entry) => entry.path!),
  )
  const seenObservationIds = new Set<string>()
  const sourceObservationTools = [
    ...ledger.sourceReads,
    ...ledger.sourceSearches,
  ].filter((entry) => {
    if (entry.status !== 'success' || entry.executionObserved === false) return false
    const identity = entry.toolCallId
      ? `call:${entry.toolCallId}`
      : `entry:${entry.tool}:${entry.order ?? entry.ts}:${entry.path ?? ''}:${entry.query ?? ''}`
    if (seenObservationIds.has(identity)) return false
    seenObservationIds.add(identity)
    return true
  }).map((entry) => entry.tool)
  return {
    sourceObservationTools,
    sourceReadPaths,
    sourceScopes: sourceEvidenceScopes(sourceReadPaths),
    searchCount: ledger.sourceSearches.filter((entry) => entry.status === 'success').length,
    successfulWritePaths: uniqueNormalizedPaths(
      ledger.artifactWrites
        .filter((entry) => entry.status === 'success' && entry.path)
        .map((entry) => entry.path!),
    ),
    artifactReadBackPaths: uniqueNormalizedPaths(
      ledger.artifactReadBacks
        .filter((entry) => entry.status === 'success' && entry.path)
        .map((entry) => entry.path!),
    ),
    validationCount: ledger.validationRuns.filter((entry) => entry.status === 'success').length,
  }
}

function ledgerHasSuccessfulWriteForPath(
  ledger: AgentEvidenceLedger | undefined,
  path: string,
): boolean {
  return ledgerEvidenceStats(ledger).successfulWritePaths.some((writtenPath) =>
    evidencePathsReferToSameFile(writtenPath, path)
  )
}

function ledgerHasArtifactReadBackForPath(
  ledger: AgentEvidenceLedger | undefined,
  path: string,
): boolean {
  return ledgerEvidenceStats(ledger).artifactReadBackPaths.some((readPath) =>
    evidencePathsReferToSameFile(readPath, path)
  )
}

function outputMentionsPath(output: string, path: string): boolean {
  const normalizedPath = normalizeEvidencePath(path)
  const claimLooksLikeFile = pathHasFileExtension(normalizedPath)
  return output
    .split(/\r?\n/)
    .map((line) =>
      normalizeEvidencePath(
        (line.trim().split(/\s+/)[0] ?? '').replace(/:\d+(?::\d+)?(?::.*)?$/g, ''),
      )
    )
    .some((linePath) =>
      linePath === normalizedPath
      || linePath.endsWith(`/${normalizedPath}`)
      || normalizedPath.endsWith(`/${linePath}`)
      || (!claimLooksLikeFile && linePath.startsWith(`${normalizedPath}/`))
    )
}

function evidencePathSupportsClaim(evidencePath: string, claimedPath: string): boolean {
  const normalizedEvidence = normalizeEvidencePath(evidencePath)
  const normalizedClaim = normalizeEvidencePath(claimedPath)
  const evidenceIsGlob = pathLooksLikeGlob(normalizedEvidence)
  const claimIsGlob = pathLooksLikeGlob(normalizedClaim)
  if (evidenceIsGlob || claimIsGlob) {
    if (evidencePathsReferToSameFile(normalizedEvidence, normalizedClaim)) {
      return true
    }
    if (evidenceIsGlob && claimIsGlob) {
      const evidencePrefix = globStaticPrefix(normalizedEvidence)
      const claimPrefix = globStaticPrefix(normalizedClaim)
      return evidencePrefix === claimPrefix
        || evidencePrefix.endsWith(`/${claimPrefix}`)
        || claimPrefix.endsWith(`/${evidencePrefix}`)
    }
    if (evidenceIsGlob && !pathHasFileExtension(normalizedClaim)) {
      const evidencePrefix = globStaticPrefix(normalizedEvidence)
      return evidencePrefix === normalizedClaim
        || evidencePrefix.startsWith(`${normalizedClaim}/`)
        || evidencePrefix.endsWith(`/${normalizedClaim}/`)
    }
    return false
  }
  return evidencePathsReferToSameFile(normalizedEvidence, normalizedClaim)
    || (!pathHasFileExtension(normalizedClaim) && normalizedEvidence.startsWith(`${normalizedClaim}/`))
}

function structuredGitObservedPaths(toolCall: ToolCall, output: string): string[] {
  const paths = new Set<string>()
  if (toolCall.name === 'git.log' && toolCall.arguments.detailed === true) {
    for (const line of output.split(/\r?\n/)) {
      const statPath = line.match(/^\s*(.+?)\s+\|\s+(?:\d+|Bin\b)/)?.[1]?.trim()
      if (statPath && looksLikeRepositoryPath(statPath, 'bare')) {
        paths.add(normalizeEvidencePath(statPath))
      }
      const numstatPath = line.match(/^(?:\d+|-)\t(?:\d+|-)\t(.+)$/)?.[1]?.trim()
      if (numstatPath && looksLikeRepositoryPath(numstatPath, 'bare')) {
        paths.add(normalizeEvidencePath(numstatPath))
      }
      const lifecyclePath = line.match(/^\s*(?:create|delete) mode \d+ (.+)$/)?.[1]?.trim()
      if (lifecyclePath && looksLikeRepositoryPath(lifecyclePath, 'bare')) {
        paths.add(normalizeEvidencePath(lifecyclePath))
      }
    }
  }
  if (toolCall.name === 'git.diff') {
    for (const line of output.split(/\r?\n/)) {
      const header = line.match(/^diff --git a\/(.+?) b\/(.+)$/)
      if (header) {
        if (header[1]) paths.add(normalizeEvidencePath(header[1]))
        if (header[2]) paths.add(normalizeEvidencePath(header[2]))
        continue
      }
      const marker = line.match(/^(?:---|\+\+\+) [ab]\/(.+)$/)?.[1]
      if (marker) paths.add(normalizeEvidencePath(marker))
      const rename = line.match(/^rename (?:from|to) (.+)$/)?.[1]
      if (rename) paths.add(normalizeEvidencePath(rename))
    }
  }
  return [...paths]
}

function hasStructuredGitPathEvidence(toolCall: ToolCall): boolean {
  return toolCall.name === 'git.diff'
    || (toolCall.name === 'git.log' && toolCall.arguments.detailed === true)
}

export function findUnsupportedRepositoryPathClaims(options: {
  text: string
  messages: Message[]
  evidenceScope?: 'recent' | 'all'
  extraEvidencedPaths?: string[]
}): string[] {
  if (!options.text.trim()) {
    return []
  }
  const claimedPaths = extractClaimedRepositoryPaths(options.text)
  if (claimedPaths.length === 0) {
    return []
  }

  const evidenceMessages = options.evidenceScope === 'all'
    ? options.messages
    : messagesSinceLastUser(options.messages)
  const toolCallsById = new Map<string, ToolCall>()
  const evidencedPaths = new Set<string>(
    (options.extraEvidencedPaths ?? [])
      .filter((path) => path.trim().length > 0)
      .map(normalizeEvidencePath),
  )
  const evidenceOutputs: string[] = []
  for (const message of evidenceMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
    }

    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    const output = extractMessageText(message)
    if (isNonEvidentialPathToolResultText(output)) {
      continue
    }
    const args = toolCall?.arguments ?? {}
    if (
      toolCall
      && (toolCall.name === 'fs.read' || toolCall.name === 'fs.write' || toolCall.name === 'fs.append' || toolCall.name === 'fs.edit')
      && typeof args.path === 'string'
    ) {
      evidencedPaths.add(normalizeEvidencePath(args.path))
    }
    if (toolCall?.name === 'fs.glob' && typeof args.pattern === 'string') {
      evidencedPaths.add(normalizeEvidencePath(args.pattern))
      if (typeof args.cwd === 'string' && args.cwd.trim()) {
        evidencedPaths.add(normalizeEvidencePath(`${args.cwd.replace(/\/+$/g, '')}/${args.pattern}`))
      }
    }
    if (toolCall?.name === 'fs.search' && typeof args.glob === 'string') {
      evidencedPaths.add(normalizeEvidencePath(args.glob))
      if (typeof args.cwd === 'string' && args.cwd.trim()) {
        evidencedPaths.add(normalizeEvidencePath(`${args.cwd.replace(/\/+$/g, '')}/${args.glob}`))
      }
    }
    if (toolCall?.name === 'apply_patch' && typeof args.patch === 'string') {
      for (const path of extractApplyPatchPaths(args.patch)) {
        evidencedPaths.add(normalizeEvidencePath(path))
      }
    }
    if (toolCall?.name === 'fs.glob' || toolCall?.name === 'fs.search') {
      evidenceOutputs.push(output)
    }
    if (
      toolCall?.name === 'fs.read'
      || toolCall?.name === 'fs.glob'
      || toolCall?.name === 'fs.search'
    ) {
      for (const observedPath of extractClaimedRepositoryPaths(output)) {
        evidencedPaths.add(normalizeEvidencePath(observedPath))
      }
    }
    if (toolCall && hasStructuredGitPathEvidence(toolCall)) {
      for (const observedPath of structuredGitObservedPaths(toolCall, output)) {
        evidencedPaths.add(observedPath)
      }
    }
    if (toolCall?.name === 'fs.read' && typeof toolCall.arguments.path === 'string') {
      evidencedPaths.add(normalizeEvidencePath(toolCall.arguments.path))
    }
  }

  return claimedPaths.filter((claimedPath) => {
    if (Array.from(evidencedPaths).some((evidencePath) =>
      evidencePathSupportsClaim(evidencePath, claimedPath)
    )) {
      return false
    }
    return !evidenceOutputs.some((output) => outputMentionsPath(output, claimedPath))
  })
}

function formatWriteEvidence(write: WriteEvidence): string {
  const path = write.path ? ` ${write.path}` : ''
  const status = ` [${write.status}]`
  const chars = typeof write.chars === 'number' ? ` (${write.chars} chars)` : ''
  const preview = write.preview ? ` preview="${write.preview}"` : ''
  const result = write.result ? ` result="${write.result}"` : ''
  return `${write.tool}${path}${status}${chars}${preview}${result}`
}

function stripToolSummaryPrefix(content: string): string {
  return content.replace(/^\[[^\]]+ output summarized for agent context:[^\]]+\]\s*/i, '')
}

function summarizeMarkdownStructure(content: string): string | null {
  const body = stripToolSummaryPrefix(content)
  const headings: string[] = []
  const fenceLanguages: string[] = []
  const seenFenceLanguages = new Set<string>()
  for (const line of body.split(/\r?\n/)) {
    const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*$/)
    if (heading?.[2]) {
      headings.push(`${heading[1].length}:${compact(heading[2], 90)}`)
    }
    const fence = line.match(/^```\s*([A-Za-z0-9_.-]+)?/)
    if (fence) {
      const language = (fence[1] ?? '(plain)').trim() || '(plain)'
      if (!seenFenceLanguages.has(language)) {
        seenFenceLanguages.add(language)
        fenceLanguages.push(language)
      }
    }
  }
  if (headings.length === 0 && fenceLanguages.length === 0) {
    return null
  }
  const lineCount = body.split(/\r?\n/).length
  return [
    `${body.length} chars`,
    `${lineCount} lines`,
    `headings=${headings.slice(0, 18).join(' | ') || '(none)'}`,
    `codeFences=${fenceLanguages.slice(0, 12).join(', ') || '(none)'}`,
  ].join('; ')
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

function markdownHeadingTitles(content: string): string[] {
  const body = stripToolSummaryPrefix(content)
  return body
    .split(/\r?\n/)
    // fs.read prefixes text with display-only `NNN<TAB>` line numbers. The
    // evidence floor consumes the tool result, not raw file bytes, so remove
    // that presentation prefix before recognising Markdown headings.
    .map((line) => line.replace(/^\s*\d+\t/, ''))
    .map((line) => line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/)?.[1] ?? '')
    .filter(Boolean)
    .map(normalizeArtifactHeading)
    .filter(Boolean)
}

function artifactSectionTokens(value: string): string[] {
  return normalizeArtifactHeading(value)
    .split(/\s+/)
    .filter((token) => token.length >= 2)
}

function artifactHeadingMatchesSection(heading: string, expected: string): boolean {
  if (heading === expected || heading.includes(expected)) {
    return true
  }
  const expectedTokens = artifactSectionTokens(expected)
  if (expectedTokens.length <= 1) {
    return false
  }
  const headingTokens = new Set(artifactSectionTokens(heading))
  return expectedTokens.every((token) => headingTokens.has(token))
}

function artifactHasSection(content: string, title: string): boolean {
  const expected = normalizeArtifactHeading(title)
  if (!expected) {
    return false
  }
  return markdownHeadingTitles(content).some((heading) =>
    artifactHeadingMatchesSection(heading, expected)
  )
}

const FILE_MUTATION_TOOL_NAMES = new Set(['fs.write', 'fs.append', 'fs.edit', 'fs.move', 'apply_patch'])
const VALIDATION_TOOL_NAMES = new Set([
  'terminal.run',
  'assistant.status',
  'gitea.actions.runs.inspect',
  'system.info',
  'usage.report',
  'process.list',
  'process.sessions',
  'process.follow',
  'service.status',
  'service.logs',
  'service.healthcheck',
  'schedule_list',
  'schedule_runs',
])

function isFileMutationTool(name: string): boolean {
  return FILE_MUTATION_TOOL_NAMES.has(name)
}

function isValidationTool(name: string): boolean {
  return VALIDATION_TOOL_NAMES.has(name)
}

function extractApplyPatchPaths(patch: string): string[] {
  const paths = new Set<string>()
  const pattern = /^\*\*\* (?:Add|Update|Delete) File: (.+)$/gm
  let match: RegExpExecArray | null
  while ((match = pattern.exec(patch)) !== null) {
    const path = match[1]?.trim()
    if (path) {
      paths.add(path)
    }
  }
  return Array.from(paths)
}

function fileMutationPaths(toolName: string, args: Record<string, unknown>): string[] {
  if ((toolName === 'fs.write' || toolName === 'fs.append' || toolName === 'fs.edit') && typeof args.path === 'string') {
    return [args.path]
  }
  // A move mutates both ends, so both count as touched paths.
  if (toolName === 'fs.move') {
    return [args.from, args.to].filter((value): value is string => typeof value === 'string')
  }
  if (toolName === 'apply_patch' && typeof args.patch === 'string') {
    return extractApplyPatchPaths(args.patch)
  }
  return []
}

function isFailedToolResultText(content: string): boolean {
  const trimmed = content.trim()
  return trimmed.length === 0
    || /^\[approval:(?:denied|needs-changes)\]/i.test(trimmed)
    || /^\[error:/i.test(trimmed)
    || /\bTool\s+[A-Za-z0-9_.-]+\s+blocked:/i.test(trimmed)
}

interface SuccessfulToolExecutions {
  names: Set<string>
  counts: Map<string, number>
}

function toolResultMessageSucceeded(message: Message): boolean {
  const status = message.metadata?.toolResultStatus
  if (status === 'error' || status === 'blocked') return false
  if (status === 'success') return true
  return !isFailedToolResultText(extractMessageText(message))
}

function collectSuccessfulToolExecutions(
  messages: Message[],
  evidenceLedger?: AgentEvidenceLedger,
): SuccessfulToolExecutions {
  const toolCallsById = new Map<string, ToolCall>()
  const seenToolCallIds = new Set<string>()
  const names = new Set<string>()
  const counts = new Map<string, number>()
  const record = (name: string, toolCallId?: string): void => {
    const normalized = name.trim()
    if (!normalized) return
    if (toolCallId) {
      if (seenToolCallIds.has(toolCallId)) return
      seenToolCallIds.add(toolCallId)
    }
    names.add(normalized)
    counts.set(normalized, (counts.get(normalized) ?? 0) + 1)
  }

  for (const message of messagesSinceLastUser(messages)) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
    }
    if (message.role !== 'tool' || !toolResultMessageSucceeded(message)) continue
    const toolName = message.toolCallId
      ? toolCallsById.get(message.toolCallId)?.name ?? message.name
      : message.name
    if (toolName) record(toolName, message.toolCallId)
  }

  if (evidenceLedger) {
    const successfulEntries = [
      ...evidenceLedger.sourceReads,
      ...evidenceLedger.sourceSearches,
      ...evidenceLedger.artifactWrites,
      ...evidenceLedger.artifactReadBacks,
      ...evidenceLedger.validationRuns,
    ].filter((entry) => entry.status === 'success' && entry.executionObserved !== false)
    for (const entry of successfulEntries) {
      record(entry.tool, entry.toolCallId)
    }
  }

  return { names, counts }
}

function formatSuccessfulToolExecutions(executions: SuccessfulToolExecutions): string {
  if (executions.counts.size === 0) return '(none)'
  return Array.from(executions.counts.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, count]) => `${name} x${count}`)
    .join(', ')
}

function isNonEvidentialPathToolResultText(content: string): boolean {
  const trimmed = content.trim()
  return isFailedToolResultText(trimmed)
    || /^\[no matches(?: after offset \d+)?\]$/i.test(trimmed)
}

function isSuccessfulFileMutationResult(toolName: string, content: string): boolean {
  if (isFailedToolResultText(content)) {
    return false
  }
  if (toolName === 'fs.write') {
    return /\bWrote\s+\d+\s+bytes\s+to\s+/i.test(content)
  }
  if (toolName === 'fs.append') {
    return /\bAppended\s+\d+\s+bytes\s+to\s+/i.test(content)
  }
  if (toolName === 'fs.edit') {
    return /\bReplaced\s+\d+\s+occurrences?\s+in\s+/i.test(content)
  }
  if (toolName === 'apply_patch') {
    return /\bApplied patch to\s+\d+\s+file\(s\):/i.test(content)
  }
  return false
}

function classifyFileMutationResult(
  toolName: string,
  content: string | undefined,
): WriteEvidence['status'] {
  if (content === undefined) {
    return 'unknown'
  }
  return isSuccessfulFileMutationResult(toolName, content) ? 'success' : 'error'
}

export function unavailableOutcomeReviewReason(messages: Message[]): string | null {
  return collectFileMutationEvidence(messages).length > 0
    ? 'Completion review was unavailable after file-edit attempts. The current changes have not been validated as a complete result.'
    : null
}

function collectFileMutationEvidence(messages: Message[]): WriteEvidence[] {
  const recentMessages = messagesSinceLastUser(messages)
  const resultByToolCallId = new Map<string, { text: string; index: number }>()
  for (const [index, message] of recentMessages.entries()) {
    if (message.role === 'tool' && message.toolCallId) {
      resultByToolCallId.set(message.toolCallId, {
        text: extractMessageText(message),
        index,
      })
    }
  }

  const writeOperations: WriteEvidence[] = []
  for (const message of recentMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      if (!isFileMutationTool(toolCall.name)) {
        continue
      }
      const args = toolCall.arguments
      const textPayload = typeof args.content === 'string'
        ? args.content
        : typeof args.contents === 'string'
          ? args.contents
          : typeof args.patch === 'string'
            ? args.patch
            : undefined
      const paths = fileMutationPaths(toolCall.name, args)
      const result = resultByToolCallId.get(toolCall.id)
      const resultText = result?.text
      const status = classifyFileMutationResult(toolCall.name, resultText)
      const base = {
        tool: toolCall.name,
        status,
        ...(textPayload ? summarizeTextArgument(textPayload) : {}),
        ...(resultText ? { result: compact(resultText, 220) } : {}),
        ...(result ? { resultMessageIndex: result.index } : {}),
      }
      if (paths.length === 0) {
        writeOperations.push(base)
      } else {
        writeOperations.push(...paths.map((path) => ({ ...base, path })))
      }
    }
  }
  return writeOperations
}

function isSuccessfulReadToolResult(content: string): boolean {
  const trimmed = content.trim()
  return trimmed.length > 0
    && !trimmed.startsWith('[error:')
    && !/^\[fs\.read:\s.*\b(?:is a directory|past end of file)\b/i.test(trimmed)
}

function isSuccessfulValidationToolResult(content: string): boolean {
  const trimmed = content.trim()
  return trimmed.length > 0
    && !isFailedToolResultText(trimmed)
    && !/\bExecution failed\b/i.test(trimmed)
    && !/\bProcess exited with code\s+(?!0\b)\d+\b/i.test(trimmed)
    && !/\bexit code\s+(?!0\b)\d+\b/i.test(trimmed)
}

/**
 * Read-only lookups whose successful output is what an answer can be grounded
 * in when the run produces nothing to validate. Deliberately narrow: retrieval
 * and inspection tools only, never a mutation or a command run.
 */
const LOOKUP_EVIDENCE_TOOL_NAMES = new Set([
  'memory.search',
  'memory.list',
  'memory.graph.search',
  'memory.graph.page',
  'memory.graph.neighbors',
  'memory.documents.search',
  'memory.daily.read',
  'memory.daily.search',
  'memory.context.snapshot',
  'fs.read',
  'fs.list',
  'fs.glob',
  'fs.search',
  'git.status',
  'git.diff',
  'git.log',
  'code.symbols',
  'code.dependencies',
  'web.search',
  'webfetch',
  'browser.navigate',
  'browser.extract',
  'apps.read',
  'apps.search',
  'doc.get',
  'doc.outline',
])

function collectSuccessfulLookupEvidence(messages: Message[]): string[] {
  const recentMessages = messagesSinceLastUser(messages)
  const toolCallsById = new Map<string, ToolCall>()
  for (const message of recentMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
    }
  }

  const evidence: string[] = []
  for (const message of recentMessages) {
    if (message.role !== 'tool' || !message.toolCallId) continue
    const toolCall = toolCallsById.get(message.toolCallId)
    if (!toolCall || !LOOKUP_EVIDENCE_TOOL_NAMES.has(toolCall.name)) continue
    const text = extractMessageText(message)
    if (!text.trim() || isFailedToolResultText(text.trim())) continue
    evidence.push(toolCall.name)
  }
  return evidence
}

function collectSuccessfulValidationEvidence(messages: Message[]): string[] {
  const recentMessages = messagesSinceLastUser(messages)
  const toolCallsById = new Map<string, ToolCall>()
  for (const message of recentMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
    }
  }

  const evidence: string[] = []
  for (const message of recentMessages) {
    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    if (!toolCall || !isValidationTool(toolCall.name)) {
      continue
    }
    const text = extractMessageText(message)
    if (!isSuccessfulValidationToolResult(text)) {
      continue
    }
    const command = typeof toolCall.arguments.command === 'string'
      ? toolCall.arguments.command
      : typeof toolCall.arguments.executable === 'string'
        ? [
            toolCall.arguments.executable,
            ...(Array.isArray(toolCall.arguments.args)
              ? toolCall.arguments.args.filter((entry): entry is string => typeof entry === 'string')
              : []),
          ].join(' ')
        : undefined
    evidence.push(command ? `${toolCall.name}: ${compact(command, 160)}` : toolCall.name)
  }
  return evidence
}

function isBroadGlobPattern(value: unknown): boolean {
  if (typeof value !== 'string') {
    return false
  }
  const trimmed = value.trim()
  return trimmed === '*' || trimmed === '**' || trimmed === '**/*' || trimmed === './**/*'
}

function isSummarizedBroadGlobResult(content: string): boolean {
  return /\bfs\.glob output summarized for agent context\b/i.test(content)
    || /\[\d+\s+omitted lines\]/i.test(content)
    || content.length > 20_000
}

function buildTargetedInventoryToolCalls(
  cwd: string | undefined,
  candidates: string[] = [],
  readFiles: Set<string> = new Set(),
): NonNullable<RunOutcomeReview['toolCalls']> {
  const readCalls = selectRepresentativeUnreadSourcePaths(candidates, readFiles, 4)
    .map((path) => ({
      name: 'fs.read',
      arguments: { path },
    }))
  if (readCalls.length > 0) {
    return readCalls
  }

  const args = cwd ? { cwd } : {}
  return [
    {
      name: 'fs.glob',
      arguments: {
        ...args,
        patterns: [
          'packages/*/package.json',
          'packages/*/src/index.ts',
          'packages/*/src/main.ts',
          'docs/architecture/*.md',
          '*.md',
        ],
        limit: 500,
      },
    },
    {
      name: 'fs.search',
      arguments: {
        ...args,
        query: 'export |class |interface |function |route',
        glob: 'packages/*/src/**/*.{ts,tsx,js,jsx}',
        fixedStrings: false,
        limit: 5,
      },
    },
  ]
}

function isSourceCodeEvidencePath(path: string): boolean {
  const normalized = normalizeEvidencePath(path)
  if (
    normalized.startsWith('docs/')
    || normalized.includes('/docs/')
    || normalized.startsWith('docs-')
    || normalized.endsWith('.md')
    || normalized.endsWith('.mdx')
  ) {
    return false
  }
  return /\.(?:[cm]?[jt]sx?|mjs|cjs|go|rs|py|java|kt|cs|rb|php|swift|sh|bash|zsh|fish|sql|vue|svelte|css|scss)$/i.test(normalized)
}

function packageScopeForPath(path: string): string | null {
  const normalized = normalizeEvidencePath(path)
  const match = normalized.match(/(?:^|\/)(packages\/[A-Za-z0-9_.-]+)(?:\/|$)/)
  return match?.[1] ?? null
}

function addPackageScopesFromToolOutput(output: string, scopes: Set<string>): void {
  const pattern =
    /(?:^|[\s"'(])((?:\.\/)?packages\/[A-Za-z0-9_.-]+)\/(?:package\.json|src\/[^\s"'`),]+)/g
  let match: RegExpExecArray | null
  while ((match = pattern.exec(output)) !== null) {
    const scope = packageScopeForPath(match[1] ?? '')
    if (scope) {
      scopes.add(scope)
    }
  }
}

function collectSourcePathsFromToolOutput(output: string): string[] {
  const paths = new Set<string>()
  const pattern =
    /(?:^|[\s"'(])((?:\.\/)?packages\/[A-Za-z0-9_.-]+\/src\/[^\s"'`),:]+\.(?:[cm]?[jt]sx?|mjs|cjs|vue|svelte|css|scss))(?:[:\s"'`),]|$)/g
  let match: RegExpExecArray | null
  while ((match = pattern.exec(output)) !== null) {
    const path = normalizeEvidencePath((match[1] ?? '').replace(/^\.\//, ''))
    if (isSourceCodeEvidencePath(path)) {
      paths.add(path)
    }
  }
  return Array.from(paths)
}

function selectRepresentativeUnreadSourcePaths(
  candidates: string[],
  readFiles: Set<string>,
  limit: number,
): string[] {
  const selected: string[] = []
  const selectedScopes = new Set<string>()
  const seen = new Set<string>()
  const normalizedReadFiles = Array.from(readFiles).map(normalizeEvidencePath)
  const uniqueCandidates = candidates
    .map(normalizeEvidencePath)
    .filter((path) =>
      isSourceCodeEvidencePath(path)
      && !normalizedReadFiles.some((readPath) => evidencePathsReferToSameFile(readPath, path))
    )
    .filter((path) => {
      if (seen.has(path)) {
        return false
      }
      seen.add(path)
      return true
    })

  for (const path of uniqueCandidates) {
    const scope = packageScopeForPath(path)
    if (scope && selectedScopes.has(scope)) {
      continue
    }
    selected.push(path)
    if (scope) {
      selectedScopes.add(scope)
    }
    if (selected.length >= limit) {
      return selected
    }
  }

  for (const path of uniqueCandidates) {
    if (selected.includes(path)) {
      continue
    }
    selected.push(path)
    if (selected.length >= limit) {
      return selected
    }
  }
  return selected
}

function repositoryEvidenceRequirement(
  contract: AgentRunContract | undefined,
): {
  minSourceFiles: number
  minSourceScopes: number
  requiresArtifactEvidenceMap: boolean
  requiresSearch: boolean
  description: string
} | null {
  const requirements = contract?.evidenceRequirements?.filter((entry) =>
    (entry.kind === 'source' || entry.kind === 'repository')
    && evidenceRequirementUsesRepositoryBreadth(entry)
  ) ?? []
  if (requirements.length === 0) {
    return null
  }
  return {
    minSourceFiles: Math.max(...requirements.map(evidenceRequirementMinSourceFiles)),
    minSourceScopes: Math.max(...requirements.map(evidenceRequirementMinSourceScopes)),
    requiresArtifactEvidenceMap: requirements.some((entry) => entry.requiresArtifactEvidenceMap === true),
    requiresSearch: requirements.some(evidenceRequirementRequiresSearch),
    description: requirements.map((entry) => entry.description).join('; '),
  }
}

function artifactEvidenceMapRequirements(
  contract: AgentRunContract | undefined,
): Array<{
  minSourceScopes: number
  description: string
}> {
  return contract?.evidenceRequirements
    ?.filter((entry) =>
      (entry.kind === 'source' || entry.kind === 'repository')
      && entry.requiresArtifactEvidenceMap === true
    )
    .map((entry) => ({
      minSourceScopes: entry.minSourceScopes ?? 1,
      description: entry.description,
    })) ?? []
}

function requiresArtifactSelfReview(contract: AgentRunContract | undefined): boolean {
  return contract?.evidenceRequirements?.some((entry) => entry.requiresArtifactSelfReview === true) ?? false
}

function validationEvidenceRequirements(contract: AgentRunContract | undefined): string[] {
  return contract?.evidenceRequirements
    ?.filter((entry) => entry.kind === 'validation')
    .map((entry) => entry.description)
    .filter(Boolean) ?? []
}

/**
 * A run that wrote nothing, and whose contract asks for no durable artifact,
 * has nothing a validation tool could check: every validation tool verifies
 * something the run produced or acts on. Demanding a separate validation run
 * there is unsatisfiable by construction — a live "what is my primary
 * language, look it up in memory" run searched memory successfully and was
 * still blocked to INCOMPLETE. For those runs the successful lookup the answer
 * is grounded in settles the requirement. Any run that wrote something, or was
 * contracted to, keeps the strict rule.
 *
 * The graph's own contract-gap check carries the same rule; this is the second
 * gate, on the path that keeps no evidence ledger, so the signals come from the
 * messages as well.
 */
function retrievalSettlesValidation(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): boolean {
  if ((contract?.requiredArtifacts?.length ?? 0) > 0) return false
  const stats = ledgerEvidenceStats(evidenceLedger)
  if (stats.successfulWritePaths.length > 0) return false
  if (collectFileMutationEvidence(messages).length > 0) return false
  return stats.searchCount > 0
    || stats.sourceObservationTools.length > 0
    || stats.sourceReadPaths.length > 0
    || collectSuccessfulLookupEvidence(messages).length > 0
}

function evaluateValidationEvidenceFloor(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const requirements = validationEvidenceRequirements(contract)
  if (requirements.length === 0) {
    return null
  }

  const validationEvidenceCount =
    ledgerEvidenceStats(evidenceLedger).validationCount
    + collectSuccessfulValidationEvidence(messages).length
  if (validationEvidenceCount > 0) {
    return null
  }

  if (retrievalSettlesValidation(messages, contract, evidenceLedger)) {
    return null
  }

  return {
    reason: `The run contract requires validation/check evidence (${requirements.join('; ')}), but no successful validation/check run is recorded.`,
    instruction: 'Run the appropriate validation, health, test, build, or check command/tool for the task before claiming completion. If validation is impossible because a tool, permission, or environment is unavailable, answer INCOMPLETE with the concrete blocker.',
  }
}

function evaluateEvidenceFloor(
  messages: Message[],
  contract?: AgentRunContract,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const ledgerStats = ledgerEvidenceStats(evidenceLedger)
  const toolCallsById = new Map<string, ToolCall>()
  const readFiles = new Set<string>(ledgerStats.sourceReadPaths)
  const successfulWritePaths = collectFileMutationEvidence(messages)
    .filter((write) => write.status === 'success' && typeof write.path === 'string')
    .map((write) => write.path!)
    .concat(ledgerStats.successfulWritePaths)
  const successfulWriteCount = successfulWritePaths.length
  let searchCount = ledgerStats.searchCount
  let broadInventory = false
  let summarizedBroadInventory = false
  let broadInventoryCwd: string | undefined
  const observedPackageScopes = new Set<string>()
  const observedSourcePaths: string[] = []

  for (const message of messagesSinceLastUser(messages)) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
      const args = toolCall.arguments
      if (toolCall.name === 'fs.read') {
        continue
      } else if (toolCall.name === 'fs.glob') {
        const patterns = [
          typeof args.pattern === 'string' ? args.pattern : undefined,
          ...(Array.isArray(args.patterns)
            ? args.patterns.filter((entry): entry is string => typeof entry === 'string')
            : []),
        ].filter((entry): entry is string => typeof entry === 'string')
        if (patterns.some(isBroadGlobPattern) && typeof args.cwd === 'string') {
          broadInventory = true
          broadInventoryCwd = args.cwd
        }
      }
    }

    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    const text = extractMessageText(message)
    if (toolCall?.name === 'fs.read' && typeof toolCall.arguments.path === 'string') {
      if (isSuccessfulReadToolResult(text)) {
        readFiles.add(toolCall.arguments.path)
      }
      continue
    }
    if (toolCall?.name === 'fs.glob') {
      addPackageScopesFromToolOutput(text, observedPackageScopes)
      observedSourcePaths.push(...collectSourcePathsFromToolOutput(text))
      const patterns = [
        typeof toolCall.arguments.pattern === 'string' ? toolCall.arguments.pattern : undefined,
        ...(Array.isArray(toolCall.arguments.patterns)
          ? toolCall.arguments.patterns.filter((entry): entry is string => typeof entry === 'string')
          : []),
      ].filter((entry): entry is string => typeof entry === 'string')
      if (patterns.some(isBroadGlobPattern) && isSummarizedBroadGlobResult(text)) {
        broadInventory = true
        summarizedBroadInventory = true
        if (typeof toolCall.arguments.cwd === 'string') {
          broadInventoryCwd = toolCall.arguments.cwd
        }
      }
      continue
    }
    if (toolCall?.name === 'fs.search') {
      const glob = typeof toolCall.arguments.glob === 'string' ? toolCall.arguments.glob : ''
      if (!isFailedToolResultText(text)) {
        if (!glob || isSourceCodeEvidencePath(glob.replace(/\*+/g, 'index.ts'))) {
          searchCount += 1
        }
        addPackageScopesFromToolOutput(text, observedPackageScopes)
        observedSourcePaths.push(...collectSourcePathsFromToolOutput(text))
      }
    }
  }

  const genericSourceRequirements = contract?.evidenceRequirements?.filter((requirement) =>
    requirement.kind === 'source'
    && !evidenceRequirementUsesRepositoryBreadth(requirement)
  ) ?? []
  const messageObservationTools = collectSuccessfulLookupEvidence(messages)
  const observedSourceTools = ledgerStats.sourceObservationTools.length > 0
    ? ledgerStats.sourceObservationTools
    : messageObservationTools
  for (const requirement of genericSourceRequirements) {
    const matchingObservations = observedSourceTools.filter((tool) =>
      sourceToolMatchesEvidenceRequirement(requirement, tool)
    )
    const requiredObservations = evidenceRequirementMinSourceObservations(requirement)
    const matchingSearchCount = evidenceLedger
      ? evidenceLedger.sourceSearches.filter((entry) =>
          entry.status === 'success'
          && sourceToolMatchesEvidenceRequirement(requirement, entry.tool)
        ).length
      : searchCount
    if (
      matchingObservations.length < requiredObservations
      || (evidenceRequirementRequiresSearch(requirement) && matchingSearchCount === 0)
    ) {
      const sourceTools = requirement.sourceToolNames?.length
        ? ` using ${requirement.sourceToolNames.join(' or ')}`
        : ''
      return {
        reason: `The run contract requires observed source evidence (${requirement.description}), but only ${matchingObservations.length}/${requiredObservations} successful source observation(s)${sourceTools} are recorded.`,
        instruction: `Gather the missing successful source observation${sourceTools}, then continue to the requested artifact or final answer. If the source is unavailable, answer INCOMPLETE with the concrete blocker.`,
      }
    }
  }

  const requiredSourceEvidence = repositoryEvidenceRequirement(contract)
  const shouldEnforceSourceEvidence = Boolean(requiredSourceEvidence)
    || (broadInventory && successfulWriteCount > 0)
  if (!shouldEnforceSourceEvidence) {
    return null
  }

  const sourceReadCount = Array.from(readFiles)
    .filter((readPath) =>
      isSourceCodeEvidencePath(readPath)
      && !successfulWritePaths.some((writePath) => evidencePathsReferToSameFile(readPath, writePath))
    )
    .length
  const sourceReadPaths = Array.from(readFiles).filter((readPath) =>
    isSourceCodeEvidencePath(readPath)
    && !successfulWritePaths.some((writePath) => evidencePathsReferToSameFile(readPath, writePath))
  )
  const sourceScopes = sourceEvidenceScopes(sourceReadPaths)
  const minPackageScopes = observedPackageScopes.size >= 4
    ? Math.min(6, observedPackageScopes.size)
    : 0
  const minSourceFiles = requiredSourceEvidence?.minSourceFiles ?? 4
  const minSourceScopes = Math.max(requiredSourceEvidence?.minSourceScopes ?? 0, minPackageScopes)
  const requiresSearch = requiredSourceEvidence?.requiresSearch ?? true
  if (
    sourceReadCount >= minSourceFiles
    && (!requiresSearch || searchCount > 0)
    && sourceScopes.length >= minSourceScopes
  ) {
    return null
  }

  return {
    reason: [
      requiredSourceEvidence
        ? `The run contract requires source/repository evidence (${requiredSourceEvidence.description}),`
        : `The run wrote an artifact after ${summarizedBroadInventory ? 'a summarized ' : ''}broad workspace inventory,`,
      `but only ${sourceReadCount}/${minSourceFiles} source-code file(s) were read and ${searchCount} search step(s) were gathered;`,
      minSourceScopes > 0
        ? `source scope coverage is ${sourceScopes.length}/${minSourceScopes} across distinct observed source areas;`
        : '',
      'this evidence floor cannot support a complete broad repository artifact.',
    ].filter(Boolean).join(' '),
    instruction: 'Continue with targeted repository inventory, representative cross-package source reads, and search evidence before claiming the artifact is complete. If the requested scope cannot be completed in this run, update the artifact as partial and answer INCOMPLETE with the remaining scope.',
    toolCalls: buildTargetedInventoryToolCalls(broadInventoryCwd, observedSourcePaths, readFiles),
  }
}

function contractArtifactPaths(contract: AgentRunContract | undefined): string[] {
  return contract?.requiredArtifacts?.map((artifact) => artifact.path).filter(Boolean) ?? []
}

function requiredFileArtifactPaths(contract: AgentRunContract | undefined): string[] {
  return contract?.requiredArtifacts
    ?.filter((artifact) => artifact.kind !== 'directory')
    .map((artifact) => artifact.path)
    .filter(Boolean) ?? []
}

function requestedPathsForReview(
  messages: Message[],
  contract: AgentRunContract | undefined,
): string[] {
  const artifactPaths = contractArtifactPaths(contract)
  return artifactPaths.length > 0
    ? artifactPaths
    : extractRequestedPaths(latestUserMessageText(messages))
}

function evaluateFailedRequestedArtifactMutation(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const requestedPaths = requestedPathsForReview(messages, contract)
  if (requestedPaths.length === 0) {
    return null
  }

  const mutations = collectFileMutationEvidence(messages)
  const missingSuccessfulWrites = requestedPaths.filter((requestedPath) => {
    const relevantMutations = mutations.filter((mutation) =>
      mutation.path && evidencePathsReferToSameFile(mutation.path, requestedPath)
    )
    if (relevantMutations.length === 0) {
      return false
    }
    return !relevantMutations.some((mutation) => mutation.status === 'success')
      && !ledgerHasSuccessfulWriteForPath(evidenceLedger, requestedPath)
  })

  if (missingSuccessfulWrites.length === 0) {
    return null
  }

  const failedEvidence = mutations
    .filter((mutation) =>
      mutation.status !== 'success'
      && mutation.path
      && missingSuccessfulWrites.some((requestedPath) =>
        evidencePathsReferToSameFile(mutation.path!, requestedPath)
      )
    )
    .map(formatWriteEvidence)

  return {
    reason: [
      `The run attempted to mutate requested artifact path(s) ${missingSuccessfulWrites.join(', ')},`,
      'but the recent tool trace has no successful file mutation result for those path(s).',
      'Denied, failed, or still-pending mutation attempts cannot satisfy a file deliverable.',
      failedEvidence.length > 0 ? `Observed failed/pending mutation(s): ${failedEvidence.join(' | ')}` : '',
    ].filter(Boolean).join(' '),
    instruction: 'Do not claim the requested artifact was created or updated. If approval or policy blocked the write, answer INCOMPLETE with that blocker; otherwise retry with a valid write/edit after resolving the failed mutation.',
  }
}

function evaluateMissingRequiredArtifactWrite(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const requiredPaths = requiredFileArtifactPaths(contract)
  if (requiredPaths.length === 0) {
    return null
  }

  const successfulWritePaths = collectFileMutationEvidence(messages)
    .filter((mutation) => mutation.status === 'success' && mutation.path)
    .map((mutation) => mutation.path!)

  const missing = requiredPaths.filter((requiredPath) =>
    !successfulWritePaths.some((writtenPath) =>
      evidencePathsReferToSameFile(writtenPath, requiredPath)
    )
    && !ledgerHasSuccessfulWriteForPath(evidenceLedger, requiredPath)
  )

  if (missing.length === 0) {
    return null
  }

  return {
    reason: `Required artifact path(s) have no successful file mutation evidence: ${missing.join(', ')}.`,
    instruction: 'Create or update the required artifact path(s) with a successful file-writing/editing tool call, then read the artifact back before finalizing. If policy, permissions, or missing context prevents the write, answer INCOMPLETE with the concrete blocker.',
  }
}

function evaluateCompactedRequiredArtifactReadBack(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const requiredPaths = requiredFileArtifactPaths(contract)
  if (requiredPaths.length === 0) {
    return null
  }

  const recentSuccessfulWritePaths = collectFileMutationEvidence(messages)
    .filter((mutation) => mutation.status === 'success' && mutation.path)
    .map((mutation) => mutation.path!)
  const missing = requiredPaths.filter((requiredPath) =>
    !recentSuccessfulWritePaths.some((writtenPath) =>
      evidencePathsReferToSameFile(writtenPath, requiredPath)
    )
    && ledgerHasSuccessfulWriteForPath(evidenceLedger, requiredPath)
    && !ledgerHasArtifactReadBackForPath(evidenceLedger, requiredPath)
  )

  if (missing.length === 0) {
    return null
  }

  return {
    reason: `Required artifact path(s) were written in compacted evidence but have no artifact read-back evidence: ${missing.join(', ')}.`,
    instruction: 'Read the required artifact path(s) back after the latest successful write and verify the contents against the run contract before finalizing.',
    toolCalls: missing.slice(0, 4).map((path) => ({
      name: 'fs.read',
      arguments: { path },
    })),
  }
}

function collectSuccessfulReadEvidence(messages: Message[]): Array<{ path: string; resultMessageIndex: number }> {
  const recentMessages = messagesSinceLastUser(messages)
  const toolCallsById = new Map<string, ToolCall>()
  for (const message of recentMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
    }
  }

  const reads: Array<{ path: string; resultMessageIndex: number }> = []
  for (const [index, message] of recentMessages.entries()) {
    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    const path = typeof toolCall?.arguments.path === 'string' ? toolCall.arguments.path : undefined
    if (toolCall?.name === 'fs.read' && path && isSuccessfulReadToolResult(extractMessageText(message))) {
      reads.push({ path, resultMessageIndex: index })
    }
  }
  return reads
}

function latestArtifactReadAfterWrite(
  messages: Message[],
  artifactPath: string,
  evidenceLedger?: AgentEvidenceLedger,
): {
  written: boolean
  readAfterWrite: boolean
  latestReadContent?: string
} {
  const recentMessages = messagesSinceLastUser(messages)
  const latestWriteIndex = collectFileMutationEvidence(messages)
    .filter((write) =>
      write.status === 'success'
      && write.path
      && evidencePathsReferToSameFile(write.path, artifactPath)
      && typeof write.resultMessageIndex === 'number'
    )
    .reduce((latest, write) => Math.max(latest, write.resultMessageIndex!), -1)

  if (latestWriteIndex < 0) {
    const written = ledgerHasSuccessfulWriteForPath(evidenceLedger, artifactPath)
    return {
      written,
      readAfterWrite: written && ledgerHasArtifactReadBackForPath(evidenceLedger, artifactPath),
    }
  }

  const toolCallsById = new Map<string, ToolCall>()
  let latestReadContent: string | undefined
  for (const [index, message] of recentMessages.entries()) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
    }
    if (message.role !== 'tool' || !message.toolCallId || index <= latestWriteIndex) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    const path = typeof toolCall?.arguments.path === 'string' ? toolCall.arguments.path : undefined
    const text = extractMessageText(message)
    if (
      toolCall?.name === 'fs.read'
      && path
      && evidencePathsReferToSameFile(path, artifactPath)
      && isSuccessfulReadToolResult(text)
    ) {
      latestReadContent = text
    }
  }

  const ledgerReadBack = ledgerHasArtifactReadBackForPath(evidenceLedger, artifactPath)
  return {
    written: true,
    readAfterWrite: latestReadContent != null || ledgerReadBack,
    ...(latestReadContent != null ? { latestReadContent } : {}),
  }
}

function requiredArtifactSections(
  contract: AgentRunContract | undefined,
): Array<{ section: ArtifactSection; artifactPath: string }> {
  const sections = contract?.artifactSections?.filter((section) => section.required !== false) ?? []
  if (sections.length === 0) {
    return []
  }
  const defaultArtifactPath = contract?.requiredArtifacts?.[0]?.path
  return sections
    .map((section) => {
      const artifactPath = section.artifactPath ?? defaultArtifactPath
      return artifactPath ? { section, artifactPath } : null
    })
    .filter((entry): entry is { section: ArtifactSection; artifactPath: string } => Boolean(entry))
}

function collectArtifactSectionCoverageEvidence(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): string[] {
  return requiredArtifactSections(contract).map(({ section, artifactPath }) => {
    const evidence = latestArtifactReadAfterWrite(messages, artifactPath, evidenceLedger)
    if (!evidence.written) {
      return `${section.title} (${artifactPath}) => artifact not written`
    }
    if (!evidence.readAfterWrite || !evidence.latestReadContent) {
      return `${section.title} (${artifactPath}) => not read after latest write`
    }
    return `${section.title} (${artifactPath}) => ${
      artifactHasSection(evidence.latestReadContent, section.title) ? 'covered' : 'missing'
    }`
  })
}

function evaluateArtifactSectionCoverage(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const requiredSections = requiredArtifactSections(contract)
  if (requiredSections.length === 0) {
    return null
  }

  const missing = requiredSections
    .map(({ section, artifactPath }) => {
      const evidence = latestArtifactReadAfterWrite(messages, artifactPath, evidenceLedger)
      if (!evidence.written) {
        return { section, artifactPath, reason: 'artifact not written' }
      }
      if (!evidence.readAfterWrite || !evidence.latestReadContent) {
        return { section, artifactPath, reason: 'not read after the latest write' }
      }
      if (!artifactHasSection(evidence.latestReadContent, section.title)) {
        return { section, artifactPath, reason: 'section heading not found in artifact read-back' }
      }
      return null
    })
    .filter((entry): entry is { section: ArtifactSection; artifactPath: string; reason: string } =>
      Boolean(entry)
    )

  if (missing.length === 0) {
    return null
  }

  const missingSummary = missing
    .map(({ section, artifactPath, reason }) => `${section.title} (${artifactPath}: ${reason})`)
    .join(', ')
  const readPaths = Array.from(new Set(
    missing
      .filter(({ reason }) => reason === 'not read after the latest write')
      .map(({ artifactPath }) => artifactPath),
  )).slice(0, 4)
  return {
    reason: `Required artifact section coverage is incomplete: ${missingSummary}.`,
    instruction: 'Update the required artifact so every required section named in the run contract is present as a clear markdown heading, then read the artifact back and verify section coverage before finalizing.',
    toolCalls: readPaths.map((path) => ({
      name: 'fs.read',
      arguments: { path },
    })),
  }
}

function collectObservedSourceReadPaths(
  messages: Message[],
  evidenceLedger?: AgentEvidenceLedger,
): string[] {
  const paths = new Set<string>(ledgerEvidenceStats(evidenceLedger).sourceReadPaths)
  const recentMessages = messagesSinceLastUser(messages)
  const toolCallsById = new Map<string, ToolCall>()
  for (const message of recentMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
    }
    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    const path = typeof toolCall?.arguments.path === 'string' ? toolCall.arguments.path : undefined
    if (toolCall?.name === 'fs.read' && path && isSuccessfulReadToolResult(extractMessageText(message))) {
      paths.add(path)
    }
  }
  return uniqueNormalizedPaths(Array.from(paths).filter(isSourceCodeEvidencePath))
}

function evidenceContentMentions(content: string, target: string): boolean {
  const normalizedContent = normalizeEvidencePath(content).toLowerCase()
  const normalizedTarget = normalizeEvidencePath(target).toLowerCase()
  return normalizedTarget.length > 0 && normalizedContent.includes(normalizedTarget)
}

function traceableSourceScopeCount(content: string, sourcePaths: string[]): number {
  const scopes = sourceEvidenceScopes(sourcePaths)
  const mentionedScopes = new Set<string>()
  for (const scope of scopes) {
    if (evidenceContentMentions(content, scope)) {
      mentionedScopes.add(scope)
    }
  }
  for (const path of sourcePaths) {
    if (!evidenceContentMentions(content, path)) {
      continue
    }
    const scope = sourceEvidenceScopes([path])[0]
    if (scope) {
      mentionedScopes.add(scope)
    }
  }
  return mentionedScopes.size
}

function evaluateArtifactEvidenceMapTraceability(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const requirements = artifactEvidenceMapRequirements(contract)
  if (requirements.length === 0) {
    return null
  }

  const artifactPaths = requiredFileArtifactPaths(contract)
  if (artifactPaths.length === 0) {
    return null
  }

  const sourcePaths = collectObservedSourceReadPaths(messages, evidenceLedger)
  const sourceScopes = sourceEvidenceScopes(sourcePaths)
  if (sourceScopes.length === 0) {
    return null
  }

  const requiredScopeCount = Math.min(
    sourceScopes.length,
    Math.max(1, ...requirements.map((requirement) => requirement.minSourceScopes)),
  )
  const missing: Array<{ path: string; reason: string }> = []
  for (const artifactPath of artifactPaths) {
    const evidence = latestArtifactReadAfterWrite(messages, artifactPath, evidenceLedger)
    if (!evidence.written) {
      missing.push({ path: artifactPath, reason: 'artifact not written' })
      continue
    }
    if (!evidence.readAfterWrite || !evidence.latestReadContent) {
      missing.push({ path: artifactPath, reason: 'artifact content not read after latest write' })
      continue
    }
    const traceableScopes = traceableSourceScopeCount(evidence.latestReadContent, sourcePaths)
    if (traceableScopes < requiredScopeCount) {
      missing.push({
        path: artifactPath,
        reason: `artifact evidence map traces ${traceableScopes}/${requiredScopeCount} observed source scope(s)`,
      })
    }
  }

  if (missing.length === 0) {
    return null
  }

  const readPaths = missing
    .filter((entry) => entry.reason === 'artifact content not read after latest write')
    .map((entry) => entry.path)
    .slice(0, 4)
  return {
    reason: `Required artifact evidence map traceability is incomplete: ${
      missing.map((entry) => `${entry.path}: ${entry.reason}`).join(', ')
    }.`,
    instruction: 'Update the required artifact with an evidence or coverage map that cites observed source paths/scopes for the claims it makes, then read the artifact back before finalizing. If the requested breadth cannot be completed, mark the artifact partial and answer INCOMPLETE with the remaining scope.',
    toolCalls: readPaths.map((path) => ({
      name: 'fs.read',
      arguments: { path },
    })),
  }
}

function evaluateArtifactSelfReviewCoverage(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  if (!requiresArtifactSelfReview(contract)) {
    return null
  }

  const artifactPaths = requiredFileArtifactPaths(contract)
  const criteria = contract?.acceptanceCriteria ?? []
  if (artifactPaths.length === 0 || criteria.length === 0) {
    return null
  }

  const missing: Array<{ path: string; missingCriteria: string[]; reason?: string }> = []
  for (const artifactPath of artifactPaths) {
    const evidence = latestArtifactReadAfterWrite(messages, artifactPath, evidenceLedger)
    if (!evidence.written) {
      missing.push({ path: artifactPath, missingCriteria: criteria.map((criterion) => criterion.id), reason: 'artifact not written' })
      continue
    }
    if (!evidence.readAfterWrite || !evidence.latestReadContent) {
      missing.push({ path: artifactPath, missingCriteria: criteria.map((criterion) => criterion.id), reason: 'artifact content not read after latest write' })
      continue
    }
    const content = evidence.latestReadContent
    const missingCriteria = criteria
      .map((criterion) => criterion.id)
      .filter((id) => !new RegExp(`\\b${id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(content))
    if (missingCriteria.length > 0) {
      missing.push({ path: artifactPath, missingCriteria })
    }
  }

  if (missing.length === 0) {
    return null
  }

  const readPaths = missing
    .filter((entry) => entry.reason === 'artifact content not read after latest write')
    .map((entry) => entry.path)
    .slice(0, 4)
  return {
    reason: `Required artifact self-review is incomplete: ${
      missing.map((entry) => `${entry.path}: ${
        entry.reason ?? `missing acceptance criterion id(s) ${entry.missingCriteria.join(', ')}`
      }`).join(', ')
    }.`,
    instruction: 'Update the required artifact with a self-review or acceptance checklist that explicitly references each run contract acceptance criterion id and whether it is satisfied, partial, or blocked. Then read the artifact back before finalizing.',
    toolCalls: readPaths.map((path) => ({
      name: 'fs.read',
      arguments: { path },
    })),
  }
}

function collectArtifactStructureEvidence(
  messages: Message[],
  contract: AgentRunContract | undefined,
): string[] {
  const requestedPaths = requestedPathsForReview(messages, contract)
  if (requestedPaths.length === 0) {
    return []
  }
  const recentMessages = messagesSinceLastUser(messages)
  const toolCallsById = new Map<string, ToolCall>()
  for (const message of recentMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
      if (!isFileMutationTool(toolCall.name)) {
        continue
      }
      const paths = fileMutationPaths(toolCall.name, toolCall.arguments)
      const content = typeof toolCall.arguments.content === 'string'
        ? toolCall.arguments.content
        : typeof toolCall.arguments.contents === 'string'
          ? toolCall.arguments.contents
          : undefined
      if (!content) {
        continue
      }
      for (const path of paths) {
        if (!requestedPaths.some((requestedPath) => evidencePathsReferToSameFile(path, requestedPath))) {
          continue
        }
        const structure = summarizeMarkdownStructure(content)
        if (structure) {
          toolCallsById.set(`${toolCall.id}:artifact-structure` as string, {
            id: `${toolCall.id}:artifact-structure`,
            name: 'artifact.structure',
            arguments: { path, structure },
          })
        }
      }
    }
  }

  const evidence = new Map<string, string>()
  for (const [index, message] of recentMessages.entries()) {
    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    const path = typeof toolCall?.arguments.path === 'string' ? toolCall.arguments.path : undefined
    if (
      toolCall?.name === 'fs.read'
      && path
      && requestedPaths.some((requestedPath) => evidencePathsReferToSameFile(path, requestedPath))
      && isSuccessfulReadToolResult(extractMessageText(message))
    ) {
      const structure = summarizeMarkdownStructure(extractMessageText(message))
      if (structure) {
        evidence.set(`read:${path}`, `read ${path} @${index}: ${structure}`)
      }
    }
  }
  for (const toolCall of toolCallsById.values()) {
    if (toolCall.name !== 'artifact.structure') {
      continue
    }
    const path = typeof toolCall.arguments.path === 'string' ? toolCall.arguments.path : undefined
    const structure = typeof toolCall.arguments.structure === 'string'
      ? toolCall.arguments.structure
      : undefined
    if (path && structure) {
      evidence.set(`write:${path}`, `write ${path}: ${structure}`)
    }
  }
  return Array.from(evidence.values()).slice(0, 8)
}

function evaluateUnverifiedRequestedArtifactWrite(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): EvidenceFloorResult | null {
  const requiresReadBack = Boolean(
    (contract?.requiredArtifacts?.length ?? 0) > 0
    || (contract?.artifactSections?.length ?? 0) > 0
    || (contract?.evidenceRequirements?.length ?? 0) > 0
  )
  if (!requiresReadBack) {
    return null
  }

  const requestedPaths = requestedPathsForReview(messages, contract)
  if (requestedPaths.length === 0) {
    return null
  }

  const successfulWrites = collectFileMutationEvidence(messages)
    .filter((mutation) =>
      mutation.status === 'success'
      && mutation.path
      && typeof mutation.resultMessageIndex === 'number'
      && requestedPaths.some((requestedPath) =>
        evidencePathsReferToSameFile(mutation.path!, requestedPath)
      )
    )
  if (successfulWrites.length === 0) {
    return null
  }

  const reads = collectSuccessfulReadEvidence(messages)
  const unverifiedWrites = successfulWrites.filter((write) =>
    !reads.some((read) =>
      evidencePathsReferToSameFile(read.path, write.path!)
      && read.resultMessageIndex > write.resultMessageIndex!
    )
    && !ledgerHasArtifactReadBackForPath(evidenceLedger, write.path!)
  )
  if (unverifiedWrites.length === 0) {
    return null
  }

  const uniquePaths = Array.from(new Set(
    unverifiedWrites
      .map((write) => write.path)
      .filter((path): path is string => typeof path === 'string'),
  )).slice(0, 4)

  return {
    reason: `Requested artifact path(s) were written but not read back after the write: ${uniquePaths.join(', ')}. A write result proves the file changed, but not that the artifact content satisfies the requested contract.`,
    instruction: 'Read the written artifact(s), verify the requested sections and diagrams against the run contract, then either continue editing or provide a grounded final answer.',
    toolCalls: uniquePaths.map((path) => ({
      name: 'fs.read',
      arguments: { path },
    })),
  }
}

function evaluateUnsupportedPathClaims(
  messages: Message[],
  assistantAnswer: string | undefined,
  options: {
    evidenceScope?: 'recent' | 'all'
    runContract?: AgentRunContract
  } = {},
): EvidenceFloorResult | null {
  const executionIntent = options.runContract?.executionIntent
  const readOnlyRuntimeCapabilities = new Set([
    'process',
    'service',
    'terminal',
    'network',
    'filesystem-read',
  ])
  const hasDirectRuntimeCapability = executionIntent?.capabilities.some((capability) => (
    capability === 'process'
    || capability === 'service'
    || capability === 'terminal'
  )) === true
  const externalEvidenceCapabilities = new Set(['browser', 'network'])
  const hasOnlyExternalCapabilities = (executionIntent?.capabilities.length ?? 0) > 0
    && executionIntent!.capabilities.every((capability) => externalEvidenceCapabilities.has(capability))
  const hasRepositoryEvidenceContract =
    (options.runContract?.requiredArtifacts?.length ?? 0) > 0
    || options.runContract?.evidenceRequirements?.some((requirement) => (
      requirement.kind === 'repository'
      || (requirement.kind === 'source' && (
        !hasOnlyExternalCapabilities
        || (requirement.minSourceFiles ?? 0) > 0
        || (requirement.minSourceScopes ?? 0) > 0
        || requirement.sourceToolNames?.some((name) => name.startsWith('fs.') || name.startsWith('git.') || name.startsWith('code.'))
      ))
    )) === true
  // A source observation can be a web/App document. Do not turn examples in
  // that source into workspace existence claims. Explicit repository breadth,
  // source readers, and mixed filesystem contracts retain the local floor.
  const isExplicitExternalEvidenceInspection =
    executionIntent?.kind === 'inspection'
    && executionIntent.workspaceMutation === 'forbidden'
    && executionIntent.capabilities.length > 0
    && executionIntent.capabilities.every((capability) => (
      externalEvidenceCapabilities.has(capability)
    ))
    && !hasRepositoryEvidenceContract
  const isExplicitOperationalInspection = (
    executionIntent?.kind === 'inspection'
    || executionIntent?.kind === 'operational-action'
  )
    && executionIntent.workspaceMutation === 'forbidden'
    && executionIntent.capabilities.every((capability) => readOnlyRuntimeCapabilities.has(capability))
    && hasDirectRuntimeCapability
    && !hasRepositoryEvidenceContract
  if (isExplicitOperationalInspection || isExplicitExternalEvidenceInspection) {
    return null
  }

  const unsupported = findUnsupportedRepositoryPathClaims({
    text: assistantAnswer ?? '',
    messages,
    evidenceScope: options.evidenceScope,
  })
  if (unsupported.length === 0) {
    return null
  }

  const sample = unsupported.slice(0, 6)
  return {
    reason: [
      'The candidate answer cites repository path-like items that are not grounded in the recent read/search/glob/write or structured Git evidence:',
      sample.join(', '),
      'Final answers may not introduce concrete file or directory claims that the run did not observe.',
    ].join(' '),
    instruction: 'Verify the cited paths with read/search/glob or structured Git evidence, remove unsupported path claims, or answer INCOMPLETE with the missing verification scope.',
    toolCalls: sample.slice(0, 4).map((path) => ({
      name: 'fs.glob',
      arguments: { pattern: path },
    })),
  }
}

function formatToolEvidenceForReview(
  messages: Message[],
  contract: AgentRunContract | undefined,
  evidenceLedger?: AgentEvidenceLedger,
): string {
  const ledgerStats = ledgerEvidenceStats(evidenceLedger)
  const toolCallsById = new Map<string, ToolCall>()
  const readFiles = new Set<string>(ledgerStats.sourceReadPaths)
  const failedReadFiles: string[] = []
  const globQueries: string[] = []
  const searchQueries: string[] = []
  const structuredGitEvidence: string[] = []
  const observedPackageScopes = new Set<string>()
  const writeOperations = collectFileMutationEvidence(messages)
  const artifactStructure = collectArtifactStructureEvidence(messages, contract)
  const artifactSectionCoverage = collectArtifactSectionCoverageEvidence(messages, contract, evidenceLedger)
  const validationEvidence = collectSuccessfulValidationEvidence(messages)
  const successfulToolExecutions = collectSuccessfulToolExecutions(messages, evidenceLedger)
  const otherTools: string[] = []

  for (const message of messagesSinceLastUser(messages)) {
    for (const toolCall of message.toolCalls ?? []) {
      toolCallsById.set(toolCall.id, toolCall)
      const args = toolCall.arguments
      if (toolCall.name === 'fs.read') {
        continue
      }
      if (toolCall.name === 'fs.glob') {
        const cwd = typeof args.cwd === 'string' ? args.cwd : '(default cwd)'
        const pattern = typeof args.pattern === 'string'
          ? args.pattern
          : Array.isArray(args.patterns)
            ? args.patterns.filter((entry): entry is string => typeof entry === 'string').join(', ')
            : '(unknown pattern)'
        globQueries.push(`${cwd} :: ${pattern}`)
        continue
      }
      if (toolCall.name === 'fs.search') {
        const cwd = typeof args.cwd === 'string' ? args.cwd : '(default cwd)'
        const query = typeof args.query === 'string' ? args.query : '(unknown query)'
        const glob = typeof args.glob === 'string' ? ` :: ${args.glob}` : ''
        searchQueries.push(`${cwd}${glob} :: ${query}`)
        continue
      }
      if (hasStructuredGitPathEvidence(toolCall)) {
        structuredGitEvidence.push(summarizeToolArguments(toolCall.name, args))
        continue
      }
      if (isFileMutationTool(toolCall.name)) {
        continue
      }
      otherTools.push(toolCall.name)
    }

    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }
    const toolCall = toolCallsById.get(message.toolCallId)
    const path = typeof toolCall?.arguments.path === 'string' ? toolCall.arguments.path : undefined
    const text = extractMessageText(message)
    if (toolCall?.name === 'fs.glob' || toolCall?.name === 'fs.search') {
      addPackageScopesFromToolOutput(text, observedPackageScopes)
    }
    if (toolCall?.name === 'fs.read' && path) {
      if (isSuccessfulReadToolResult(text)) {
        readFiles.add(path)
      } else {
        failedReadFiles.push(path)
      }
    }
  }

  const requestedPathCoverage = formatRequestedPathCoverage(
    requestedPathsForReview(messages, contract),
    Array.from(readFiles),
    globQueries,
    searchQueries,
  )
  const writtenPaths = writeOperations
    .filter((entry) => entry.status === 'success')
    .map((entry) => entry.path)
    .filter((entry): entry is string => typeof entry === 'string')
  const sourceReadFiles = Array.from(readFiles)
    .filter((readPath) => !writtenPaths.some((writePath) => evidencePathsReferToSameFile(readPath, writePath)))
  const sourcePackageScopes = Array.from(new Set(
    sourceReadFiles
      .filter(isSourceCodeEvidencePath)
      .map(packageScopeForPath)
      .filter((scope): scope is string => typeof scope === 'string'),
  ))
  const sourceScopes = sourceEvidenceScopes(sourceReadFiles)
  const successfulWrites = writeOperations.filter((entry) => entry.status === 'success')
  const failedWrites = writeOperations.filter((entry) => entry.status === 'error')
  const pendingWrites = writeOperations.filter((entry) => entry.status === 'unknown')
  const toolResultExcerpts = formatToolResultExcerpts(messages, toolCallsById)

  return [
    `Read file content via fs.read: ${readFiles.size > 0 ? Array.from(readFiles).join(', ') : '(none)'}`,
    `Read attempts without file content: ${failedReadFiles.length > 0 ? failedReadFiles.join(', ') : '(none)'}`,
    `Source-content reads excluding written artifacts: ${sourceReadFiles.length > 0 ? sourceReadFiles.join(', ') : '(none)'}`,
    `Source package scopes read: ${sourcePackageScopes.length > 0 ? sourcePackageScopes.join(', ') : '(none)'}`,
    `Distinct source scopes read: ${sourceScopes.length > 0 ? sourceScopes.join(', ') : '(none)'}`,
    `Observed package scopes from list/search: ${observedPackageScopes.size > 0 ? Array.from(observedPackageScopes).join(', ') : '(none)'}`,
    `Glob/list-only evidence: ${globQueries.length > 0 ? globQueries.join(' | ') : '(none)'}`,
    `Search evidence: ${searchQueries.length > 0 ? searchQueries.join(' | ') : '(none)'}`,
    `Structured Git path/change evidence: ${structuredGitEvidence.length > 0 ? structuredGitEvidence.join(' | ') : '(none)'}`,
    `Ledger evidence: source reads ${ledgerStats.sourceReadPaths.length}; source scopes ${ledgerStats.sourceScopes.length}; searches/inventory ${ledgerStats.searchCount}; artifact writes ${ledgerStats.successfulWritePaths.length}; artifact read-backs ${ledgerStats.artifactReadBackPaths.length}; validation/check runs ${ledgerStats.validationCount}`,
    `Required/requested path coverage: ${requestedPathCoverage}`,
    `Successful file writes/edits: ${successfulWrites.length > 0 ? successfulWrites.map(formatWriteEvidence).join(' | ') : '(none)'}`,
    `Failed file writes/edits: ${failedWrites.length > 0 ? failedWrites.map(formatWriteEvidence).join(' | ') : '(none)'}`,
    `Pending file writes/edits without observed tool result: ${pendingWrites.length > 0 ? pendingWrites.map(formatWriteEvidence).join(' | ') : '(none)'}`,
    `Requested artifact structure evidence: ${artifactStructure.length > 0 ? artifactStructure.join(' | ') : '(none)'}`,
    `Required artifact section coverage: ${artifactSectionCoverage.length > 0 ? artifactSectionCoverage.join(' | ') : '(none)'}`,
    `Successful validation/check evidence: ${validationEvidence.length > 0 ? validationEvidence.join(' | ') : '(none)'}`,
    `Other tools called: ${otherTools.length > 0 ? otherTools.join(', ') : '(none)'}`,
    `Current-turn successful tool executions (authoritative): ${formatSuccessfulToolExecutions(successfulToolExecutions)}`,
    `Current-turn tool result excerpts (authoritative):\n${toolResultExcerpts}`,
    'Interpretation rule: fs.glob/list output is existence or inventory evidence only; structured git.log/git.diff output establishes observed history, change statistics, and paths but not full source-content inspection. A successful write only proves the artifact exists; it does not prove the artifact is complete for the requested scope. Treat candidate claims such as "read", "verified", "directly cited", or "all file contents checked" as unsupported unless the file appears in the source-content read list, search evidence, or a write/edit result proves only that the deliverable was updated.',
  ].join('\n')
}

function formatToolResultExcerpts(
  messages: Message[],
  toolCallsById: ReadonlyMap<string, ToolCall>,
): string {
  const allResults = messagesSinceLastUser(messages).flatMap((message) => {
    if (message.role !== 'tool' || !message.toolCallId) return []
    const result = extractMessageText(message).trim()
    if (!result) return []
    const toolCall = toolCallsById.get(message.toolCallId)
    return [{
      id: message.toolCallId,
      tool: toolCall?.name ?? '(unknown tool)',
      args: toolCall ? summarizeToolArguments(toolCall.name, toolCall.arguments) : '',
      result,
    }]
  })
  if (allResults.length === 0) return '(none)'

  // Keep every result for normal and long-but-bounded operational runs. For an
  // unusually large trace, retain both ends of the observation sequence so
  // initial inventory and terminal verification remain represented without
  // making the judge request unbounded.
  const selected = allResults.length <= 80
    ? allResults
    : [...allResults.slice(0, 40), ...allResults.slice(-40)]
  // Spend the existing aggregate budget adaptively. A fixed 900-character cap
  // discarded middle scalar fields even when a short operational run had only
  // a handful of modest JSON results and the aggregate packet had ample room.
  // Keep such results complete; large traces/results remain bounded by the
  // 20k aggregate share and a 5k per-result ceiling.
  const perResultBudget = Math.min(5_000, Math.max(250, Math.floor(20_000 / selected.length)))
  const lines = selected.map((entry) => {
    const identityBudget = Math.min(220, Math.max(100, Math.floor(perResultBudget * 0.45)))
    const identity = compact(`${entry.id} ${entry.args || entry.tool}`, identityBudget)
    const resultBudget = Math.max(100, perResultBudget - identity.length - 7)
    return `- ${identity} => ${compactBoundaries(entry.result, resultBudget)}`
  })
  if (selected.length !== allResults.length) {
    lines.splice(40, 0, `- (${allResults.length - selected.length} middle tool results omitted by bounded review context)`)
  }
  return lines.join('\n')
}

function extractRequestedPaths(userText: string): string[] {
  const paths = new Set<string>()
  const patterns = [
    /(?:^|[\s`"'([])((?:\.{1,2}\/|\/|~\/)?[A-Za-z0-9_.@-]+(?:\/[A-Za-z0-9_.@*{}[\]-]+)+)(?=$|[\s`"',.)\]])/g,
    /(?:^|[\s`"'([])((?:\.{1,2}\/|~\/)?[A-Za-z0-9_.@-]+\.[A-Za-z0-9][A-Za-z0-9_-]{0,15})(?=$|[\s`"',.)\]])/g,
  ]
  let match: RegExpExecArray | null
  for (const pattern of patterns) {
    while ((match = pattern.exec(userText)) !== null) {
      const value = match[1]?.replace(/[.,;:]+$/g, '')
      if (value && !value.startsWith('http://') && !value.startsWith('https://')) {
        paths.add(value)
      }
    }
  }
  return Array.from(paths)
}

function pathCoversScope(path: string, scope: string): boolean {
  const normalizedPath = path.replace(/^\.\//, '').replace(/\/+$/g, '')
  const normalizedScope = scope.replace(/^\.\//, '').replace(/\/+$/g, '')
  return normalizedPath === normalizedScope || normalizedPath.startsWith(`${normalizedScope}/`)
}

function formatRequestedPathCoverage(
  requestedPaths: string[],
  readFiles: string[],
  globQueries: string[],
  searchQueries: string[],
): string {
  if (requestedPaths.length === 0) {
    return '(no required artifact or explicit path-like scope)'
  }

  return requestedPaths.map((requestedPath) => {
    const hasRead = readFiles.some((path) =>
      pathCoversScope(path, requestedPath) || evidencePathsReferToSameFile(path, requestedPath)
    )
    const hasGlob = globQueries.some((query) => query.includes(requestedPath))
    const hasSearch = searchQueries.some((query) => query.includes(requestedPath))
    return `${requestedPath} => ${[
      hasRead ? 'read' : undefined,
      hasGlob ? 'glob' : undefined,
      hasSearch ? 'search' : undefined,
    ].filter(Boolean).join('+') || 'no evidence'}`
  }).join(' | ')
}

function currentTurnUserMessageIndex(messages: Message[]): number {
  // Internal continuation prompts may use the user role. The runtime's
  // persisted turn marker, not their position or wording, defines the task
  // whose execution evidence the judge must review.
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.role === 'user' && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true) {
      return index
    }
  }
  // Older checkpoints and callers without a marker retain their old boundary.
  return messages.findLastIndex((message) => message.role === 'user')
}

function messagesSinceLastUser(messages: Message[]): Message[] {
  return messages.slice(currentTurnUserMessageIndex(messages) + 1)
}

function latestUserMessageText(messages: Message[]): string {
  const message = messages[currentTurnUserMessageIndex(messages)]
  return message ? compact(extractMessageText(message), 1400) : '[unknown]'
}

function formatRunContractForReview(contract: AgentRunContract | undefined): string {
  if (!contract) {
    return '(none)'
  }
  const lines = [
    `Source: ${contract.source}`,
    `Goal: ${contract.summary}`,
    'Acceptance criteria:',
    ...contract.acceptanceCriteria.map((criterion) => `- ${criterion.id}: ${criterion.text}`),
  ]
  if (contract.constraints.length > 0) {
    lines.push('Constraints:', ...contract.constraints.map((entry) => `- ${entry}`))
  }
  if (contract.outOfScope.length > 0) {
    lines.push('Out of scope:', ...contract.outOfScope.map((entry) => `- ${entry}`))
  }
  if (contract.requiredArtifacts?.length) {
    lines.push(
      'Required artifacts:',
      ...contract.requiredArtifacts.map((artifact) => {
        const description = artifact.description ? ` — ${artifact.description}` : ''
        return `- ${artifact.path} (${artifact.kind})${description}`
      }),
    )
  }
  if (contract.evidenceRequirements?.length) {
    lines.push(
      'Evidence requirements:',
      ...contract.evidenceRequirements.map((requirement) => {
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
        ].filter(Boolean)
        return `- ${details.join('; ')}`
      }),
    )
  }
  if (contract.artifactSections?.length) {
    lines.push(
      'Required artifact sections:',
      ...contract.artifactSections.map((section) => {
        const artifactPath = section.artifactPath ? `; artifact=${section.artifactPath}` : ''
        const required = section.required === false ? '; optional' : '; required'
        const description = section.description ? ` — ${section.description}` : ''
        return `- ${section.id}: ${section.title}${artifactPath}${required}${description}`
      }),
    )
  }
  return lines.join('\n')
}

export function shouldReviewOutcomeWithLLM(
  messages: Message[],
  options: { includeToollessFinal?: boolean } = {},
): boolean {
  const recentMessages = messagesSinceLastUser(messages)
  const hasToolEvidence = recentMessages.some((message) => message.role === 'tool')
  if (
    !hasToolEvidence
    && explicitlyForbidsToolUse(latestUserMessageText(messages))
  ) {
    return false
  }
  if (options.includeToollessFinal) {
    return true
  }
  return hasToolEvidence
}

function formatConversationForReview(messages: Message[]): string {
  return messagesSinceLastUser(messages)
    .slice(-10)
    .map((message) => {
      const text = compact(extractMessageText(message), 900)
      const toolCalls = formatToolCalls(message)
      const body = [text, toolCalls].filter(Boolean).join('\n')
      const toolResultId = message.role === 'tool' && message.toolCallId
        ? `(${message.toolCallId})`
        : ''
      return `${message.role}${toolResultId}: ${body || '[empty]'}`
    })
    .join('\n\n')
}

export function buildRunOutcomeReviewRequest(options: {
  model: string
  messages: Message[]
  assistantAnswer: string
  availableToolNames: string[]
  runContract?: AgentRunContract
  evidenceLedger?: AgentEvidenceLedger
  userInstructions?: readonly string[]
  maxTokens?: number
}): ChatRequest {
  const availableTools = options.availableToolNames.length > 0
    ? options.availableToolNames.join(', ')
    : '(none)'

  return {
    model: options.model,
    temperature: 0,
    maxTokens: options.maxTokens ?? OUTCOME_REVIEW_MAX_TOKENS,
    thinkingLevel: ThinkingLevel.Off,
    messages: [
      {
        role: 'system',
        content: [
          'You are an outcome judge for an autonomous agent run.',
          'Judge semantically whether the assistant answer fully satisfies the user request using the conversation and tool results.',
          'Do not use keyword matching. Decide from meaning, evidence, and whether the answer is actually complete.',
          'Return the JSON object directly. Do not include hidden reasoning, markdown, prose, or preambles.',
          'Return JSON only with this shape:',
          '{"status":"complete|needs_recovery","reason":"short reason","instruction":"what the agent should do next if recovery is needed","recoveryMode":"tool|synthesis|user-input","recoveryBasis":"missing-tool-execution|insufficient-tool-evidence|answer-synthesis|user-input","missingToolNames":["exact.registered.tool.name"],"toolCalls":[{"name":"fs.read|fs.glob|fs.search","arguments":{}}]}',
          'Use needs_recovery when the answer is only a progress update, leaves unresolved placeholders, ignores relevant tool results, stops after failed tools without a clear blocker, or still has obvious requested work pending.',
          'For needs_recovery, set recoveryMode=tool when new tool work is required, synthesis when the existing tool evidence is sufficient and only a corrected final answer is needed, or user-input when no tool can proceed without a user decision.',
          'For every needs_recovery response, set recoveryBasis to the primary cause. Use missing-tool-execution only when a necessary tool has no successful current-turn result, and list every such exact registered name in missingToolNames. Use insufficient-tool-evidence when a tool did run successfully but its returned content does not establish the needed fact. Use answer-synthesis when the evidence is sufficient but the candidate answer is wrong or incomplete. Use user-input only for a genuinely blocking user decision. Omit missingToolNames for every basis except missing-tool-execution.',
          'The authoritative successful-execution summary and result excerpts are trusted runtime facts. Never claim a listed successful tool was unattempted, unexecuted, or missing. A missing-tool-execution claim that contradicts that ledger will be discarded deterministically; request insufficient-tool-evidence instead when the result content, selector, scope, or returned fields are inadequate.',
          'The Candidate transport line states whether the candidate is complete or a bounded head/tail excerpt. Do not call the assistant answer truncated merely because a bounded transport-omission marker is present; that marker is not part of the answer. Judge visible semantic gaps normally, and request recovery for actual unfinished syntax or missing requested content only when the complete candidate is shown or the visible head/tail establishes the defect.',
          'When needs_recovery can be advanced by concrete read-only evidence collection, include up to 4 concrete toolCalls using only fs.read, fs.glob, or fs.search. Omit toolCalls when recovery requires judgment, writing, user input, or an unsafe/state-changing action.',
          'If the user requested creating, writing, updating, or saving a named file/artifact, use needs_recovery unless the recent tool trace shows a successful file-writing/editing tool call for that deliverable, or the assistant states a concrete blocker. A prose answer that merely summarizes or claims the file will be written is not complete.',
          'Hard fail for file/artifact requests: returning the artifact body inside chat, markdown fences, or a code block is NOT the same as writing the requested file. A named-file deliverable is complete only when the tool trace proves a successful fs.write/fs.append/fs.edit/apply_patch/terminal write for that deliverable, or the candidate answer states a real policy/tool blocker. If there is no such write trace, return needs_recovery even if the candidate answer looks polished.',
          'For broad or deep artifact requests such as full-codebase analysis, architecture reconstruction, requirement extraction, long reports, or presentation-style documents, also judge whether the run contract, recent tool trace, source-content read list, search evidence, and file-write preview show enough evidence and scope for that request. Use needs_recovery when a broad artifact is only a short overview, lacks the requested sections or diagrams, has no evidence trail, or follows only shallow exploration. This is a semantic quality check, not a fixed keyword or size rule.',
          'If the latest request asks for exhaustive, whole-codebase, or requirement-extraction work, the burden is higher than "a file was written": the artifact must either be grounded in broad, representative repository evidence or explicitly mark itself partial/incomplete with the remaining scope. Do not accept source-of-truth links, package-name lists, or artifact self-claims as substitutes for observed evidence.',
          'When a run contract evidence requirement has requiresArtifactEvidenceMap=true, the written artifact must trace its claims to observed source paths or source scopes from the tool evidence summary. Use needs_recovery if the artifact does not expose that traceability or coverage map.',
          'When a run contract evidence requirement has requiresArtifactSelfReview=true, the written artifact must include an explicit self-review or acceptance checklist that closes the run contract acceptance criteria by id. Use needs_recovery if those criterion ids are not traceable in the artifact read-back.',
          'Use the tool evidence summary as authoritative over the candidate answer. If the answer claims it read or verified files that only appear in glob/list evidence, the written artifact preview, or the artifact itself, treat that claim as unsupported and request recovery unless the answer clearly labels it as existence-only evidence.',
          'For factual status, measurement, identity, count, or configuration claims derived from external or runtime observations, compare the candidate claim itself with the exact current-turn tool result excerpts. If a claim reverses, changes, or invents an observed value, resource, or outcome, use needs_recovery with recoveryMode=synthesis when the existing evidence is sufficient to correct the answer.',
          'When recoveryMode=synthesis is appropriate, identify every material mismatch or omission visible in the current candidate in this one review rather than returning only the first defect. Put the full bounded correction checklist in instruction so the next synthesis can repair all observed resources, values, counts, polarity, and requested disclosure boundaries in one pass.',
          'When the run contract lists required artifacts or the latest user request names specific file or directory scopes, compare each scope against the Required/requested path coverage line. If a required scope has no evidence, or only list evidence while the answer claims content-level analysis, use needs_recovery.',
          'Use complete when the answer is adequate or when more recovery would require private user data or unavailable capabilities.',
          `Available tools: ${availableTools}`,
        ].join(' '),
      },
      {
        role: 'user',
        content: [
          '[Latest user request]',
          latestUserMessageText(options.messages),
          '',
          '[Run contract]',
          formatRunContractForReview(options.runContract),
          formatActiveUserInstructions(options.userInstructions),
          '',
          '[Tool evidence summary]',
          formatToolEvidenceForReview(options.messages, options.runContract, options.evidenceLedger),
          '',
          '[Recent conversation after the latest user request]',
          formatConversationForReview(options.messages) || '[none]',
          '',
          '[Candidate assistant answer]',
          formatCandidateAnswerForReview(options.assistantAnswer),
        ].join('\n'),
      },
    ],
  }
}

function parseCompleteRunOutcomeReviewJson(content: string): RunOutcomeReview | null {
  const match = content.match(/\{[\s\S]*\}/)
  if (!match) return null
  try {
    const parsed = JSON.parse(match[0]) as Record<string, unknown>
    return normalizeRunOutcomeReview(parsed)
  } catch {
    return null
  }
}

export function parseRunOutcomeReview(content: string): RunOutcomeReview | null {
  const complete = parseCompleteRunOutcomeReviewJson(content)
  if (complete) return complete

  // Some local providers cut short small JSON-only judge replies. A visible
  // needs_recovery status is still a conservative repair signal.
  return parsePartialRunOutcomeReview(content)
}

/**
 * Prefer the provider's visible control reply, but allow a JSON-only outcome
 * review returned in a non-visible reasoning field to pass through the same
 * schema normalization. The thinking text itself never becomes user output.
 */
export function parseRunOutcomeReviewTransport(
  visibleContent: string,
  thinking: string | undefined,
): RunOutcomeReview | null {
  return parseRunOutcomeReview(visibleContent)
    ?? (thinking?.trim() ? parseCompleteRunOutcomeReviewJson(thinking) : null)
}

export function buildEmptyRunOutcomeReviewRecovery(content: string): RunOutcomeReview | null {
  if (content.trim().length > 0) {
    return null
  }
  return {
    status: 'needs_recovery',
    reason: 'Outcome review returned an empty response before confirming completion.',
    instruction: 'Continue with grounded evidence. If the requested scope cannot be completed in this run, answer INCOMPLETE with the concrete missing evidence.',
  }
}

export function buildOutcomeReviewExhaustedMessage(reason: string): string {
  return [
    'INCOMPLETE: The run could not produce a sufficiently grounded final answer before recovery attempts were exhausted.',
    `Last review blocker: ${reason}`,
    'The partial tool evidence remains available in this session; resume the run or ask again with a narrower scope to continue gathering file evidence.',
  ].join('\n')
}

function normalizeRunOutcomeReview(parsed: Record<string, unknown>): RunOutcomeReview | null {
  const status = typeof parsed.status === 'string'
    ? parsed.status.toLowerCase()
    : ''
  if (status !== 'complete' && status !== 'needs_recovery') {
    return null
  }
  const reason = typeof parsed.reason === 'string' && parsed.reason.trim()
    ? parsed.reason.trim()
    : defaultOutcomeReviewReason(status)
  const instruction = typeof parsed.instruction === 'string' && parsed.instruction.trim()
    ? parsed.instruction.trim()
    : undefined
  const recoveryMode = parsed.recoveryMode === 'tool'
    || parsed.recoveryMode === 'synthesis'
    || parsed.recoveryMode === 'user-input'
    ? parsed.recoveryMode
    : undefined
  const missingToolNames = normalizeMissingToolNames(parsed.missingToolNames)
  const recoveryBasis = normalizeRecoveryBasis(parsed.recoveryBasis)
    ?? (missingToolNames.length > 0 ? 'missing-tool-execution' : undefined)
  const toolCalls = normalizeSuggestedToolCalls(parsed.toolCalls ?? parsed.suggestedToolCalls)
  return {
    status,
    reason,
    instruction,
    ...(recoveryMode ? { recoveryMode } : {}),
    ...(recoveryBasis ? { recoveryBasis } : {}),
    ...(missingToolNames.length > 0 ? { missingToolNames } : {}),
    ...(toolCalls.length > 0 ? { toolCalls } : {}),
  }
}

function normalizeRecoveryBasis(value: unknown): RunOutcomeReviewRecoveryBasis | undefined {
  return value === 'missing-tool-execution'
    || value === 'insufficient-tool-evidence'
    || value === 'answer-synthesis'
    || value === 'user-input'
    ? value
    : undefined
}

function normalizeMissingToolNames(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return Array.from(new Set(value
    .filter((entry): entry is string => typeof entry === 'string')
    .map((entry) => entry.trim())
    .filter((entry) => /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(entry))))
    .slice(0, 16)
}

function defaultOutcomeReviewReason(status: RunOutcomeReviewStatus): string {
  return status === 'complete'
    ? 'The answer appears complete.'
    : 'The answer needs another agent step.'
}

function extractJsonStringField(content: string, field: string): string | undefined {
  const escapedField = field.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const full = content.match(new RegExp(`"${escapedField}"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"`, 'i'))
  if (full?.[1]) {
    try {
      return JSON.parse(`"${full[1]}"`) as string
    } catch {
      return compact(full[1], 500)
    }
  }

  const partial = content.match(new RegExp(`"${escapedField}"\\s*:\\s*"([\\s\\S]*)`, 'i'))
  if (!partial?.[1]) {
    return undefined
  }
  const nextFieldIndex = partial[1].search(/"\s*,\s*"(?:status|reason|instruction)"\s*:/i)
  const raw = nextFieldIndex >= 0
    ? partial[1].slice(0, nextFieldIndex)
    : partial[1]
  const cleaned = raw
    .replace(/```[\s\S]*$/g, '')
    .replace(/[}\]]+\s*$/g, '')
    .trim()
  return cleaned ? compact(cleaned, 500) : undefined
}

function parsePartialRunOutcomeReview(content: string): RunOutcomeReview | null {
  const statusMatch = content.match(/"status"\s*:\s*"(complete|needs_recovery)"/i)
  const status = statusMatch?.[1]?.toLowerCase() as RunOutcomeReviewStatus | undefined
  if (status !== 'complete' && status !== 'needs_recovery') {
    return null
  }

  const reason = extractJsonStringField(content, 'reason')?.trim()
    || (status === 'needs_recovery'
      ? 'Outcome review response was truncated but indicated needs_recovery.'
      : defaultOutcomeReviewReason(status))
  const instruction = extractJsonStringField(content, 'instruction')?.trim()
  const recoveryModeMatch = content.match(
    /"recoveryMode"\s*:\s*"(tool|synthesis|user-input)"/i,
  )
  const recoveryMode = recoveryModeMatch?.[1]?.toLowerCase() as
    | RunOutcomeReview['recoveryMode']
    | undefined
  const recoveryBasisMatch = content.match(
    /"recoveryBasis"\s*:\s*"(missing-tool-execution|insufficient-tool-evidence|answer-synthesis|user-input)"/i,
  )
  const recoveryBasis = recoveryBasisMatch?.[1]?.toLowerCase() as
    | RunOutcomeReview['recoveryBasis']
    | undefined
  return {
    status,
    reason,
    instruction: instruction || undefined,
    ...(recoveryMode ? { recoveryMode } : {}),
    ...(recoveryBasis ? { recoveryBasis } : {}),
  }
}

function groundRunOutcomeReviewToolHistory(options: {
  review: RunOutcomeReview
  messages: Message[]
  evidenceLedger?: AgentEvidenceLedger
}): RunOutcomeReview {
  const { review } = options
  if (
    review.status !== 'needs_recovery'
    || review.recoveryBasis !== 'missing-tool-execution'
    || !review.missingToolNames?.length
  ) {
    return review
  }

  const successful = collectSuccessfulToolExecutions(options.messages, options.evidenceLedger)
  const successfulByNormalizedName = new Map(
    Array.from(successful.names, (name) => [name.toLowerCase(), name]),
  )
  const contradicted = review.missingToolNames
    .map((name) => successfulByNormalizedName.get(name.toLowerCase()))
    .filter((name): name is string => Boolean(name))
  if (contradicted.length === 0) return review

  const uniqueContradicted = Array.from(new Set(contradicted))
  return {
    status: 'needs_recovery',
    recoveryMode: 'synthesis',
    recoveryBasis: 'answer-synthesis',
    reason: `The proposed missing-tool blocker contradicts trusted successful current-turn execution evidence for: ${uniqueContradicted.join(', ')}.`,
    instruction: [
      `Use the existing successful results from ${uniqueContradicted.join(', ')} and do not claim those tools were unattempted or unexecuted.`,
      'Re-evaluate only whether their returned content is sufficient for the request.',
      'If it is sufficient, synthesize the corrected finished answer now; if it is not, state the exact missing content or evidence boundary instead of repeating the contradicted execution-history claim.',
    ].join(' '),
  }
}

export function shouldRepairRunOutcomeReview(options: {
  review: RunOutcomeReview | null
  repairedCount: number
  maxRepairs?: number
}): boolean {
  const maxRepairs = options.maxRepairs ?? MAX_OUTCOME_REVIEW_REPAIRS
  return options.review?.status === 'needs_recovery' && options.repairedCount < maxRepairs
}

export function enforceRunOutcomeReviewEvidenceFloor(options: {
  review: RunOutcomeReview | null
  messages: Message[]
  runContract?: AgentRunContract
  evidenceLedger?: AgentEvidenceLedger
  assistantAnswer?: string
  pathClaimEvidenceScope?: 'recent' | 'all'
}): RunOutcomeReview | null {
  const unverifiedArtifactWrite = evaluateUnverifiedRequestedArtifactWrite(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  const artifactSectionCoverage = evaluateArtifactSectionCoverage(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  const broadArtifactEvidenceFloor = evaluateEvidenceFloor(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  const artifactEvidenceMapTraceability = evaluateArtifactEvidenceMapTraceability(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  const artifactSelfReviewCoverage = evaluateArtifactSelfReviewCoverage(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  const validationEvidenceFloor = evaluateValidationEvidenceFloor(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  const unsupportedPathClaims = evaluateUnsupportedPathClaims(
    options.messages,
    options.assistantAnswer,
    {
      evidenceScope: options.pathClaimEvidenceScope,
      runContract: options.runContract,
    },
  )
  const failedArtifactMutation = evaluateFailedRequestedArtifactMutation(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  if (options.review?.status === 'needs_recovery' && failedArtifactMutation) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = failedArtifactMutation.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${failedArtifactMutation.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${failedArtifactMutation.instruction}`
        : failedArtifactMutation.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  const missingRequiredArtifactWrite = evaluateMissingRequiredArtifactWrite(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  const compactedArtifactReadBack = evaluateCompactedRequiredArtifactReadBack(
    options.messages,
    options.runContract,
    options.evidenceLedger,
  )
  if (options.review?.status === 'needs_recovery' && missingRequiredArtifactWrite) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = missingRequiredArtifactWrite.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${missingRequiredArtifactWrite.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${missingRequiredArtifactWrite.instruction}`
        : missingRequiredArtifactWrite.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && compactedArtifactReadBack) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = compactedArtifactReadBack.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${compactedArtifactReadBack.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${compactedArtifactReadBack.instruction}`
        : compactedArtifactReadBack.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && unverifiedArtifactWrite) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = unverifiedArtifactWrite.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${unverifiedArtifactWrite.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${unverifiedArtifactWrite.instruction}`
        : unverifiedArtifactWrite.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && artifactSectionCoverage) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = artifactSectionCoverage.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${artifactSectionCoverage.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${artifactSectionCoverage.instruction}`
        : artifactSectionCoverage.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && validationEvidenceFloor) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = validationEvidenceFloor.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${validationEvidenceFloor.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${validationEvidenceFloor.instruction}`
        : validationEvidenceFloor.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && broadArtifactEvidenceFloor) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = broadArtifactEvidenceFloor.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${broadArtifactEvidenceFloor.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${broadArtifactEvidenceFloor.instruction}`
        : broadArtifactEvidenceFloor.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && artifactEvidenceMapTraceability) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = artifactEvidenceMapTraceability.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${artifactEvidenceMapTraceability.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${artifactEvidenceMapTraceability.instruction}`
        : artifactEvidenceMapTraceability.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && artifactSelfReviewCoverage) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = artifactSelfReviewCoverage.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${artifactSelfReviewCoverage.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${artifactSelfReviewCoverage.instruction}`
        : artifactSelfReviewCoverage.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status === 'needs_recovery' && unsupportedPathClaims) {
    const existingToolCalls = options.review.toolCalls ?? []
    const additionalToolCalls = unsupportedPathClaims.toolCalls ?? []
    return {
      ...options.review,
      reason: `${options.review.reason} ${unsupportedPathClaims.reason}`,
      instruction: options.review.instruction
        ? `${options.review.instruction} ${unsupportedPathClaims.instruction}`
        : unsupportedPathClaims.instruction,
      toolCalls: [...existingToolCalls, ...additionalToolCalls].slice(0, 4),
    }
  }
  if (options.review?.status !== 'complete') {
    return options.review
      ? groundRunOutcomeReviewToolHistory({
          review: options.review,
          messages: options.messages,
          evidenceLedger: options.evidenceLedger,
        })
      : null
  }
  if (failedArtifactMutation) {
    return {
      status: 'needs_recovery',
      reason: failedArtifactMutation.reason,
      instruction: failedArtifactMutation.instruction,
      ...(failedArtifactMutation.toolCalls && failedArtifactMutation.toolCalls.length > 0
        ? { toolCalls: failedArtifactMutation.toolCalls }
        : {}),
    }
  }
  if (missingRequiredArtifactWrite) {
    return {
      status: 'needs_recovery',
      reason: missingRequiredArtifactWrite.reason,
      instruction: missingRequiredArtifactWrite.instruction,
      ...(missingRequiredArtifactWrite.toolCalls && missingRequiredArtifactWrite.toolCalls.length > 0
        ? { toolCalls: missingRequiredArtifactWrite.toolCalls }
      : {}),
    }
  }
  if (compactedArtifactReadBack) {
    return {
      status: 'needs_recovery',
      reason: compactedArtifactReadBack.reason,
      instruction: compactedArtifactReadBack.instruction,
      ...(compactedArtifactReadBack.toolCalls && compactedArtifactReadBack.toolCalls.length > 0
        ? { toolCalls: compactedArtifactReadBack.toolCalls }
        : {}),
    }
  }
  if (artifactSectionCoverage) {
    return {
      status: 'needs_recovery',
      reason: artifactSectionCoverage.reason,
      instruction: artifactSectionCoverage.instruction,
      ...(artifactSectionCoverage.toolCalls && artifactSectionCoverage.toolCalls.length > 0
        ? { toolCalls: artifactSectionCoverage.toolCalls }
        : {}),
    }
  }
  const floor = broadArtifactEvidenceFloor
  if (validationEvidenceFloor) {
    return {
      status: 'needs_recovery',
      reason: validationEvidenceFloor.reason,
      instruction: validationEvidenceFloor.instruction,
      ...(validationEvidenceFloor.toolCalls && validationEvidenceFloor.toolCalls.length > 0
        ? { toolCalls: validationEvidenceFloor.toolCalls }
        : {}),
    }
  }
  if (unsupportedPathClaims) {
    return {
      status: 'needs_recovery',
      reason: unsupportedPathClaims.reason,
      instruction: unsupportedPathClaims.instruction,
      ...(unsupportedPathClaims.toolCalls && unsupportedPathClaims.toolCalls.length > 0
        ? { toolCalls: unsupportedPathClaims.toolCalls }
        : {}),
    }
  }
  if (!floor) {
    if (artifactEvidenceMapTraceability) {
      return {
        status: 'needs_recovery',
        reason: artifactEvidenceMapTraceability.reason,
        instruction: artifactEvidenceMapTraceability.instruction,
        ...(artifactEvidenceMapTraceability.toolCalls && artifactEvidenceMapTraceability.toolCalls.length > 0
          ? { toolCalls: artifactEvidenceMapTraceability.toolCalls }
          : {}),
      }
    }
    if (artifactSelfReviewCoverage) {
      return {
        status: 'needs_recovery',
        reason: artifactSelfReviewCoverage.reason,
        instruction: artifactSelfReviewCoverage.instruction,
        ...(artifactSelfReviewCoverage.toolCalls && artifactSelfReviewCoverage.toolCalls.length > 0
          ? { toolCalls: artifactSelfReviewCoverage.toolCalls }
          : {}),
      }
    }
    if (unverifiedArtifactWrite) {
      return {
        status: 'needs_recovery',
        reason: unverifiedArtifactWrite.reason,
        instruction: unverifiedArtifactWrite.instruction,
        ...(unverifiedArtifactWrite.toolCalls && unverifiedArtifactWrite.toolCalls.length > 0
          ? { toolCalls: unverifiedArtifactWrite.toolCalls }
          : {}),
      }
    }
    return options.review
  }
  return {
    status: 'needs_recovery',
    reason: floor.reason,
    instruction: floor.instruction,
    ...(floor.toolCalls && floor.toolCalls.length > 0 ? { toolCalls: floor.toolCalls } : {}),
  }
}

/**
 * Drop earlier copies of this block before a new one is pushed.
 *
 * Every review that finds the same gap emits the same 1,300-character
 * instruction, and they were appended rather than replaced — captured requests
 * carried two verbatim copies. Only the newest can be acted on, and the text
 * does not vary with the attempt, so nothing is lost by keeping one.
 */
export function dropStaleRunOutcomeReviewRecovery(messages: Message[]): void {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'system'
      && typeof message.content === 'string'
      && message.content.includes(OUTCOME_REVIEW_RECOVERY_MARKER)
    ) {
      messages.splice(index, 1)
    }
  }
}

export function buildRunOutcomeReviewRecoveryMessage(review: RunOutcomeReview): Message {
  const recoveryMode = review.recoveryMode ?? 'tool'
  const toolRecoveryLines = recoveryMode === 'tool'
    ? [
        'If the next step references file paths, verify the paths with fs.glob/fs.search before fs.read; do not assume example paths exist.',
        'Do not reread the same validation artifact or source file with the same arguments if that exact evidence has already been collected in this turn; choose a different evidence path, perform the missing write/edit work, or state the concrete blocker.',
        'Continue now by performing the missing tool work. Do not answer from memory before the required new evidence or artifact work is available.',
        'A toolless ANSWER before the required new tool evidence will be rejected. If no valid tool can continue, reply INCOMPLETE with the concrete blocker.',
      ]
    : recoveryMode === 'synthesis'
      ? [
          'The existing tool evidence is sufficient. Do not call another tool merely to create new evidence.',
          'Apply every mismatch and omission listed above in one comprehensive correction pass, then re-check every factual claim against the existing current-turn results.',
          'Provide the corrected finished answer now, grounded only in the existing current-turn results.',
        ]
      : [
          'No tool can resolve the remaining decision without user input.',
          'Ask one concise blocking question or report the concrete decision that is required.',
        ]
  return {
    role: 'system',
    metadata: { outcomeReviewRecoveryMode: recoveryMode },
    content: [
      `${OUTCOME_REVIEW_RECOVERY_MARKER} The previous assistant reply did not fully satisfy the user request.`,
      `Reason: ${review.reason}`,
      review.instruction ? `Next step: ${review.instruction}` : '',
      ...toolRecoveryLines,
      'Do not restate a plan or progress update as the final answer.',
    ].filter(Boolean).join('\n'),
  }
}

export function hasPendingRunOutcomeReviewRecovery(messages: Message[]): boolean {
  const lastRecoveryIndex = messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) =>
      message.role === 'system'
      && typeof message.content === 'string'
      && message.content.includes(OUTCOME_REVIEW_RECOVERY_MARKER)
    )
    .at(-1)?.index
  if (lastRecoveryIndex == null) {
    return false
  }

  const recoveryMode = messages[lastRecoveryIndex]?.metadata?.outcomeReviewRecoveryMode
  if (recoveryMode === 'synthesis' || recoveryMode === 'user-input') {
    return false
  }

  return !messages.slice(lastRecoveryIndex + 1).some((message) => message.role === 'tool')
}

export function hasRunOutcomeReviewRecoverySinceLastUser(messages: Message[]): boolean {
  return messagesSinceLastUser(messages).some((message) =>
    message.role === 'system'
    && typeof message.content === 'string'
    && message.content.includes(OUTCOME_REVIEW_RECOVERY_MARKER)
  )
}

export function buildRunOutcomeReviewNoProgressMessage(): Message {
  return {
    role: 'system',
    content: [
      '[LLM outcome review follow-up] The previous recovery instruction still has no new tool evidence after it.',
      'Do not repeat the rejected final answer.',
      'Call the appropriate tool(s) now to gather the missing evidence or perform the missing work.',
      'If no valid tool can make progress, reply INCOMPLETE with the concrete blocker and the specific missing evidence.',
    ].join('\n'),
  }
}

const OUTCOME_REVIEW_AUTORUN_TOOL_NAMES = new Set(['fs.read', 'fs.glob', 'fs.search'])

function stringArg(args: Record<string, unknown>, key: string): string | undefined {
  const value = args[key]
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined
}

function normalizeSuggestedToolArguments(
  name: string,
  args: Record<string, unknown>,
): Record<string, unknown> | null {
  if (name === 'fs.read') {
    const path = stringArg(args, 'path') ?? stringArg(args, 'file')
    if (!path) return null
    return { ...args, path }
  }

  if (name === 'fs.glob') {
    const pattern = stringArg(args, 'pattern') ?? stringArg(args, 'glob') ?? stringArg(args, 'path')
    if (!pattern) return null
    const normalized: Record<string, unknown> = { ...args, pattern }
    delete normalized.glob
    return normalized
  }

  if (name === 'fs.search') {
    const query = stringArg(args, 'query') ?? stringArg(args, 'pattern')
    if (!query) return null
    const cwd = stringArg(args, 'cwd') ?? stringArg(args, 'path')
    const glob = stringArg(args, 'glob')
    const normalized: Record<string, unknown> = { ...args, query }
    delete normalized.path
    delete normalized.pattern
    if (cwd) normalized.cwd = cwd
    if (glob) normalized.glob = glob
    return normalized
  }

  return null
}

function normalizeSuggestedToolCalls(value: unknown): NonNullable<RunOutcomeReview['toolCalls']> {
  if (!Array.isArray(value)) {
    return []
  }

  const calls: NonNullable<RunOutcomeReview['toolCalls']> = []
  for (const entry of value) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      continue
    }
    const record = entry as Record<string, unknown>
    const name = typeof record.name === 'string' ? record.name : ''
    const args = record.arguments
    if (
      !OUTCOME_REVIEW_AUTORUN_TOOL_NAMES.has(name)
      || !args
      || typeof args !== 'object'
      || Array.isArray(args)
    ) {
      continue
    }
    const normalizedArgs = normalizeSuggestedToolArguments(name, args as Record<string, unknown>)
    if (!normalizedArgs) {
      continue
    }
    calls.push({ name, arguments: normalizedArgs })
  }
  return calls
}

export function buildRunOutcomeReviewSuggestedToolCalls(
  review: RunOutcomeReview,
  availableToolNames: Set<string>,
  options: { idPrefix: string; maxToolCalls?: number; messages?: Message[] },
): ToolCall[] {
  return buildRunOutcomeReviewSuggestedToolCallPlan(review, availableToolNames, options).toolCalls
}

export interface RunOutcomeReviewSuggestedToolCallPlan {
  toolCalls: ToolCall[]
  skippedDuplicateCount: number
}

function normalizeStableJson(value: unknown): unknown {
  if (!value || typeof value !== 'object') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map(normalizeStableJson)
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, normalizeStableJson(entry)]),
  )
}

function toolCallSignature(name: string, args: Record<string, unknown>): string {
  if (name === 'fs.read' && typeof args.path === 'string') {
    const offset = typeof args.offset === 'number' ? args.offset : ''
    return `${name}:${args.path}:${offset}`
  }
  return `${name}:${JSON.stringify(normalizeStableJson(args))}`
}

function attemptedOutcomeReviewAutorunSignatures(messages: Message[] | undefined): Set<string> {
  const signatures = new Set<string>()
  if (!messages) {
    return signatures
  }

  for (const message of messagesSinceLastUser(messages)) {
    for (const toolCall of message.toolCalls ?? []) {
      if (!OUTCOME_REVIEW_AUTORUN_TOOL_NAMES.has(toolCall.name)) {
        continue
      }
      signatures.add(toolCallSignature(toolCall.name, toolCall.arguments))
    }
  }
  return signatures
}

export function buildRunOutcomeReviewSuggestedToolCallPlan(
  review: RunOutcomeReview,
  availableToolNames: Set<string>,
  options: { idPrefix: string; maxToolCalls?: number; messages?: Message[] },
): RunOutcomeReviewSuggestedToolCallPlan {
  const maxToolCalls = options.maxToolCalls ?? 4
  const attemptedSignatures = attemptedOutcomeReviewAutorunSignatures(options.messages)
  let skippedDuplicateCount = 0
  const toolCalls = (review.toolCalls ?? [])
    .map((toolCall) => {
      const normalizedArgs = normalizeSuggestedToolArguments(toolCall.name, toolCall.arguments)
      return normalizedArgs ? { ...toolCall, arguments: normalizedArgs } : null
    })
    .filter((toolCall): toolCall is NonNullable<typeof toolCall> => Boolean(toolCall))
    .filter((toolCall) =>
      OUTCOME_REVIEW_AUTORUN_TOOL_NAMES.has(toolCall.name)
      && availableToolNames.has(toolCall.name)
    )
    .filter((toolCall) => {
      const signature = toolCallSignature(toolCall.name, toolCall.arguments)
      if (attemptedSignatures.has(signature)) {
        skippedDuplicateCount += 1
        return false
      }
      attemptedSignatures.add(signature)
      return true
    })
    .slice(0, maxToolCalls)
    .map((toolCall, index) => ({
      id: `${options.idPrefix}-${index + 1}`,
      name: toolCall.name,
      arguments: toolCall.arguments,
    }))
  return { toolCalls, skippedDuplicateCount }
}

export function buildRunOutcomeReviewDuplicateSuggestedToolCallsMessage(
  review: RunOutcomeReview,
): Message {
  return {
    role: 'system',
    content: [
      '[LLM outcome review follow-up] The review suggested only read-only tool calls that have already been attempted with the same arguments in this turn.',
      `Reason: ${review.reason}`,
      'Do not repeat those calls. Choose different source evidence, inspect a different relevant scope, update the requested artifact with the evidence already gathered, or reply INCOMPLETE with the specific remaining blocker.',
      'A toolless ANSWER that merely repeats the rejected conclusion will be rejected.',
    ].join('\n'),
  }
}
