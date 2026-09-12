import { activeUserInstructions, formatActiveUserInstructions } from './user-steering.js'
import { createHash } from 'node:crypto'
import { ThinkingLevel, type ChatRequest } from '@sepilotd/core'
import type {
  AgentRunContract,
  AgentStateBoardSnapshot,
  Message,
  ToolExecutionPosture,
  ToolSecurityEffect,
} from '@sepilotd/core'
import type {
  AgentCriterionVerdictSnapshot,
  AgentState,
} from './graph/types.js'
import { TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY } from './policy-failure.js'
import { isExecutorConfirmedReadOnlyObservation } from './read-only-observation-evidence.js'
import { isDocumentArtifactPath } from './task-contract.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

export const CRITERION_EVIDENCE_REVIEW_MAX_TOKENS = 2048
const MAX_REVIEW_OBSERVATIONS = 32
const MAX_REVIEW_FAILURES = 16
const MAX_CRITERION_EVIDENCE_REFERENCES = 16

export interface CriterionEvidenceEpisode {
  toolResultCount: number
  toolResultFingerprint: string
}

export interface CriterionReferenceableObservation {
  toolCallId: string
  tool: string
  output: string
}

export type CriterionVerdict = AgentCriterionVerdictSnapshot['criterionVerdicts'][number]

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

const DOCUMENT_ARTIFACT_WRITE_TOOLS = new Set([
  'apply_patch',
  'fs.append',
  'fs.edit',
  'fs.write',
])

function normalizeEvidencePath(value: string): string {
  return value.trim().replaceAll('\\', '/').replace(/^\.\//u, '').replace(/\/+$/u, '')
}

function evidencePathsReferToSameFile(left: string, right: string): boolean {
  const normalizedLeft = normalizeEvidencePath(left)
  const normalizedRight = normalizeEvidencePath(right)
  return normalizedLeft.length > 0
    && normalizedRight.length > 0
    && (
      normalizedLeft === normalizedRight
      || normalizedLeft.endsWith(`/${normalizedRight}`)
      || normalizedRight.endsWith(`/${normalizedLeft}`)
    )
}

function toolHistoryEntryPaths(
  entry: NonNullable<AgentState['toolCallHistory']>[number],
): string[] {
  const paths = [entry.input.path, entry.input.file]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
  if (entry.tool !== 'apply_patch' || typeof entry.input.patch !== 'string') {
    return [...new Set(paths)]
  }
  for (const match of entry.input.patch.matchAll(/^\*\*\* (?:Add|Update|Delete) File: (.+)$/gmu)) {
    const path = match[1]?.trim()
    if (path) paths.push(path)
  }
  return [...new Set(paths)]
}

/**
 * Document production can use the semantic criterion judge after the exact
 * requested artifacts were written and then observed again. Mutation results
 * themselves never become criterion evidence: the judge sees only
 * executor-confirmed observe calls, including the post-write read-back.
 */
export function documentArtifactCriterionReviewEligible(
  state: Pick<AgentState, 'seedContract' | 'toolCallHistory'>,
): boolean {
  const contract = state.seedContract
  const artifacts = contract?.requiredArtifacts ?? []
  if (
    !contract
    || artifacts.length === 0
    || artifacts.some((artifact) => (
      artifact.kind === 'directory'
      || !(artifact.kind === 'document' || isDocumentArtifactPath(artifact.path))
    ))
    || (contract.evidenceRequirements ?? []).some((requirement) => requirement.kind === 'validation')
  ) {
    return false
  }

  const history = state.toolCallHistory ?? []
  return artifacts.every((artifact) => {
    let latestWriteIndex = -1
    history.forEach((entry, index) => {
      if (
        entry.status === 'success'
        && DOCUMENT_ARTIFACT_WRITE_TOOLS.has(entry.tool)
        && toolHistoryEntryPaths(entry).some((path) => evidencePathsReferToSameFile(path, artifact.path))
      ) {
        latestWriteIndex = index
      }
    })
    if (latestWriteIndex < 0) return false
    return history.some((entry, index) => (
      entry.status === 'success'
      && entry.tool === 'fs.read'
      && index > latestWriteIndex
      && Boolean(entry.output?.trim())
      && toolHistoryEntryPaths(entry).some((path) => evidencePathsReferToSameFile(path, artifact.path))
      && isExecutorConfirmedReadOnlyObservation({
        tool: entry.tool,
        securityEffect: entry.securityEffect,
        executionObserved: entry.executionObserved,
        actionPurpose: entry.input.actionPurpose,
        executionPosture: entry.executionPosture,
      })
    ))
  })
}

/**
 * Runtime-owned tool-result metadata can carry the same execution posture
 * that Graph journals on its tool-call ledger. Validate the complete shape
 * before using it: arbitrary tool result metadata is never copied into this
 * field by the executor, and malformed persisted messages fail closed.
 */
function executionPostureFromMessage(message: Message): ToolExecutionPosture | undefined {
  const candidate = record(message.metadata?.executionPosture)
  const sandbox = record(candidate?.sandbox)
  const filesystem = record(candidate?.filesystem)
  const network = record(candidate?.network)
  if (
    typeof sandbox?.requested !== 'boolean'
    || typeof sandbox.active !== 'boolean'
    || typeof sandbox.mode !== 'string'
    || typeof filesystem?.boundary !== 'string'
    || typeof filesystem.isolated !== 'boolean'
    || (filesystem.readOnly !== undefined && typeof filesystem.readOnly !== 'boolean')
    || typeof network?.isolated !== 'boolean'
    || typeof network.mode !== 'string'
  ) {
    return undefined
  }
  return {
    sandbox: {
      requested: sandbox.requested,
      active: sandbox.active,
      mode: sandbox.mode,
      ...(typeof sandbox.fallbackReason === 'string'
        ? { fallbackReason: sandbox.fallbackReason }
        : {}),
    },
    filesystem: {
      boundary: filesystem.boundary,
      isolated: filesystem.isolated,
      ...(typeof filesystem.cwd === 'string' ? { cwd: filesystem.cwd } : {}),
      ...(typeof filesystem.readOnly === 'boolean' ? { readOnly: filesystem.readOnly } : {}),
      ...(typeof filesystem.note === 'string' ? { note: filesystem.note } : {}),
    },
    network: {
      isolated: network.isolated,
      mode: network.mode,
    },
  }
}

/**
 * Reconstruct the same exact-id observation packet for the lean ReAct loop
 * that Graph keeps in `toolCallHistory`. Only executor-confirmed successful
 * calls whose registry-owned effect is statically `observe` are admitted.
 * Dynamic tools require the daemon-owned posture copied by the executor plus
 * their structured `actionPurpose=observe`; missing or malformed posture fails
 * closed instead of trusting provider text or tool-supplied result metadata.
 */
export function criterionEvidenceStateFromMessages(options: {
  messages: readonly Message[]
  contract: AgentRunContract
  securityEffectForTool: (tool: string) => ToolSecurityEffect
}): Pick<AgentState, 'seedContract' | 'toolCallHistory'> {
  let turnStart = -1
  for (let index = options.messages.length - 1; index >= 0; index -= 1) {
    const message = options.messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      turnStart = index
      break
    }
  }
  if (turnStart < 0) {
    for (let index = options.messages.length - 1; index >= 0; index -= 1) {
      if (options.messages[index]?.role === 'user') {
        turnStart = index
        break
      }
    }
  }

  const calls = new Map<string, { name: string; input: Record<string, unknown> }>()
  const toolCallHistory: NonNullable<AgentState['toolCallHistory']> = []
  const scoped = turnStart >= 0 ? options.messages.slice(turnStart) : options.messages
  scoped.forEach((message, index) => {
    for (const call of message.toolCalls ?? []) {
      calls.set(call.id, { name: call.name, input: { ...call.arguments } })
    }
    if (!message.toolCallId || message.role !== 'tool') return
    const call = calls.get(message.toolCallId)
    const tool = message.name ?? call?.name
    if (!tool) return
    const status = message.metadata?.toolResultStatus === 'success'
      ? 'success' as const
      : 'error' as const
    const executionPosture = executionPostureFromMessage(message)
    toolCallHistory.push({
      toolCallId: message.toolCallId,
      tool,
      input: call?.input ?? {},
      status,
      executionObserved:
        message.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
      securityEffect: options.securityEffectForTool(tool),
      ...(executionPosture ? { executionPosture } : {}),
      ts: index + 1,
      output: messageText(message),
    })
  })

  return {
    seedContract: {
      ...options.contract,
      acceptanceCriteria: options.contract.acceptanceCriteria.map((criterion) => ({
        ...criterion,
      })),
      constraints: [...options.contract.constraints],
      outOfScope: [...options.contract.outOfScope],
    },
    toolCallHistory,
  }
}

/** Build the durable, mode-neutral completion packet consumed by session evaluation. */
export function buildCriterionEvidenceBoardSnapshot(options: {
  contract: AgentRunContract
  state: Pick<AgentState, 'toolCallHistory'>
  verdicts: readonly CriterionVerdict[]
}): AgentStateBoardSnapshot {
  const verdictsById = new Map(options.verdicts.map((verdict) => [verdict.id, verdict]))
  const unmet = options.contract.acceptanceCriteria
    .filter((criterion) => verdictsById.get(criterion.id)?.verdict !== 'met')
    .map((criterion) => criterion.id)
  const observations = collectCriterionReferenceableObservations(options.state)
  return {
    goal: options.contract.summary,
    completionCriteria: options.contract.acceptanceCriteria.map((criterion) => ({ ...criterion })),
    contract: structuredClone(options.contract),
    plan: [],
    todos: [],
    decisions: [],
    failedAttempts: [],
    openQuestions: [],
    evidenceSection: observations.length > 0
      ? [
          '[Criterion-referenceable read-only observations]',
          ...observations.map((entry) => `- [evidence ${entry.toolCallId}] ${entry.tool}`),
        ].join('\n')
      : null,
    completion: {
      criterionVerdicts: options.verdicts.map((verdict) => ({
        ...verdict,
        ...(verdict.evidenceToolCallIds
          ? { evidenceToolCallIds: [...verdict.evidenceToolCallIds] }
          : {}),
      })),
      gate: {
        decision: unmet.length === 0 ? 'pass' : 'block',
        unmet,
        ...(unmet.length > 0
          ? { reason: 'One or more acceptance criteria lack exact current-turn observation evidence.' }
          : {}),
      },
    },
  }
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

function boundedEnds<T>(values: readonly T[], max: number): T[] {
  if (values.length <= max) return [...values]
  const head = Math.ceil(max / 2)
  return [...values.slice(0, head), ...values.slice(-(max - head))]
}

export function criterionEvidenceEpisode(
  state: Pick<AgentState, 'toolCallHistory'>,
): CriterionEvidenceEpisode {
  const history = state.toolCallHistory ?? []
  const hash = createHash('sha256')
  for (const entry of history) {
    hash.update(entry.toolCallId ?? '')
    hash.update('\0')
    hash.update(entry.tool)
    hash.update('\0')
    hash.update(entry.status)
    hash.update('\0')
    hash.update(String(entry.ts))
    hash.update('\0')
    hash.update(entry.output ?? '')
    hash.update('\0')
  }
  return {
    toolResultCount: history.length,
    toolResultFingerprint: hash.digest('hex'),
  }
}

export function collectCriterionReferenceableObservations(
  state: Pick<AgentState, 'toolCallHistory'>,
): CriterionReferenceableObservation[] {
  return (state.toolCallHistory ?? []).flatMap((entry) => {
    if (
      !entry.toolCallId
      || entry.status !== 'success'
      || !isExecutorConfirmedReadOnlyObservation({
        tool: entry.tool,
        securityEffect: entry.securityEffect,
        executionObserved: entry.executionObserved,
        actionPurpose: entry.input.actionPurpose,
        executionPosture: entry.executionPosture,
      })
      || !entry.output?.trim()
    ) {
      return []
    }
    return [{
      toolCallId: entry.toolCallId,
      tool: entry.tool,
      output: entry.output,
    }]
  })
}

export function buildCriterionEvidenceReviewRequest(options: {
  model: string
  state: Pick<AgentState, 'seedContract' | 'toolCallHistory' | 'steeringNotes'>
  assistantAnswer: string
  maxTokens?: number
}): ChatRequest | null {
  const contract = options.state.seedContract
  const observations = collectCriterionReferenceableObservations(options.state)
  const readOnlyContract = contract?.executionIntent?.workspaceMutation === 'forbidden'
  const verifiedDocumentContract = documentArtifactCriterionReviewEligible(options.state)
  if (
    !contract
    || contract.acceptanceCriteria.length === 0
    || (!readOnlyContract && !verifiedDocumentContract)
    || observations.length === 0
  ) {
    return null
  }

  const selectedObservations = boundedEnds(observations, MAX_REVIEW_OBSERVATIONS)
  const failedResults = boundedEnds(
    (options.state.toolCallHistory ?? []).filter((entry) => (
      entry.status === 'error' && Boolean(entry.output?.trim())
    )),
    MAX_REVIEW_FAILURES,
  )
  const observationLines = selectedObservations.map((entry) =>
    `- ${entry.toolCallId} | ${entry.tool} => ${compactBoundaries(entry.output, 700)}`
  )
  const failureLines = failedResults.map((entry) =>
    `- ${entry.toolCallId ?? '(no call id)'} | ${entry.tool} => ${compactBoundaries(entry.output ?? '', 500)}`
  )

  return {
    model: options.model,
    temperature: 0,
    thinkingLevel: ThinkingLevel.Off,
    maxTokens: options.maxTokens ?? CRITERION_EVIDENCE_REVIEW_MAX_TOKENS,
    messages: [
      {
        role: 'system',
        content: [
          'You are the criterion-evidence judge for an autonomous agent run.',
          'Evaluate every acceptance criterion independently from meaning and the exact current-turn result excerpts. Do not use keyword matching.',
          'A criterion is met only when one or more listed referenceable observations or required-artifact read-backs directly establish it and the candidate answer does not contradict that evidence.',
          'Mutation acknowledgements are never referenceable evidence. For document production, a post-write read-back may establish the artifact content or materialization, while source observations establish its factual grounding.',
          'A successful observation is not evidence for an unrelated criterion. Never attach every observation to every criterion.',
          'Use unmet when the available evidence is missing, failed, contradictory, stale for the requested scope, or only establishes a different criterion.',
          'For met, cite the smallest sufficient set of exact toolCallId values from Referenceable successful observations. Never invent, shorten, or cite a failed-result id.',
          'Return each contract criterion exactly once. Do not add unknown criterion ids.',
          'Return JSON only, with no markdown or hidden reasoning, using this shape:',
          '{"criteria":[{"id":"AC1","verdict":"met|unmet","evidenceToolCallIds":["exact-id"]}]}',
          'For unmet, omit evidenceToolCallIds or return an empty array.',
        ].join(' '),
      },
      {
        role: 'user',
        content: [
          '[Run goal]',
          contract.summary,
          formatActiveUserInstructions(activeUserInstructions(options.state.steeringNotes)),
          '',
          '[Acceptance criteria]',
          ...contract.acceptanceCriteria.map((criterion) => `- ${criterion.id}: ${criterion.text}`),
          '',
          '[Referenceable successful observations and artifact read-backs]',
          ...observationLines,
          ...(observations.length > selectedObservations.length
            ? [`- (${observations.length - selectedObservations.length} middle observations omitted by bounded review context)`]
            : []),
          '',
          '[Failed results — blocker context only, never MET evidence]',
          ...(failureLines.length > 0 ? failureLines : ['- (none)']),
          '',
          '[Candidate assistant answer]',
          compactBoundaries(options.assistantAnswer, 6000) || '(empty)',
        ].join('\n'),
      },
    ],
  }
}

/**
 * Re-ask the same semantic judgment once when the provider returned only a
 * partially valid structured packet. This repairs representation only: the
 * contract, candidate answer, and bounded evidence excerpts are unchanged,
 * no tool is exposed, and missing criteria are named from parser output.
 */
export function buildCriterionEvidenceReviewRepairRequest(options: {
  model: string
  state: Pick<AgentState, 'seedContract' | 'toolCallHistory' | 'steeringNotes'>
  assistantAnswer: string
  acceptedVerdicts: readonly CriterionVerdict[]
  maxTokens?: number
}): ChatRequest | null {
  const request = buildCriterionEvidenceReviewRequest(options)
  const criteria = options.state.seedContract?.acceptanceCriteria ?? []
  const acceptedIds = new Set(options.acceptedVerdicts.map((verdict) => verdict.id))
  const missingIds = criteria
    .map((criterion) => criterion.id)
    .filter((id) => !acceptedIds.has(id))
  if (!request || missingIds.length === 0 || request.messages.length === 0) return null

  const first = request.messages[0]
  if (!first || typeof first.content !== 'string') return null
  request.messages[0] = {
    ...first,
    content: [
      first.content,
      `The previous response was structurally incomplete or used invalid evidence references for these criteria: ${missingIds.join(', ')}.`,
      'This is the only protocol-repair attempt. Re-evaluate the full unchanged packet and return every criterion exactly once.',
      'Every met verdict MUST include a non-empty evidenceToolCallIds array containing only exact ids from the referenceable list; otherwise return unmet for that criterion.',
    ].join(' '),
  }
  return request
}

function jsonObject(content: string): Record<string, unknown> | null {
  const match = content.match(/\{[\s\S]*\}/)
  if (!match) return null
  try {
    const parsed = JSON.parse(match[0])
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null
  } catch {
    return null
  }
}

export function parseCriterionEvidenceReview(
  content: string,
  state: Pick<AgentState, 'seedContract' | 'toolCallHistory'>,
): CriterionVerdict[] {
  const parsed = jsonObject(content)
  const rawCriteria = parsed?.criteria
  const contractCriteria = state.seedContract?.acceptanceCriteria ?? []
  if (!Array.isArray(rawCriteria) || contractCriteria.length === 0) return []

  const canonicalIds = new Map(contractCriteria.map((criterion) => [
    criterion.id.toUpperCase(),
    criterion.id,
  ]))
  const referenceableIds = new Set(
    collectCriterionReferenceableObservations(state).map((entry) => entry.toolCallId),
  )
  const candidatesById = new Map<string, CriterionVerdict[]>()

  for (const raw of rawCriteria) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const candidate = raw as Record<string, unknown>
    const rawId = typeof candidate.id === 'string' ? candidate.id.trim().toUpperCase() : ''
    const id = canonicalIds.get(rawId)
    const verdict = typeof candidate.verdict === 'string'
      ? candidate.verdict.trim().toLowerCase()
      : ''
    if (!id || (verdict !== 'met' && verdict !== 'unmet')) continue

    let normalized: CriterionVerdict | null = null
    if (verdict === 'unmet') {
      normalized = { id, verdict: 'unmet' }
    } else {
      const refs = Array.isArray(candidate.evidenceToolCallIds)
        ? [...new Set(candidate.evidenceToolCallIds.filter(
            (value): value is string => typeof value === 'string' && value.trim().length > 0,
          ).map((value) => value.trim()))].slice(0, MAX_CRITERION_EVIDENCE_REFERENCES)
        : []
      if (refs.length > 0 && refs.every((ref) => referenceableIds.has(ref))) {
        normalized = { id, verdict: 'met', evidenceToolCallIds: refs }
      }
    }
    if (!normalized) continue
    const current = candidatesById.get(id) ?? []
    current.push(normalized)
    candidatesById.set(id, current)
  }

  return contractCriteria.flatMap((criterion) => {
    const candidates = candidatesById.get(criterion.id) ?? []
    // Duplicate output is protocol-ambiguous even when the values happen to
    // match. Ignore it so a malformed judge response cannot overwrite the
    // ordinary completion-gate recovery path.
    return candidates.length === 1 ? candidates : []
  })
}

/**
 * Reasoning-capable transports can place a requested JSON-only control reply
 * in their non-visible thinking field while leaving message content empty.
 * Prefer visible content, then admit thinking only through the same strict
 * criterion/id/evidence validation. The raw thinking text is never surfaced.
 */
export function parseCriterionEvidenceReviewTransport(
  visibleContent: string,
  thinking: string | undefined,
  state: Pick<AgentState, 'seedContract' | 'toolCallHistory'>,
): CriterionVerdict[] {
  const visibleVerdicts = parseCriterionEvidenceReview(visibleContent, state)
  if (visibleVerdicts.length > 0 || !thinking?.trim()) return visibleVerdicts
  return parseCriterionEvidenceReview(thinking, state)
}

export function renderCriterionEvidenceReviewProtocol(
  verdicts: readonly CriterionVerdict[],
): string {
  return verdicts.map((entry) => entry.verdict === 'met'
    ? `CRITERION ${entry.id}: MET EVIDENCE ${(entry.evidenceToolCallIds ?? []).join(',')}`
    : `CRITERION ${entry.id}: UNMET`
  ).join('\n')
}
