import type { GatewayClient } from '../gateway/client.js'
import type { SepilotdConfig } from '../config/schema.js'
import { randomUUID } from 'node:crypto'
import type { AgentRunContract } from '@sepilotd/core'
import type { SubagentFindings } from './subagent-findings.js'
import { emptyEvidenceLedger } from './graph/evidence-ledger.js'
import { formatRunContractForPrompt } from './task-contract.js'
import {
  buildDelegationAssertionPayload,
  verifyDelegationAssertion,
  type DelegationAssertionFields,
  type DelegationKeyResolver,
  type DelegationSigner,
} from './delegation-signing.js'

export interface DelegatorSecurityOptions {
  /** Signs claim/status assertions written by this device. */
  signer?: DelegationSigner
  /** Resolves paired-device public keys for verifying inbound assertions. */
  keyResolver?: DelegationKeyResolver
  /**
   * When true, inbound claim/status/cancel/answer assertions must carry a valid
   * paired-device signature or they are ignored (fail-closed). Default false so
   * an un-upgraded peer that does not yet sign is not silently cut off; operators
   * opt in via SEPILOTD_DELEGATION_REQUIRE_SIGNATURE once every device signs.
   */
  requireSignature?: boolean
}

export interface DelegationRequest {
  targetDevice?: string  // device name or ID, empty = any available
  instruction: string
  priority?: 'high' | 'medium' | 'low'
  timeoutMs?: number
  executionId?: string
  /** Parent run contract forwarded to the remote worker as hard task boundaries. */
  runContract?: AgentRunContract
  /**
   * Scoped board slice (goal + acceptance criteria + known failed-attempts +
   * open-questions) built by {@link buildScopedBoardSlice}, transmitted down to
   * the delegate so a cross-device run starts from the parent's structured state
   * instead of `previousMessages: []`. PLAN_065 T5.
   */
  boardSlice?: string
}

/**
 * Append a scoped board slice to a delegation instruction so it rides down in
 * the dispatched payload (parsed back into the worker's `task.instruction`).
 * Returns the instruction unchanged when there is no slice. PLAN_065 T5.
 */
export function appendBoardSliceToDelegationInstruction(
  instruction: string,
  boardSlice?: string,
): string {
  const slice = boardSlice?.trim()
  if (!slice) return instruction
  return [instruction, slice].join('\n\n')
}

export function appendRunContractToDelegationInstruction(
  instruction: string,
  runContract?: AgentRunContract,
): string {
  const contractPrompt = formatRunContractForPrompt(runContract)
  if (!contractPrompt) return instruction
  return [
    contractPrompt,
    '[Delegated user task]',
    instruction,
  ].join('\n\n')
}

const DELEGATION_FINDINGS_FENCE = 'findings'

/**
 * Parse a structured findings block a delegate emitted in its completion
 * comment. This is a self-describing fenced-JSON protocol block (like the
 * ANSWER: stem), NOT free-text extraction — findings are only accepted when the
 * delegate deliberately emitted the block. Missing/invalid block -> undefined so
 * the caller falls back to the legacy text-only path. PLAN_065 T5.
 */
export function parseDelegationFindings(commentBody: string): SubagentFindings | undefined {
  const match = commentBody.match(
    new RegExp('```' + DELEGATION_FINDINGS_FENCE + '\\s*\\n([\\s\\S]*?)```'),
  )
  if (!match) return undefined
  let parsed: unknown
  try {
    parsed = JSON.parse(match[1]!.trim())
  } catch {
    return undefined
  }
  if (!parsed || typeof parsed !== 'object') return undefined
  const record = parsed as Record<string, unknown>
  if (typeof record.sessionId !== 'string') return undefined
  const evidenceRecord = (record.evidence ?? {}) as Record<string, unknown>
  const empty = emptyEvidenceLedger()
  const evidenceBucket = (key: keyof typeof empty) =>
    Array.isArray(evidenceRecord[key]) ? (evidenceRecord[key] as typeof empty[typeof key]) : empty[key]
  return {
    sessionId: record.sessionId,
    category: typeof record.category === 'string' ? record.category : 'general',
    evidence: {
      sourceReads: evidenceBucket('sourceReads'),
      sourceSearches: evidenceBucket('sourceSearches'),
      artifactWrites: evidenceBucket('artifactWrites'),
      artifactReadBacks: evidenceBucket('artifactReadBacks'),
      validationRuns: evidenceBucket('validationRuns'),
      errors: evidenceBucket('errors'),
    },
    failedAttempts: Array.isArray(record.failedAttempts)
      ? (record.failedAttempts as SubagentFindings['failedAttempts'])
      : [],
    openQuestions: Array.isArray(record.openQuestions)
      ? (record.openQuestions as SubagentFindings['openQuestions'])
      : [],
  }
}

export interface DelegationResult {
  delegationId: string
  targetDevice: string
  status: 'dispatched' | 'picked_up' | 'completed' | 'failed' | 'timeout' | 'cancelled'
  result?: string
}

export interface DelegationTask extends DelegationResult {
  executionId?: string
  instruction: string
}

export interface DelegationStatusUpdate {
  delegationId: string
  targetDevice: string
  status: 'picked_up' | 'completed' | 'failed' | 'timeout' | 'cancelled'
  message?: string
  claimId?: string
  generation?: number
  /** Authenticated writer device id, when the comment is signed. */
  deviceId?: string
  signature?: string
}

export interface DelegationQuestionInput {
  delegationId: string
  targetDevice: string
  sessionId: string
  prompt: string
  choices?: string[]
}

export interface DelegationQuestionAnswerInput {
  delegationId: string
  questionId: string
  answer: string
}

export interface DelegationClaim {
  delegationId: string
  claimId: string
  targetDevice: string
  message?: string
  createdAt?: string
  /** Authenticated writer device id, when the comment is signed. */
  deviceId?: string
  signature?: string
}

export interface DelegationClaimAttemptResult {
  claimId: string
  claimed: boolean
  source: 'gateway' | 'comments'
  generation?: number
}

export interface DelegationClaimRenewalResult {
  renewed: boolean
  source: 'gateway' | 'comments'
  generation?: number
}

export interface DelegationClaimReleaseResult {
  released: boolean
  source: 'gateway' | 'comments'
}

interface DelegationClaimRelease {
  delegationId: string
  claimId: string
  targetDevice: string
  deviceId?: string
  signature?: string
}

interface GatewayClaimApi {
  claimDelegation: (
    delegationId: string,
    input: {
      claimId: string
      targetDevice: string
      ttlMs?: number
      generation?: number
    },
  ) => Promise<{
    claimed: boolean
    activeClaim: {
      claimId: string
      generation?: number
    } | null
  }>
  getDelegationClaim: (
    delegationId: string,
  ) => Promise<{
    claimId: string
    generation?: number
  } | null>
  releaseDelegationClaim: (
    delegationId: string,
    input: {
      claimId: string
      targetDevice?: string
    },
  ) => Promise<{
    released: boolean
    activeClaim: {
      claimId: string
    } | null
  }>
}

function parseDelegationDispatch(
  body: string,
): {
  delegationId: string
  executionId?: string
  targetDevice: string
  instruction: string
} | null {
  const match = body.match(
    /^\[delegation:([^\]]+)\](?:\[execution:([^\]]+)\])?\[target:([^\]]+)\]\s*([\s\S]*)$/,
  )
  if (!match) {
    return null
  }

  return {
    delegationId: match[1]!,
    executionId: match[2] || undefined,
    targetDevice: match[3]!,
    instruction: match[4] ?? '',
  }
}

function parseDelegationStatusUpdate(
  body: string,
): DelegationStatusUpdate | null {
  const match = body.match(
    /^\[delegation:([^\]]+)\]\[status:(picked_up|completed|failed|timeout|cancelled)\]\[target:([^\]]+)\](?:\[generation:(\d+)\])?(?:\[device:([^\]]+)\]\[sig:([^\]]+)\])?\s*([\s\S]*)$/,
  )
  if (!match) {
    return null
  }

  return {
    delegationId: match[1]!,
    status: match[2]! as DelegationStatusUpdate['status'],
    targetDevice: match[3]!,
    generation: match[4] ? Number(match[4]) : undefined,
    deviceId: match[5] || undefined,
    signature: match[6] || undefined,
    message: match[7] || undefined,
  }
}

function parseDelegationClaim(
  body: string,
): Omit<DelegationClaim, 'createdAt'> | null {
  const match = body.match(
    /^\[delegation:([^\]]+)\]\[claim:([^\]]+)\]\[target:([^\]]+)\](?:\[device:([^\]]+)\]\[sig:([^\]]+)\])?\s*([\s\S]*)$/,
  )
  if (!match) {
    return null
  }

  return {
    delegationId: match[1]!,
    claimId: match[2]!,
    targetDevice: match[3]!,
    deviceId: match[4] || undefined,
    signature: match[5] || undefined,
    message: match[6] || undefined,
  }
}

function parseDelegationClaimRelease(
  body: string,
): DelegationClaimRelease | null {
  const match = body.match(
    /^\[delegation:([^\]]+)\]\[claim_release:([^\]]+)\]\[target:([^\]]+)\](?:\[device:([^\]]+)\]\[sig:([^\]]+)\])?\s*([\s\S]*)$/,
  )
  if (!match) {
    return null
  }

  return {
    delegationId: match[1]!,
    claimId: match[2]!,
    targetDevice: match[3]!,
    deviceId: match[4] || undefined,
    signature: match[5] || undefined,
  }
}

function parseDelegationQuestionAnswer(
  body: string,
): { delegationId: string; questionId: string; answer: string; deviceId?: string; signature?: string } | null {
  const match = body.match(
    /^\[delegation:([^\]]+)\]\[answer:([^\]]+)\](?:\[device:([^\]]+)\]\[sig:([^\]]+)\])?\s*([\s\S]*)$/,
  )
  if (!match) {
    return null
  }

  return {
    delegationId: match[1]!,
    questionId: match[2]!,
    deviceId: match[3] || undefined,
    signature: match[4] || undefined,
    answer: match[5] ?? '',
  }
}

function parseDelegationCancel(
  body: string,
): { delegationId: string; message?: string; deviceId?: string; signature?: string } | null {
  const match = body.match(
    /^\[delegation:([^\]]+)\]\[cancel\](?:\[device:([^\]]+)\]\[sig:([^\]]+)\])?\s*([\s\S]*)$/,
  )
  if (!match) {
    return null
  }

  return {
    delegationId: match[1]!,
    deviceId: match[2] || undefined,
    signature: match[3] || undefined,
    message: match[4] || undefined,
  }
}

function isClaimFresh(
  claim: DelegationClaim,
  claimTtlMs: number,
  now = Date.now(),
): boolean {
  if (!claim.createdAt) {
    return true
  }

  const createdAt = new Date(claim.createdAt).getTime()
  if (Number.isNaN(createdAt)) {
    return true
  }

  return now - createdAt <= claimTtlMs
}

function isUnavailableGatewayClaimApi(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)
  return /^40(4|5):/.test(message) || /^501:/.test(message)
}

function getGatewayClaimApi(client: GatewayClient): GatewayClaimApi | null {
  const candidate = client as Partial<GatewayClaimApi>
  if (
    typeof candidate.claimDelegation === 'function'
    && typeof candidate.getDelegationClaim === 'function'
    && typeof candidate.releaseDelegationClaim === 'function'
  ) {
    return candidate as GatewayClaimApi
  }
  return null
}

function generationResult(
  activeClaim: { generation?: number } | null,
): { generation?: number } {
  return typeof activeClaim?.generation === 'number'
    ? { generation: activeClaim.generation }
    : {}
}

interface DelegationCommentVerifier {
  verifyClaim(claim: Omit<DelegationClaim, 'createdAt'>): boolean
  verifyRelease(release: DelegationClaimRelease): boolean
}

function buildActiveClaimsByDelegation(
  comments: Array<{ body: string; createdAt?: string }>,
  claimTtlMs: number,
  now: number,
  verifier?: DelegationCommentVerifier,
): Map<string, DelegationClaim[]> {
  const activeClaims = new Map<string, Map<string, { claim: DelegationClaim; lastSeenAt?: string }>>()

  for (const comment of comments) {
    const claim = parseDelegationClaim(comment.body)
    if (claim) {
      // Fail-closed: an unsigned or badly-signed claim from an unpaired/forged
      // device is ignored so it can neither win nor block a delegation.
      if (verifier && !verifier.verifyClaim(claim)) {
        continue
      }
      const claimsForDelegation = activeClaims.get(claim.delegationId) ?? new Map()
      const existing = claimsForDelegation.get(claim.claimId)
      claimsForDelegation.set(claim.claimId, {
        claim: {
          ...claim,
          createdAt: existing?.claim.createdAt ?? comment.createdAt,
        },
        lastSeenAt: comment.createdAt ?? existing?.lastSeenAt,
      })
      activeClaims.set(claim.delegationId, claimsForDelegation)
      continue
    }

    const release = parseDelegationClaimRelease(comment.body)
    if (!release) {
      continue
    }
    // Only an authenticated release may drop a claim; a forged release must not
    // be able to evict the legitimate holder.
    if (verifier && !verifier.verifyRelease(release)) {
      continue
    }

    activeClaims.get(release.delegationId)?.delete(release.claimId)
  }

  const result = new Map<string, DelegationClaim[]>()
  for (const [delegationId, claims] of activeClaims.entries()) {
    const active = Array.from(claims.values())
      .filter(({ claim, lastSeenAt }) =>
        isClaimFresh({ ...claim, createdAt: lastSeenAt ?? claim.createdAt }, claimTtlMs, now))
      .map(({ claim }) => claim)
      .sort((left, right) => {
        const leftTime = new Date(left.createdAt ?? 0).getTime()
        const rightTime = new Date(right.createdAt ?? 0).getTime()
        return leftTime - rightTime
      })

    if (active.length > 0) {
      result.set(delegationId, active)
    }
  }

  return result
}

export class TaskDelegator {
  private gatewayClient: GatewayClient
  private config: SepilotdConfig
  private readonly signer?: DelegationSigner
  private readonly keyResolver?: DelegationKeyResolver
  private readonly requireSignature: boolean

  constructor(
    gatewayClient: GatewayClient,
    config: SepilotdConfig,
    security: DelegatorSecurityOptions = {},
  ) {
    this.gatewayClient = gatewayClient
    this.config = config
    this.signer = security.signer
    this.keyResolver = security.keyResolver
    this.requireSignature = security.requireSignature ?? false
  }

  /**
   * Whether inbound assertions must be verified. Only active when a key resolver
   * is available AND the operator opted in — otherwise verification is skipped so
   * an un-upgraded peer is not cut off.
   */
  private get verificationEnabled(): boolean {
    return this.requireSignature && !!this.keyResolver
  }

  /** Marker suffix authenticating an outbound assertion, or '' when unsigned. */
  private signMarkerSuffix(fields: Omit<DelegationAssertionFields, 'deviceId'>): string {
    if (!this.signer) return ''
    const payload = buildDelegationAssertionPayload({ ...fields, deviceId: this.signer.deviceId })
    const signature = this.signer.sign(payload)
    return `[device:${this.signer.deviceId}][sig:${signature}]`
  }

  private commentVerifier(): DelegationCommentVerifier | undefined {
    if (!this.verificationEnabled || !this.keyResolver) return undefined
    const resolver = this.keyResolver
    return {
      verifyClaim: (claim) =>
        verifyDelegationAssertion(
          resolver,
          {
            kind: 'claim',
            delegationId: claim.delegationId,
            deviceId: claim.deviceId ?? '',
            claimId: claim.claimId,
          },
          claim.signature,
        ),
      verifyRelease: (release) =>
        verifyDelegationAssertion(
          resolver,
          {
            kind: 'claim_release',
            delegationId: release.delegationId,
            deviceId: release.deviceId ?? '',
            claimId: release.claimId,
          },
          release.signature,
        ),
    }
  }

  private isStatusUpdateAuthentic(update: DelegationStatusUpdate): boolean {
    if (!this.verificationEnabled || !this.keyResolver) return true
    return verifyDelegationAssertion(
      this.keyResolver,
      {
        kind: 'status',
        delegationId: update.delegationId,
        deviceId: update.deviceId ?? '',
        claimId: update.claimId,
        detail: update.status,
      },
      update.signature,
    )
  }

  /**
   * Delegate a task to another device via Gateway (GitHub Issue comment).
   * Creates a ticket for the target device to pick up.
   */
  async delegate(request: DelegationRequest): Promise<DelegationResult> {
    const delegationId = randomUUID()
    const targetDevice = request.targetDevice ?? 'pending'
    const marker = request.executionId
      ? `[delegation:${delegationId}][execution:${request.executionId}][target:${targetDevice}]`
      : `[delegation:${delegationId}][target:${targetDevice}]`

    try {
      // Use Gateway to create a ticket (Issue) for delegation. Ride the scoped
      // board slice down inside the instruction payload so the delegate starts
      // from the parent's structured state (PLAN_065 T5).
      const instructionWithContract = appendRunContractToDelegationInstruction(
        request.instruction,
        request.runContract,
      )
      const instructionWithBoard = appendBoardSliceToDelegationInstruction(
        instructionWithContract,
        request.boardSlice,
      )
      // Do NOT swallow the gateway write failure: an un-written delegation that
      // reports 'dispatched' is a silent black-hole (the target never sees the
      // task). Let it propagate so the status is reported as failed.
      await this.gatewayClient.addComment(
        'delegation',  // Special ticket for inter-device communication
        `${marker} ${instructionWithBoard}`,
        'general',
      )

      return {
        delegationId,
        targetDevice,
        status: 'dispatched',
      }
    } catch (error) {
      return {
        delegationId,
        targetDevice: request.targetDevice ?? 'unknown',
        status: 'failed',
        result: error instanceof Error ? error.message : String(error),
      }
    }
  }

  async findDelegationByExecutionId(
    executionId: string,
  ): Promise<DelegationResult | null> {
    try {
      const tasks = await this.listDelegations()
      const match = tasks.find(task => task.executionId === executionId)
      return match ?? null
    } catch {
      return null
    }
  }

  async cancelDelegation(executionId: string): Promise<DelegationResult | null> {
    const delegation = await this.findDelegationByExecutionId(executionId)
    if (!delegation) {
      return null
    }

    const cancelSuffix = this.signMarkerSuffix({ kind: 'cancel', delegationId: delegation.delegationId })
    await this.gatewayClient.addComment(
      'delegation',
      `[delegation:${delegation.delegationId}][cancel]${cancelSuffix} Cancel requested for execution ${executionId}`,
      'progress',
    )
    return {
      ...delegation,
      status: 'cancelled',
      result: 'Cancellation requested.',
    }
  }

  async listDelegations(): Promise<DelegationTask[]> {
    const comments = await this.gatewayClient.getComments('delegation')
    const delegations = new Map<string, DelegationTask>()

    for (const comment of comments) {
      const dispatch = parseDelegationDispatch(comment.body)
      if (dispatch) {
        delegations.set(dispatch.delegationId, {
          delegationId: dispatch.delegationId,
          executionId: dispatch.executionId,
          targetDevice: dispatch.targetDevice,
          instruction: dispatch.instruction,
          status: 'dispatched',
        })
        continue
      }

      const update = parseDelegationStatusUpdate(comment.body)
      if (!update) {
        continue
      }

      // Fail-closed: a forged/unsigned status comment cannot flip a delegation
      // to completed/failed/cancelled when signature enforcement is active.
      if (!this.isStatusUpdateAuthentic(update)) {
        continue
      }

      const existing = delegations.get(update.delegationId)
      if (!existing) {
        continue
      }

      delegations.set(update.delegationId, {
        ...existing,
        targetDevice: update.targetDevice,
        status: update.status,
        result: update.message,
      })
    }

    return Array.from(delegations.values())
  }

  async listPendingDelegationsForDevice(
    deviceAliases: string[],
    options?: {
      claimTtlMs?: number
      now?: number
    },
  ): Promise<DelegationTask[]> {
    const comments = await this.gatewayClient.getComments('delegation')
    const candidates = new Set(
      deviceAliases
        .map(alias => alias.trim())
        .filter(Boolean),
    )
    const claimTtlMs = options?.claimTtlMs ?? 120_000
    const now = options?.now ?? Date.now()
    const delegations = await this.listDelegations()
    const activeClaims = buildActiveClaimsByDelegation(comments, claimTtlMs, now, this.commentVerifier())
    // When verification is on, authorize a foreign claim by its authenticated
    // device id, not the free-text [target] name a forger controls.
    const claimMatchesDevice = (claim: DelegationClaim): boolean =>
      this.verificationEnabled
        ? !!claim.deviceId && candidates.has(claim.deviceId)
        : candidates.has(claim.targetDevice)

    return delegations.filter((delegation) =>
      delegation.status === 'dispatched'
      && (
        delegation.targetDevice === 'pending'
        || candidates.has(delegation.targetDevice)
      ),
    ).filter((delegation) => {
      const claim = (activeClaims.get(delegation.delegationId) ?? []).find((candidate) =>
        isClaimFresh(candidate, claimTtlMs, now),
      )
      return !claim || claimMatchesDevice(claim)
    })
  }

  async claimDelegation(input: {
    delegationId: string
    targetDevice: string
    message?: string
  }): Promise<string> {
    const claimId = randomUUID()
    await this.writeClaimComment({
      delegationId: input.delegationId,
      claimId,
      targetDevice: input.targetDevice,
      message: input.message,
    })
    return claimId
  }

  private async writeClaimComment(input: {
    delegationId: string
    claimId: string
    targetDevice: string
    message?: string
  }): Promise<void> {
    const marker = `[delegation:${input.delegationId}][claim:${input.claimId}][target:${input.targetDevice}]${
      this.signMarkerSuffix({ kind: 'claim', delegationId: input.delegationId, claimId: input.claimId })
    }`

    await this.gatewayClient.addComment(
      'delegation',
      `${marker}${input.message ? ` ${input.message}` : ''}`,
      'progress',
    )
  }

  async tryAcquireDelegationClaim(input: {
    delegationId: string
    targetDevice: string
    ttlMs?: number
  }): Promise<DelegationClaimAttemptResult> {
    const claimId = randomUUID()
    const gatewayClaimApi = getGatewayClaimApi(this.gatewayClient)

    if (gatewayClaimApi) {
      try {
        const result = await gatewayClaimApi.claimDelegation(input.delegationId, {
          claimId,
          targetDevice: input.targetDevice,
          ttlMs: input.ttlMs,
        })
        const activeClaim = result.activeClaim
        return {
          claimId,
          claimed: result.claimed && activeClaim?.claimId === claimId,
          source: 'gateway',
          ...generationResult(activeClaim),
        }
      } catch (error) {
        if (!isUnavailableGatewayClaimApi(error)) {
          throw error
        }
      }
    }

    const fallbackClaimId = await this.claimDelegation({
      delegationId: input.delegationId,
      targetDevice: input.targetDevice,
      message: `Claimed by ${input.targetDevice}`,
    })
    const claimed = await this.verifyDelegationClaim(
      input.delegationId,
      fallbackClaimId,
    )
    return {
      claimId: fallbackClaimId,
      claimed,
      source: 'comments',
    }
  }

  async renewDelegationClaim(input: {
    delegationId: string
    claimId: string
    targetDevice: string
    ttlMs?: number
    generation?: number
  }): Promise<DelegationClaimRenewalResult> {
    const gatewayClaimApi = getGatewayClaimApi(this.gatewayClient)

    if (gatewayClaimApi) {
      try {
        const result = await gatewayClaimApi.claimDelegation(input.delegationId, {
          claimId: input.claimId,
          targetDevice: input.targetDevice,
          ttlMs: input.ttlMs,
          generation: input.generation,
        })
        const activeClaim = result.activeClaim
        return {
          renewed: result.claimed && activeClaim?.claimId === input.claimId,
          source: 'gateway',
          ...generationResult(activeClaim),
        }
      } catch (error) {
        if (!isUnavailableGatewayClaimApi(error)) {
          throw error
        }
      }
    }

    await this.writeClaimComment({
      delegationId: input.delegationId,
      claimId: input.claimId,
      targetDevice: input.targetDevice,
      message: `Renewed by ${input.targetDevice}`,
    })
    return {
      renewed: await this.verifyDelegationClaim(input.delegationId, input.claimId, {
        claimTtlMs: input.ttlMs,
        generation: input.generation,
      }),
      source: 'comments',
    }
  }

  async releaseDelegationClaim(input: {
    delegationId: string
    claimId: string
    targetDevice: string
  }): Promise<DelegationClaimReleaseResult> {
    const gatewayClaimApi = getGatewayClaimApi(this.gatewayClient)

    if (gatewayClaimApi) {
      try {
        const result = await gatewayClaimApi.releaseDelegationClaim(input.delegationId, {
          claimId: input.claimId,
          targetDevice: input.targetDevice,
        })
        return {
          released: result.released,
          source: 'gateway',
        }
      } catch (error) {
        if (!isUnavailableGatewayClaimApi(error)) {
          throw error
        }
      }
    }

    const releaseSuffix = this.signMarkerSuffix({
      kind: 'claim_release',
      delegationId: input.delegationId,
      claimId: input.claimId,
    })
    await this.gatewayClient.addComment(
      'delegation',
      `[delegation:${input.delegationId}][claim_release:${input.claimId}][target:${input.targetDevice}]${releaseSuffix} Released by ${input.targetDevice}`,
      'progress',
    )
    return {
      released: !(await this.verifyDelegationClaim(input.delegationId, input.claimId)),
      source: 'comments',
    }
  }

  async verifyDelegationClaim(
    delegationId: string,
    claimId: string,
    options?: {
      claimTtlMs?: number
      generation?: number
    },
  ): Promise<boolean> {
    const gatewayClaimApi = getGatewayClaimApi(this.gatewayClient)

    if (gatewayClaimApi) {
      try {
        const activeClaim = await gatewayClaimApi.getDelegationClaim(delegationId)
        if (activeClaim) {
          if (activeClaim.claimId !== claimId) {
            return false
          }
          if (options?.generation !== undefined) {
            return activeClaim.generation === options.generation
          }
          return true
        }
      } catch (error) {
        if (!isUnavailableGatewayClaimApi(error)) {
          throw error
        }
      }
    }

    const terminal = await this.listDelegations()
    const task = terminal.find(candidate => candidate.delegationId === delegationId)
    if (!task || task.status !== 'dispatched') {
      return false
    }

    const claims = await this.listActiveClaims(delegationId, options)
    return claims[0]?.claimId === claimId
  }

  async listActiveClaims(
    delegationId: string,
    options?: {
      claimTtlMs?: number
    },
  ): Promise<DelegationClaim[]> {
    const comments = await this.gatewayClient.getComments('delegation')
    const claimTtlMs = options?.claimTtlMs ?? 120_000
    const now = Date.now()
    return buildActiveClaimsByDelegation(comments, claimTtlMs, now, this.commentVerifier()).get(delegationId) ?? []
  }

  private async canWriteDelegationStatus(update: DelegationStatusUpdate): Promise<boolean> {
    if (!update.claimId || update.generation === undefined) {
      return true
    }

    const gatewayClaimApi = getGatewayClaimApi(this.gatewayClient)
    if (!gatewayClaimApi) {
      return true
    }

    try {
      const activeClaim = await gatewayClaimApi.getDelegationClaim(update.delegationId)
      return activeClaim?.claimId === update.claimId
        && activeClaim.generation === update.generation
    } catch (error) {
      if (!isUnavailableGatewayClaimApi(error)) {
        throw error
      }
      return true
    }
  }

  async recordDelegationStatus(update: DelegationStatusUpdate): Promise<void> {
    if (!(await this.canWriteDelegationStatus(update))) {
      return
    }

    const marker = `[delegation:${update.delegationId}][status:${update.status}][target:${update.targetDevice}]${
      update.generation !== undefined ? `[generation:${update.generation}]` : ''
    }${this.signMarkerSuffix({
      kind: 'status',
      delegationId: update.delegationId,
      claimId: update.claimId,
      detail: update.status,
    })}`
    const type = update.status === 'failed' || update.status === 'cancelled'
      ? 'error'
      : update.status === 'completed'
        ? 'result'
        : 'progress'

    await this.gatewayClient.addComment(
      'delegation',
      `${marker}${update.message ? ` ${update.message}` : ''}`,
      type,
    )
  }

  async postDelegationQuestion(input: DelegationQuestionInput): Promise<string> {
    const questionId = randomUUID()
    const marker = `[delegation:${input.delegationId}][question:${questionId}][target:${input.targetDevice}][session:${input.sessionId}]`
    const choices = input.choices?.length
      ? `\nChoices:\n${input.choices.map(choice => `- ${choice}`).join('\n')}`
      : ''

    await this.gatewayClient.addComment(
      'delegation',
      `${marker} ${input.prompt}${choices}`,
      'progress',
    )
    return questionId
  }

  async getDelegationQuestionAnswer(
    delegationId: string,
    questionId: string,
  ): Promise<string | null> {
    const comments = await this.gatewayClient.getComments('delegation')
    for (const comment of comments.slice().reverse()) {
      const answer = parseDelegationQuestionAnswer(comment.body)
      if (
        answer
        && answer.delegationId === delegationId
        && answer.questionId === questionId
      ) {
        if (
          this.verificationEnabled
          && this.keyResolver
          && !verifyDelegationAssertion(
            this.keyResolver,
            { kind: 'answer', delegationId, deviceId: answer.deviceId ?? '', detail: questionId },
            answer.signature,
          )
        ) {
          continue
        }
        return answer.answer
      }
    }
    return null
  }

  async getDelegationCancel(
    delegationId: string,
  ): Promise<{ message?: string } | null> {
    const comments = await this.gatewayClient.getComments('delegation')
    for (const comment of comments.slice().reverse()) {
      const cancel = parseDelegationCancel(comment.body)
      if (cancel?.delegationId === delegationId) {
        if (
          this.verificationEnabled
          && this.keyResolver
          && !verifyDelegationAssertion(
            this.keyResolver,
            { kind: 'cancel', delegationId, deviceId: cancel.deviceId ?? '' },
            cancel.signature,
          )
        ) {
          continue
        }
        return { message: cancel.message }
      }
    }
    return null
  }

  async answerDelegationQuestion(input: DelegationQuestionAnswerInput): Promise<void> {
    const answerSuffix = this.signMarkerSuffix({
      kind: 'answer',
      delegationId: input.delegationId,
      detail: input.questionId,
    })
    await this.gatewayClient.addComment(
      'delegation',
      `[delegation:${input.delegationId}][answer:${input.questionId}]${answerSuffix} ${input.answer}`,
      'progress',
    )
  }

  /** Check if a task should be delegated based on device capabilities */
  shouldDelegate(_instruction: string, requiredCapabilities: string[]): boolean {
    const myRole = this.config.device.role
    const myCapabilities = new Set(['terminal', 'filesystem'])

    // Add capabilities based on role
    if (myRole === 'server') {
      myCapabilities.add('docker')
      myCapabilities.add('gpu')
      myCapabilities.add('kubernetes')
    }
    if (myRole === 'desktop') {
      myCapabilities.add('browser')
      myCapabilities.add('gui')
    }

    // Check if we have all required capabilities
    for (const cap of requiredCapabilities) {
      if (!myCapabilities.has(cap)) return true  // Should delegate
    }
    return false
  }
}
