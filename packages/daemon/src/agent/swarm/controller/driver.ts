import { setTimeout as sleep } from 'node:timers/promises'
import type { ToolResult } from '../../../tools/registry.js'
import type {
  AgentRunContract,
  SwarmAgentHandle,
  SwarmAgentRecoveryHint,
  SwarmAgentRecoveryStep,
  SwarmAgentStartupEvidence,
  SwarmAgentStartupState,
  SwarmEvent,
} from '@sepilotd/core'
import type { SwarmToolContext } from '../tools/context.js'
import type { SwarmRun } from '../run/swarm-run.js'
import { TmuxSwarmAgentRuntimeAdapter, type SwarmAgentRuntimeAdapter } from '../launcher/runtime-adapter.js'
import { getAgentConfig } from '../config/agents.js'
import { IdleDetector } from '../tmux/idle-detector.js'
import {
  createSwarmPromptAcceptedEvidence,
  createSwarmPromptAcceptanceTimeoutEvidence,
  createSwarmPromptSentEvidence,
  createSwarmStartupFailureEvidence,
  createSwarmStartupOutputEvidence,
} from '../run/startup-evidence.js'
import { formatRunContractForPrompt } from '../../task-contract.js'

export type SwarmDriveTurnStatus =
  | 'idle'
  | 'done'
  | 'blocked'
  | 'timeout'
  | 'dead'

export type SwarmDriveStatus =
  | 'done'
  | 'max_turns'
  | 'blocked'
  | 'timeout'
  | 'dead'

export type SwarmDriveRequiredActionType =
  | 'trust_prompt'
  | 'tool_permission'

export interface SwarmDriveRequiredAction {
  type: SwarmDriveRequiredActionType
  lifecycleState: Extract<SwarmAgentStartupState, 'trust_required' | 'tool_permission_required'>
  action: Extract<SwarmAgentRecoveryStep, 'review_trust_prompt' | 'review_tool_permission'>
  reason: string
  suggested_input: string
  output_preview?: string
  snapshot: string
  recovery_hint?: SwarmAgentRecoveryHint
}

export interface SwarmDriveInput {
  agent_handle?: string
  prompt: string
  followups?: string[]
  continue_prompt?: string
  max_turns?: number
  stop_patterns?: string[]
  timeout_sec?: number
  stable_sec?: number
  min_wait_ms?: number
  poll_ms?: number
  timeout_retries?: number
  runContract?: AgentRunContract
}

export interface SwarmDriveTurn {
  turn: number
  prompt: string
  status: SwarmDriveTurnStatus
  reason: string
  output: string
  snapshot: string
  durationMs: number
  required_action?: SwarmDriveRequiredAction
}

export interface SwarmDriveResult {
  runId: string
  handle: string
  status: SwarmDriveStatus
  turns: SwarmDriveTurn[]
  lastOutput: string
  nextPrompt?: string
  recoveryHint?: SwarmAgentRecoveryHint
  required_action?: SwarmDriveRequiredAction
}

interface WaitOptions {
  timeoutMs: number
  stableSeconds: number
  minWaitMs: number
  pollMs: number
}

interface WaitOutcome {
  status: SwarmDriveTurnStatus
  reason: string
  output: string
  snapshot: string
  required_action?: SwarmDriveRequiredAction
}

function nowMs(): number { return Date.now() }
function durationSince(start: number): number { return Date.now() - start }

function runtime(ctx: SwarmToolContext): SwarmAgentRuntimeAdapter {
  return ctx.runtime ?? new TmuxSwarmAgentRuntimeAdapter(ctx.pool, ctx.launcher)
}

export function formatSwarmWorkerPrompt(
  prompt: string,
  runContract?: AgentRunContract,
): string {
  const contractPrompt = formatRunContractForPrompt(runContract)
  if (!contractPrompt) return prompt
  return [
    contractPrompt,
    '[Swarm worker task]',
    prompt,
  ].join('\n\n')
}

function emit(run: SwarmRun, event: SwarmEvent): void {
  run.emit('event', event)
}

function emitMessage(run: SwarmRun, text: string): void {
  emit(run, { type: 'supervisor.message', runId: run.id, text, ts: nowMs() })
}

function emitSnapshot(run: SwarmRun, handle: string, text: string): void {
  emit(run, { type: 'pane.snapshot', runId: run.id, handle, text, ts: nowMs() })
}

function requiredActionFromEvidence(
  evidence: SwarmAgentStartupEvidence,
  snapshot: string,
): SwarmDriveRequiredAction | undefined {
  if (evidence.lifecycleState !== 'trust_required'
    && evidence.lifecycleState !== 'tool_permission_required') {
    return undefined
  }
  if (evidence.lifecycleState === 'trust_required') {
    return {
      type: 'trust_prompt',
      lifecycleState: evidence.lifecycleState,
      action: 'review_trust_prompt',
      reason: evidence.recoveryHint?.reason ?? 'worker is waiting for a workspace trust decision',
      suggested_input: 'attach to the tmux pane and answer the trust prompt only after reviewing the workspace',
      output_preview: evidence.lastOutputPreview,
      snapshot,
      recovery_hint: evidence.recoveryHint,
    }
  }
  return {
    type: 'tool_permission',
    lifecycleState: evidence.lifecycleState,
    action: 'review_tool_permission',
    reason: evidence.recoveryHint?.reason ?? 'worker is waiting for a tool permission decision',
    suggested_input: 'attach to the tmux pane and approve or deny the requested tool only after reviewing the command',
    output_preview: evidence.lastOutputPreview,
    snapshot,
    recovery_hint: evidence.recoveryHint,
  }
}

function activeBlockerEvidence(handle: SwarmAgentHandle): SwarmAgentStartupEvidence | null {
  const evidence = handle.startupEvidence
  if (evidence?.lifecycleState !== 'trust_required'
    && evidence?.lifecycleState !== 'tool_permission_required') {
    return null
  }
  return evidence
}

async function existingBlockerOutcome(
  ctx: SwarmToolContext,
  run: SwarmRun,
  handle: SwarmAgentHandle,
): Promise<WaitOutcome | null> {
  const evidence = activeBlockerEvidence(handle)
  if (!evidence) return null
  run.setStatus(handle.handle, 'blocked')
  const rt = runtime(ctx)
  const snapshot = await rt.capture(handle, 5000)
    .catch(() => evidence.lastOutputPreview ?? '')
  emitSnapshot(run, handle.handle, snapshot)
  return {
    status: 'blocked',
    reason: evidence.lifecycleState,
    output: snapshot || (evidence.lastOutputPreview ?? ''),
    snapshot,
    required_action: requiredActionFromEvidence(evidence, snapshot),
  }
}

async function waitForInteractiveLeaseRelease(input: {
  run: SwarmRun
  handle: SwarmAgentHandle
  turn: number
  pollMs: number
  signal?: AbortSignal
}): Promise<boolean> {
  const { run, handle, turn, pollMs, signal } = input
  let logged = false
  while (run.isInteractiveHeld(handle.handle)) {
    if (!logged) {
      const owner = run.interactiveLeaseOwner(handle.handle) ?? 'unknown'
      emitMessage(run, `drive turn ${turn}: paused_for_human (${handle.handle}, owner=${owner})`)
      run.setStatus(handle.handle, 'idle')
      logged = true
    }
    if (signal?.aborted) return false
    await sleep(pollMs)
  }
  if (logged) {
    emitMessage(run, `drive turn ${turn}: resumed_after_human (${handle.handle})`)
  }
  return true
}

function recordOutputBlocker(
  run: SwarmRun,
  handle: SwarmAgentHandle,
  text: string,
): SwarmAgentStartupEvidence | null {
  const evidence = createSwarmStartupOutputEvidence(handle, text)
  if (!evidence) return null
  run.recordStartupEvidence(handle.handle, evidence)
  run.setStatus(handle.handle, 'blocked')
  return evidence
}

function recordPromptAccepted(run: SwarmRun, handle: SwarmAgentHandle, text: string): void {
  const evidence = createSwarmPromptAcceptedEvidence(handle, text)
  if (evidence) run.recordStartupEvidence(handle.handle, evidence)
}

function recordPromptTimeout(
  run: SwarmRun,
  handle: SwarmAgentHandle,
  text: string,
): SwarmAgentStartupEvidence | null {
  const evidence = createSwarmPromptAcceptanceTimeoutEvidence(handle, text)
  if (!evidence) return null
  run.recordStartupEvidence(handle.handle, evidence)
  run.setStatus(handle.handle, 'blocked')
  return evidence
}

function recordAgentDead(
  run: SwarmRun,
  handle: SwarmAgentHandle,
  reason: string,
): SwarmAgentStartupEvidence {
  const evidence = createSwarmStartupFailureEvidence({
    handle: handle.handle,
    agent: handle.agent,
    cwd: handle.cwd,
    runtime: handle.runtime ?? 'tmux',
    startedAt: handle.spawnedAt,
    error: new Error(`transport disconnected: ${reason}`),
  })
  run.recordStartupEvidence(handle.handle, evidence)
  run.setStatus(handle.handle, 'dead')
  return evidence
}

function isDead(ctx: SwarmToolContext, handle: SwarmAgentHandle): boolean {
  if (handle.status === 'dead' || handle.status === 'killed') return true
  if (handle.runtime && handle.runtime !== 'tmux') return false
  return !ctx.pool.isAlive(handle.tmuxSessionName)
}

function numberInput(value: unknown, fallback: number, min: number, max: number): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.max(min, Math.min(max, value))
    : fallback
}

function waitOptions(input: SwarmDriveInput, handle: SwarmAgentHandle): WaitOptions {
  const cfg = getAgentConfig(handle.agent)
  return {
    timeoutMs: numberInput(input.timeout_sec, 300, 0.1, 1800) * 1000,
    stableSeconds: numberInput(input.stable_sec, cfg.stableSeconds, 0.1, 60),
    minWaitMs: numberInput(input.min_wait_ms, cfg.minWaitMs, 0, 60_000),
    pollMs: numberInput(input.poll_ms, 500, 25, 5000),
  }
}

function compilePatterns(patterns: string[] | undefined): RegExp[] {
  const source = patterns?.length
    ? patterns
    : [
        '\\bDONE\\b',
        '\\bcomplete(?:d)?\\b',
        '\\bfinished\\b',
        '작업\\s*완료',
        '완료',
      ]
  return source.map((pattern) => {
    try {
      return new RegExp(pattern, 'i')
    } catch {
      return new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
    }
  })
}

function isDone(output: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(output))
}

function contractRequiresWorkerCompletionEvidence(contract: AgentRunContract | undefined): boolean {
  return Boolean(
    contract?.requiredArtifacts?.length
    || contract?.artifactSections?.length
    || contract?.evidenceRequirements?.length,
  )
}

const WORKER_COMPLETION_EVIDENCE_PATTERN =
  /\b(?:fs\.(?:write|append|edit|read)|apply_patch|terminal\.run|browser\.(?:screenshot|click|evaluate)|validated?|verified|tests?\s+(?:pass|passed|run)|passed\s+tests?|lint(?:ed)?|typecheck(?:ed)?|screenshot|artifact|wrote|written|created|updated|modified|changed|fixed)\b|(?:검증|테스트|통과|수정|작성|생성|업데이트|스크린샷)|(?:^|[\s`])(?:\.{1,2}\/|\/|[A-Za-z0-9_.@-]+\/)[A-Za-z0-9_.@/-]+\.[A-Za-z0-9_-]{1,16}\b/iu

function workerDoneOutputHasContractEvidence(
  output: string,
  contract: AgentRunContract | undefined,
): boolean {
  if (!contractRequiresWorkerCompletionEvidence(contract)) return true
  return WORKER_COMPLETION_EVIDENCE_PATTERN.test(output)
}

function contractEvidenceFollowupPrompt(): string {
  return [
    'Your previous reply matched a completion marker, but it did not include evidence required by the run contract.',
    'Continue the task if work remains.',
    'If it is actually complete, reply with DONE plus concrete evidence: files/artifacts changed, validation/tests/browser checks run, and any remaining gaps.',
    'Do not repeat DONE by itself.',
  ].join(' ')
}

function nextPrompt(input: SwarmDriveInput, completedTurns: number): string | null {
  const followup = input.followups?.[completedTurns - 1]
  if (typeof followup === 'string' && followup.trim()) return followup
  if (input.continue_prompt?.trim()) return input.continue_prompt
  return 'Continue the task. If the work is complete, reply with DONE and a concise summary. Otherwise perform the next concrete step.'
}

function incompleteNextPrompt(
  input: SwarmDriveInput,
  completedTurns: number,
  missingContractEvidence: boolean,
): string {
  if (missingContractEvidence) return contractEvidenceFollowupPrompt()
  return nextPrompt(input, completedTurns)
    ?? 'Continue from the latest worker output. If the work is complete, reply with DONE plus concrete evidence; otherwise state the remaining blocker and perform the next concrete step.'
}

async function waitForState(input: {
  ctx: SwarmToolContext
  run: SwarmRun
  handle: SwarmAgentHandle
  options: WaitOptions
  signal?: AbortSignal
}): Promise<WaitOutcome> {
  const { ctx, run, handle, options, signal } = input
  const rt = runtime(ctx)
  if (isDead(ctx, handle)) {
    run.setStatus(handle.handle, 'dead')
    return { status: 'dead', reason: 'agent process is not alive', output: '', snapshot: '' }
  }
  const direct = rt.waitForIdle?.(handle, { timeoutMs: options.timeoutMs, signal })
  if (direct) {
    const waited = await direct
    emitSnapshot(run, handle.handle, waited.snapshot)
    const blocker = recordOutputBlocker(run, handle, waited.snapshot)
    if (blocker) {
      return {
        status: 'blocked',
        reason: blocker.lifecycleState,
        output: waited.output,
        snapshot: waited.snapshot,
        required_action: requiredActionFromEvidence(blocker, waited.snapshot),
      }
    }
    if (waited.status === 'idle') {
      run.setStatus(handle.handle, 'idle')
      recordPromptAccepted(run, handle, waited.output || waited.snapshot)
      return { status: 'idle', reason: 'idle prompt observed', output: waited.output, snapshot: waited.snapshot }
    }
    return { status: 'timeout', reason: 'wait_idle timed out', output: waited.output, snapshot: waited.snapshot }
  }

  const cfg = getAgentConfig(handle.agent)
  const detector = new IdleDetector({
    idlePatterns: cfg.idlePatterns,
    busyPatterns: cfg.busyPatterns,
    minWaitMs: options.minWaitMs,
    stableSeconds: options.stableSeconds,
  }, nowMs())
  const deadline = nowMs() + options.timeoutMs
  let lastSnapshot = ''
  while (nowMs() < deadline) {
    if (signal?.aborted) {
      return { status: 'timeout', reason: 'aborted', output: lastSnapshot, snapshot: lastSnapshot }
    }
    if (isDead(ctx, handle)) {
      run.setStatus(handle.handle, 'dead')
      return { status: 'dead', reason: 'agent process is not alive', output: '', snapshot: lastSnapshot }
    }
    const text = await rt.capture(handle, 5000)
    lastSnapshot = text
    const blocker = recordOutputBlocker(run, handle, text)
    if (blocker) {
      emitSnapshot(run, handle.handle, text)
      return {
        status: 'blocked',
        reason: blocker.lifecycleState,
        output: text,
        snapshot: text,
        required_action: requiredActionFromEvidence(blocker, text),
      }
    }
    const decision = detector.observe(text, nowMs())
    if (decision === 'idle') {
      run.setStatus(handle.handle, 'idle')
      const output = await rt.readNewOutput(handle)
      recordPromptAccepted(run, handle, output || text)
      emitSnapshot(run, handle.handle, text)
      return { status: 'idle', reason: 'idle prompt observed', output, snapshot: text }
    }
    await sleep(Math.min(options.pollMs, Math.max(0, deadline - nowMs())))
  }
  const snapshot = await rt.capture(handle, 5000).catch(() => lastSnapshot)
  return { status: 'timeout', reason: 'wait_idle timed out', output: snapshot, snapshot }
}

export async function driveSwarmAgent(
  ctx: SwarmToolContext,
  runId: string,
  input: SwarmDriveInput,
  signal?: AbortSignal,
): Promise<SwarmDriveResult> {
  const run = ctx.registry.get(runId)
  if (!run) throw new Error(`swarm run not found: ${runId}`)
  const handleId = input.agent_handle ?? run.snapshot().activeHandle
  if (!handleId) throw new Error('no active agent; pass agent_handle')
  const handle = run.getAgent(handleId)
  if (!handle) throw new Error(`unknown handle: ${handleId}`)
  const maxTurns = Math.floor(numberInput(input.max_turns, 8, 1, 100))
  const timeoutRetries = Math.floor(numberInput(input.timeout_retries, 1, 0, 5))
  const donePatterns = compilePatterns(input.stop_patterns)
  const turns: SwarmDriveTurn[] = []
  let prompt = input.prompt
  let lastOutput = ''
  const options = waitOptions(input, handle)

  for (let turn = 1; turn <= maxTurns; turn++) {
    const startedAt = nowMs()
    const rt = runtime(ctx)
    if (isDead(ctx, handle)) {
      const evidence = recordAgentDead(run, handle, 'agent process is not alive before send')
      turns.push({
        turn,
        prompt,
        status: 'dead',
        reason: 'agent process is not alive before send',
        output: '',
        snapshot: '',
        durationMs: durationSince(startedAt),
      })
      return {
        runId,
        handle: handle.handle,
        status: 'dead',
        turns,
        lastOutput,
        recoveryHint: evidence.recoveryHint,
      }
    }
    const existingBlocker = await existingBlockerOutcome(ctx, run, handle)
    if (existingBlocker) {
      turns.push({
        turn,
        prompt,
        status: 'blocked',
        reason: existingBlocker.reason,
        output: existingBlocker.output,
        snapshot: existingBlocker.snapshot,
        durationMs: durationSince(startedAt),
        required_action: existingBlocker.required_action,
      })
      emitMessage(run, `drive turn ${turn}: blocked (${existingBlocker.reason})`)
      lastOutput = existingBlocker.output || existingBlocker.snapshot
      return {
        runId,
        handle: handle.handle,
        status: 'blocked',
        turns,
        lastOutput,
        required_action: existingBlocker.required_action,
      }
    }

    const leaseReleased = await waitForInteractiveLeaseRelease({
      run,
      handle,
      turn,
      pollMs: options.pollMs,
      signal,
    })
    if (!leaseReleased) {
      turns.push({
        turn,
        prompt,
        status: 'timeout',
        reason: 'aborted while paused_for_human',
        output: lastOutput,
        snapshot: lastOutput,
        durationMs: durationSince(startedAt),
      })
      return {
        runId,
        handle: handle.handle,
        status: 'timeout',
        turns,
        lastOutput,
        recoveryHint: handle.startupEvidence?.recoveryHint,
      }
    }

    emitMessage(run, `drive turn ${turn}: send prompt to ${handle.handle}`)
    await rt.sendPrompt(handle, formatSwarmWorkerPrompt(prompt, input.runContract))
    run.setStatus(handle.handle, 'busy')
    run.recordStartupEvidence(handle.handle, createSwarmPromptSentEvidence(handle))

    let outcome: WaitOutcome | null = null
    for (let attempt = 0; attempt <= timeoutRetries; attempt++) {
      outcome = await waitForState({
        ctx,
        run,
        handle,
        options,
        signal,
      })
      if (outcome.status !== 'timeout') break
      if (attempt < timeoutRetries) {
        emitMessage(run, `drive turn ${turn}: timeout; recapturing and waiting again`)
      }
    }
    if (!outcome) throw new Error('drive wait produced no outcome')
    lastOutput = outcome.output || outcome.snapshot
    const doneMatched = outcome.status === 'idle' && isDone(lastOutput, donePatterns)
    const doneHasRequiredEvidence = doneMatched
      ? workerDoneOutputHasContractEvidence(lastOutput, input.runContract)
      : false
    const finalStatus = doneMatched && doneHasRequiredEvidence ? 'done' : outcome.status
    const missingContractEvidence = doneMatched && !doneHasRequiredEvidence
    turns.push({
      turn,
      prompt,
      status: finalStatus,
      reason: finalStatus === 'done'
        ? 'stop pattern matched'
        : missingContractEvidence
          ? 'stop pattern matched without required contract evidence'
          : outcome.reason,
      output: outcome.output,
      snapshot: outcome.snapshot,
      durationMs: durationSince(startedAt),
      required_action: outcome.required_action,
    })
    emitMessage(run, `drive turn ${turn}: ${finalStatus} (${turns[turns.length - 1].reason})`)

    if (finalStatus === 'done') return { runId, handle: handle.handle, status: 'done', turns, lastOutput }
    if (finalStatus === 'blocked') {
      return {
        runId,
        handle: handle.handle,
        status: 'blocked',
        turns,
        lastOutput,
        required_action: outcome.required_action,
      }
    }
    if (finalStatus === 'dead') {
      const evidence = recordAgentDead(run, handle, outcome.reason)
      return {
        runId,
        handle: handle.handle,
        status: 'dead',
        turns,
        lastOutput,
        recoveryHint: evidence.recoveryHint,
      }
    }
    if (finalStatus === 'timeout') {
      const timeoutEvidence = recordPromptTimeout(run, handle, outcome.snapshot || outcome.output)
      return {
        runId,
        handle: handle.handle,
        status: 'timeout',
        turns,
        lastOutput,
        recoveryHint: timeoutEvidence?.recoveryHint ?? handle.startupEvidence?.recoveryHint,
      }
    }

    if (turn >= maxTurns) {
      return {
        runId,
        handle: handle.handle,
        status: 'max_turns',
        turns,
        lastOutput,
        nextPrompt: incompleteNextPrompt(input, turn, missingContractEvidence),
      }
    }
    prompt = incompleteNextPrompt(input, turn, missingContractEvidence)
  }

  return {
    runId,
    handle: handle.handle,
    status: 'max_turns',
    turns,
    lastOutput,
    nextPrompt: incompleteNextPrompt(input, turns.length, false),
  }
}

export function driveResultToToolResult(result: SwarmDriveResult, start: number): ToolResult {
  const status = result.status === 'done' ? 'success' : 'error'
  const code = status === 'success' ? undefined : `DRIVE_${result.status.toUpperCase()}`
  return {
    output: JSON.stringify(result),
    status,
    code,
    durationMs: durationSince(start),
  }
}
