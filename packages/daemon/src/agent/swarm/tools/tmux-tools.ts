import { randomBytes } from 'node:crypto'
import type {
  ToolDefinitionRuntime,
  ToolResult,
  ToolExecutionContext,
} from '../../../tools/registry.js'
import type {
  SwarmAgentHandle,
  SwarmAgentName,
  SwarmAgentRecoveryStep,
} from '@sepilotd/core'
import type { SwarmToolContext } from './context.js'
import { setTimeout as sleep } from 'node:timers/promises'
import { IdleDetector } from '../tmux/idle-detector.js'
import { getAgentConfig } from '../config/agents.js'
import { TmuxSwarmAgentRuntimeAdapter, type SwarmAgentRuntimeAdapter } from '../launcher/runtime-adapter.js'
import type { SwarmRun } from '../run/swarm-run.js'
import {
  createSwarmPromptAcceptedEvidence,
  createSwarmPromptAcceptanceTimeoutEvidence,
  createSwarmPromptSentEvidence,
  createSwarmStartupFailureEvidence,
  createSwarmStartupOutputEvidence,
} from '../run/startup-evidence.js'
import {
  runSwarmAgentPreflight,
  summarizeSwarmPreflight,
} from './preflight.js'
import {
  driveResultToToolResult,
  driveSwarmAgent,
  formatSwarmWorkerPrompt,
} from '../controller/driver.js'

function nowMs(): number { return Date.now() }
function durationSince(start: number): number { return Date.now() - start }
function fail(message: string, start: number): ToolResult {
  return { output: message, status: 'error', durationMs: durationSince(start) }
}

function recordOutputEvidence(
  run: SwarmRun,
  handle: SwarmAgentHandle,
  text: string,
): ReturnType<typeof createSwarmStartupOutputEvidence> {
  const evidence = createSwarmStartupOutputEvidence(handle, text)
  if (!evidence) return null
  run.recordStartupEvidence(handle.handle, evidence)
  run.setStatus(handle.handle, 'blocked')
  return evidence
}

function recordPromptAccepted(
  run: SwarmRun,
  handle: SwarmAgentHandle,
  text: string,
): boolean {
  const evidence = createSwarmPromptAcceptedEvidence(handle, text)
  if (!evidence) return false
  run.recordStartupEvidence(handle.handle, evidence)
  return true
}

function recordPromptAcceptanceTimeout(
  run: SwarmRun,
  handle: SwarmAgentHandle,
  text: string,
): boolean {
  const evidence = createSwarmPromptAcceptanceTimeoutEvidence(handle, text)
  if (!evidence) return false
  run.recordStartupEvidence(handle.handle, evidence)
  run.setStatus(handle.handle, 'blocked')
  return true
}

function blockerResult(evidence: NonNullable<ReturnType<typeof createSwarmStartupOutputEvidence>>, start: number): ToolResult {
  return {
    output: `agent blocked: ${evidence.lifecycleState}${evidence.lastOutputPreview ? `\n${evidence.lastOutputPreview}` : ''}`,
    status: 'error',
    code: 'AGENT_BLOCKED',
    durationMs: durationSince(start),
  }
}

function existingBlockerResult(handle: SwarmAgentHandle, start: number): ToolResult | null {
  const evidence = handle.startupEvidence
  if (evidence?.lifecycleState !== 'trust_required'
    && evidence?.lifecycleState !== 'tool_permission_required') {
    return null
  }
  return {
    output: `agent blocked: ${evidence.lifecycleState}${evidence.lastOutputPreview ? `\n${evidence.lastOutputPreview}` : ''}`,
    status: 'error',
    code: 'AGENT_BLOCKED',
    durationMs: durationSince(start),
  }
}

function agentRuntime(ctx: SwarmToolContext): SwarmAgentRuntimeAdapter {
  return ctx.runtime ?? new TmuxSwarmAgentRuntimeAdapter(ctx.pool, ctx.launcher)
}

const VALID_AGENTS: readonly SwarmAgentName[] = ['claude', 'codex', 'gemini', 'opencode']
const RECOVERY_ACTIONS: SwarmAgentRecoveryStep[] = [
  'inspect_worker_output',
  'review_trust_prompt',
  'review_tool_permission',
  'resend_prompt',
  'fix_configuration',
  'check_transport',
  'inspect_protocol_logs',
  'restart_worker',
  'escalate_to_human',
]
function preflightEnv(ctx: SwarmToolContext): NodeJS.ProcessEnv {
  return ctx.env ?? process.env
}

function recoveryAction(value: unknown): SwarmAgentRecoveryStep | null {
  return typeof value === 'string' && RECOVERY_ACTIONS.includes(value as SwarmAgentRecoveryStep)
    ? value as SwarmAgentRecoveryStep
    : null
}

function recoveryOutputPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  return normalized.length <= 500 ? normalized : normalized.slice(-500)
}

/**
 * Hard cap on agents per swarm run. A misbehaving supervisor LLM can otherwise
 * loop on swarm.spawn_agent and exhaust file descriptors / tmux server memory.
 * Override with `SEPILOTD_SWARM_MAX_AGENTS_PER_RUN` env var if needed.
 */
const MAX_AGENTS_PER_RUN = (() => {
  const raw = process.env.SEPILOTD_SWARM_MAX_AGENTS_PER_RUN
  const n = raw ? Number(raw) : NaN
  return Number.isFinite(n) && n > 0 ? n : 8
})()

function emitToolCall(
  ctx: SwarmToolContext,
  runId: string,
  tool: string,
  input: Record<string, unknown>,
): void {
  const run = ctx.registry.get(runId)
  if (!run) return
  // SwarmRun extends EventEmitter and only the registry's own listener is
  // wired through `run.on('event', ...)`. Emit directly so the event reaches
  // the jsonl store + any active SSE listeners.
  run.emit('event', { type: 'tool.call', runId, tool, input, ts: Date.now() })
}

function emitToolResult(
  ctx: SwarmToolContext,
  runId: string,
  tool: string,
  output: string,
): void {
  const run = ctx.registry.get(runId)
  if (!run) return
  const preview = output.length <= 200 ? output : output.slice(0, 200) + '…'
  run.emit('event', { type: 'tool.result', runId, tool, outputPreview: preview, ts: Date.now() })
}

function emitPaneSnapshot(
  ctx: SwarmToolContext,
  runId: string,
  handle: string,
  text: string,
): void {
  const run = ctx.registry.get(runId)
  if (!run) return
  run.emit('event', { type: 'pane.snapshot', runId, handle, text, ts: Date.now() })
}

export function createSpawnAgentTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.spawn_agent',
    description: 'Spawn a new external CLI agent (claude/codex/gemini/opencode) in this swarm run. Returns its handle.',
    inputSchema: {
      type: 'object',
      properties: {
        agent: { type: 'string', enum: VALID_AGENTS },
        role: { type: 'string' },
      },
      required: ['agent'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.spawn_agent', input)
      const agent = (input.agent as SwarmAgentName)
      if (!VALID_AGENTS.includes(agent)) {
        const r = fail(`unknown agent: ${String(agent)}`, start)
        emitToolResult(ctx, runId, 'swarm.spawn_agent', r.output)
        return r
      }
      const currentCount = run.snapshot().agents.length
      if (currentCount >= MAX_AGENTS_PER_RUN) {
        const r = fail(
          `agent quota exceeded: ${currentCount}/${MAX_AGENTS_PER_RUN}; kill an existing agent first`,
          start,
        )
        emitToolResult(ctx, runId, 'swarm.spawn_agent', r.output)
        return r
      }
      const handleId = `a_${randomBytes(3).toString('hex')}`
      const cwd = run.snapshot().worktree.path
      const preflight = runSwarmAgentPreflight({ agent, cwd, env: preflightEnv(ctx) })
      if (preflight.status === 'blocked') {
        const output = JSON.stringify({ code: 'PREFLIGHT_BLOCKED', preflight })
        run.recordStartupEvidence(handleId, createSwarmStartupFailureEvidence({
          handle: handleId,
          agent,
          cwd,
          runtime: preflight.runtime,
          startedAt: Date.now(),
          error: new Error(`configuration preflight blocked: ${summarizeSwarmPreflight(preflight)}`),
        }))
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'PREFLIGHT_BLOCKED',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.spawn_agent', r.output)
        return r
      }
      const startedAt = Date.now()
      let handle: SwarmAgentHandle
      try {
        handle = await agentRuntime(ctx).launch({
          runId,
          handle: handleId,
          agent,
          cwd,
          role: typeof input.role === 'string' ? input.role : undefined,
          autoApprove: true,
        })
      } catch (error) {
        const evidence = createSwarmStartupFailureEvidence({
          handle: handleId,
          agent,
          cwd,
          startedAt,
          error,
        })
        run.recordStartupEvidence(handleId, evidence)
        const r = fail(evidence.failureMessage ?? 'agent startup failed', start)
        emitToolResult(ctx, runId, 'swarm.spawn_agent', r.output)
        return r
      }
      run.addAgent(handle)
      const out = JSON.stringify({ handle: handle.handle, agent: handle.agent, role: handle.role })
      emitToolResult(ctx, runId, 'swarm.spawn_agent', out)
      return { output: out, status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createPreflightTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.preflight',
    description: 'Check whether a swarm agent lane can start before spawning an external worker.',
    inputSchema: {
      type: 'object',
      properties: {
        agent: { type: 'string', enum: VALID_AGENTS },
      },
      required: ['agent'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.preflight', input)
      const agent = input.agent as SwarmAgentName
      if (!VALID_AGENTS.includes(agent)) {
        const r = fail(`unknown agent: ${String(agent)}`, start)
        emitToolResult(ctx, runId, 'swarm.preflight', r.output)
        return r
      }
      const report = runSwarmAgentPreflight({
        agent,
        cwd: run.snapshot().worktree.path,
        env: preflightEnv(ctx),
      })
      const output = JSON.stringify(report)
      emitToolResult(ctx, runId, 'swarm.preflight', output)
      return {
        output,
        status: report.status === 'blocked' ? 'error' : 'success',
        code: report.status === 'blocked' ? 'PREFLIGHT_BLOCKED' : undefined,
        durationMs: durationSince(start),
      }
    },
  }
}

export function createKillAgentTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.kill_agent',
    description: 'Stop and remove an agent from the swarm run by handle.',
    inputSchema: {
      type: 'object',
      properties: { agent_handle: { type: 'string' } },
      required: ['agent_handle'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.kill_agent', input)
      const handleId = String(input.agent_handle)
      const handle = run.getAgent(handleId)
      if (!handle) {
        const r = fail(`unknown handle: ${handleId}`, start)
        emitToolResult(ctx, runId, 'swarm.kill_agent', r.output)
        return r
      }
      await agentRuntime(ctx).stop(handle)
      run.removeAgent(handleId)
      emitToolResult(ctx, runId, 'swarm.kill_agent', 'killed')
      return { output: 'killed', status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createListAgentsTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.list_agents',
    description: 'List all agents in the current swarm run with status and role.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    async execute(_input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.list_agents', {})
      const snap = run.snapshot()
      const out = JSON.stringify({
        activeHandle: snap.activeHandle ?? null,
        agents: snap.agents.map((a) => ({
          handle: a.handle,
          agent: a.agent,
          role: a.role ?? null,
          status: a.status,
          runtime: a.runtime ?? null,
          startup: a.startupEvidence?.lifecycleState ?? null,
          trustPromptDetected: a.startupEvidence?.trustPromptDetected ?? false,
          toolPermissionPromptDetected: a.startupEvidence?.toolPermissionPromptDetected ?? false,
          recoveryHint: a.startupEvidence?.recoveryHint ?? null,
        })),
      })
      emitToolResult(ctx, runId, 'swarm.list_agents', out)
      return { output: out, status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createRecoverTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.recover',
    description: 'Inspect or perform one bounded recovery action for a blocked swarm agent.',
    inputSchema: {
      type: 'object',
      properties: {
        agent_handle: { type: 'string' },
        action: { type: 'string', enum: RECOVERY_ACTIONS },
        prompt: { type: 'string' },
      },
      required: ['agent_handle'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.recover', input)
      const handle = run.getAgent(String(input.agent_handle))
      if (!handle) {
        const r = fail(`unknown handle: ${String(input.agent_handle)}`, start)
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }
      const hint = handle.startupEvidence?.recoveryHint
      if (!hint) {
        const output = JSON.stringify({
          code: 'RECOVERY_HINT_MISSING',
          message: 'agent has no recovery hint',
        })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_HINT_MISSING',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }
      const action = recoveryAction(input.action)
        ?? hint.steps[0]
        ?? 'escalate_to_human'
      if (!hint.steps.includes(action)) {
        const output = JSON.stringify({
          code: 'RECOVERY_ACTION_UNAVAILABLE',
          scenario: hint.scenario,
          action,
          allowedActions: hint.steps,
        })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_ACTION_UNAVAILABLE',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }
      if (action === 'inspect_worker_output') {
        const text = await agentRuntime(ctx).capture(handle)
        const outputPreview = recoveryOutputPreview(text)
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt: 1,
          maxAttempts: hint.maxAttempts,
          status: 'succeeded',
          message: 'captured worker output for recovery inspection',
          outputPreview,
        })
        emitPaneSnapshot(ctx, runId, handle.handle, text)
        const output = JSON.stringify({ scenario: hint.scenario, action, output: text })
        emitToolResult(ctx, runId, 'swarm.recover', output)
        return { output, status: 'success', durationMs: durationSince(start) }
      }
      if (action === 'inspect_protocol_logs') {
        let text = ''
        let captureError: string | undefined
        try {
          text = await agentRuntime(ctx).capture(handle, 500)
        } catch (error) {
          captureError = error instanceof Error ? error.message : String(error)
        }
        const outputPreview = recoveryOutputPreview([
          handle.startupEvidence?.failureMessage,
          handle.startupEvidence?.lastOutputPreview,
          text,
          captureError ? `capture failed: ${captureError}` : undefined,
        ].filter(Boolean).join('\n'))
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt: 1,
          maxAttempts: hint.maxAttempts,
          status: captureError ? 'failed' : 'succeeded',
          message: captureError
            ? `protocol log capture failed: ${captureError}`
            : 'captured protocol failure context',
          outputPreview,
        })
        if (text) emitPaneSnapshot(ctx, runId, handle.handle, text)
        const output = JSON.stringify({
          scenario: hint.scenario,
          action,
          startupEvidence: handle.startupEvidence ?? null,
          output: text,
          captureError,
          recommendedAction: captureError ? 'escalate_to_human' : 'restart_worker',
        })
        emitToolResult(ctx, runId, 'swarm.recover', output)
        return {
          output,
          status: captureError ? 'error' : 'success',
          code: captureError ? 'RECOVERY_PROTOCOL_LOGS_UNAVAILABLE' : undefined,
          durationMs: durationSince(start),
        }
      }
      if (action === 'check_transport') {
        const preflight = runSwarmAgentPreflight({
          agent: handle.agent,
          cwd: handle.cwd,
          env: preflightEnv(ctx),
        })
        const processAlive = handle.runtime && handle.runtime !== 'tmux'
          ? true
          : ctx.pool.isAlive(handle.tmuxSessionName)
        const output = JSON.stringify({
          scenario: hint.scenario,
          action,
          processAlive,
          preflight,
          summary: summarizeSwarmPreflight(preflight),
          recommendedAction: preflight.status === 'blocked'
            ? 'escalate_to_human'
            : processAlive
              ? 'inspect_worker_output'
              : 'restart_worker',
        })
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt: 1,
          maxAttempts: hint.maxAttempts,
          status: preflight.status === 'blocked' ? 'failed' : 'succeeded',
          message: `transport check: processAlive=${processAlive}; ${summarizeSwarmPreflight(preflight)}`,
        })
        emitToolResult(ctx, runId, 'swarm.recover', output)
        return {
          output,
          status: preflight.status === 'blocked' ? 'error' : 'success',
          code: preflight.status === 'blocked' ? 'RECOVERY_TRANSPORT_CHECK_FAILED' : undefined,
          durationMs: durationSince(start),
        }
      }
      if (action === 'resend_prompt') {
        const prompt = typeof input.prompt === 'string' && input.prompt.trim()
          ? input.prompt
          : null
        if (!prompt) {
          const output = JSON.stringify({
            code: 'RECOVERY_PROMPT_REQUIRED',
            scenario: hint.scenario,
            action,
            requiredInput: 'prompt',
            message: 'resend_prompt requires an explicit prompt; inspect the worker output first, then pass the last failed or revised prompt.',
          })
          const r: ToolResult = {
            output,
            status: 'error',
            code: 'RECOVERY_PROMPT_REQUIRED',
            durationMs: durationSince(start),
          }
          emitToolResult(ctx, runId, 'swarm.recover', r.output)
          return r
        }
        const currentAttempts = run.recoveryAttemptCount(handle.handle, hint.scenario, action)
        if (currentAttempts >= hint.maxAttempts) {
          run.recordRecoveryEvent({
            handle: handle.handle,
            scenario: hint.scenario,
            action,
            attempt: currentAttempts,
            maxAttempts: hint.maxAttempts,
            status: 'escalated',
            message: `max recovery attempts (${hint.maxAttempts}) exceeded for ${hint.scenario}`,
          })
          const output = JSON.stringify({
            code: 'RECOVERY_ESCALATION_REQUIRED',
            scenario: hint.scenario,
            action,
            maxAttempts: hint.maxAttempts,
          })
          const r: ToolResult = {
            output,
            status: 'error',
            code: 'RECOVERY_ESCALATION_REQUIRED',
            durationMs: durationSince(start),
          }
          emitToolResult(ctx, runId, 'swarm.recover', r.output)
          return r
        }
        const attempt = run.claimRecoveryAttempt(handle.handle, hint.scenario, action)
        const runtime = agentRuntime(ctx)
        await runtime.sendPrompt(
          handle,
          formatSwarmWorkerPrompt(prompt, execCtx?.runContract),
        )
        run.setStatus(handle.handle, 'busy')
        run.recordStartupEvidence(handle.handle, createSwarmPromptSentEvidence(handle))
        let outputPreview: string | undefined
        let blocker: ReturnType<typeof recordOutputEvidence> = null
        try {
          const snap = await runtime.capture(handle)
          outputPreview = recoveryOutputPreview(snap)
          blocker = recordOutputEvidence(run, handle, snap)
          emitPaneSnapshot(ctx, runId, handle.handle, snap)
        } catch { /* best-effort */ }
        if (blocker) {
          run.recordRecoveryEvent({
            handle: handle.handle,
            scenario: hint.scenario,
            action,
            attempt,
            maxAttempts: hint.maxAttempts,
            status: 'escalated',
            message: `resent prompt but worker is blocked: ${blocker.lifecycleState}`,
            outputPreview,
          })
          const output = JSON.stringify({
            code: 'AGENT_BLOCKED',
            scenario: hint.scenario,
            action,
            lifecycleState: blocker.lifecycleState,
            recoveryHint: blocker.recoveryHint ?? null,
            outputPreview,
          })
          const r: ToolResult = {
            output,
            status: 'error',
            code: 'AGENT_BLOCKED',
            durationMs: durationSince(start),
          }
          emitToolResult(ctx, runId, 'swarm.recover', r.output)
          return r
        }
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt,
          maxAttempts: hint.maxAttempts,
          status: 'succeeded',
          message: 'resent prompt to worker',
          outputPreview,
        })
        const output = JSON.stringify({ scenario: hint.scenario, action, status: 'resent' })
        emitToolResult(ctx, runId, 'swarm.recover', output)
        return { output, status: 'success', durationMs: durationSince(start) }
      }
      if (action === 'escalate_to_human') {
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt: run.recoveryAttemptCount(handle.handle, hint.scenario, action),
          maxAttempts: hint.maxAttempts,
          status: 'escalated',
          message: hint.reason,
        })
        const output = JSON.stringify({
          code: 'RECOVERY_ESCALATED',
          scenario: hint.scenario,
          reason: hint.reason,
        })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_ESCALATED',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }
      if (action !== 'restart_worker') {
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt: run.recoveryAttemptCount(handle.handle, hint.scenario, action),
          maxAttempts: hint.maxAttempts,
          status: 'escalated',
          message: `recovery action ${action} requires human intervention`,
        })
        const output = JSON.stringify({
          code: 'RECOVERY_REQUIRES_HUMAN',
          scenario: hint.scenario,
          action,
          allowedActions: hint.steps,
          reason: hint.reason,
          suggestedNextStep: `Resolve recovery action ${action} outside swarm.recover, or explicitly choose a later recovery action after confirming it is safe.`,
        })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_REQUIRES_HUMAN',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }

      const currentAttempts = run.recoveryAttemptCount(handle.handle, hint.scenario, action)
      if (currentAttempts >= hint.maxAttempts) {
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt: currentAttempts,
          maxAttempts: hint.maxAttempts,
          status: 'escalated',
          message: `max recovery attempts (${hint.maxAttempts}) exceeded for ${hint.scenario}`,
        })
        const output = JSON.stringify({
          code: 'RECOVERY_ESCALATION_REQUIRED',
          scenario: hint.scenario,
          action,
          maxAttempts: hint.maxAttempts,
        })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_ESCALATION_REQUIRED',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }

      const preflight = runSwarmAgentPreflight({
        agent: handle.agent,
        cwd: handle.cwd,
        env: preflightEnv(ctx),
      })
      if (preflight.status === 'blocked') {
        const message = `restart preflight blocked: ${summarizeSwarmPreflight(preflight)}`
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt: currentAttempts,
          maxAttempts: hint.maxAttempts,
          status: 'failed',
          message,
        })
        const output = JSON.stringify({ code: 'RECOVERY_FAILED', scenario: hint.scenario, action, message })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_FAILED',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }

      const attempt = run.claimRecoveryAttempt(handle.handle, hint.scenario, action)
      const replacementId = `a_${randomBytes(3).toString('hex')}`
      const runtime = agentRuntime(ctx)
      let replacement: SwarmAgentHandle
      try {
        replacement = await runtime.launch({
          runId,
          handle: replacementId,
          agent: handle.agent,
          cwd: handle.cwd,
          role: handle.role,
          autoApprove: true,
        })
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt,
          maxAttempts: hint.maxAttempts,
          status: 'failed',
          message,
        })
        const output = JSON.stringify({ code: 'RECOVERY_FAILED', scenario: hint.scenario, action, message })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_FAILED',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }

      const wasActive = run.snapshot().activeHandle === handle.handle
      run.addAgent(replacement)
      if (wasActive) run.setActive(replacement.handle)
      try { await runtime.stop(handle) } catch { /* best-effort; replacement is already live */ }
      run.removeAgent(handle.handle)
      const activeHandle = run.snapshot().activeHandle ?? null
      const replacementStartup = replacement.startupEvidence?.lifecycleState
        ?? (replacement.status === 'idle' ? 'ready_for_prompt' : null)
      const replacementRecoveryHint = replacement.startupEvidence?.recoveryHint ?? null
      const replacementBlocked = replacement.status === 'blocked'
        || replacementStartup === 'trust_required'
        || replacementStartup === 'tool_permission_required'
      const replacementUnhealthy = replacement.status === 'dead'
        || replacement.status === 'killed'
        || replacementStartup === 'failed'
      if (replacementBlocked) {
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt,
          maxAttempts: hint.maxAttempts,
          status: 'escalated',
          message: `replacement worker is blocked: ${replacementStartup ?? replacement.status}`,
          replacementHandle: replacement.handle,
        })
        const output = JSON.stringify({
          code: 'AGENT_BLOCKED',
          scenario: hint.scenario,
          action,
          previousHandle: handle.handle,
          replacementHandle: replacement.handle,
          activeHandle,
          replacementStatus: replacement.status,
          replacementStartup,
          recoveryHint: replacementRecoveryHint,
          recommendedAction: 'resolve_required_action',
          suggestedNextStep: 'Resolve the replacement worker required action, or hand off with the interrupted work summary before sending more prompts.',
        })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'AGENT_BLOCKED',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }
      if (replacementUnhealthy) {
        run.recordRecoveryEvent({
          handle: handle.handle,
          scenario: hint.scenario,
          action,
          attempt,
          maxAttempts: hint.maxAttempts,
          status: 'failed',
          message: `replacement worker is not healthy: ${replacementStartup ?? replacement.status}`,
          replacementHandle: replacement.handle,
        })
        const output = JSON.stringify({
          code: 'RECOVERY_REPLACEMENT_UNHEALTHY',
          scenario: hint.scenario,
          action,
          previousHandle: handle.handle,
          replacementHandle: replacement.handle,
          activeHandle,
          replacementStatus: replacement.status,
          replacementStartup,
          recoveryHint: replacementRecoveryHint,
          recommendedAction: 'escalate_to_human',
          suggestedNextStep: 'Do not continue driving this replacement worker. Report the replacement startup failure and hand off or escalate.',
        })
        const r: ToolResult = {
          output,
          status: 'error',
          code: 'RECOVERY_REPLACEMENT_UNHEALTHY',
          durationMs: durationSince(start),
        }
        emitToolResult(ctx, runId, 'swarm.recover', r.output)
        return r
      }
      run.recordRecoveryEvent({
        handle: handle.handle,
        scenario: hint.scenario,
        action,
        attempt,
        maxAttempts: hint.maxAttempts,
        status: 'succeeded',
        message: 'replacement worker launched',
        replacementHandle: replacement.handle,
      })
      const nextDrivePrompt = 'Continue the interrupted work from the previous worker. Summarize what you inspect or change, and report DONE only with concrete completion evidence.'
      const output = JSON.stringify({
        scenario: hint.scenario,
        action,
        previousHandle: handle.handle,
        replacementHandle: replacement.handle,
        activeHandle,
        replacementStatus: replacement.status,
        replacementStartup,
        recoveryHint: replacementRecoveryHint,
        recommendedAction: 'drive_replacement',
        nextDriveInput: {
          agent_handle: replacement.handle,
          prompt: nextDrivePrompt,
        },
        suggestedNextStep: 'Call swarm.drive with nextDriveInput, or adapt its prompt with the interrupted work summary, prior evidence, and remaining task.',
      })
      emitToolResult(ctx, runId, 'swarm.recover', output)
      return { output, status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createSendTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.send',
    description: 'Low-level escape hatch: send a prompt to a swarm agent (Enter is appended). Supervisors should use swarm.drive for ordinary worker task turns.',
    inputSchema: {
      type: 'object',
      properties: {
        agent_handle: { type: 'string' },
        prompt: { type: 'string' },
      },
      required: ['agent_handle', 'prompt'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.send', input)
      const handle = run.getAgent(String(input.agent_handle))
      if (!handle) {
        const r = fail(`unknown handle: ${String(input.agent_handle)}`, start)
        emitToolResult(ctx, runId, 'swarm.send', r.output)
        return r
      }
      const runtime = agentRuntime(ctx)
      await runtime.sendPrompt(
        handle,
        formatSwarmWorkerPrompt(String(input.prompt), execCtx?.runContract),
      )
      run.setStatus(handle.handle, 'busy')
      run.recordStartupEvidence(handle.handle, createSwarmPromptSentEvidence(handle))
      // Capture pane snapshot right after send so attach clients see the prompt land.
      try {
        const snap = await runtime.capture(handle)
        recordOutputEvidence(run, handle, snap)
        emitPaneSnapshot(ctx, runId, handle.handle, snap)
      } catch { /* best-effort */ }
      emitToolResult(ctx, runId, 'swarm.send', 'sent')
      return { output: 'sent', status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createInterruptTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.interrupt',
    description: 'Send an interrupt key to a swarm agent (Ctrl-C by default, or Escape for menu/backout flows).',
    inputSchema: {
      type: 'object',
      properties: {
        agent_handle: { type: 'string' },
        keyName: { type: 'string', enum: ['C-c', 'ctrl-c', 'Escape', 'esc'] },
      },
      required: ['agent_handle'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.interrupt', input)
      const handle = run.getAgent(String(input.agent_handle))
      if (!handle) {
        const r = fail(`unknown handle: ${String(input.agent_handle)}`, start)
        emitToolResult(ctx, runId, 'swarm.interrupt', r.output)
        return r
      }
      const keyName = typeof input.keyName === 'string' ? input.keyName : 'C-c'
      await agentRuntime(ctx).sendNamedKeys(handle, keyName)
      run.setStatus(handle.handle, 'idle')
      emitToolResult(ctx, runId, 'swarm.interrupt', 'interrupted')
      return { output: 'interrupted', status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createCaptureTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.capture',
    description: 'Capture the current pane content of a swarm agent (cleaned of ANSI escapes).',
    inputSchema: {
      type: 'object',
      properties: {
        agent_handle: { type: 'string' },
        lines: { type: 'number', minimum: 1, maximum: 5000 },
      },
      required: ['agent_handle'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.capture', input)
      const handle = run.getAgent(String(input.agent_handle))
      if (!handle) {
        const r = fail(`unknown handle: ${String(input.agent_handle)}`, start)
        emitToolResult(ctx, runId, 'swarm.capture', r.output)
        return r
      }
      const text = await agentRuntime(ctx).capture(handle, typeof input.lines === 'number' ? input.lines : 200)
      recordOutputEvidence(run, handle, text)
      emitPaneSnapshot(ctx, runId, handle.handle, text)
      emitToolResult(ctx, runId, 'swarm.capture', text)
      return { output: text, status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createSetActiveTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.set_active',
    description: 'Mark an agent as the active worker for handoff bookkeeping.',
    inputSchema: {
      type: 'object',
      properties: { agent_handle: { type: 'string' } },
      required: ['agent_handle'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.set_active', input)
      const handleId = String(input.agent_handle)
      if (!run.getAgent(handleId)) {
        const r = fail(`unknown handle: ${handleId}`, start)
        emitToolResult(ctx, runId, 'swarm.set_active', r.output)
        return r
      }
      run.setActive(handleId)
      emitToolResult(ctx, runId, 'swarm.set_active', 'active')
      return { output: 'active', status: 'success', durationMs: durationSince(start) }
    },
  }
}

export function createWaitIdleTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.wait_idle',
    description: 'Low-level escape hatch: block until the agent becomes idle (or timeout), then return new output. Supervisors should prefer swarm.drive.',
    inputSchema: {
      type: 'object',
      properties: {
        agent_handle: { type: 'string' },
        timeout_sec: { type: 'number', minimum: 0.1, maximum: 1800 },
        stable_sec: {
          type: 'number',
          minimum: 0.1,
          maximum: 60,
          description: 'Optional idle stability window override. Defaults to the agent config.',
        },
        min_wait_ms: {
          type: 'number',
          minimum: 0,
          maximum: 60000,
          description: 'Optional minimum wait before idle detection. Defaults to the agent config.',
        },
        poll_ms: {
          type: 'number',
          minimum: 25,
          maximum: 5000,
          description: 'Optional polling interval while waiting for idle. Defaults to 500 ms.',
        },
      },
      required: ['agent_handle'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.wait_idle', input)
      const handle = run.getAgent(String(input.agent_handle))
      if (!handle) {
        const r = fail(`unknown handle: ${String(input.agent_handle)}`, start)
        emitToolResult(ctx, runId, 'swarm.wait_idle', r.output)
        return r
      }
      const existingBlocker = existingBlockerResult(handle, start)
      if (existingBlocker) {
        emitToolResult(ctx, runId, 'swarm.wait_idle', existingBlocker.output)
        return existingBlocker
      }
      const cfg = getAgentConfig(handle.agent)
      const timeoutSeconds = typeof input.timeout_sec === 'number'
        ? Math.max(0.1, Math.min(1800, input.timeout_sec))
        : 300
      const timeoutMs = timeoutSeconds * 1000
      const stableSeconds = typeof input.stable_sec === 'number'
        ? Math.max(0.1, Math.min(60, input.stable_sec))
        : cfg.stableSeconds
      const minWaitMs = typeof input.min_wait_ms === 'number'
        ? Math.max(0, Math.min(60_000, input.min_wait_ms))
        : cfg.minWaitMs
      const pollMs = typeof input.poll_ms === 'number'
        ? Math.max(25, Math.min(5000, input.poll_ms))
        : 500
      const runtime = agentRuntime(ctx)
      const directWait = runtime.waitForIdle?.(handle, {
        timeoutMs,
        signal: execCtx?.signal,
      })
      if (directWait) {
        const waited = await directWait
        const blocker = recordOutputEvidence(run, handle, waited.snapshot)
        emitPaneSnapshot(ctx, runId, handle.handle, waited.snapshot)
        if (blocker) {
          const r = blockerResult(blocker, start)
          emitToolResult(ctx, runId, 'swarm.wait_idle', r.output)
          return r
        }
        if (waited.status === 'idle') {
          run.setStatus(handle.handle, 'idle')
          recordPromptAccepted(run, handle, waited.output || waited.snapshot)
        } else if (!recordPromptAcceptanceTimeout(run, handle, waited.snapshot || waited.output)) {
          run.setStatus(handle.handle, 'busy')
        }
        emitToolResult(ctx, runId, 'swarm.wait_idle', waited.output)
        return {
          output: waited.output,
          status: waited.status === 'idle' ? 'success' : 'error',
          code: waited.status === 'idle' ? undefined : 'TIMEOUT_TRANSIENT',
          durationMs: durationSince(start),
        }
      }
      const detector = new IdleDetector({
        idlePatterns: cfg.idlePatterns,
        busyPatterns: cfg.busyPatterns,
        minWaitMs,
        stableSeconds,
      }, start)

      const deadline = start + timeoutMs
      while (Date.now() < deadline) {
        if (execCtx?.signal?.aborted) {
          const r = fail('aborted', start)
          emitToolResult(ctx, runId, 'swarm.wait_idle', r.output)
          return r
        }
        const text = await runtime.capture(handle)
        const blocker = recordOutputEvidence(run, handle, text)
        if (blocker) {
          emitPaneSnapshot(ctx, runId, handle.handle, text)
          const r = blockerResult(blocker, start)
          emitToolResult(ctx, runId, 'swarm.wait_idle', r.output)
          return r
        }
        const decision = detector.observe(text, Date.now())
        if (decision === 'idle') {
          run.setStatus(handle.handle, 'idle')
          const newOut = await runtime.readNewOutput(handle)
          recordPromptAccepted(run, handle, newOut || text)
          emitPaneSnapshot(ctx, runId, handle.handle, text)
          emitToolResult(ctx, runId, 'swarm.wait_idle', newOut)
          return { output: newOut, status: 'success', durationMs: durationSince(start) }
        }
        await sleep(Math.min(pollMs, Math.max(0, deadline - Date.now())))
      }
      const tail = await agentRuntime(ctx).capture(handle)
      const blocker = recordOutputEvidence(run, handle, tail)
      emitPaneSnapshot(ctx, runId, handle.handle, tail)
      if (blocker) {
        const r = blockerResult(blocker, start)
        emitToolResult(ctx, runId, 'swarm.wait_idle', r.output)
        return r
      }
      recordPromptAcceptanceTimeout(run, handle, tail)
      emitToolResult(ctx, runId, 'swarm.wait_idle', tail)
      return {
        output: tail,
        status: 'error',
        code: 'TIMEOUT_TRANSIENT',
        durationMs: durationSince(start),
      }
    },
  }
}

export function createDriveTool(ctx: SwarmToolContext): ToolDefinitionRuntime {
  return {
    name: 'swarm.drive',
    description: 'Default supervisor path: drive one agent through a human-like send/wait/observe loop for multiple bounded turns.',
    inputSchema: {
      type: 'object',
      properties: {
        agent_handle: { type: 'string' },
        prompt: { type: 'string' },
        followups: { type: 'array', items: { type: 'string' } },
        continue_prompt: { type: 'string' },
        max_turns: { type: 'number', minimum: 1, maximum: 100 },
        stop_patterns: { type: 'array', items: { type: 'string' } },
        timeout_sec: { type: 'number', minimum: 0.1, maximum: 1800 },
        stable_sec: { type: 'number', minimum: 0.1, maximum: 60 },
        min_wait_ms: { type: 'number', minimum: 0, maximum: 60000 },
        poll_ms: { type: 'number', minimum: 25, maximum: 5000 },
        timeout_retries: { type: 'number', minimum: 0, maximum: 5 },
      },
      required: ['prompt'],
    },
    async execute(input, execCtx?: ToolExecutionContext): Promise<ToolResult> {
      const start = nowMs()
      const runId = ctx.resolveRunId(execCtx?.sessionId)
      if (!runId) return fail('not in a swarm run', start)
      const run = ctx.registry.get(runId)
      if (!run) return fail('swarm run not found', start)
      emitToolCall(ctx, runId, 'swarm.drive', input)
      try {
        const result = await driveSwarmAgent(ctx, runId, {
          agent_handle: typeof input.agent_handle === 'string' ? input.agent_handle : undefined,
          prompt: String(input.prompt ?? ''),
          followups: Array.isArray(input.followups)
            ? input.followups.filter((item): item is string => typeof item === 'string')
            : undefined,
          continue_prompt: typeof input.continue_prompt === 'string' ? input.continue_prompt : undefined,
          max_turns: typeof input.max_turns === 'number' ? input.max_turns : undefined,
          stop_patterns: Array.isArray(input.stop_patterns)
            ? input.stop_patterns.filter((item): item is string => typeof item === 'string')
            : undefined,
          timeout_sec: typeof input.timeout_sec === 'number' ? input.timeout_sec : undefined,
          stable_sec: typeof input.stable_sec === 'number' ? input.stable_sec : undefined,
          min_wait_ms: typeof input.min_wait_ms === 'number' ? input.min_wait_ms : undefined,
          poll_ms: typeof input.poll_ms === 'number' ? input.poll_ms : undefined,
          timeout_retries: typeof input.timeout_retries === 'number' ? input.timeout_retries : undefined,
          runContract: execCtx?.runContract,
        }, execCtx?.signal)
        const toolResult = driveResultToToolResult(result, start)
        emitToolResult(ctx, runId, 'swarm.drive', toolResult.output)
        return toolResult
      } catch (error) {
        const r = fail(error instanceof Error ? error.message : String(error), start)
        emitToolResult(ctx, runId, 'swarm.drive', r.output)
        return r
      }
    },
  }
}
