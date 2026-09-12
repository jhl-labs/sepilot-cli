import type { DaemonChatStreamPayload } from './types.js'
import type {
  TerminalChatStreamFrame,
  TerminalChatStreamFrameConsumer,
  TerminalChatStreamFrameConsumerOptions,
  TerminalChatStreamPresenter,
  TerminalChatStreamRenderer,
  TerminalChatStreamRendererOptions,
} from './chat-surface-types.js'
import {
  formatAcceptanceCriteriaCount,
  formatRunContractDetail,
  formatToolCall,
  toolExecutionPostureLabel,
  toolResultRecoveryLabel,
} from './chat-surface-utils.js'

function truncate(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text
}

function formatDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '0ms'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`
}

type PlannerStepLike = {
  id: string
  title: string
  status?: string
  children?: PlannerStepLike[]
}

function flattenPlannerSteps(
  steps: readonly PlannerStepLike[] | undefined,
  rows: PlannerStepLike[] = [],
): PlannerStepLike[] {
  for (const step of steps ?? []) {
    rows.push(step)
    flattenPlannerSteps(step.children, rows)
  }
  return rows
}

function formatCheckpointFiles(files: Array<{ path: string }>): string {
  if (files.length === 0) return '0 files'
  const visible = files.slice(0, 3).map((file) => file.path).join(', ')
  const more = files.length > 3 ? `, +${files.length - 3} more` : ''
  return `${files.length} file${files.length === 1 ? '' : 's'}: ${visible}${more}`
}

function summarizeSubagentProgressInner(
  inner: Extract<DaemonChatStreamPayload, { type: 'subagent_progress' }>['inner'],
): { detail: string; failed: boolean } {
  switch (inner.type) {
    case 'reasoning_step':
      return {
        detail: `reasoning: ${inner.label}${inner.detail ? ` · ${truncate(inner.detail, 160)}` : ''}`,
        failed: false,
      }
    case 'action_progress':
      return {
        detail: `progress: ${truncate(inner.summary, 100)} → ${truncate(inner.nextStep, 100)}`,
        failed: false,
      }
    case 'thinking':
      return { detail: `thinking: ${truncate(inner.content ?? 'reasoning', 180)}`, failed: false }
    case 'tool_call':
      return { detail: `tool: ${inner.toolCall?.name ?? 'tool'}`, failed: false }
    case 'tool_result':
      return {
        detail: `result ${inner.status}: ${truncate(inner.output ?? '', 180)}`,
        failed: inner.status !== 'success',
      }
    case 'message':
      return { detail: `message: ${truncate(inner.content ?? '', 180)}`, failed: false }
    default:
      return { detail: inner.type, failed: false }
  }
}

export function formatTerminalChatStreamFrame(frame: TerminalChatStreamFrame): string {
  switch (frame.kind) {
    case 'session':
      return `[session] ${frame.sessionId}\n`
    case 'artifacts':
      return `\n[artifacts] ${frame.count}\n`
    case 'context':
      return `\n[context] ${frame.items.map((item) => item.citationLabel).join(' · ')}\n`
    case 'run_contract':
      return `\n[run contract] ${formatAcceptanceCriteriaCount(frame.contract.acceptanceCriteria.length)}: ${formatRunContractDetail(frame.contract, 220)}\n`
    case 'state_board': {
      const current = frame.todos?.find((todo) => todo.status === 'in_progress')
      const currentSegment = current ? ` · 현재: ${current.content}` : ''
      return `\n[진행] criteria ${frame.criteriaTotal} · plan ${frame.planDone}/${frame.planTotal} · todos ${frame.todosDone}/${frame.todosTotal}${currentSegment}\n`
    }
    case 'inline_text_start':
    case 'inline_text_end':
      return ''
    case 'text':
      return frame.text
    case 'message':
      return `\n[message] ${frame.content}\n`
    case 'llm_request': {
      const tools = frame.toolNames.length
        ? ` tools=${frame.toolNames.slice(0, 5).join(',')}${frame.toolNames.length > 5 ? `,+${frame.toolNames.length - 5}` : ''}`
        : ''
      const trace = frame.traceRef ? ` trace=${frame.traceRef}` : ''
      const target = frame.providerId ? `${frame.providerId}/${frame.model}` : frame.model
      const role = frame.auxiliary ? 'aux' : 'main'
      const source = frame.source ? ` source=${frame.source}` : ''
      return `\n[model:${role}] ${target} iteration=${frame.iteration}${source}${tools}${trace}\n`
    }
    case 'node_trace': {
      const next = frame.nextEdge ? ` -> ${frame.nextEdge}` : ''
      return `\n[node] ${frame.node} ${formatDurationMs(frame.durationMs)}${next}\n`
    }
    case 'planner_working_memory': {
      const current = frame.currentStepTitle ? ` current=${truncate(frame.currentStepTitle, 80)}` : ''
      const risks = frame.risks.length ? ` risks=${frame.risks.length}` : ''
      const assumptions = frame.openAssumptions ? ` assumptions=${frame.openAssumptions}` : ''
      return `\n[planner] plan ${frame.planDone}/${frame.planTotal} decisions=${frame.decisions}${risks}${assumptions}${current}: ${truncate(frame.taskSummary, 160)}\n`
    }
    case 'thinking':
      return `\n[thinking] ${frame.content}\n`
    case 'reasoning_step':
      return `\n[reasoning] ${frame.label}${frame.detail ? `: ${truncate(frame.detail, 220)}` : ''}\n`
    case 'action_progress': {
      const tools = frame.toolNames.length > 0 ? ` (${frame.toolNames.join(', ')})` : ''
      return `\n[progress] ${truncate(frame.summary, 180)} → ${truncate(frame.nextStep, 180)}${tools}\n`
    }
    case 'mode_route_decision': {
      const confidence = frame.confidence === undefined ? '' : ` confidence=${frame.confidence.toFixed(2)}`
      const persona = frame.persona ? ` persona=${frame.persona}` : ''
      const fallback = frame.fallback ? ' fallback' : ''
      const candidates = frame.candidates?.length ? ` candidates=${frame.candidates.join(',')}` : ''
      const reason = frame.reason ? `: ${truncate(frame.reason, 220)}` : ''
      return `\n[mode route] ${frame.chosen}${persona}${confidence}${fallback}${candidates}${reason}\n`
    }
    case 'router_decision': {
      const skills = frame.skillIds.length ? ` skills=${frame.skillIds.join(',')}` : ''
      const fallback = frame.fallback ? ' fallback' : ''
      const reason = frame.reason ? `: ${truncate(frame.reason, 220)}` : ''
      return `\n[router] ${frame.mode}/${frame.persona} confidence=${frame.confidence}${fallback}${skills}${reason}\n`
    }
    case 'quality_gate_verdict': {
      const reason = frame.blockingReason ? `: ${truncate(frame.blockingReason, 220)}` : ''
      return `\n[quality gate:${frame.decision}] ${frame.phase} backtracks=${frame.backtrackCount}${reason}\n`
    }
    case 'backtrack':
      return `\n[backtrack] ${frame.phase} attempt ${frame.attempt}: ${truncate(frame.reason, 220)}\n`
    case 'recovery':
      return `\n[recovery:${frame.scope}/${frame.recoveryKind}] ${frame.action}: ${truncate(frame.message, 220)}\n`
    case 'panel_open':
      return `\n[panel] ${frame.personas.map((persona) => persona.name).join(' · ')}\n`
    case 'panel_turn_start':
      return `\n[panel:${frame.personaName}] responding\n`
    case 'panel_turn_complete':
      return `\n[panel:${frame.personaName}] ${frame.text}\n`
    case 'panel_turn_failed':
      return `\n[panel:${frame.personaName}:error] ${frame.error}\n`
    case 'panel_synthesizing':
      return `\n[panel] synthesizing ${frame.panelists} replies\n`
    case 'debate_round': {
      const decision = frame.round.finalDecision ? ` ${frame.round.finalDecision}` : ''
      const rationale = frame.round.rationale ? `: ${truncate(frame.round.rationale, 240)}` : ''
      return `\n[debate${decision}] ${truncate(frame.round.topic, 120)}${rationale}\n`
    }
    case 'edit_checkpoint_opened':
      return `\n[edit checkpoint:open] ${frame.checkpoint.checkpointId} ${frame.checkpoint.label} (${formatCheckpointFiles(frame.checkpoint.files)})\n`
    case 'edit_checkpoint_resolved':
      return `\n[edit checkpoint:${frame.checkpoint.status}] ${frame.checkpoint.checkpointId} ${frame.checkpoint.label} (${formatCheckpointFiles(frame.checkpoint.files)})\n`
    case 'cowork_plan': {
      const steps = frame.plan
        .slice(0, 6)
        .map((step, index) => `${index + 1}. ${step.role}: ${step.instruction}`)
        .join('\n')
      const more = frame.plan.length > 6 ? `\n… ${frame.plan.length - 6} more` : ''
      return `\n[cowork plan] ${frame.plan.length} task${frame.plan.length === 1 ? '' : 's'}\n${steps}${more}\n`
    }
    case 'cowork_task_start':
      return `\n[cowork:${frame.role}] started ${truncate(frame.instruction, 180)}\n`
    case 'cowork_task_complete':
      return `\n[cowork:${frame.role}] completed ${truncate(frame.result, 220)}\n`
    case 'cowork_task_failed':
      return `\n[cowork:${frame.role}:failed] ${truncate(frame.error, 220)}\n`
    case 'cowork_synthesizing':
      return `\n[cowork] synthesizing${frame.summary ? `: ${truncate(frame.summary, 180)}` : ''}\n`
    case 'cowork_discuss_request': {
      const choices = frame.choices?.length ? ` choices: ${frame.choices.join(' · ')}` : ''
      return `\n[cowork question] ${truncate(frame.prompt, 220)}${choices}\n`
    }
    case 'cowork_discuss_response':
      return `\n[cowork answer] ${truncate(frame.response, 220)}\n`
    case 'subagent_progress': {
      const label = frame.label ? `:${frame.label}` : ''
      const status = frame.failed ? ':error' : ''
      return `\n[subagent${label}${status}] ${frame.subagentId} ${frame.detail}\n`
    }
    case 'question_request': {
      const choices = frame.choices?.length ? ` choices: ${frame.choices.join(' · ')}` : ''
      return `\n[question] ${truncate(frame.prompt, 220)}${choices}\n  answer with: sepilot answer ${frame.sessionId ?? '<session-id>'} ${frame.questionId} <reply>\n`
    }
    case 'tool_call':
      return `\n[tool] ${frame.toolName}\n`
    case 'approval_request':
      return `\n[approval-request] ${frame.preview}\n  decide with: /approve ${frame.requestId} or /deny ${frame.requestId}\n`
    case 'auto_approval':
      return `\n[auto-${frame.decision}] ${frame.toolName} via ${frame.scope} rule '${frame.pattern}'\n`
    case 'tool_result':
      return `\n[tool:${frame.status}${frame.recoveryLabel ? `, ${frame.recoveryLabel}` : ''}${frame.postureLabel ? `, ${frame.postureLabel}` : ''}] ${frame.output}\n`
    case 'done':
      return `\n[done] ${frame.usage.inputTokens}->${frame.usage.outputTokens}\n`
    case 'error':
      return `\n[error] ${frame.message}\n`
    case 'state_change':
      return `\n[state] ${frame.state}\n`
    case 'context_compact':
      // Render: "[context compacted: 16,000 → 4,000 tokens] <summary>"
      // Numbers are localized so 16k+ ranges read at a glance.
      return `\n[context compacted: ${frame.beforeTokens.toLocaleString()} → ${frame.afterTokens.toLocaleString()} tokens] ${frame.summary}\n`
    case 'phase_change': {
      const entered = frame.enteredPhase ?? 'finalize'
      const closed = frame.closedPhase
        ? ` (closed ${frame.closedPhase.phase}: ${frame.closedPhase.closedTokens.toLocaleString()} tokens)`
        : ''
      return `\n[phase] ${entered}${closed}\n`
    }
    case 'post_edit_findings': {
      const lines = ['\n[post-edit]']
      if (frame.editedFiles.length > 0) {
        lines.push(`  edited: ${frame.editedFiles.slice(0, 3).join(', ')}`)
      }
      if (frame.brokenCallers.length > 0) {
        lines.push(`  broken callers (${frame.brokenCallers.length}):`)
        for (const c of frame.brokenCallers.slice(0, 3)) {
          lines.push(`    - ${c.file}: ${c.summary.split('\n')[0]}`)
        }
      } else if (frame.reverseCallers.length > 0) {
        lines.push(`  likely callers: ${frame.reverseCallers.slice(0, 3).join(', ')}`)
      }
      if (frame.ownDiagnostics.length > 0) {
        lines.push(`  diagnostics (${frame.ownDiagnostics.length}):`)
        for (const d of frame.ownDiagnostics.slice(0, 3)) {
          lines.push(`    - ${d.file}: ${d.summary.split('\n')[0]}`)
        }
      }
      return lines.join('\n') + '\n'
    }
    case 'steering_ack': {
      const detail = frame.steeringMessage ? `: ${truncate(frame.steeringMessage, 80)}` : ''
      return `\n[steering ack] ${frame.noteId} (${frame.steeringKind})${detail}\n`
    }
    case 'steering_consumed':
      return `\n[steering consumed] ${frame.noteId}\n`
    default:
      return ''
  }
}

export function createTerminalChatStreamFrameConsumer(
  options: TerminalChatStreamFrameConsumerOptions,
): TerminalChatStreamFrameConsumer {
  const presenter = createTerminalChatStreamPresenter()

  return {
    handleEvent(event: DaemonChatStreamPayload) {
      for (const frame of presenter.handleEvent(event)) {
        options.onFrame(frame)
      }
    },
    getContent() {
      return presenter.getContent()
    },
  }
}

export function createTerminalChatStreamRenderer(
  options: TerminalChatStreamRendererOptions,
): TerminalChatStreamRenderer {
  return createTerminalChatStreamFrameConsumer({
    onFrame(frame) {
      const text = formatTerminalChatStreamFrame(frame)
      if (text) {
        options.write(text)
      }
    },
  })
}

export function createTerminalChatStreamPresenter(): TerminalChatStreamPresenter {
  let fullContent = ''
  let inlineTextActive = false
  let currentSessionId: string | undefined

  const closeInlineText = (frames: TerminalChatStreamFrame[]) => {
    if (!inlineTextActive) return
    inlineTextActive = false
    frames.push({ kind: 'inline_text_end' })
  }

  const captureSession = (
    frames: TerminalChatStreamFrame[],
    sessionId: string | undefined,
  ) => {
    if (!sessionId || sessionId === currentSessionId) return
    currentSessionId = sessionId
    frames.push({ kind: 'session', sessionId })
  }

  const payloadSessionId = (event: DaemonChatStreamPayload): string | undefined => (
    typeof (event as { sessionId?: unknown }).sessionId === 'string'
      ? (event as { sessionId: string }).sessionId
      : undefined
  )

  return {
    handleEvent(event: DaemonChatStreamPayload): TerminalChatStreamFrame[] {
      const frames: TerminalChatStreamFrame[] = []

      if ('sessionId' in event && !('type' in event)) {
        captureSession(frames, event.sessionId)
        return frames
      }

      if ('artifacts' in event) {
        captureSession(frames, event.sessionId)
        frames.push({ kind: 'artifacts', count: event.artifacts.length })
        return frames
      }

      if (!('type' in event)) {
        return frames
      }

      captureSession(frames, payloadSessionId(event))

      switch (event.type) {
        case 'memory_context':
          closeInlineText(frames)
          frames.push({ kind: 'context', items: event.items })
          break
        case 'text_delta':
          if (!inlineTextActive) {
            inlineTextActive = true
            frames.push({ kind: 'inline_text_start' })
          }
          fullContent += event.text ?? ''
          frames.push({ kind: 'text', text: event.text ?? '' })
          break
        case 'message':
          if (inlineTextActive) {
            fullContent = event.content ?? fullContent
            break
          }
          // A new non-streamed assistant message supersedes text from an
          // earlier LLM iteration (often a progress sentence before a tool
          // call). `getContent()` is the final answer contract used by
          // one-shot/output-file callers, not a transcript accumulator.
          fullContent = event.content ?? fullContent
          frames.push({
            kind: 'message',
            content: event.content ?? '',
            streamed: false,
          })
          break
        case 'llm_request':
          closeInlineText(frames)
          frames.push({
            kind: 'llm_request',
            turnId: event.turnId,
            iteration: event.iteration,
            model: event.requestDigest.model,
            providerId: event.requestDigest.providerId,
            source: event.requestDigest.source,
            startedAt: event.requestDigest.startedAt,
            timeoutMs: event.requestDigest.timeoutMs,
            auxiliary: event.requestDigest.auxiliary,
            toolNames: event.requestDigest.toolNames,
            traceRef: event.requestDigest.traceRef,
          })
          break
        case 'node_trace':
          closeInlineText(frames)
          frames.push({
            kind: 'node_trace',
            node: event.node,
            durationMs: event.durationMs,
            nextEdge: event.nextEdge,
          })
          break
        case 'planner_working_memory_updated': {
          closeInlineText(frames)
          const steps = flattenPlannerSteps(event.workingMemory.plan)
          const current = steps.find((step) => step.id === event.workingMemory.currentSubtaskId)
          frames.push({
            kind: 'planner_working_memory',
            taskSummary: event.workingMemory.taskSummary,
            currentStepTitle: current?.title,
            currentStepRationale: event.workingMemory.currentStepRationale,
            planTotal: steps.length,
            planDone: steps.filter((step) => step.status === 'done').length,
            decisions: event.workingMemory.decisions.length,
            risks: event.workingMemory.risks,
            openAssumptions: event.workingMemory.openAssumptions?.length ?? 0,
            abandonedAlternatives: event.workingMemory.abandonedAlternatives?.length ?? 0,
          })
          break
        }
        case 'thinking':
          closeInlineText(frames)
          frames.push({ kind: 'thinking', content: event.content ?? '' })
          break
        case 'reasoning_step':
          closeInlineText(frames)
          frames.push({
            kind: 'reasoning_step',
            label: event.label,
            detail: event.detail,
          })
          break
        case 'action_progress':
          closeInlineText(frames)
          frames.push({
            kind: 'action_progress',
            summary: event.summary,
            nextStep: event.nextStep,
            toolNames: event.toolNames,
          })
          break
        case 'mode_route_decision':
          closeInlineText(frames)
          frames.push({
            kind: 'mode_route_decision',
            chosen: event.chosen,
            persona: event.persona,
            candidates: event.candidates,
            reason: event.reason,
            confidence: event.confidence,
            fallback: event.fallback,
          })
          break
        case 'router_decision':
          closeInlineText(frames)
          frames.push({
            kind: 'router_decision',
            id: event.id,
            mode: event.decision.mode,
            persona: event.decision.persona,
            skillIds: event.decision.skillIds,
            reason: event.decision.reason,
            confidence: event.decision.confidence,
            fallback: event.decision.fallback,
          })
          break
        case 'quality_gate_verdict':
          closeInlineText(frames)
          frames.push({
            kind: 'quality_gate_verdict',
            phase: event.phase,
            decision: event.decision,
            blockingReason: event.blockingReason,
            backtrackCount: event.backtrackCount,
          })
          break
        case 'backtrack':
          closeInlineText(frames)
          frames.push({
            kind: 'backtrack',
            phase: event.phase,
            reason: event.reason,
            attempt: event.attempt,
          })
          break
        case 'recovery':
          closeInlineText(frames)
          frames.push({
            kind: 'recovery',
            scope: event.scope,
            recoveryKind: event.kind,
            action: event.action,
            message: event.message,
            recoverable: event.recoverable,
          })
          break
        case 'panel_open':
          closeInlineText(frames)
          frames.push({ kind: 'panel_open', personas: event.personas ?? [] })
          break
        case 'panel_turn_start':
          closeInlineText(frames)
          frames.push({ kind: 'panel_turn_start', personaName: event.personaName })
          break
        case 'panel_turn_complete':
          closeInlineText(frames)
          frames.push({
            kind: 'panel_turn_complete',
            personaName: event.personaName,
            text: event.text,
          })
          break
        case 'panel_turn_failed':
          closeInlineText(frames)
          frames.push({
            kind: 'panel_turn_failed',
            personaName: event.personaName,
            error: event.error,
          })
          break
        case 'panel_synthesizing':
          closeInlineText(frames)
          frames.push({ kind: 'panel_synthesizing', panelists: event.panelists })
          break
        case 'debate_round':
          closeInlineText(frames)
          frames.push({ kind: 'debate_round', round: event.round })
          break
        case 'edit_checkpoint_opened':
          closeInlineText(frames)
          frames.push({ kind: 'edit_checkpoint_opened', checkpoint: event.checkpoint })
          break
        case 'edit_checkpoint_resolved':
          closeInlineText(frames)
          frames.push({ kind: 'edit_checkpoint_resolved', checkpoint: event.checkpoint })
          break
        case 'cowork_plan':
          closeInlineText(frames)
          frames.push({ kind: 'cowork_plan', plan: event.plan ?? [] })
          break
        case 'cowork_task_start':
          closeInlineText(frames)
          frames.push({
            kind: 'cowork_task_start',
            role: event.role,
            instruction: event.instruction,
          })
          break
        case 'cowork_task_complete':
          closeInlineText(frames)
          frames.push({
            kind: 'cowork_task_complete',
            role: event.role,
            instruction: event.instruction,
            result: event.result,
          })
          break
        case 'cowork_task_failed':
          closeInlineText(frames)
          frames.push({
            kind: 'cowork_task_failed',
            role: event.role,
            instruction: event.instruction,
            error: event.error,
          })
          break
        case 'cowork_synthesizing':
          closeInlineText(frames)
          frames.push({ kind: 'cowork_synthesizing', summary: event.summary ?? '' })
          break
        case 'cowork_discuss_request':
          closeInlineText(frames)
          frames.push({
            kind: 'cowork_discuss_request',
            prompt: event.prompt,
            choices: event.choices,
          })
          break
        case 'cowork_discuss_response':
          closeInlineText(frames)
          frames.push({
            kind: 'cowork_discuss_response',
            prompt: event.prompt,
            response: event.response,
          })
          break
        case 'subagent_progress': {
          closeInlineText(frames)
          const summary = summarizeSubagentProgressInner(event.inner)
          frames.push({
            kind: 'subagent_progress',
            subagentId: event.subagentId,
            label: event.label,
            detail: summary.detail,
            failed: summary.failed,
          })
          break
        }
        case 'question_request':
          closeInlineText(frames)
          frames.push({
            kind: 'question_request',
            questionId: event.questionId,
            prompt: event.prompt,
            choices: event.choices,
            sessionId: event.sessionId,
          })
          break
        case 'tool_call':
          closeInlineText(frames)
          frames.push({
            kind: 'tool_call',
            toolName: event.toolCall?.name ?? 'tool',
            preview: formatToolCall(event.toolCall ?? {}, 80),
          })
          break
        case 'approval_request':
          closeInlineText(frames)
          frames.push({
            kind: 'approval_request',
            toolName: event.toolCall?.name ?? 'tool',
            requestId: event.requestId,
            preview: formatToolCall(event.toolCall ?? {}, 200),
            input: event.toolCall?.arguments as Record<string, unknown> | undefined,
            sessionId: event.sessionId,
          })
          break
        case 'auto_approval':
          closeInlineText(frames)
          frames.push({
            kind: 'auto_approval',
            toolName: event.toolCall?.name ?? 'tool',
            requestId: event.requestId,
            preview: formatToolCall(event.toolCall ?? {}, 200),
            decision: event.decision,
            pattern: event.rule?.pattern ?? '',
            scope: event.scope,
          })
          break
        case 'tool_result':
          closeInlineText(frames)
          frames.push({
            kind: 'tool_result',
            status: event.status,
            output: event.output ?? '',
            recoveryLabel: toolResultRecoveryLabel(event.recovery),
            postureLabel: toolExecutionPostureLabel(event.executionPosture),
          })
          break
        case 'done':
          closeInlineText(frames)
          frames.push({
            kind: 'done',
            usage: event.usage,
            ...(event.stopReason ? { stopReason: event.stopReason } : {}),
          })
          break
        case 'error':
          closeInlineText(frames)
          frames.push({
            kind: 'error',
            message: event.error?.message ?? 'Unknown error',
            code: event.error?.code,
            ...(event.error?.reason ? { reason: event.error.reason } : {}),
            ...(event.error?.pendingDecision
              ? { pendingDecision: event.error.pendingDecision }
              : {}),
            ...(event.error?.stopReason ? { stopReason: event.error.stopReason } : {}),
          })
          break
        case 'state_change':
          closeInlineText(frames)
          frames.push({ kind: 'state_change', state: event.state })
          break
        case 'context_compact':
          // Without this case the terminal stream silently swallowed
          // every compaction event — cli surfaces had no way to tell
          // the user their token totals just dropped because the
          // daemon summarised earlier turns.
          closeInlineText(frames)
          frames.push({
            kind: 'context_compact',
            summary: event.summary ?? '',
            beforeTokens: event.beforeTokens ?? 0,
            afterTokens: event.afterTokens ?? 0,
          })
          break
        case 'run_contract':
          closeInlineText(frames)
          frames.push({ kind: 'run_contract', contract: event.contract })
          break
        case 'state_board':
          closeInlineText(frames)
          frames.push({
            kind: 'state_board',
            criteriaTotal: event.criteriaTotal,
            planTotal: event.planTotal,
            planDone: event.planDone,
            todosTotal: event.todosTotal,
            todosDone: event.todosDone,
            ...(event.currentNode ? { currentNode: event.currentNode } : {}),
            ...(event.nodeState ? { nodeState: event.nodeState } : {}),
            ...(event.iteration !== undefined ? { iteration: event.iteration } : {}),
            todos: event.todos,
            plan: event.plan,
          })
          break
        case 'phase_change': {
          // Surface phase transitions on the cli stream so operators
          // see live progress through long runs (implementation →
          // validation → review → finalize).
          closeInlineText(frames)
          const closed = (
            event as {
              closedPhase?: {
                phase: string
                usage: { inputTokens: number; outputTokens: number }
              }
            }
          ).closedPhase
          frames.push({
            kind: 'phase_change',
            enteredPhase: (event as { enteredPhase?: string | null }).enteredPhase ?? null,
            closedPhase: closed
              ? {
                  phase: closed.phase,
                  closedTokens:
                    (closed.usage?.inputTokens ?? 0) + (closed.usage?.outputTokens ?? 0),
                }
              : undefined,
          })
          break
        }
        case 'post_edit_findings': {
          // Surface the post-edit analysis to the operator. Splits the
          // diagnostics list into "broken callers" (`[caller]` prefix)
          // and "own diagnostics" so surfaces can highlight downstream
          // damage without re-grepping the summary text.
          closeInlineText(frames)
          const e = event as {
            editedFiles?: string[]
            reverseCallers?: string[]
            diagnostics?: Array<{ file: string; summary: string }>
          }
          const diagnostics = e.diagnostics ?? []
          frames.push({
            kind: 'post_edit_findings',
            editedFiles: e.editedFiles ?? [],
            reverseCallers: e.reverseCallers ?? [],
            brokenCallers: diagnostics.filter((d) => d.summary.startsWith('[caller]')),
            ownDiagnostics: diagnostics.filter((d) => !d.summary.startsWith('[caller]')),
          })
          break
        }
        case 'steering_ack':
          closeInlineText(frames)
          frames.push({
            kind: 'steering_ack',
            noteId: event.noteId,
            steeringKind: event.kind,
            steeringMessage: event.message,
          })
          break
        case 'steering_consumed':
          closeInlineText(frames)
          frames.push({ kind: 'steering_consumed', noteId: event.noteId })
          break
        default:
          return frames
      }

      return frames
    },
    getContent() {
      return fullContent
    },
  }
}
