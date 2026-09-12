import { randomUUID } from 'node:crypto'
import { assertNever, type AgentEvent, type ISessionStore } from '@sepilotd/core'
import { logAgentDebugTrace } from '../observability/agent-trace.js'

function debugEventIdentifiers(event: AgentEvent): {
  turnId?: string
  requestId?: string
  toolCallId?: string
} {
  switch (event.type) {
    case 'llm_request':
      return { turnId: event.turnId }
    case 'tool_call':
      return { toolCallId: event.toolCall.id }
    case 'tool_result':
      return { toolCallId: event.toolCallId }
    case 'approval_request':
    case 'auto_approval':
      return { requestId: event.requestId, toolCallId: event.toolCall.id }
    case 'approval_response':
      return { requestId: event.requestId }
    default:
      return {}
  }
}

function debugEventData(event: AgentEvent): Record<string, unknown> {
  if (event.type === 'text_delta') {
    return {
      type: event.type,
      chars: event.text.length,
      preview: event.text.slice(0, 240),
    }
  }
  if (event.type === 'tool_result') {
    return {
      ...event,
      contentParts: event.contentParts?.map((part) => ({
        type: part.type,
        ...(part.type === 'text' ? { chars: part.text.length } : { omitted: true }),
      })),
    }
  }
  return { ...event }
}

/**
 * Raw provider text/reasoning arrives in token-sized deltas. The correlated
 * llm.call trace already stores the bounded, redacted request/response, while
 * graph/node/tool events preserve control-flow evidence. Writing every delta
 * again made a single model call produce thousands of JSONL records and hid
 * the useful events in operator tooling. Keep the deltas live for clients,
 * but omit only those two duplicate stream variants from the debug journal.
 */
export function shouldTraceAgentSessionEvent(event: AgentEvent): boolean {
  return event.type !== 'thinking' && event.type !== 'text_delta'
}

export async function persistAgentSessionEvent(
  sessions: ISessionStore,
  sessionId: string,
  event: AgentEvent,
): Promise<void> {
  const timestamp = new Date().toISOString()

  if (shouldTraceAgentSessionEvent(event)) {
    await logAgentDebugTrace({
      event: 'agent.event',
      source: 'session-events',
      sessionId,
      ...debugEventIdentifiers(event),
      status: event.type,
      data: debugEventData(event),
    })
  }

  switch (event.type) {
    case 'memory_context':
      await sessions.appendEvent(sessionId, {
        type: 'memory_context',
        id: event.id,
        timestamp,
        items: event.items,
      })
      break
    case 'llm_request':
      await sessions.appendEvent(sessionId, {
        type: 'llm_request',
        id: randomUUID(),
        timestamp,
        turnId: event.turnId,
        iteration: event.iteration,
        requestDigest: event.requestDigest,
      })
      break
    case 'mode_route_decision':
      await sessions.appendEvent(sessionId, {
        type: 'mode_route_decision',
        id: randomUUID(),
        timestamp,
        chosen: event.chosen,
        persona: event.persona,
        candidates: event.candidates,
        reason: event.reason,
        confidence: event.confidence,
        fallback: event.fallback,
      })
      break
    case 'quality_gate_verdict':
      await sessions.appendEvent(sessionId, {
        type: 'quality_gate_verdict',
        id: randomUUID(),
        timestamp,
        phase: event.phase,
        decision: event.decision,
        blockingReason: event.blockingReason,
        backtrackCount: event.backtrackCount,
      })
      break
    case 'backtrack':
      await sessions.appendEvent(sessionId, {
        type: 'backtrack',
        id: randomUUID(),
        timestamp,
        phase: event.phase,
        reason: event.reason,
        attempt: event.attempt,
      })
      break
    case 'node_trace':
      await sessions.appendEvent(sessionId, {
        type: 'node_trace',
        id: randomUUID(),
        timestamp,
        node: event.node,
        durationMs: event.durationMs,
        nextEdge: event.nextEdge,
      })
      break
    case 'tool_call':
      await sessions.appendEvent(sessionId, {
        type: 'tool_call',
        id: event.toolCall.id,
        timestamp,
        tool: event.toolCall.name,
        input: event.toolCall.arguments,
        status: 'executing',
      })
      break
    case 'approval_request':
      await sessions.appendEvent(sessionId, {
        type: 'approval_request',
        id: event.requestId,
        timestamp,
        toolCallId: event.toolCall.id,
        tool: event.toolCall.name,
        input: event.toolCall.arguments,
        status: 'pending',
        ...(event.context ? { context: event.context } : {}),
      })
      break
    case 'approval_response':
      await sessions.appendEvent(sessionId, {
        type: 'approval_response',
        id: randomUUID(),
        timestamp,
        requestId: event.requestId,
        decision: event.decision,
        approved: event.approved,
        note: event.note,
        approvedBy: 'agent',
      })
      break
    case 'auto_approval':
      // Durable counterpart of the transient AgentEvent: the cli prints
      // a `[auto-approved] ... rule '<pattern>'` banner from the
      // AgentEvent, but a later `sessions show` audit needs the same
      // decision in the journal so an operator can reconstruct *which*
      // remembered rule authorised each silent tool execution. Without
      // this, tool_call/tool_result pairs in the journal would have no
      // consent trail.
      await sessions.appendEvent(sessionId, {
        type: 'auto_approval',
        id: randomUUID(),
        timestamp,
        requestId: event.requestId,
        toolCallId: event.toolCall.id,
        tool: event.toolCall.name,
        input: event.toolCall.arguments,
        decision: event.decision,
        scope: event.scope,
        rule: event.rule,
      })
      break
    case 'tool_result':
      await sessions.appendEvent(sessionId, {
        type: 'tool_result',
        id: randomUUID(),
        timestamp,
        toolCallId: event.toolCallId,
        output: event.output,
        status: event.status,
        duration_ms: 0,
        recovery: event.recovery,
        ...(event.executionPosture ? { executionPosture: event.executionPosture } : {}),
        ...(event.metadata ? { metadata: event.metadata } : {}),
      })
      break
    case 'recovery':
      await sessions.appendEvent(sessionId, {
        type: 'recovery',
        id: randomUUID(),
        timestamp,
        scope: event.scope,
        kind: event.kind,
        action: event.action,
        message: event.message,
        recoverable: event.recoverable,
        details: event.details,
      })
      break
    case 'cowork_plan':
      await sessions.appendEvent(sessionId, {
        type: 'cowork_plan',
        id: randomUUID(),
        timestamp,
        plan: event.plan,
      })
      break
    case 'run_contract':
      await sessions.appendEvent(sessionId, {
        type: 'run_contract',
        id: randomUUID(),
        timestamp,
        contract: event.contract,
      })
      break
    case 'cowork_task_start':
      await sessions.appendEvent(sessionId, {
        type: 'cowork_task_start',
        id: randomUUID(),
        timestamp,
        role: event.role,
        instruction: event.instruction,
      })
      break
    case 'cowork_task_complete':
      await sessions.appendEvent(sessionId, {
        type: 'cowork_task_complete',
        id: randomUUID(),
        timestamp,
        role: event.role,
        instruction: event.instruction,
        result: event.result,
      })
      break
    case 'cowork_task_failed':
      await sessions.appendEvent(sessionId, {
        type: 'cowork_task_failed',
        id: randomUUID(),
        timestamp,
        role: event.role,
        instruction: event.instruction,
        error: event.error,
      })
      break
    case 'cowork_synthesizing':
      await sessions.appendEvent(sessionId, {
        type: 'cowork_synthesizing',
        id: randomUUID(),
        timestamp,
        summary: event.summary,
      })
      break
    case 'cowork_discuss_request':
      await sessions.appendEvent(sessionId, {
        type: 'cowork_discuss_request',
        id: randomUUID(),
        timestamp,
        prompt: event.prompt,
        choices: event.choices,
      })
      break
    case 'cowork_discuss_response':
      await sessions.appendEvent(sessionId, {
        type: 'cowork_discuss_response',
        id: randomUUID(),
        timestamp,
        prompt: event.prompt,
        response: event.response,
      })
      break
    case 'edit_checkpoint_opened':
      await sessions.appendEvent(sessionId, {
        type: 'edit_checkpoint_opened',
        id: event.checkpoint.checkpointId,
        timestamp,
        checkpoint: event.checkpoint,
      })
      break
    case 'edit_checkpoint_resolved':
      await sessions.appendEvent(sessionId, {
        type: 'edit_checkpoint_resolved',
        id: randomUUID(),
        timestamp,
        checkpoint: event.checkpoint,
      })
      break
    case 'debate_round':
      await sessions.appendEvent(sessionId, {
        type: 'debate_round',
        id: event.round.roundId,
        timestamp,
        round: event.round,
      })
      break
    case 'planner_working_memory_updated':
      await sessions.appendEvent(sessionId, {
        type: 'planner_working_memory_updated',
        id: randomUUID(),
        timestamp,
        workingMemory: event.workingMemory,
      })
      break
    case 'phase_change':
      await sessions.appendEvent(sessionId, {
        type: 'phase_change',
        id: randomUUID(),
        timestamp,
        enteredPhase: event.enteredPhase,
        closedPhase: event.closedPhase,
        phaseUsages: event.phaseUsages,
      })
      break
    case 'post_edit_findings':
      await sessions.appendEvent(sessionId, {
        type: 'post_edit_findings',
        id: randomUUID(),
        timestamp,
        editedFiles: event.editedFiles,
        impactedExternalModules: event.impactedExternalModules,
        impactedLocalModules: event.impactedLocalModules,
        reverseCallers: event.reverseCallers,
        diagnostics: event.diagnostics,
        analyzedAt: event.analyzedAt,
      })
      break
    case 'context_compact':
      await sessions.appendEvent(sessionId, {
        type: 'context_compact',
        id: randomUUID(),
        timestamp,
        summary: event.summary,
        beforeTokens: event.beforeTokens,
        afterTokens: event.afterTokens,
      })
      break
    case 'router_decision':
      await sessions.appendEvent(sessionId, {
        type: 'router_decision',
        id: event.id,
        timestamp,
        decision: event.decision,
      })
      break
    // persona-panel 이벤트 4종. desktop 트랜스크립트가 live stream 때만 panel UI를
    // 보이고 reload 후엔 단일 통합 답변으로 collapse되던 문제 때문에 journal로 둠.
    // panel_turn_start는 UI 효과가 없어 저장 생략 (turn_complete가 그 자리를 메움).
    case 'panel_open':
      await sessions.appendEvent(sessionId, {
        type: 'panel_open',
        id: randomUUID(),
        timestamp,
        personas: event.personas,
      })
      break
    case 'panel_turn_complete':
      await sessions.appendEvent(sessionId, {
        type: 'panel_turn_complete',
        id: randomUUID(),
        timestamp,
        personaId: event.personaId,
        personaName: event.personaName,
        text: event.text,
      })
      break
    case 'panel_turn_failed':
      await sessions.appendEvent(sessionId, {
        type: 'panel_turn_failed',
        id: randomUUID(),
        timestamp,
        personaId: event.personaId,
        personaName: event.personaName,
        error: event.error,
      })
      break
    case 'panel_synthesizing':
      await sessions.appendEvent(sessionId, {
        type: 'panel_synthesizing',
        id: randomUUID(),
        timestamp,
        panelists: event.panelists,
      })
      break
    case 'done':
      await sessions.appendEvent(sessionId, {
        type: 'session_end',
        id: randomUUID(),
        timestamp,
        totalTokens: {
          input: event.usage.inputTokens,
          output: event.usage.outputTokens,
        },
        totalCost: 0,
        duration_ms: 0,
        ...(event.stopReason ? { stopReason: event.stopReason } : {}),
      })
      break
    case 'state_change':
    // Effective permissions are request-scoped telemetry. Persisting an old
    // run's clamp as though it still applied after config changes would be
    // misleading, while SEPILOT_DEBUG traces retain the full decision.
    case 'execution_policy':
    // Context pressure is request-scoped live telemetry. The durable session
    // already stores provider usage in session_end; replaying an old request's
    // pressure as current context would be misleading.
    case 'context_usage':
    case 'thinking':
    case 'reasoning_step':
    case 'action_progress':
    case 'text_delta':
    case 'message':
    // Pending human questions are owned by PendingQuestionStore and exposed
    // through `/sessions/:id/questions` plus live `question_request` frames.
    // Do not duplicate them into the session event log.
    case 'question_request':
    case 'panel_turn_start':
    case 'subagent_progress':
    case 'error':
    // 'state_board' is stream-only here: the full board snapshot is
    // already journaled separately via `appendStateBoardEvent` (session
    // resume/watch), so this lightweight counts-only AgentEvent only needs
    // to reach the live chat stream for the cli/gui progress heartbeat.
    case 'state_board':
    // 'steering_consumed' (AgentEvent) is stream-only here too: the durable
    // acknowledgement is already journaled as a `SteeringConsumedEvent`
    // `SessionEvent` via `createJournalSteeringConsumed` at the consume site
    // (see `agent/graph/nodes.ts` `buildAgentMessages`), so this lightweight
    // AgentEvent only needs to reach the live chat stream so cli/gui surfaces
    // can render a "steering note applied" line.
    case 'steering_consumed':
      // Stream-only or persisted elsewhere by the caller. Keep this list
      // explicit so adding a new AgentEvent requires a persistence decision.
      break
    default:
      assertNever(event, 'session-events')
  }
}
