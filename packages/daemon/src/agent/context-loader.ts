import type { Message, ISessionStore, PlannerWorkingMemory, TodoItem } from '@sepilotd/core'
import { getCompactContinuationMessage } from './session-compaction.js'
import { summarizeToolOutputForAgentContext } from './tool-output.js'
import { coalesceAdjacentAssistantToolCallMessages } from './tool-protocol-context.js'
import { replayConversationTextControls } from './turn-context.js'
import type { AgentSeedContract } from './graph/types.js'
import {
  buildStateBoard,
  formatStateBoard,
  isStateBoardEnabled,
  type AgentStateBoardSource,
} from './graph/state-board.js'

/**
 * Load previous conversation messages from a session.
 * Returns Message[] suitable for passing to the LLM.
 * Limits to last `maxMessages` to stay within context window.
 */
export async function loadSessionContext(
  sessionStore: ISessionStore,
  sessionId: string,
  maxMessages?: number,
): Promise<Message[]> {
  const events = await sessionStore.getEvents(sessionId)
  const messages: Message[] = []
  // Durable board seed (P022-T5): the latest contract/todo/planner-working-memory
  // survive context compaction, so track them across the whole event stream —
  // not inside `messages`, which `context_compact` clears — and rehydrate the
  // state board out-of-band on follow-up turns. Previously these events were
  // dropped, so a new turn lost the prior goal/plan/todo.
  let rehydratedContract: AgentSeedContract | undefined
  let rehydratedTodos: TodoItem[] | undefined
  let rehydratedPlannerMemory: PlannerWorkingMemory | undefined
  // Map tool_call id -> tool name so a reloaded tool_result can be summarized
  // with the same per-tool budget the live loop applies (see below).
  const toolNameByCallId = new Map<string, string>()

  for (const event of events) {
    switch (event.type) {
      case 'user_message':
        messages.push({ role: 'user', content: replayConversationTextControls(event.content) })
        break
      case 'run_contract':
        rehydratedContract = event.contract
        break
      case 'todo_list':
        rehydratedTodos = event.items
        break
      case 'planner_working_memory_updated':
        rehydratedPlannerMemory = event.workingMemory
        break
      case 'assistant_message':
        messages.push({ role: 'assistant', content: replayConversationTextControls(event.content) })
        break
      case 'tool_call':
        toolNameByCallId.set(event.id, event.tool)
        messages.push({
          role: 'assistant',
          content: '',
          toolCalls: [{ id: event.id, name: event.tool, arguments: event.input }],
        })
        break
      case 'tool_result':
        {
          const toolName = toolNameByCallId.get(event.toolCallId)
          messages.push({
            role: 'tool',
            // Re-summarize large tool output on reload so the reloaded view
            // matches the live loop (which stores the summarized form via
            // summarizeToolOutputForAgentContext). Without this, a resumed
            // session re-injected the full multi-KB raw output, bloating the
            // context and diverging from what the model saw live.
            content: summarizeToolOutputForAgentContext(
              toolName ?? '',
              event.output,
            ),
            toolCallId: event.toolCallId,
            ...(toolName ? { name: toolName } : {}),
            metadata: {
              ...(event.metadata ?? {}),
              status: event.status,
            },
          })
        }
        break
      case 'context_compact':
        // Replace all previous messages with the summary
        messages.length = 0
        messages.push({
          role: 'system',
          content: getCompactContinuationMessage(event.summary, {
            recentMessagesPreserved: (event.preservedMessages?.length ?? 0) > 0,
            suppressFollowUpQuestions: true,
          }),
        })
        if (event.preservedMessages?.length) {
          messages.push(...event.preservedMessages)
        }
        break
    }
  }

  // Trim to last N messages if needed
  const limit = maxMessages ?? DEFAULT_SESSION_CONTEXT_MAX_MESSAGES
  const normalizedMessages = coalesceAdjacentAssistantToolCallMessages(messages)
  const trimmed = stripStalePlannerBlocks(trimMessagesToLimit(normalizedMessages, limit))
  // Rehydrate the board after trimming so it is never trimmed away, and prepend
  // it as an out-of-band system block. `buildAgentMessages` replaces it with the
  // fresh live board when the run rebuilds one this turn (same STATE_BOARD_PREFIX).
  const boardMessage = buildRehydratedBoardMessage(
    rehydratedContract,
    rehydratedTodos,
    rehydratedPlannerMemory,
  )
  return boardMessage ? [boardMessage, ...trimmed] : trimmed
}

/**
 * Rebuild the state board from the durable contract/todo/planner seed and
 * render it as a system message, or `null` when there is nothing durable to
 * rehydrate (or the board is disabled via `SEPILOTD_STATE_BOARD=0`). The
 * evidence section is dropped: the evidence ledger is per-run and not part of
 * the durable seed, so a rehydrated board carries only the durable goal/plan/
 * todo/failed/open-question state, never stale contract-evidence gaps.
 */
export function buildRehydratedBoardMessage(
  contract: AgentSeedContract | undefined,
  todos: TodoItem[] | undefined,
  plannerWorkingMemory: PlannerWorkingMemory | undefined,
): Message | null {
  if (!isStateBoardEnabled()) return null
  if (!contract && !(todos && todos.length > 0) && !plannerWorkingMemory) return null
  const source: AgentStateBoardSource = {
    seedContract: contract,
    todoList: todos,
    plannerWorkingMemory,
  }
  const board = buildStateBoard(source)
  board.evidenceSection = null
  const formatted = formatStateBoard(board)
  return formatted ? { role: 'system', content: formatted } : null
}

export const DEFAULT_SESSION_CONTEXT_MAX_MESSAGES = 50

/**
 * Default message-count cap for session context loads. Overridable with
 * `SEPILOTD_SESSION_CONTEXT_MAX_MESSAGES` for operators running models with
 * unusually small or large context windows.
 */
export function resolveSessionContextMaxMessages(): number {
  const raw = Number(process.env.SEPILOTD_SESSION_CONTEXT_MAX_MESSAGES)
  if (Number.isFinite(raw) && raw >= 4) {
    return Math.floor(raw)
  }
  return DEFAULT_SESSION_CONTEXT_MAX_MESSAGES
}

/**
 * Trim to the last `maxMessages` entries without starting the window on an
 * orphaned `tool` message. A tool result whose `tool_call` assistant message
 * fell outside the window is rejected by strict providers (OpenAI-style 400),
 * so the window grows backwards until it starts on a non-tool message.
 */
export function trimMessagesToLimit(
  messages: Message[],
  maxMessages: number | undefined,
): Message[] {
  if (maxMessages === undefined || messages.length <= maxMessages) {
    return messages
  }
  let keepFrom = messages.length - maxMessages
  while (keepFrom > 0 && messages[keepFrom]?.role === 'tool') {
    keepFrom -= 1
  }
  return messages.slice(keepFrom)
}

/**
 * Drop ````planner` JSON blocks from every assistant message
 * except the most recent one. The model is asked to emit a planner snapshot
 * at the end of each non-trivial turn (see `system-prompt.ts`); without
 * stripping, every prior turn's snapshot rides along on every subsequent
 * LLM call and the planner JSON ends up dominating the prompt budget. The
 * latest plan is the only one with current state, so that's what we keep.
 *
 * The session jsonl event log is unchanged — this only trims the
 * in-memory copy fed to the model.
 */
const PLANNER_BLOCK_REGEX = /\n?```(?:planner|json:planner)\s*\n[\s\S]*?\n?```\s*$/i

export function stripStalePlannerBlocks(messages: Message[]): Message[] {
  // Select the most recent *substantive* assistant message to preserve. A turn
  // that ends in tool calls emits a trailing assistant message with empty
  // content (and a `toolCalls` array); picking that as the "latest" would strip
  // the planner block from the real preceding assistant message and lose the
  // current plan. Skip empty / tool-call-only assistant messages and keep the
  // last one that actually carries a planner block (falling back to the last
  // non-empty assistant message).
  let lastAssistantIdx = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]
    if (message.role !== 'assistant' || typeof message.content !== 'string') continue
    if (message.toolCalls && message.toolCalls.length > 0) continue
    if (message.content.trim().length === 0) continue
    if (lastAssistantIdx < 0) lastAssistantIdx = i
    if (PLANNER_BLOCK_REGEX.test(message.content)) {
      lastAssistantIdx = i
      break
    }
  }
  if (lastAssistantIdx < 0) return messages
  let changed = false
  const next = messages.map((message, i) => {
    if (i === lastAssistantIdx) return message
    if (message.role !== 'assistant' || typeof message.content !== 'string') return message
    if (!PLANNER_BLOCK_REGEX.test(message.content)) return message
    changed = true
    return { ...message, content: message.content.replace(PLANNER_BLOCK_REGEX, '').trimEnd() }
  })
  return changed ? next : messages
}

/**
 * Estimate token count for messages (rough approximation: 1 token ≈ 4 chars)
 */
export function estimateTokens(messages: Message[]): number {
  let chars = 0
  for (const msg of messages) {
    if (typeof msg.content === 'string') chars += msg.content.length
    else if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === 'text') chars += part.text.length
      }
    }
  }
  return Math.ceil(chars / 4)
}
