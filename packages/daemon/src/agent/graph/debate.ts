import { randomUUID } from 'node:crypto'
import type {
  AgentEvent,
  DebateDecision,
  DebateRole,
  DebateRoundEntry,
  DebateRoundSummary,
  ChatRequest,
  ILLMProvider,
} from '@sepilotd/core'
import type { AgentState, GraphExecutionContext } from './types.js'
import { guardedProviderChat } from '../../providers/circuit-breaker.js'
import { throwIfAborted } from '../../abort.js'

const ROLE_PROMPTS: Record<DebateRole, string> = {
  proposer:
    'You are the proposer. Take the current draft answer and produce the strongest, most concrete version of it. Keep the structure and intent of the draft, but make it sharper. Output only the proposed answer text.',
  critic:
    "You are the critic. Identify the most important weaknesses, risks, missing checks, or correctness issues in the proposer's draft. Cite specifics. Be direct. Output only the critique text.",
  resolver:
    "You are the resolver. Given the proposer's draft and the critic's feedback, produce the final answer. Start the response with one of [ACCEPT], [REVISE], or [REJECT] on its own line, followed by the final text.",
}

export interface DebateNodeDeps {
  provider: ILLMProvider
  model?: string
  maxTokens?: number
  systemPrompt?: string
}

export function debateNode(deps: DebateNodeDeps) {
  return async function* (
    state: AgentState,
    context?: GraphExecutionContext,
  ): AsyncGenerator<AgentEvent, AgentState, void> {
    const model = resolveDebateModel(deps, context)
    const topic = state.input
    const draft =
      state.output
      || state.reviewSummary
      || state.implementationSummary
      || state.findingsSummary
      || ''
    const entries: DebateRoundEntry[] = []
    let proposerOut = draft
    let criticOut = ''

    for (const role of ['proposer', 'critic', 'resolver'] as const) {
      const startedAt = new Date().toISOString()
      const userContent = buildUserMessage(role, topic, draft, proposerOut, criticOut)
      throwIfAborted(context?.signal, 'Provider request aborted')
      const request: ChatRequest = {
        model,
        systemPrompt: deps.systemPrompt,
        messages: [
          { role: 'system', content: ROLE_PROMPTS[role] },
          { role: 'user', content: userContent },
        ],
        maxTokens: deps.maxTokens ?? 1024,
      }
      const response = await guardedProviderChat({
        provider: deps.provider,
        request,
        signal: context?.signal,
        breaker: context?.providerCircuitBreaker,
      })
      const content = extractText(response.message?.content)
      const usage = normalizeUsage(response.usage)
      state.totalUsage.inputTokens += usage.inputTokens
      state.totalUsage.outputTokens += usage.outputTokens
      const tokensUsed = usage.totalTokens
      const endedAt = new Date().toISOString()
      entries.push({ role, content, tokensUsed, startedAt, endedAt })
      if (role === 'proposer') proposerOut = content
      if (role === 'critic') criticOut = content
    }

    const final = entries[2].content
    const decision = inferDecision(final)
    const round: DebateRoundSummary = {
      roundId: randomUUID(),
      topic,
      entries,
      finalDecision: decision,
      rationale: stripDecisionPrefix(final),
      createdAt: entries[0].startedAt,
    }
    state.debateRounds = [...(state.debateRounds ?? []), round]
    if (decision !== 'reject') {
      state.output = round.rationale || final
    } else {
      appendDebateRejectSignal(state, round.rationale || final)
    }
    yield { type: 'debate_round', round }
    return state
  }
}

function resolveDebateModel(
  deps: DebateNodeDeps,
  context?: GraphExecutionContext,
): string {
  const auxModel = context?.auxModel?.trim()
  if (auxModel && providerHasModel(deps.provider, auxModel)) {
    return auxModel
  }
  return context?.agentContext.model
    ?? deps.model
    ?? deps.provider.models[0]?.id
    ?? 'default'
}

function providerHasModel(provider: ILLMProvider, modelId: string): boolean {
  return provider.models.some((model) => model.id === modelId)
}

function normalizeUsage(
  usage: { totalTokens?: number; inputTokens?: number; outputTokens?: number } | undefined,
): { inputTokens: number; outputTokens: number; totalTokens: number } {
  const inputTokens = usage?.inputTokens ?? (usage?.totalTokens ?? 0)
  const outputTokens = usage?.outputTokens ?? 0
  const totalTokens = usage?.totalTokens ?? (inputTokens + outputTokens)
  return { inputTokens, outputTokens, totalTokens }
}

function compactSignalLine(text: string): string {
  return text.replace(/\s+/g, ' ').trim().slice(0, 400)
}

function appendDebateRejectSignal(state: AgentState, rationale: string): void {
  const reason = compactSignalLine(rationale) || 'Debate resolver rejected the reviewed answer.'
  const signal = `UNVERIFIED: Debate rejected final answer: ${reason}`
  state.reviewSummary = state.reviewSummary
    ? `${state.reviewSummary}\n${signal}`
    : signal
}

function buildUserMessage(
  role: DebateRole,
  topic: string,
  originalDraft: string,
  proposerOut: string,
  criticOut: string,
): string {
  if (role === 'proposer') {
    return `Topic:\n${topic}\n\nCurrent draft:\n${originalDraft || '(empty)'}`
  }
  if (role === 'critic') {
    return `Topic:\n${topic}\n\nProposer draft:\n${proposerOut || '(empty)'}`
  }
  return `Topic:\n${topic}\n\nProposer draft:\n${proposerOut || '(empty)'}\n\nCritic feedback:\n${criticOut || '(empty)'}`
}

function extractText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part && typeof part === 'object' && 'text' in part) {
          const text = (part as { text?: unknown }).text
          return typeof text === 'string' ? text : ''
        }
        return ''
      })
      .join('')
  }
  return ''
}

function inferDecision(text: string): DebateDecision {
  const head = text.trimStart().slice(0, 32).toUpperCase()
  if (head.startsWith('[REJECT]')) return 'reject'
  if (head.startsWith('[REVISE]')) return 'revise'
  if (head.startsWith('[ACCEPT]')) return 'accept'
  return 'accept'
}

function stripDecisionPrefix(text: string): string {
  return text.replace(/^\s*\[(?:ACCEPT|REVISE|REJECT)\]\s*/i, '')
}
