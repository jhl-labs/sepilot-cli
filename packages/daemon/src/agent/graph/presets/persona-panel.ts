// Persona-panel graph: a moderator first gathers meeting constraints, then
// runs a moderated discussion where the next speaker is selected from the
// evolving transcript instead of walking the roster once in order.
//
// Why a dedicated graph?
//
// The cowork graph also runs multiple agents in sequence, but it
// *decomposes* one task into role-specific subtasks. Persona-panel does
// the opposite — every panelist participates in the *same* meeting from
// their own system-prompted point of view. Reusing the cowork pipeline
// would have meant either teaching the LLM decomposition prompt to "split"
// the question into N copies of itself, or introducing a branch flag that
// swaps half the orchestrator behavior. A small dedicated graph is clearer
// and keeps the cowork specialist contract untouched.
//
// Streaming contract
//
// - `panel_open`   — emitted once with the resolved panelist roster
//   (id / name / avatar) so the UI can paint a header before any LLM call.
// - `panel_turn_start` / `panel_turn_complete` when the moderator grants
//   the floor to a persona. `_failed` is emitted instead when the provider
//   call throws, so one bad panelist doesn't stop the run.
// - `panel_synthesizing` then a final assistant message via state.output.

import { AgentGraph } from '../engine.js'
import { recordUsage, type Deps } from '../nodes.js'
import { ThinkingLevel, type ChatRequest, type Message } from '@sepilotd/core'
import type { AgentState, GraphExecutionContext } from '../types.js'
import { guardedProviderChat } from '../../../providers/circuit-breaker.js'
import { logLlmCallTrace } from '../../../observability/agent-trace.js'
import type { Persona } from '../../personas.js'
import { runUserFacingTextCall } from '../streaming.js'
import { getAbortError, isAbortError } from '../../../abort.js'
import { stopReasonCompletionGate } from '../../stop-reason.js'
import { reviewPanelTurn } from '../../panel-turn-review.js'

const MAX_PANELISTS = 6
const PANELIST_MAX_TOKENS = 1200
const SYNTHESIZE_MAX_TOKENS = 4096
const MAX_QUALITY_REPAIR_TURNS = 2
const SETUP_MARKER = '<!-- sepilotd:persona-panel-setup:v1 -->'
const HUMAN_QUESTION_MARKER_PREFIX = '<!-- sepilotd:persona-panel-human-question:v1:'
const HUMAN_QUESTION_MARKER_SUFFIX = ' -->'
const MODERATOR_ID = 'meeting-moderator'
const MODERATOR_NAME = '회의 진행자'

interface PanelTurn {
  personaId: string
  personaName: string
  text: string
  status?: 'success' | 'failed'
  instruction?: string
  reason?: string
}

interface FloorDecision {
  action: 'speak' | 'close' | 'ask_user'
  personaId?: string
  instruction?: string
  question?: string
  choices?: string[]
  reason?: string
}

interface QualityDecision {
  status: 'ok' | 'needs_more_discussion'
  personaId?: string
  instruction?: string
  reason?: string
  missing?: string[]
}

interface MeetingContext {
  topic: string
  setupAnswer: string
  maxTurns: number
}

interface SuspendedMeetingState extends MeetingContext {
  turns: PanelTurn[]
  qualityReviewed: boolean
  qualityRepairTurns: number
  humanDelegated: boolean
}

function buildPanelistSystemPrompt(persona: Persona): string {
  const base = persona.systemPromptAddition?.trim()
  return [
    base || `You are ${persona.name}.`,
    'You are one participant in a moderated multi-persona meeting.',
    'Speak only from your own role and expertise.',
    'Respond to the live discussion: build on useful points, challenge weak assumptions, and ask for missing constraints when your role would naturally do so.',
    'Do not summarize the whole meeting and do not impersonate other participants.',
    'Keep each intervention concise enough for a live meeting turn.',
  ].join(' ')
}

function buildSequentialPanelistSystemPrompt(persona: Persona): string {
  const base = persona.systemPromptAddition?.trim()
  return [
    base || `You are ${persona.name}.`,
    'You are one invited perspective in a multi-persona review.',
    'Answer only from your own role and expertise.',
    'Read the earlier perspectives before responding: build on useful points, challenge weak assumptions, and avoid repeating what is already covered.',
    'Do not impersonate the other participants and do not synthesize the entire panel.',
    'Give one self-contained, concise perspective on the original request.',
    'Apply the user’s language, length and format requirements to your own intervention. Those requirements override your persona’s default report template. Other participants have separate turns; never write their replies.',
  ].join(' ')
}

function textOfMessage(message: Message | undefined): string {
  const content = message?.content
  return typeof content === 'string' ? content : ''
}

function lastSetupPromptIndex(messages: Message[]): number {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i]
    if (message?.role === 'assistant' && textOfMessage(message).includes(SETUP_MARKER)) {
      return i
    }
  }
  return -1
}

function shouldAskForMeetingSetup(state: AgentState): boolean {
  return lastSetupPromptIndex(state.messages) === -1
    && !readSuspendedMeeting(state.messages)
    && !shouldDelegateWithoutQuestions(state.input)
}

function decodeSuspendedMeetingState(text: string): SuspendedMeetingState | null {
  const escapedPrefix = HUMAN_QUESTION_MARKER_PREFIX.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const escapedSuffix = HUMAN_QUESTION_MARKER_SUFFIX.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = text.match(new RegExp(`${escapedPrefix}([A-Za-z0-9_-]+)${escapedSuffix}`))
  if (!match?.[1]) return null
  try {
    const parsed = JSON.parse(Buffer.from(match[1], 'base64url').toString('utf8')) as Partial<SuspendedMeetingState>
    if (
      typeof parsed.topic !== 'string'
      || typeof parsed.setupAnswer !== 'string'
      || typeof parsed.maxTurns !== 'number'
      || !Array.isArray(parsed.turns)
    ) {
      return null
    }
    return {
      topic: parsed.topic,
      setupAnswer: parsed.setupAnswer,
      maxTurns: parsed.maxTurns,
      turns: parsed.turns.filter((turn): turn is PanelTurn => (
        typeof turn === 'object'
        && turn !== null
        && typeof turn.personaId === 'string'
        && typeof turn.personaName === 'string'
        && typeof turn.text === 'string'
      )),
      qualityReviewed: parsed.qualityReviewed === true,
      qualityRepairTurns: typeof parsed.qualityRepairTurns === 'number' ? parsed.qualityRepairTurns : 0,
      humanDelegated: parsed.humanDelegated === true,
    }
  } catch {
    return null
  }
}

function readSuspendedMeeting(messages: Message[]): SuspendedMeetingState | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i]
    if (message?.role !== 'assistant') continue
    const decoded = decodeSuspendedMeetingState(textOfMessage(message))
    if (decoded) return decoded
  }
  return null
}

function resolveMeetingContext(state: AgentState): MeetingContext {
  const setupIndex = lastSetupPromptIndex(state.messages)
  let topic = state.input
  if (setupIndex >= 0) {
    for (let i = setupIndex - 1; i >= 0; i -= 1) {
      const message = state.messages[i]
      if (message?.role === 'user') {
        const text = textOfMessage(message).trim()
        if (text) {
          topic = text
          break
        }
      }
    }
  }

  return {
    topic,
    setupAnswer: setupIndex >= 0 ? state.input : '',
    maxTurns: inferMaxTurns(state.input),
  }
}

function inferMaxTurns(answer: string): number {
  const lower = answer.toLowerCase()
  const turnMatch = lower.match(/(\d{1,2})\s*(?:턴|turns?|rounds?)/iu)
  if (turnMatch?.[1]) {
    const turns = Number.parseInt(turnMatch[1], 10)
    if (Number.isFinite(turns)) return Math.min(Math.max(turns, 2), 16)
  }

  const timeMatch = lower.match(/(\d{1,3})\s*(?:분|minutes?|mins?|min|m)/iu)
  if (timeMatch?.[1]) {
    const minutes = Number.parseInt(timeMatch[1], 10)
    if (Number.isFinite(minutes)) {
      if (minutes <= 5) return 4
      if (minutes <= 10) return 6
      if (minutes <= 20) return 10
      return 14
    }
  }
  if (/(짧게|간단|brief|quick|short)/iu.test(answer)) return 4
  if (/(길게|상세|thorough|deep|long|워크샵|workshop)/iu.test(answer)) return 12
  return 8
}

function buildSetupQuestion(personas: Persona[]): string {
  const roster = personas.map((persona) => `- ${persona.name}: ${persona.description ?? persona.id}`).join('\n')
  return [
    `${MODERATOR_NAME}입니다. 바로 결론을 만들기 전에 회의 조건을 정하겠습니다.`,
    '',
    '아래 항목에 답해 주세요. 모르면 "기본값으로 진행"이라고 답해도 됩니다.',
    '',
    '1. 회의 시간: 몇 분짜리 회의로 진행할까요? 예: 10분, 20분, 30분',
    '2. 산출물: 최종 결과는 무엇이어야 하나요? 예: 의사결정, 리스크 목록, 실행계획, 회의록',
    '3. 의사결정 방식: 합의, 진행자 권고안, 다수 의견, 장단점 비교 중 무엇을 원하나요?',
    '4. 회의 규칙/제약: 반드시 다뤄야 할 관점, 제외할 범위, 성공 기준이 있나요?',
    '',
    '참석 예정 페르소나:',
    roster,
    '',
    SETUP_MARKER,
  ].join('\n')
}

function formatTranscript(turns: PanelTurn[]): string {
  if (turns.length === 0) return '(no meeting turns yet)'
  return turns
    .map((turn, index) => {
      const meta = [
        `#${index + 1}`,
        turn.status === 'failed' ? 'status: failed' : '',
        turn.reason ? `floor reason: ${turn.reason}` : '',
        turn.instruction ? `instruction: ${turn.instruction}` : '',
      ].filter(Boolean).join(' | ')
      return `[${meta}]\n${turn.personaName}: ${turn.text}`
    })
    .join('\n\n')
}

function formatPanelMemories(turns: PanelTurn[]): string[] {
  return turns.map((turn) => (
    `[${turn.personaName}${turn.status === 'failed' ? ' - 응답 실패' : ''}]\n${turn.text}`
  ))
}

function formatRosterForPrompt(personas: Persona[], speakingCounts: Map<string, number>): string {
  return personas
    .map((persona) => [
      `- id=${persona.id}`,
      `name=${persona.name}`,
      `spoken=${speakingCounts.get(persona.id) ?? 0}`,
      `description=${persona.description ?? ''}`,
    ].join(' | '))
    .join('\n')
}

function extractJsonObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim()
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim()
  const candidate = fenced ?? trimmed.match(/\{[\s\S]*\}/)?.[0] ?? trimmed
  try {
    const parsed = JSON.parse(candidate) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null
  } catch {
    return null
  }
}

function parseStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const strings = value.filter((item): item is string => typeof item === 'string')
  return strings.length > 0 ? strings : undefined
}

function parseFloorDecision(text: string): FloorDecision | null {
  const parsed = extractJsonObject(text)
  if (!parsed) return null
  const action = parsed.action === 'close'
    ? 'close'
    : parsed.action === 'ask_user'
      ? 'ask_user'
      : 'speak'
  return {
    action,
    personaId: typeof parsed.personaId === 'string' ? parsed.personaId : undefined,
    instruction: typeof parsed.instruction === 'string' ? parsed.instruction : undefined,
    question: typeof parsed.question === 'string' ? parsed.question : undefined,
    choices: parseStringArray(parsed.choices),
    reason: typeof parsed.reason === 'string' ? parsed.reason : undefined,
  }
}

function parseQualityDecision(text: string): QualityDecision | null {
  const parsed = extractJsonObject(text)
  if (!parsed) return null
  const rawStatus = typeof parsed.status === 'string' ? parsed.status.toLowerCase() : ''
  const status = rawStatus === 'needs_more_discussion' || rawStatus === 'needs_more'
    ? 'needs_more_discussion'
    : 'ok'
  return {
    status,
    personaId: typeof parsed.personaId === 'string' ? parsed.personaId : undefined,
    instruction: typeof parsed.instruction === 'string' ? parsed.instruction : undefined,
    reason: typeof parsed.reason === 'string' ? parsed.reason : undefined,
    missing: parseStringArray(parsed.missing),
  }
}

function countPersonaTurns(turns: PanelTurn[]): number {
  return turns.filter((turn) => turn.personaId !== MODERATOR_ID).length
}

function shouldDelegateWithoutQuestions(text: string): boolean {
  return /(알아서|맡길게|위임|자율|임의|묻지\s*마|물어보지\s*마|질문하지\s*마|나에게\s*묻지|너희끼리|dont ask|don't ask|do not ask|no questions|decide yourself|use your judgment|use your judgement|up to you)/iu.test(text)
}

function rebuildSpeakingCounts(turns: PanelTurn[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const turn of turns) {
    if (turn.personaId === MODERATOR_ID) continue
    counts.set(turn.personaId, (counts.get(turn.personaId) ?? 0) + 1)
  }
  return counts
}

function encodeSuspendedMeetingState(snapshot: SuspendedMeetingState): string {
  return [
    HUMAN_QUESTION_MARKER_PREFIX,
    Buffer.from(JSON.stringify(snapshot), 'utf8').toString('base64url'),
    HUMAN_QUESTION_MARKER_SUFFIX,
  ].join('')
}

function buildHumanQuestionOutput(decision: FloorDecision, snapshot: SuspendedMeetingState): string {
  const question = decision.question
    ?? decision.instruction
    ?? '회의를 계속 진행하기 전에 사용자의 의사결정이 필요합니다.'
  const choices = decision.choices?.length
    ? ['', '선택지:', ...decision.choices.map((choice, index) => `${index + 1}. ${choice}`)]
    : []
  return [
    `${MODERATOR_NAME}입니다. 회의를 잠시 멈추고 확인할 사항이 있습니다.`,
    '',
    question,
    ...choices,
    '',
    '답변해 주시면 그 내용으로 이어서 회의하겠습니다.',
    '"그냥 알아서 해" 또는 "나에게 묻지마"라고 답하면 이번 회의의 남은 진행에서는 추가 질문 없이 페르소나끼리 판단합니다.',
    '',
    encodeSuspendedMeetingState(snapshot),
  ].join('\n')
}

function formatQualityReview(decision: QualityDecision, canContinue: boolean): string {
  const missing = decision.missing?.length
    ? `누락/보완 항목: ${decision.missing.join(', ')}`
    : ''
  if (decision.status === 'ok') {
    return [
      '완성도 리뷰: OK',
      decision.reason ? `판단 근거: ${decision.reason}` : '판단 근거: 요청한 산출물을 만들 수 있을 만큼 토론이 수렴되었습니다.',
    ].join('\n')
  }
  if (!canContinue) {
    return [
      '완성도 리뷰: 보완 필요',
      decision.reason ? `판단 근거: ${decision.reason}` : '판단 근거: 산출물 품질을 더 높일 보완점이 남아 있습니다.',
      missing,
      '다만 설정한 시간/턴 제한에 도달했으므로 현재 근거와 남은 공백을 명시해 최종 산출물을 작성합니다.',
    ].filter(Boolean).join('\n')
  }
  return [
    '완성도 리뷰: 보완 필요',
    decision.reason ? `판단 근거: ${decision.reason}` : '판단 근거: 산출물 품질을 높이려면 추가 발언이 필요합니다.',
    missing,
    decision.instruction ? `다음 보완 발언: ${decision.instruction}` : '다음 보완 발언: 누락된 근거와 실행 가능성을 보강합니다.',
  ].filter(Boolean).join('\n')
}

function leastHeardPersona(
  personas: Persona[],
  speakingCounts: Map<string, number>,
  avoidPersonaId?: string,
): Persona {
  const eligible = personas.filter((persona) => persona.id !== avoidPersonaId)
  const candidates = eligible.length > 0 ? eligible : personas
  return candidates
    .slice()
    .sort((a, b) => {
      const countDelta = (speakingCounts.get(a.id) ?? 0) - (speakingCounts.get(b.id) ?? 0)
      if (countDelta !== 0) return countDelta
      return personas.indexOf(a) - personas.indexOf(b)
    })[0] ?? personas[0]
}

function enforceFloorBalance(
  decision: FloorDecision | null,
  personas: Persona[],
  turns: PanelTurn[],
  speakingCounts: Map<string, number>,
  maxTurns: number,
): FloorDecision {
  const distinctSpeakers = new Set(
    turns.filter((turn) => turn.personaId !== MODERATOR_ID).map((turn) => turn.personaId),
  )
  const panelTurns = turns.filter((turn) => turn.personaId !== MODERATOR_ID)
  const minTurns = personas.length === 1 ? 1 : Math.min(Math.max(personas.length, 2), 4)
  if (panelTurns.length >= maxTurns) {
    return {
      action: 'close',
      reason: `timebox reached after ${panelTurns.length} persona turns`,
    }
  }

  if (
    decision?.action === 'close'
    && (panelTurns.length < minTurns || (personas.length > 1 && distinctSpeakers.size < 2))
  ) {
    const persona = leastHeardPersona(personas, speakingCounts)
    return {
      action: 'speak',
      personaId: persona.id,
      instruction: 'Open a missing perspective before the moderator closes the meeting.',
      reason: 'minimum discussion depth not reached',
    }
  }

  if (decision?.action === 'close') {
    return {
      action: 'close',
      reason: decision.reason ?? 'moderator closed the meeting',
    }
  }

  if (decision?.action === 'ask_user') {
    return decision
  }

  const requested = decision?.personaId
    ? personas.find((persona) => persona.id === decision.personaId)
    : undefined
  const lastSpeaker = [...turns].reverse().find((turn) => turn.personaId !== MODERATOR_ID)
  const requestedCount = requested ? speakingCounts.get(requested.id) ?? 0 : 0
  const minCount = Math.min(...personas.map((persona) => speakingCounts.get(persona.id) ?? 0))
  const overDominating = Boolean(
    requested
    && personas.length > 1
    && lastSpeaker?.personaId === requested.id
    && requestedCount > minCount,
  )
  const selected = requested && !overDominating
    ? requested
    : leastHeardPersona(personas, speakingCounts, lastSpeaker?.personaId)

  return {
    action: 'speak',
    personaId: selected.id,
    instruction: decision?.instruction ?? 'Add the next necessary perspective for this meeting.',
    reason: overDominating
      ? 'moderator rebalanced speaking time'
      : decision?.reason ?? 'moderator selected the next useful speaker',
  }
}

export function buildPersonaPanelGraph(deps: Deps): AgentGraph {
  const graph = new AgentGraph()

  const logCall = async (
    context: GraphExecutionContext | undefined,
    node: string,
    model: string,
    request: ChatRequest,
    response?: Awaited<ReturnType<typeof guardedProviderChat>>,
    error?: unknown,
  ): Promise<void> => {
    if (response) recordUsage(deps, context, model, response.usage)
    const errorMessage = error instanceof Error
      ? error.message
      : error !== undefined
        ? String(error)
        : undefined
    await logLlmCallTrace({
      source: 'graph',
      mode: context?.graphId ?? 'persona-panel',
      graphId: context?.graphId,
      node,
      sessionId: context?.agentContext.sessionId,
      provider: context?.agentContext.provider ?? deps.provider.id,
      model: context?.agentContext.model ?? model,
      request,
      response,
      error: errorMessage,
    })
  }

  // A successful HTTP response is not a completed intervention. In particular,
  // reasoning tokens can consume the small speech budget before any usable text.
  // Retain usage for both attempts, publish only a complete turn, and never let
  // a moderator summary hide a missing required perspective.
  const completePanelistTurn = async (
    state: AgentState,
    context: GraphExecutionContext | undefined,
    node: string,
    initialRequest: ChatRequest,
  ): Promise<string> => {
    let request = initialRequest
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const response = await guardedProviderChat({
        provider: deps.provider, request, signal: context?.signal,
        breaker: deps.providerCircuitBreaker,
      })
      await logCall(context, attempt === 0 ? node : `${node}.repair`, request.model, request, response)
      state.totalUsage.inputTokens += response.usage.inputTokens
      state.totalUsage.outputTokens += response.usage.outputTokens
      const text = typeof response.message.content === 'string' ? response.message.content.trim() : ''
      const complete = response.finishReason === 'stop' && text && !response.message.toolCalls?.length
      let repairReason = `The intervention was empty or incomplete (${response.finishReason}).`
      if (complete) {
        const review = await reviewPanelTurn({ provider: deps.provider, request: initialRequest, candidate: text,
          sessionId: context?.agentContext.sessionId, signal: context?.signal, breaker: deps.providerCircuitBreaker })
        if (review.response) {
          await logCall(context, `${node}.review.${attempt + 1}`, request.model, review.request, review.response)
          state.totalUsage.inputTokens += review.response.usage.inputTokens
          state.totalUsage.outputTokens += review.response.usage.outputTokens
        }
        if (review.accepted) return text
        repairReason = review.reason
      }
      if (attempt > 0 || (!complete && response.finishReason !== 'length' && (text || response.finishReason !== 'stop'))) {
        throw new Error(`Panelist intervention was not validated: ${repairReason}`)
      }
      const modelLimit = deps.provider.models.find((model) => model.id === request.model)?.maxOutputTokens ?? 8000
      request = {
        ...initialRequest,
        maxTokens: Math.min(modelLimit, context?.maxTokens ?? 8000),
        thinkingLevel: ThinkingLevel.Off,
        messages: [...initialRequest.messages, { role: 'system', content: `Correct the intervention: ${repairReason} Return one complete, concise intervention satisfying your own assignment and the user’s per-speaker constraints. Do not write other speakers, continue a fragment, or describe this repair.` }],
      }
    }
    throw new Error('Panelist response repair exhausted.')
  }

  const runSequentialPanel = async function* (
    state: AgentState,
    context: GraphExecutionContext | undefined,
    personas: Persona[],
  ) {
    yield {
      type: 'panel_open' as const,
      personas: personas.map((persona) => ({
        id: persona.id,
        name: persona.name,
        description: persona.description,
      })),
    }

    const turns: PanelTurn[] = []
    const model = context?.agentContext.model ?? deps.provider.models[0]?.id ?? 'default'
    for (const [index, persona] of personas.entries()) {
      yield {
        type: 'panel_turn_start' as const,
        personaId: persona.id,
        personaName: persona.name,
      }
      const request: ChatRequest = {
        model,
        messages: [
          { role: 'system', content: buildSequentialPanelistSystemPrompt(persona) },
          {
            role: 'user',
            content: [
              `Original request:\n${state.input}`,
              'Earlier perspectives (including failures):',
              formatTranscript(turns),
              '',
              'Give your perspective now.',
            ].join('\n\n'),
          },
        ],
        temperature: context?.temperature,
        maxTokens: context?.maxTokens ?? PANELIST_MAX_TOKENS,
      }

      try {
        const text = await completePanelistTurn(
          state, context, `persona-panel.sequential.${index + 1}.${persona.id}`, request,
        )
        turns.push({
          personaId: persona.id,
          personaName: persona.name,
          text,
          status: 'success',
        })
        yield {
          type: 'panel_turn_complete' as const,
          personaId: persona.id,
          personaName: persona.name,
          text,
        }
      } catch (error) {
        if (isAbortError(error) || context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Persona panel aborted')
        }
        await logCall(
          context,
          `persona-panel.sequential.${index + 1}.${persona.id}`,
          model,
          request,
          undefined,
          error,
        )
        const message = error instanceof Error ? error.message : String(error)
        state.stopReason ??= stopReasonCompletionGate({ unmet: [`${persona.name}: ${message}`] })
        turns.push({
          personaId: persona.id,
          personaName: persona.name,
          text: `응답 실패: ${message}`,
          status: 'failed',
          reason: 'provider call failed',
        })
        yield {
          type: 'panel_turn_failed' as const,
          personaId: persona.id,
          personaName: persona.name,
          error: message,
        }
      }
    }

    state.memories = formatPanelMemories(turns)
    return state
  }

  graph.addNode(
    'panel',
    async function* (state: AgentState, context?: GraphExecutionContext) {
      const personas = (context?.panelPersonas ?? []).slice(0, MAX_PANELISTS)
      if (personas.length === 0) {
        // No panelists configured — degrade to a single plain assistant
        // turn instead of failing the whole stream. The user sees the
        // model's default voice; the UI can warn that the panel was empty.
        const model = context?.agentContext.model ?? deps.provider.models[0]?.id ?? 'default'
        const request: ChatRequest = {
          model,
          messages: [{ role: 'user', content: state.input }],
          temperature: context?.temperature,
          maxTokens: context?.maxTokens ?? PANELIST_MAX_TOKENS,
        }
        const response = yield* runUserFacingTextCall({
          provider: deps.provider,
          request,
          signal: context?.signal,
          breaker: deps.providerCircuitBreaker,
          live: context?.textDeltaMode === 'live',
        })
        await logCall(context, 'persona-panel.fallback', model, request, response)
        state.output =
          typeof response.message.content === 'string'
            ? response.message.content
            : ''
        state.totalUsage.inputTokens += response.usage.inputTokens
        state.totalUsage.outputTokens += response.usage.outputTokens
        if (response.finishReason !== 'stop' || !state.output.trim()) {
          state.stopReason ??= stopReasonCompletionGate({ unmet: ['The panel fallback response was empty or incomplete.'] })
        }
        return state
      }

      if (context?.panelStrategy === 'sequential') {
        return yield* runSequentialPanel(state, context, personas)
      }

      if (shouldAskForMeetingSetup(state)) {
        state.output = buildSetupQuestion(personas)
        return state
      }

      yield {
        type: 'panel_open',
        personas: [
          {
            id: MODERATOR_ID,
            name: MODERATOR_NAME,
            description: '회의 진행, 발언권 조정, 결론 및 회의록 작성',
          },
          ...personas.map((p) => ({
            id: p.id,
            name: p.name,
            description: p.description,
          })),
        ],
      }

      const suspended = readSuspendedMeeting(state.messages)
      const meeting = suspended ?? resolveMeetingContext(state)
      const turns: PanelTurn[] = suspended?.turns.slice() ?? []
      const speakingCounts = rebuildSpeakingCounts(turns)
      let qualityReviewed = suspended?.qualityReviewed ?? false
      let qualityRepairTurns = suspended?.qualityRepairTurns ?? 0
      let humanDelegated = suspended?.humanDelegated
        ?? shouldDelegateWithoutQuestions(`${meeting.topic}\n${meeting.setupAnswer}`)

      if (suspended) {
        const delegatedNow = shouldDelegateWithoutQuestions(state.input)
        humanDelegated = humanDelegated || delegatedNow
        const moderatorResume = delegatedNow
          ? '사용자가 남은 회의 진행을 위임했습니다. 이후 추가 질문 없이 페르소나끼리 판단해 진행합니다.'
          : `사용자 답변을 받았습니다. 답변: ${state.input}`
        turns.push({
          personaId: MODERATOR_ID,
          personaName: MODERATOR_NAME,
          text: moderatorResume,
          reason: 'human answer',
        })
        yield {
          type: 'panel_turn_complete',
          personaId: MODERATOR_ID,
          personaName: MODERATOR_NAME,
          text: moderatorResume,
        }
      } else {
        const moderatorOpening = [
          `회의를 시작합니다. 시간/운영 조건: ${meeting.setupAnswer || '기본값'}`,
          `주제: ${meeting.topic}`,
          humanDelegated
            ? '사용자가 진행을 위임했으므로 중간 확인 질문 없이 페르소나끼리 판단해 진행하겠습니다.'
            : '발언권은 필요한 관점이 생길 때마다 조정하고, 필요한 의사결정은 사용자에게 확인하겠습니다.',
          '특정 페르소나가 독점하지 않도록 하겠습니다.',
        ].join('\n')
        turns.push({
          personaId: MODERATOR_ID,
          personaName: MODERATOR_NAME,
          text: moderatorOpening,
        })
        yield {
          type: 'panel_turn_complete',
          personaId: MODERATOR_ID,
          personaName: MODERATOR_NAME,
          text: moderatorOpening,
        }
      }

      const runQualityReview = async (
        model: string,
        reason: string,
        canContinue: boolean,
      ): Promise<QualityDecision> => {
        const remainingTurns = Math.max(meeting.maxTurns - countPersonaTurns(turns), 0)
        const request: ChatRequest = {
          model,
          messages: [
            {
              role: 'system',
              content: [
                'You are the quality reviewer for a facilitated multi-persona meeting.',
                'Decide whether the current transcript is ready to produce the requested output.',
                'Return ok only when the transcript has enough evidence, disagreement handling, decision basis, risks, and next steps for the user setup.',
                'If one targeted follow-up would materially improve the output and discussion budget remains, return needs_more_discussion.',
                'Return JSON only: {"status":"ok"|"needs_more_discussion","personaId":"...","instruction":"...","reason":"...","missing":["..."]}',
              ].join(' '),
            },
            {
              role: 'user',
              content: [
                `Review trigger:\n${reason}`,
                `Meeting topic:\n${meeting.topic}`,
                `User meeting setup answer:\n${meeting.setupAnswer || '(default requested)'}`,
                `Remaining persona turns before hard cap: ${remainingTurns}`,
                `Can continue discussion: ${canContinue ? 'yes' : 'no'}`,
                'Roster:',
                formatRosterForPrompt(personas, speakingCounts),
                'Transcript so far:',
                formatTranscript(turns),
              ].join('\n\n'),
            },
          ],
          temperature: context?.temperature,
          maxTokens: 700,
        }
        try {
          const response = await guardedProviderChat({
            provider: deps.provider,
            request,
            signal: context?.signal,
            breaker: deps.providerCircuitBreaker,
          })
          await logCall(context, 'persona-panel.moderator.quality', model, request, response)
          state.totalUsage.inputTokens += response.usage.inputTokens
          state.totalUsage.outputTokens += response.usage.outputTokens
          const parsed = parseQualityDecision(
            typeof response.message.content === 'string' ? response.message.content : '',
          )
          if (parsed) return parsed
          return {
            status: 'needs_more_discussion',
            instruction: 'Briefly identify any missing evidence, risk, decision basis, or next step before final synthesis.',
            reason: 'quality review response was not parseable',
            missing: ['quality review response could not be parsed'],
          }
        } catch (error) {
          if (isAbortError(error) || context?.signal?.aborted) {
            throw getAbortError(context?.signal, 'Persona panel aborted')
          }
          await logCall(
            context,
            'persona-panel.moderator.quality',
            model,
            request,
            undefined,
            error,
          )
          return {
            status: 'needs_more_discussion',
            instruction: 'Briefly identify any missing evidence, risk, decision basis, or next step before final synthesis.',
            reason: 'quality review failed',
            missing: ['quality review call failed'],
          }
        }
      }

      let floorAttempt = 0
      while (countPersonaTurns(turns) < meeting.maxTurns) {
        floorAttempt += 1
        const model = context?.agentContext.model ?? deps.provider.models[0]?.id ?? 'default'
        const floorRequest: ChatRequest = {
          model,
          messages: [
            {
              role: 'system',
              content: [
                'You are the moderator of a realistic multi-persona meeting.',
                'Pick the next persona who has the strongest reason to speak based on the live transcript.',
                'Treat each persona as self-nominating when their role has a material objection, clarification, synthesis, or decision pressure to add.',
                'Do not simply rotate through the roster.',
                'Protect airtime: avoid the same persona speaking twice in a row when others have had fewer turns.',
                humanDelegated
                  ? 'The user delegated decisions. Do not ask the user; resolve uncertainty inside the meeting.'
                  : 'If a user-only decision is blocking useful progress, you may ask the user instead of forcing a persona turn.',
                'Close only when the meeting has enough disagreement, convergence, and actionable next steps for the requested outcome.',
                'Return JSON only: {"action":"speak"|"close"|"ask_user","personaId":"...","instruction":"...","question":"...","choices":["..."],"reason":"..."}',
              ].join(' '),
            },
            {
              role: 'user',
              content: [
                `Meeting topic:\n${meeting.topic}`,
                `User meeting setup answer:\n${meeting.setupAnswer || '(default requested)'}`,
                `Remaining persona turns before hard cap: ${Math.max(meeting.maxTurns - countPersonaTurns(turns), 0)}`,
                `Can ask user: ${humanDelegated ? 'no' : 'yes'}`,
                'Roster:',
                formatRosterForPrompt(personas, speakingCounts),
                'Transcript so far:',
                formatTranscript(turns),
              ].join('\n\n'),
            },
          ],
          temperature: context?.temperature,
          maxTokens: 700,
        }

        let rawDecision: FloorDecision | null = null
        try {
          const response = await guardedProviderChat({
            provider: deps.provider,
            request: floorRequest,
            signal: context?.signal,
            breaker: deps.providerCircuitBreaker,
          })
          await logCall(context, `persona-panel.moderator.floor.${floorAttempt}`, model, floorRequest, response)
          state.totalUsage.inputTokens += response.usage.inputTokens
          state.totalUsage.outputTokens += response.usage.outputTokens
          rawDecision = parseFloorDecision(
            typeof response.message.content === 'string' ? response.message.content : '',
          )
        } catch (error) {
          if (isAbortError(error) || context?.signal?.aborted) {
            throw getAbortError(context?.signal, 'Persona panel aborted')
          }
          await logCall(
            context,
            `persona-panel.moderator.floor.${floorAttempt}`,
            model,
            floorRequest,
            undefined,
            error,
          )
        }

        let decision = enforceFloorBalance(
          rawDecision,
          personas,
          turns,
          speakingCounts,
          meeting.maxTurns,
        )
        if (decision.action === 'ask_user') {
          if (!humanDelegated) {
            const questionText = decision.question
              ?? decision.instruction
              ?? '회의를 계속 진행하기 전에 사용자의 의사결정이 필요합니다.'
            turns.push({
              personaId: MODERATOR_ID,
              personaName: MODERATOR_NAME,
              text: `사용자 확인 요청: ${questionText}`,
              reason: decision.reason ?? 'moderator requested human decision',
            })
            yield {
              type: 'panel_turn_complete',
              personaId: MODERATOR_ID,
              personaName: MODERATOR_NAME,
              text: questionText,
            }
            state.output = buildHumanQuestionOutput(decision, {
              topic: meeting.topic,
              setupAnswer: meeting.setupAnswer,
              maxTurns: meeting.maxTurns,
              turns,
              qualityReviewed,
              qualityRepairTurns,
              humanDelegated,
            })
            return state
          }

          const delegatedPersona = leastHeardPersona(personas, speakingCounts)
          decision = {
            action: 'speak',
            personaId: delegatedPersona.id,
            instruction: decision.instruction
              ?? decision.question
              ?? 'Resolve the blocked decision internally using your role perspective.',
            reason: 'user delegated decisions, so the moderator kept the meeting moving without asking',
          }
        }
        if (decision.action === 'close') {
          const remainingTurns = Math.max(meeting.maxTurns - countPersonaTurns(turns), 0)
          const canContinue = remainingTurns > 0 && qualityRepairTurns < MAX_QUALITY_REPAIR_TURNS
          const review = await runQualityReview(
            model,
            `moderator proposed close: ${decision.reason ?? 'no reason supplied'}`,
            canContinue,
          )
          qualityReviewed = true
          const reviewText = formatQualityReview(review, canContinue)
          turns.push({
            personaId: MODERATOR_ID,
            personaName: MODERATOR_NAME,
            text: reviewText,
            reason: 'quality review',
          })
          yield {
            type: 'panel_turn_complete',
            personaId: MODERATOR_ID,
            personaName: MODERATOR_NAME,
            text: reviewText,
          }

          if (review.status === 'ok' || !canContinue) break

          qualityRepairTurns += 1
          decision = enforceFloorBalance(
            {
              action: 'speak',
              personaId: review.personaId,
              instruction: review.instruction ?? 'Address the quality review gaps before final synthesis.',
              reason: review.reason ?? 'quality review requested one targeted follow-up',
            },
            personas,
            turns,
            speakingCounts,
            meeting.maxTurns,
          )
          if (decision.action === 'close') break
        }

        const persona = personas.find((candidate) => candidate.id === decision.personaId)
          ?? leastHeardPersona(personas, speakingCounts)
        yield {
          type: 'panel_turn_start',
          personaId: persona.id,
          personaName: persona.name,
        }
        const request: ChatRequest = {
          model,
          messages: [
            { role: 'system', content: buildPanelistSystemPrompt(persona) },
            {
              role: 'user',
              content: [
                `Meeting topic:\n${meeting.topic}`,
                `User meeting setup answer:\n${meeting.setupAnswer || '(default requested)'}`,
                `Moderator floor instruction:\n${decision.instruction ?? 'Speak where your perspective is needed.'}`,
                `Why you have the floor:\n${decision.reason ?? 'Your perspective is needed now.'}`,
                'You have the floor because your persona has an active reason to intervene now, not because of roster order.',
                'Transcript so far:',
                formatTranscript(turns),
                '',
                'Give your live meeting intervention now.',
              ].join('\n\n'),
            },
          ],
          temperature: context?.temperature,
          maxTokens: context?.maxTokens ?? PANELIST_MAX_TOKENS,
        }
        try {
          const text = await completePanelistTurn(state, context, `persona-panel.turn.${persona.id}`, request)
          speakingCounts.set(persona.id, (speakingCounts.get(persona.id) ?? 0) + 1)
          turns.push({
            personaId: persona.id,
            personaName: persona.name,
            text,
            instruction: decision.instruction,
            reason: decision.reason,
          })
          yield {
            type: 'panel_turn_complete',
            personaId: persona.id,
            personaName: persona.name,
            text,
          }
        } catch (error) {
          if (isAbortError(error) || context?.signal?.aborted) {
            throw getAbortError(context?.signal, 'Persona panel aborted')
          }
          await logCall(
            context,
            `persona-panel.turn.${persona.id}`,
            model,
            request,
            undefined,
            error,
          )
          const message = error instanceof Error ? error.message : String(error)
          state.stopReason ??= stopReasonCompletionGate({ unmet: [`${persona.name}: ${message}`] })
          // A failed provider attempt consumes a turn too. Otherwise repeated
          // failures leave the loop counter unchanged and can run forever.
          speakingCounts.set(persona.id, (speakingCounts.get(persona.id) ?? 0) + 1)
          turns.push({
            personaId: persona.id,
            personaName: persona.name,
            text: `응답 실패: ${message}`,
            status: 'failed',
            reason: 'provider call failed',
          })
          yield {
            type: 'panel_turn_failed',
            personaId: persona.id,
            personaName: persona.name,
            error: message,
          }
        }
      }

      if (!qualityReviewed && countPersonaTurns(turns) > 0) {
        const model = context?.agentContext.model ?? deps.provider.models[0]?.id ?? 'default'
        const review = await runQualityReview(
          model,
          'hard timebox reached before moderator close',
          false,
        )
        const reviewText = formatQualityReview(review, false)
        turns.push({
          personaId: MODERATOR_ID,
          personaName: MODERATOR_NAME,
          text: reviewText,
          reason: 'quality review',
        })
        yield {
          type: 'panel_turn_complete',
          personaId: MODERATOR_ID,
          personaName: MODERATOR_NAME,
          text: reviewText,
        }
      }

      // Stash the per-turn payload so the synthesize node can read it
      // without rerunning the provider; agent-state has no first-class
      // panel field, so we serialize into `coworkPlan`-style memories.
      state.input = `Topic: ${meeting.topic}\nSetup: ${meeting.setupAnswer || 'default'}`
      state.memories = formatPanelMemories(turns)
      return state
    },
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'synthesize',
    async function* (state: AgentState, context?: GraphExecutionContext) {
      // panel node already populated state.output for the single-panelist
      // path; in that case skip the extra LLM round-trip.
      if (state.output) return state
      if (context?.panelStrategy === 'sequential' && state.memories.length > 0) {
        // The sequential contract is one real intervention per selected
        // persona; later speakers already receive earlier perspectives.
        // Re-authoring these turns as a simulated meeting invents speakers,
        // timestamps and additional rebuttals. Publish the recorded turns.
        state.output = state.memories.join('\n\n')
        return state
      }
      if (state.memories.length === 0) {
        state.output = '[Persona panel produced no responses]'
        return state
      }
      yield {
        type: 'panel_synthesizing',
        panelists: state.memories.length,
      }
      const model = context?.agentContext.model ?? deps.provider.models[0]?.id ?? 'default'
      const request: ChatRequest = {
        model,
        messages: [
          {
            role: 'system',
            content: [
              'You are the moderator of a multi-persona panel.',
              'Synthesize the recorded discussion faithfully, following the user-requested scope, format and length.',
              'Include detailed minutes only when requested. Never simulate extra dialogue or invent participant identities, timestamps, decisions or unspoken rebuttals.',
              'Attribute distinctive points to each participant by name and call out genuine disagreements explicitly.',
              'Do not invent positions; if a panelist did not address something, do not pretend they did.',
            ].join(' '),
          },
          {
            role: 'user',
            content: `Meeting request and setup:\n${state.input}\n\nMeeting transcript:\n\n${state.memories.join('\n\n')}`,
          },
        ],
        temperature: context?.temperature,
        maxTokens: context?.maxTokens ?? SYNTHESIZE_MAX_TOKENS,
      }
      try {
        const response = yield* runUserFacingTextCall({
          provider: deps.provider,
          request,
          signal: context?.signal,
          breaker: deps.providerCircuitBreaker,
          live: context?.textDeltaMode === 'live',
        })
        await logCall(context, 'persona-panel.synthesize', model, request, response)
        state.totalUsage.inputTokens += response.usage.inputTokens
        state.totalUsage.outputTokens += response.usage.outputTokens
        state.output =
          typeof response.message.content === 'string'
            ? response.message.content
            : ''
        if (response.finishReason !== 'stop' || !state.output.trim()) {
          state.stopReason ??= stopReasonCompletionGate({ unmet: ['The panel synthesis was empty or incomplete.'] })
        }
      } catch (error) {
        if (isAbortError(error) || context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Persona panel aborted')
        }
        await logCall(
          context,
          'persona-panel.synthesize',
          model,
          request,
          undefined,
          error,
        )
        // Fall back to concatenated panel notes so the user still sees
        // something, even if the moderator pass failed.
        state.output = state.memories.join('\n\n')
        state.stopReason ??= stopReasonCompletionGate({ unmet: ['The panel synthesis failed; only individual notes are available.'] })
      }
      return state
    },
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'reporter',
    async (state: AgentState) => {
      if (!state.output) {
        state.output = '[Persona panel completed without output]'
      }
      state.shouldStop = true
      return state
    },
    { lifecycleState: 'done' },
  )

  graph.setStart('panel')
  graph.addEdge('panel', 'synthesize')
  graph.addEdge('synthesize', 'reporter')
  graph.addEdge('reporter', '__end__')

  return graph
}
