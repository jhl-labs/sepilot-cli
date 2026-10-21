import type { AgentState, GraphExecutionContext } from './types.js'
import { AutonomyLevel } from '@sepilotd/core'
import { isAbortError, raceWithAbort, throwIfAborted } from '../../abort.js'

export const MAX_LOOP_CONTROL_QUESTIONS = 2

export type LoopControlDecision = 'continue' | 'different_approach' | 'stop'
export const LOOP_CONTROL_CHOICES: readonly LoopControlDecision[] = [
  'continue',
  'different_approach',
  'stop',
]

export interface LoopControlQuestionInput {
  kind: 'stuck_repeat' | 'no_progress'
  tool?: string
  count: number
}

export interface LoopControlAnswer {
  decision: LoopControlDecision
  guidance?: string
}

function containsHangul(value: string | undefined): boolean {
  return typeof value === 'string' && /[ㄱ-ㆎ가-힣]/u.test(value)
}

export function buildLoopControlQuestionPrompt(
  input: LoopControlQuestionInput,
  userInput?: string,
): string {
  const korean = containsHangul(userInput)
  if (input.kind === 'stuck_repeat') {
    const tool = input.tool ? `\`${input.tool}\`` : (korean ? '같은 도구 호출' : 'the same tool call')
    return korean
      ? `에이전트가 진전 없이 ${tool}을(를) ${input.count}번 반복했습니다. 계속할까요, 다른 접근을 시도할까요, 아니면 중단할까요? (continue / different_approach / stop)`
      : `The agent repeated ${tool} ${input.count} times without progress. Continue, try a different approach, or stop? (continue / different_approach / stop)`
  }
  return korean
    ? `에이전트가 ${input.count}번 연속으로 도구 호출이나 최종 답변 없이 계획만 반복했습니다. 계속할까요, 다른 접근을 시도할까요, 아니면 중단할까요? (continue / different_approach / stop)`
    : `The agent spent ${input.count} consecutive iterations without a tool call or a final answer. Continue, try a different approach, or stop? (continue / different_approach / stop)`
}

export function parseLoopControlAnswer(raw: string | undefined): LoopControlAnswer {
  const answer = (raw ?? '').trim()
  if (!answer) return { decision: 'stop' }
  const normalized = answer.toLowerCase().replace(/[\s-]+/g, '_')
  if (LOOP_CONTROL_CHOICES.includes(normalized as LoopControlDecision)) {
    return { decision: normalized as LoopControlDecision }
  }
  return { decision: 'different_approach', guidance: answer }
}

export function canAskLoopControlQuestion(
  state: Pick<AgentState, 'loopControlQuestionCount'>,
  context: (Pick<GraphExecutionContext, 'requestQuestion'> & Partial<Pick<GraphExecutionContext, 'autonomy'>>) | undefined,
): boolean {
  return Boolean(context?.requestQuestion)
    && context?.autonomy !== AutonomyLevel.Autonomous
    && (state.loopControlQuestionCount ?? 0) < MAX_LOOP_CONTROL_QUESTIONS
}

export async function askLoopControlQuestion(
  state: Pick<AgentState, 'loopControlQuestionCount' | 'input'>,
  context: Pick<GraphExecutionContext, 'requestQuestion' | 'signal'> & Partial<Pick<GraphExecutionContext, 'autonomy'>> & { sessionId: string },
  input: LoopControlQuestionInput,
): Promise<LoopControlAnswer | undefined> {
  if (!canAskLoopControlQuestion(state, context) || !context.requestQuestion) return undefined
  throwIfAborted(context.signal, 'Loop-control question aborted')
  state.loopControlQuestionCount = (state.loopControlQuestionCount ?? 0) + 1
  let raw: string | undefined
  try {
    raw = await raceWithAbort(context.requestQuestion({
      sessionId: context.sessionId,
      prompt: buildLoopControlQuestionPrompt(input, state.input),
      choices: [...LOOP_CONTROL_CHOICES],
    }), context.signal, 'Loop-control question aborted')
  } catch (error) {
    if (context.signal?.aborted || isAbortError(error) || isParkedDecisionError(error)) throw error
    return { decision: 'stop' }
  }
  return parseLoopControlAnswer(raw)
}

function isParkedDecisionError(error: unknown): boolean {
  const code = (error as { code?: unknown } | null)?.code
  return code === 'QUESTION_TIMEOUT' || code === 'APPROVAL_TIMEOUT'
}

export function buildDifferentApproachMessage(
  input: LoopControlQuestionInput & { signature?: string; guidance?: string },
): string {
  const lines = ['[Loop-control] The user asked for a different approach.']
  if (input.kind === 'stuck_repeat') {
    lines.push(
      `Do NOT call ${input.tool ? `\`${input.tool}\`` : 'the repeated tool'} with the same arguments again`
      + (input.signature ? ` (blocked signature: ${input.signature}).` : '.'),
      'Choose a materially different tool or arguments, or answer from the evidence already gathered.',
    )
  } else {
    lines.push(
      'Do not restate the plan or describe tool calls in prose.',
      'In your next reply either call a tool directly or reply with ANSWER:/INCOMPLETE: and the concrete result or blocker.',
    )
  }
  if (input.guidance) lines.push(`User guidance: ${input.guidance}`)
  return lines.join(' ')
}
