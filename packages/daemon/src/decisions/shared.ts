import type {
  DecisionQuestion,
  DecisionRequest,
  DecisionRequestOptions,
} from '@sepilotd/core'
import { getProviderSdkFetch } from '../providers/http-timeout.js'

export type DecisionErrorCode =
  | 'INVALID_DECISION_REQUEST'
  | 'DECISION_PROVIDER_HTTP'
  | 'DECISION_PROVIDER_TIMEOUT'
  | 'DECISION_PROVIDER_RESPONSE'

export class DecisionProviderError extends Error {
  constructor(
    readonly code: DecisionErrorCode,
    message: string,
    readonly status?: number,
  ) {
    super(message)
    this.name = 'DecisionProviderError'
  }
}

export function assertValidDecisionRequest(request: DecisionRequest): void {
  if (!Array.isArray(request.questions) || request.questions.length === 0) {
    throw new DecisionProviderError('INVALID_DECISION_REQUEST', 'at least one question is required')
  }
  const names = new Set<string>()
  for (const question of request.questions) {
    if (!question.name?.trim()) {
      throw new DecisionProviderError('INVALID_DECISION_REQUEST', 'every question needs a name')
    }
    if (names.has(question.name)) {
      throw new DecisionProviderError('INVALID_DECISION_REQUEST', `duplicate question name '${question.name}'`)
    }
    names.add(question.name)
    if (!question.instructions?.trim()) {
      throw new DecisionProviderError('INVALID_DECISION_REQUEST', `question '${question.name}' needs instructions`)
    }
    assertQuestionOptions(question)
  }
}

function assertQuestionOptions(question: DecisionQuestion): void {
  if (question.type === 'choice') {
    const values = question.choices.map((choice) => choice.value)
    if (values.length < 2 || new Set(values).size !== values.length) {
      throw new DecisionProviderError(
        'INVALID_DECISION_REQUEST',
        `choice question '${question.name}' needs at least two unique options`,
      )
    }
  } else if (question.type === 'score' && question.levels.length < 2) {
    throw new DecisionProviderError(
      'INVALID_DECISION_REQUEST',
      `score question '${question.name}' needs at least two levels`,
    )
  }
}

export function contextAsText(context: DecisionRequest['context']): string {
  return typeof context === 'string' ? context : JSON.stringify(context)
}

export interface DecisionHttpTarget {
  url: string
  headers: Record<string, string>
  body: unknown
}

export async function postDecisionJson(
  target: DecisionHttpTarget,
  options: DecisionRequestOptions | undefined,
  defaultTimeoutMs: number,
): Promise<unknown> {
  const timeoutMs = options?.timeoutMs ?? defaultTimeoutMs
  const timeout = AbortSignal.timeout(timeoutMs)
  const signal = options?.signal ? AbortSignal.any([options.signal, timeout]) : timeout
  let response: Response
  try {
    response = await getProviderSdkFetch()(target.url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...target.headers },
      body: JSON.stringify(target.body),
      signal,
    })
  } catch (error) {
    if (timeout.aborted && !options?.signal?.aborted) {
      throw new DecisionProviderError(
        'DECISION_PROVIDER_TIMEOUT',
        `decision provider did not answer within ${timeoutMs}ms`,
      )
    }
    throw error
  }
  const text = await response.text()
  if (!response.ok) {
    throw new DecisionProviderError(
      'DECISION_PROVIDER_HTTP',
      `decision provider responded ${response.status}: ${summarizeBody(text)}`,
      response.status,
    )
  }
  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new DecisionProviderError('DECISION_PROVIDER_RESPONSE', 'decision provider returned non-JSON body')
  }
}

function summarizeBody(text: string): string {
  const trimmed = text.replace(/\s+/g, ' ').trim()
  return trimmed.length > 200 ? `${trimmed.slice(0, 200)}…` : trimmed
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

export function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function joinUrl(baseUrl: string, path: string): string {
  const base = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
  return `${base}${path}`
}
