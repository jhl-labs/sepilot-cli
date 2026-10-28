import type {
  DecisionAnswer,
  DecisionQuestion,
  DecisionRequest,
  DecisionRequestOptions,
  DecisionResponse,
  IDecisionProvider,
} from '@sepilotd/core'
import {
  asNumber,
  asRecord,
  assertValidDecisionRequest,
  DecisionProviderError,
  joinUrl,
  postDecisionJson,
} from './shared.js'

export interface SystemOneDecisionProviderConfig {
  id: string
  name?: string
  baseUrl: string
  apiKey?: string
  headers?: Record<string, string>
  models: string[]
  timeoutMs: number
  endpointPath?: string
}

export const SYSTEMONE_DEFAULT_PATH = '/v1/systemone'
export const AI_GATEWAY_DECISIONS_PATH = '/v1/decisions'

export class SystemOneDecisionProvider implements IDecisionProvider {
  readonly id: string
  readonly name: string
  readonly kind = 'systemone'
  readonly models: string[]
  private readonly baseUrl: string
  private readonly headers: Record<string, string>
  private readonly timeoutMs: number
  private readonly endpointPath: string

  constructor(config: SystemOneDecisionProviderConfig) {
    this.id = config.id
    this.name = config.name ?? 'System One'
    this.endpointPath = config.endpointPath ?? SYSTEMONE_DEFAULT_PATH
    this.models = [...config.models]
    this.baseUrl = config.baseUrl
    this.timeoutMs = config.timeoutMs
    this.headers = {
      ...(config.headers ?? {}),
      ...(config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : {}),
    }
  }

  async decide(request: DecisionRequest, options?: DecisionRequestOptions): Promise<DecisionResponse> {
    assertValidDecisionRequest(request)
    const model = request.model ?? this.models[0]
    if (!model) {
      throw new DecisionProviderError('INVALID_DECISION_REQUEST', `decision provider '${this.id}' has no model configured`)
    }
    const startedAt = Date.now()
    const raw = await postDecisionJson(
      {
        url: joinUrl(this.baseUrl, this.endpointPath),
        headers: this.headers,
        body: {
          model,
          state: request.context,
          questions: Object.fromEntries(request.questions.map((question) => [question.name, toSystemOneQuestion(question)])),
        },
      },
      options,
      this.timeoutMs,
    )
    return {
      providerId: this.id,
      model,
      answers: request.questions.map((question) => parseSystemOneAnswer(question, raw)),
      usage: parseUsage(raw),
      latencyMs: Date.now() - startedAt,
    }
  }
}

function toSystemOneQuestion(question: DecisionQuestion): Record<string, unknown> {
  switch (question.type) {
    case 'predicate':
      return { type: 'noul', instructions: question.instructions }
    case 'choice':
      return {
        type: 'choice',
        instructions: question.instructions,
        criteria: Object.fromEntries(question.choices.map((choice) => [choice.value, choice.description ?? choice.value])),
      }
    case 'score':
      return {
        type: 'score',
        instructions: question.instructions,
        criteria: question.levels.map((level) => (level.description ? `${level.label}: ${level.description}` : level.label)),
      }
  }
}

function parseSystemOneAnswer(question: DecisionQuestion, raw: unknown): DecisionAnswer {
  const answers = asRecord(asRecord(raw)?.answers)
  const answer = asRecord(answers?.[question.name])
  if (!answer) {
    return { type: 'refusal', name: question.name, reason: 'no answer returned' }
  }
  switch (question.type) {
    case 'predicate': {
      const probability = asNumber(answer.noul)
      if (probability === undefined) return malformed(question.name, 'noul')
      return { type: 'predicate', name: question.name, probability }
    }
    case 'choice': {
      const probabilities = numberRecord(answer.probabilities)
      const choice = typeof answer.choice === 'string' ? answer.choice : undefined
      if (!choice || !probabilities) return malformed(question.name, 'choice')
      return { type: 'choice', name: question.name, choice, probabilities, confidence: asNumber(answer.confidence) }
    }
    case 'score': {
      const score = asNumber(answer.score)
      const probabilities = numberRecord(answer.probabilities)
      if (score === undefined || !probabilities) return malformed(question.name, 'score')
      return { type: 'score', name: question.name, score, probabilities, confidence: asNumber(answer.confidence) }
    }
  }
}

function malformed(name: string, field: string): never {
  throw new DecisionProviderError('DECISION_PROVIDER_RESPONSE', `answer '${name}' is missing a valid '${field}' field`)
}

function numberRecord(value: unknown): Record<string, number> | null {
  const record = asRecord(value)
  if (!record) return null
  const entries: Array<[string, number]> = []
  for (const [key, entry] of Object.entries(record)) {
    const probability = asNumber(entry)
    if (probability === undefined) return null
    entries.push([key, probability])
  }
  return Object.fromEntries(entries)
}

function parseUsage(raw: unknown): DecisionResponse['usage'] {
  const usage = asRecord(asRecord(raw)?.usage)
  const inputTokens = asNumber(usage?.input_tokens) ?? asNumber(usage?.prompt_tokens)
  if (inputTokens === undefined) return undefined
  return {
    inputTokens,
    outputTokens: asNumber(usage?.output_tokens) ?? asNumber(usage?.completion_tokens) ?? 0,
  }
}
