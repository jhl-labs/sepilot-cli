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
  contextAsText,
  DecisionProviderError,
  joinUrl,
  postDecisionJson,
} from './shared.js'

export interface OpenAIDecisionsProviderConfig {
  id: string
  name?: string
  apiKey: string
  baseUrl?: string
  headers?: Record<string, string>
  models: string[]
  timeoutMs: number
}

export const OPENAI_DECISIONS_DEFAULT_BASE_URL = 'https://api.openai.com'
export const OPENAI_DECISIONS_DEFAULT_MODEL = 'gpt-6-luna'

export class OpenAIDecisionsProvider implements IDecisionProvider {
  readonly id: string
  readonly name: string
  readonly kind = 'openai-decisions'
  readonly models: string[]
  private readonly baseUrl: string
  private readonly headers: Record<string, string>
  private readonly timeoutMs: number

  constructor(config: OpenAIDecisionsProviderConfig) {
    this.id = config.id
    this.name = config.name ?? 'OpenAI Decisions'
    this.models = config.models.length > 0 ? [...config.models] : [OPENAI_DECISIONS_DEFAULT_MODEL]
    this.baseUrl = config.baseUrl ?? OPENAI_DECISIONS_DEFAULT_BASE_URL
    this.timeoutMs = config.timeoutMs
    this.headers = { ...(config.headers ?? {}), authorization: `Bearer ${config.apiKey}` }
  }

  async decide(request: DecisionRequest, options?: DecisionRequestOptions): Promise<DecisionResponse> {
    assertValidDecisionRequest(request)
    const model = request.model ?? this.models[0]
    const startedAt = Date.now()
    const raw = await postDecisionJson(
      {
        url: joinUrl(this.baseUrl, '/v1/decisions'),
        headers: this.headers,
        body: {
          model,
          input: contextAsText(request.context),
          questions: request.questions.map(toOpenAIQuestion),
        },
      },
      options,
      this.timeoutMs,
    )
    const rawAnswers = asRecord(raw)?.answers
    if (!Array.isArray(rawAnswers)) {
      throw new DecisionProviderError('DECISION_PROVIDER_RESPONSE', 'decisions response has no answers array')
    }
    return {
      providerId: this.id,
      model: typeof asRecord(raw)?.model === 'string' ? (asRecord(raw)?.model as string) : model,
      answers: request.questions.map((question, index) => parseOpenAIAnswer(question, rawAnswers, index)),
      usage: parseUsage(raw),
      latencyMs: Date.now() - startedAt,
    }
  }
}

function toOpenAIQuestion(question: DecisionQuestion): Record<string, unknown> {
  switch (question.type) {
    case 'predicate':
      return { type: 'predicate', name: question.name, instructions: question.instructions }
    case 'choice':
      return {
        type: 'choice',
        name: question.name,
        instructions: question.instructions,
        choices: question.choices.map((choice) => ({
          value: choice.value,
          ...(choice.description ? { description: choice.description } : {}),
        })),
      }
    case 'score':
      return {
        type: 'score',
        name: question.name,
        instructions: question.instructions,
        levels: question.levels.map((level) => ({
          label: level.label,
          ...(level.description ? { description: level.description } : {}),
        })),
      }
  }
}

function parseOpenAIAnswer(question: DecisionQuestion, rawAnswers: unknown[], index: number): DecisionAnswer {
  const byName = rawAnswers.map(asRecord).find((entry) => entry?.name === question.name)
  const answer = byName ?? asRecord(rawAnswers[index])
  if (!answer) {
    return { type: 'refusal', name: question.name, reason: 'no answer returned' }
  }
  if (answer.type === 'refusal') {
    return { type: 'refusal', name: question.name }
  }
  switch (question.type) {
    case 'predicate': {
      const probability = asNumber(answer.probability)
      if (probability === undefined) return malformed(question.name, 'probability')
      return { type: 'predicate', name: question.name, probability }
    }
    case 'choice': {
      const probabilities = probabilityList(answer.probabilities, 'value')
      const choice = answer.choice
      if (!probabilities || (typeof choice !== 'string' && typeof choice !== 'boolean')) {
        return malformed(question.name, 'choice')
      }
      return {
        type: 'choice',
        name: question.name,
        choice: String(choice),
        probabilities,
        confidence: asNumber(answer.confidence),
      }
    }
    case 'score': {
      const score = asNumber(answer.score)
      const probabilities = probabilityList(answer.probabilities, 'value')
      if (score === undefined || !probabilities) return malformed(question.name, 'score')
      return { type: 'score', name: question.name, score, probabilities, confidence: asNumber(answer.confidence) }
    }
  }
}

function malformed(name: string, field: string): never {
  throw new DecisionProviderError('DECISION_PROVIDER_RESPONSE', `answer '${name}' is missing a valid '${field}' field`)
}

function probabilityList(value: unknown, keyField: string): Record<string, number> | null {
  if (!Array.isArray(value)) return null
  const entries: Array<[string, number]> = []
  for (const item of value) {
    const record = asRecord(item)
    const probability = asNumber(record?.probability)
    const key = record?.[keyField]
    if (probability === undefined || (typeof key !== 'string' && typeof key !== 'number' && typeof key !== 'boolean')) {
      return null
    }
    entries.push([String(key), probability])
  }
  return Object.fromEntries(entries)
}

function parseUsage(raw: unknown): DecisionResponse['usage'] {
  const usage = asRecord(asRecord(raw)?.usage)
  const inputTokens = asNumber(usage?.input_tokens)
  if (inputTokens === undefined) return undefined
  const details = asRecord(usage?.input_tokens_details)
  return {
    inputTokens,
    outputTokens: asNumber(usage?.output_tokens) ?? 0,
    cachedInputTokens: asNumber(details?.cached_tokens),
  }
}
