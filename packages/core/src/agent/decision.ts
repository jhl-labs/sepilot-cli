export type DecisionQuestionType = 'predicate' | 'choice' | 'score'

export interface DecisionChoiceOption {
  value: string
  description?: string
}

export interface DecisionScoreLevel {
  label: string
  description?: string
}

export interface DecisionPredicateQuestion {
  type: 'predicate'
  name: string
  instructions: string
}

export interface DecisionChoiceQuestion {
  type: 'choice'
  name: string
  instructions: string
  choices: DecisionChoiceOption[]
}

export interface DecisionScoreQuestion {
  type: 'score'
  name: string
  instructions: string
  levels: DecisionScoreLevel[]
}

export type DecisionQuestion =
  | DecisionPredicateQuestion
  | DecisionChoiceQuestion
  | DecisionScoreQuestion

export interface DecisionRequest {
  model?: string
  context: string | Record<string, unknown>
  questions: DecisionQuestion[]
}

export interface DecisionPredicateAnswer {
  type: 'predicate'
  name: string
  probability: number
}

export interface DecisionChoiceAnswer {
  type: 'choice'
  name: string
  choice: string
  probabilities: Record<string, number>
  confidence?: number
}

export interface DecisionScoreAnswer {
  type: 'score'
  name: string
  score: number
  probabilities: Record<string, number>
  confidence?: number
}

export interface DecisionRefusalAnswer {
  type: 'refusal'
  name: string
  reason?: string
}

export type DecisionAnswer =
  | DecisionPredicateAnswer
  | DecisionChoiceAnswer
  | DecisionScoreAnswer
  | DecisionRefusalAnswer

export interface DecisionUsage {
  inputTokens: number
  outputTokens: number
  cachedInputTokens?: number
}

export interface DecisionResponse {
  providerId: string
  model: string
  answers: DecisionAnswer[]
  usage?: DecisionUsage
  latencyMs: number
}

export interface DecisionRequestOptions {
  signal?: AbortSignal
  timeoutMs?: number
}

export interface IDecisionProvider {
  readonly id: string
  readonly name: string
  readonly kind: string
  readonly models: string[]
  decide(request: DecisionRequest, options?: DecisionRequestOptions): Promise<DecisionResponse>
}
