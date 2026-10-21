import type { ChatRequest, Message } from '@sepilotd/core'

const DEFAULT_CHARS_PER_TOKEN = 4
const MIN_RATIO = 1.5
const MAX_RATIO = 8
const EMA_ALPHA = 0.3
const MIN_SAMPLE_TOKENS = 200
const NON_TEXT_PART_CHARS = 400

export function messageChars(messages: Message[]): number {
  let chars = 0
  for (const msg of messages) {
    if (typeof msg.content === 'string') {
      chars += msg.content.length
    } else if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        chars += part.type === 'text' ? part.text.length : NON_TEXT_PART_CHARS
      }
    }
    for (const toolCall of msg.toolCalls ?? []) {
      chars += `${toolCall.name} ${JSON.stringify(toolCall.arguments ?? {})}`.length
    }
  }
  return chars
}

const PER_MESSAGE_OVERHEAD_TOKENS = 4
const PER_TOOL_OVERHEAD_TOKENS = 4

export function requestChars(request: Pick<ChatRequest, 'messages' | 'systemPrompt' | 'tools'>): number {
  return messageChars(request.messages)
    + (request.systemPrompt?.length ?? 0)
    + (request.tools?.length ? JSON.stringify(request.tools).length : 0)
}

export class TokenRatioCalibrator {
  private ratios = new Map<string, number>()

  observe(providerId: string, model: string, chars: number, inputTokens: number): void {
    if (!Number.isFinite(chars) || !Number.isFinite(inputTokens)) return
    if (inputTokens < MIN_SAMPLE_TOKENS || chars <= 0) return
    const observed = Math.min(MAX_RATIO, Math.max(MIN_RATIO, chars / inputTokens))
    const key = `${providerId}:${model}`
    const prev = this.ratios.get(key)
    this.ratios.set(key, prev == null ? observed : prev + EMA_ALPHA * (observed - prev))
  }

  observeRequest(
    providerId: string,
    model: string,
    request: Pick<ChatRequest, 'messages' | 'systemPrompt' | 'tools'>,
    inputTokens: number,
  ): void {
    const overhead = request.messages.length * PER_MESSAGE_OVERHEAD_TOKENS
      + (request.tools?.length ?? 0) * PER_TOOL_OVERHEAD_TOKENS
    this.observe(providerId, model, requestChars(request), inputTokens - overhead)
  }

  charsPerToken(providerId?: string, model?: string): number {
    if (!providerId || !model) return DEFAULT_CHARS_PER_TOKEN
    return this.ratios.get(`${providerId}:${model}`) ?? DEFAULT_CHARS_PER_TOKEN
  }

  reset(): void {
    this.ratios.clear()
  }
}

export const tokenCalibration = new TokenRatioCalibrator()
