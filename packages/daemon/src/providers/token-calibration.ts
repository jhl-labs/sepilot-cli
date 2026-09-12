import type { Message } from '@sepilotd/core'

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

/**
 * Learns a chars-per-token ratio per provider:model from provider usage.
 * In-memory only; ratios reset on daemon restart and re-converge from usage.
 */
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

  charsPerToken(providerId?: string, model?: string): number {
    if (!providerId || !model) return DEFAULT_CHARS_PER_TOKEN
    return this.ratios.get(`${providerId}:${model}`) ?? DEFAULT_CHARS_PER_TOKEN
  }

  reset(): void {
    this.ratios.clear()
  }
}

export const tokenCalibration = new TokenRatioCalibrator()
