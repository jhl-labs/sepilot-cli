import type { ChatResponse, ILLMProvider, Message } from '@sepilotd/core'
import { extractContent } from './utils.js'

export const MODEL_PROBE_TIMEOUT_MS = 30_000

const MODEL_PROBE_INITIAL_MAX_TOKENS = 64
const MODEL_PROBE_RETRY_MAX_TOKENS = 1_024

export interface ModelProbeTarget {
  providerId: string
  modelId: string
}

export interface ModelProbeResult extends ModelProbeTarget {
  ok: boolean
  latencyMs: number
  reason?: string
}

function probeMaxTokens(provider: ILLMProvider, modelId: string, requested: number): number {
  const configuredMaximum = provider.models.find((model) => model.id === modelId)?.maxOutputTokens
  if (!Number.isFinite(configuredMaximum) || !configuredMaximum || configuredMaximum < 1) {
    return requested
  }
  return Math.max(1, Math.min(requested, Math.floor(configuredMaximum)))
}

function extractProbeContent(message: Message): string {
  const normalized = extractContent(message)
  if (normalized.trim() || !Array.isArray(message.content)) return normalized

  // Keep the readiness probe tolerant of provider adapters that return a
  // text-bearing block before normalizing it to the core `{ type: 'text' }`
  // shape. It is still visible text, while thinking/reasoning fields remain
  // intentionally excluded.
  return message.content
    .map((part) => {
      const text = (part as { text?: unknown }).text
      return typeof text === 'string' ? text : ''
    })
    .join('\n')
}

function emptyResponseReason(finishReason: ChatResponse['finishReason'] | undefined): string {
  const reason = 'model returned an empty response'
  return finishReason ? `${reason} (finishReason: ${finishReason})` : reason
}

/**
 * Exercise one concrete provider/model pair and require user-visible text.
 *
 * Reasoning models can spend a very small output allowance entirely on hidden
 * thinking. Give every probe a practical initial budget, then retry one time
 * with a larger allowance before declaring the model unusable.
 */
export async function probeProviderModel(
  provider: ILLMProvider,
  target: ModelProbeTarget,
  timeoutMs = MODEL_PROBE_TIMEOUT_MS,
): Promise<ModelProbeResult> {
  const startedAt = Date.now()
  const controller = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  let lastFinishReason: ChatResponse['finishReason'] | undefined
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const error = new Error(`model probe timed out after ${timeoutMs}ms`)
      // Reject independently of provider AbortSignal support. Some provider
      // SDKs do not cancel an in-flight non-streaming request when the signal
      // changes.
      reject(error)
      controller.abort(error)
    }, timeoutMs)
    timer.unref?.()
  })

  try {
    const tokenBudgets = Array.from(new Set([
      probeMaxTokens(provider, target.modelId, MODEL_PROBE_INITIAL_MAX_TOKENS),
      probeMaxTokens(provider, target.modelId, MODEL_PROBE_RETRY_MAX_TOKENS),
    ]))
    for (const maxTokens of tokenBudgets) {
      const response = await Promise.race([
        provider.chat({
          model: target.modelId,
          messages: [{ role: 'user', content: 'Reply with exactly: OK' }],
          temperature: 0,
          maxTokens,
          timeoutMs,
        }, { signal: controller.signal }),
        deadline,
      ])
      lastFinishReason = response.finishReason
      if (extractProbeContent(response.message).trim()) {
        return {
          ok: true,
          providerId: target.providerId,
          modelId: target.modelId,
          latencyMs: Date.now() - startedAt,
        }
      }
    }

    return {
      ok: false,
      providerId: target.providerId,
      modelId: target.modelId,
      latencyMs: Date.now() - startedAt,
      reason: emptyResponseReason(lastFinishReason),
    }
  } catch (error) {
    return {
      ok: false,
      providerId: target.providerId,
      modelId: target.modelId,
      latencyMs: Date.now() - startedAt,
      reason: error instanceof Error ? error.message : String(error),
    }
  } finally {
    if (timer) clearTimeout(timer)
  }
}
