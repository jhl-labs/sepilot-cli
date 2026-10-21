import { ThinkingLevel, type ILLMProvider } from '@sepilotd/core'
import { runAuxiliaryLlmChat, type AuxiliaryLlmTurnBudget } from './auxiliary-llm.js'
import type { ProviderCircuitBreaker } from '../providers/circuit-breaker.js'

export async function generateSessionTitle(
  provider: ILLMProvider,
  model: string,
  firstMessage: string,
  assistantReply: string,
  options: { signal?: AbortSignal; breaker?: ProviderCircuitBreaker; budget?: AuxiliaryLlmTurnBudget } = {},
): Promise<string> {
  try {
    const response = await runAuxiliaryLlmChat({
      provider,
      request: {
        model,
        messages: [{
          role: 'user',
          content: `Generate a short title (max 50 chars) for a conversation that starts with:\nUser: ${firstMessage.slice(0, 200)}\nAssistant: ${assistantReply.slice(0, 200)}\n\nReturn ONLY the title, no quotes, no explanation.`,
        }],
        maxTokens: 30,
        thinkingLevel: ThinkingLevel.Off,
      },
      label: 'Session title generator',
      signal: options.signal,
      breaker: options.breaker,
      budget: options.budget,
    })
    const title = typeof response.message.content === 'string' ? response.message.content.trim() : ''
    return title.length > 0 && title.length <= 60 ? title : firstMessage.slice(0, 50)
  } catch {
    return firstMessage.slice(0, 50)
  }
}
