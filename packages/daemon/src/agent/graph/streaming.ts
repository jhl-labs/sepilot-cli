import type { AgentEvent, ChatRequest, ChatResponse, ILLMProvider, TokenUsage } from '@sepilotd/core'
import {
  guardedProviderChat,
  guardedProviderStream,
  type ProviderCircuitBreaker,
} from '../../providers/circuit-breaker.js'
import { extractContent } from '../../providers/utils.js'
import { stripFinalAnswerStem } from '../interim-progress.js'
import {
  containsPromptToolCallEnvelope,
  resolveExplicitFinalTransportText,
  UNUSABLE_PROMPT_TOOL_CALL_OUTPUT,
} from '../prompt-react.js'
import { INTERRUPTED_USER_FACING_RESPONSE } from '../response-safety.js'

function cleanUserFacingText(text: string): string {
  const tagged = text.match(/^\s*<(final|answer)>\s*([\s\S]*?)\s*<\/\1>\s*$/i)
  return stripFinalAnswerStem(tagged?.[2] ?? text).trim()
}

function withUserFacingContent(response: ChatResponse, content: string): ChatResponse {
  return {
    ...response,
    message: {
      ...response.message,
      content,
    },
  }
}

export async function* runUserFacingTextCall(params: {
  provider: ILLMProvider
  request: ChatRequest
  signal?: AbortSignal
  breaker?: ProviderCircuitBreaker
  live?: boolean
}): AsyncGenerator<AgentEvent, ChatResponse, void> {
  if (!params.live) {
    return yield* chatAsFinalDelta(params)
  }

  const usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
  let text = ''
  let thinking = ''
  let finishReason: ChatResponse['finishReason'] = 'stop'
  let sawStreamPayload = false
  let sawDone = false

  try {
    for await (const chunk of guardedProviderStream({
      provider: params.provider,
      request: params.request,
      signal: params.signal,
      breaker: params.breaker,
    })) {
      switch (chunk.type) {
        case 'text':
          sawStreamPayload = true
          text += chunk.text
          break
        case 'thinking':
          sawStreamPayload = true
          thinking += chunk.text
          yield { type: 'thinking', content: chunk.text }
          break
        case 'usage':
          sawStreamPayload = true
          usage.inputTokens += chunk.usage.inputTokens
          usage.outputTokens += chunk.usage.outputTokens
          break
        case 'done':
          sawDone = true
          finishReason = chunk.finishReason
          break
        case 'error':
          throw new Error(chunk.error.message)
        case 'tool_call_start':
        case 'tool_call_delta':
        case 'tool_call_end':
          sawStreamPayload = true
          break
      }
    }
  } catch (error) {
    if (params.signal?.aborted) {
      throw error
    }
    if (text.length > 0) {
      yield { type: 'text_delta', text: INTERRUPTED_USER_FACING_RESPONSE }
      return {
        message: { role: 'assistant', content: INTERRUPTED_USER_FACING_RESPONSE },
        thinking: thinking || undefined,
        usage,
        finishReason: 'length',
      }
    }
    return yield* chatAsFinalDelta(params)
  }

  if (!sawStreamPayload || (text.length === 0 && thinking.length === 0)) {
    return yield* chatAsFinalDelta(params)
  }

  if (!sawDone || finishReason !== 'stop') {
    yield { type: 'text_delta', text: INTERRUPTED_USER_FACING_RESPONSE }
    return {
      message: { role: 'assistant', content: INTERRUPTED_USER_FACING_RESPONSE },
      thinking: thinking || undefined,
      usage,
      finishReason,
    }
  }

  const protocolText = resolveExplicitFinalTransportText(text, thinking)
  if (containsPromptToolCallEnvelope(protocolText)) {
    return {
      message: { role: 'assistant', content: UNUSABLE_PROMPT_TOOL_CALL_OUTPUT },
      thinking: thinking || undefined,
      usage,
      finishReason,
    }
  }

  const content = cleanUserFacingText(protocolText)
  if (content) {
    // User-facing graph calls are fail-closed: publish one validated delta
    // only after the provider completed normally and the whole response
    // passed protocol-envelope inspection.
    yield { type: 'text_delta', text: content }
  }

  return {
    message: { role: 'assistant', content },
    thinking: thinking || undefined,
    usage,
    finishReason,
  }
}

async function* chatAsFinalDelta(params: {
  provider: ILLMProvider
  request: ChatRequest
  signal?: AbortSignal
  breaker?: ProviderCircuitBreaker
  live?: boolean
}): AsyncGenerator<AgentEvent, ChatResponse, void> {
  const response = await guardedProviderChat({
    provider: params.provider,
    request: params.request,
    signal: params.signal,
    breaker: params.breaker,
  })
  const rawContent = extractContent(response.message)
  const protocolText = resolveExplicitFinalTransportText(rawContent, response.thinking)
  const containsToolMarkup = containsPromptToolCallEnvelope(protocolText)
  const wasInterrupted = response.finishReason !== 'stop'
  const content = wasInterrupted
    ? INTERRUPTED_USER_FACING_RESPONSE
    : containsToolMarkup
      ? UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
      : cleanUserFacingText(protocolText)
  if (params.live && content && (wasInterrupted || !containsToolMarkup)) {
    yield { type: 'text_delta', text: content }
  }
  return withUserFacingContent(response, content)
}
