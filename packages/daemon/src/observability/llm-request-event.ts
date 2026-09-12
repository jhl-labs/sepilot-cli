import type {
  AgentEvent,
  ChatRequest,
  ContentPart,
  LlmRequestDigest,
  Message,
} from '@sepilotd/core'

function textLength(content: Message['content'] | undefined): number {
  if (typeof content === 'string') {
    return content.length
  }

  if (!Array.isArray(content)) {
    return 0
  }

  return content.reduce((total, part: ContentPart) => (
    part.type === 'text' ? total + part.text.length : total
  ), 0)
}

function turnIdComponent(value: string): string {
  return value.trim().replace(/[:\s]+/g, '-').replace(/[^A-Za-z0-9._/-]/g, '_') || 'unknown'
}

export function buildLlmTurnId(
  sessionId: string | undefined,
  iteration: number | undefined,
  source: string,
): string {
  const safeSessionId = turnIdComponent(sessionId ?? 'unknown-session')
  const safeIteration = Number.isFinite(iteration)
    ? Math.max(0, Math.trunc(iteration ?? 0))
    : 0
  return `${safeSessionId}:${safeIteration}:${turnIdComponent(source)}`
}

export function buildLlmRequestDigest(
  request: ChatRequest,
  traceRef?: string,
  metadata?: Pick<LlmRequestDigest, 'providerId' | 'source' | 'startedAt' | 'timeoutMs' | 'auxiliary'>,
): LlmRequestDigest {
  const systemPromptChars = [
    request.systemPrompt?.length ?? 0,
    ...request.messages
      .filter((message) => message.role === 'system')
      .map((message) => textLength(message.content)),
  ].reduce((total, length) => total + length, 0)

  const toolSchemaChars = (request.tools ?? []).reduce((total, tool) => (
    total + tool.name.length + (tool.description?.length ?? 0)
    + JSON.stringify(tool.inputSchema ?? {}).length
  ), 0)

  const digest: LlmRequestDigest = {
    model: request.model,
    messageCount: request.messages.length,
    systemPromptChars,
    ...(toolSchemaChars > 0 ? { toolSchemaChars } : {}),
    toolNames: [...new Set(request.tools?.map((tool) => tool.name) ?? [])].sort(),
    ...metadata,
  }
  if (traceRef) {
    digest.traceRef = traceRef
  }
  return digest
}

export function buildLlmRequestEvent(params: {
  sessionId?: string
  iteration?: number
  source: string
  request: ChatRequest
  turnId?: string
  providerId?: string
  startedAt?: number
  timeoutMs?: number | null
  auxiliary?: boolean
}): Extract<AgentEvent, { type: 'llm_request' }> {
  const iteration = Number.isFinite(params.iteration)
    ? Math.max(0, Math.trunc(params.iteration ?? 0))
    : 0
  const turnId = params.turnId ?? buildLlmTurnId(params.sessionId, iteration, params.source)
  return {
    type: 'llm_request',
    turnId,
    iteration,
    requestDigest: buildLlmRequestDigest(params.request, turnId, {
      providerId: params.providerId,
      source: params.source,
      startedAt: params.startedAt ?? Date.now(),
      ...(typeof params.timeoutMs === 'number' ? { timeoutMs: params.timeoutMs } : {}),
      auxiliary: params.auxiliary ?? false,
    }),
  }
}
