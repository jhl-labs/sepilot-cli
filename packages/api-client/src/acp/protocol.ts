const ACP_PROTOCOL_VERSION = 1

const ACP_AGENT_INFO = {
  name: 'sepilotd',
  title: 'sepilotd',
  version: '0.0.0',
} as const

export function createAcpInitializeResult() {
  return {
    protocolVersion: ACP_PROTOCOL_VERSION,
    agentCapabilities: {
      loadSession: false,
      promptCapabilities: {
        image: false,
        audio: false,
        embeddedContext: true,
      },
    },
    agentInfo: ACP_AGENT_INFO,
    authMethods: [],
    capabilities: { threads: true, streaming: true },
    _meta: {
      'sepilotd/acp': {
        legacyMethods: ['newThread', 'sendMessage', 'cancelThread'],
        methods: ['session/new', 'session/prompt', 'session/cancel'],
        notifications: ['session/update'],
      },
    },
  }
}

export function createAcpTextContent(text: string): { type: 'text'; text: string } {
  return { type: 'text', text }
}

export function extractAcpPromptText(params: unknown): string | null {
  if (!params || typeof params !== 'object') return null
  const record = params as Record<string, unknown>
  if (typeof record.content === 'string') return record.content
  if (typeof record.text === 'string') return record.text

  const prompt = record.prompt
  if (typeof prompt === 'string') return prompt
  if (!Array.isArray(prompt)) return null

  const parts: string[] = []
  for (const block of prompt) {
    if (!block || typeof block !== 'object') continue
    const item = block as Record<string, unknown>
    if (item.type === 'text' && typeof item.text === 'string') {
      parts.push(item.text)
      continue
    }
    if (item.type === 'resource' && item.resource && typeof item.resource === 'object') {
      const resource = item.resource as Record<string, unknown>
      if (typeof resource.text === 'string') {
        parts.push(resource.text)
      } else if (typeof resource.uri === 'string') {
        parts.push(`[resource] ${resource.uri}`)
      }
      continue
    }
    if (item.type === 'resource_link' && typeof item.uri === 'string') {
      parts.push(`[resource] ${item.uri}`)
    }
  }

  const content = parts.join('\n\n').trim()
  return content.length > 0 ? content : null
}

export function getAcpSessionId(
  params: unknown,
  legacyKey = 'threadId',
): string | null {
  if (!params || typeof params !== 'object') return null
  const record = params as Record<string, unknown>
  const value = record.sessionId ?? record[legacyKey]
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}
