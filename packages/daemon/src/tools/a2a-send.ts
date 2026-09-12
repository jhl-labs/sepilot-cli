import { A2AHttpClient } from '../a2a/client.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

function stringRecord(value: unknown): Record<string, string> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const entries = Object.entries(value)
    .filter((entry): entry is [string, string] =>
      typeof entry[0] === 'string' && typeof entry[1] === 'string',
    )
  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

export function createA2ASendTool(client = new A2AHttpClient()): ToolDefinitionRuntime {
  return {
    name: 'a2a.send',
    description:
      'Send a text message to a remote Agent2Agent (A2A) server. Fetches its Agent Card, selects a JSONRPC interface, and calls SendMessage with A2A-Version.',
    inputSchema: {
      type: 'object',
      properties: {
        agentCardUrl: {
          type: 'string',
          description: 'Absolute Agent Card URL, or an agent base URL with /.well-known/agent-card.json.',
        },
        message: { type: 'string', description: 'Text message to send through A2A SendMessage' },
        headers: {
          type: 'object',
          additionalProperties: { type: 'string' },
          description: 'Optional HTTP headers for the remote A2A server, such as Authorization.',
        },
        timeoutMs: { type: 'number', description: 'Optional request timeout in milliseconds.' },
      },
      required: ['agentCardUrl', 'message'],
    },
    async execute(input: Record<string, unknown>): Promise<ToolResult> {
      const startedAt = Date.now()
      try {
        const agentCardUrl = typeof input.agentCardUrl === 'string' ? input.agentCardUrl.trim() : ''
        const message = typeof input.message === 'string' ? input.message.trim() : ''
        if (!agentCardUrl) throw new Error('agentCardUrl is required')
        if (!message) throw new Error('message is required')
        const result = await client.sendMessage({
          agentCardUrl,
          message,
          headers: stringRecord(input.headers),
          timeoutMs: typeof input.timeoutMs === 'number' ? input.timeoutMs : undefined,
        })
        const head = `[a2a ${result.card.name}]`
        return {
          status: 'success',
          output: result.output ? `${head}\n\n${result.output}` : `${head}\n\n${JSON.stringify(result.result, null, 2)}`,
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          status: 'error',
          output: `[a2a error] ${error instanceof Error ? error.message : String(error)}`,
          code: 'A2A_SEND_FAILED',
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}
