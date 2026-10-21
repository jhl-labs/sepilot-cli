import { randomUUID } from 'node:crypto'
import type { ChatRequest, Message } from '@sepilotd/core'

const MAX_CALLS = 16

export function codexToolOutputSchema(request: ChatRequest): Record<string, unknown> {
  return {
    type: 'object',
    properties: {
      toolCalls: {
        type: 'array', minItems: request.toolChoice === 'required' ? 1 : 0, maxItems: MAX_CALLS,
        items: {
          type: 'object', additionalProperties: false,
          properties: {
            name: { type: 'string', enum: [...new Set(request.tools!.map(tool => tool.name))] },
            argumentsJson: { type: 'string' },
          },
          required: ['name', 'argumentsJson'],
        },
      },
      answer: { type: 'string' },
    },
    required: ['toolCalls', 'answer'], additionalProperties: false,
  }
}

export function withCodexToolInstructions(request: ChatRequest): ChatRequest {
  return {
    ...request,
    systemPrompt: [
      request.systemPrompt,
      '[Host response format]',
      'Return a JSON object with toolCalls and answer. Each call has name and argumentsJson, a JSON string encoding its argument object.',
      'The host tools below are available. The host executes them under its own permissions. Request tools when observations or actions are missing; their results arrive as tool messages in the host transcript.',
      'Continue after completed host calls using their returned results. Do not repeat a completed request merely because this is a new transport turn.',
      'For tool requests, leave answer empty. For a final response, leave toolCalls empty and put the substantive next assistant response in answer, preserving the host answer protocol.',
      request.toolChoice === 'required' ? 'This control turn requires at least one host tool request.' : '',
      `Host tool definitions: ${JSON.stringify(request.tools)}`,
    ].filter(Boolean).join('\n'),
  }
}

const object = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value)
const parseJson = (value: string, label: string): unknown => {
  try { return JSON.parse(value) as unknown } catch {}
  try { return JSON.parse(repairStringEncoding(value)) as unknown } catch { throw new Error(`Invalid Codex ${label} JSON`) }
}

export function repairStringEncoding(text: string): string {
  let out = '', inString = false
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!
    if (!inString) { if (char === '"') inString = true; out += char; continue }
    if (char === '"') { inString = false; out += char; continue }
    if (char === '\\') {
      const next = text[index + 1]
      if (next !== undefined && '"\\/bfnrt'.includes(next)) { out += char + next; index++; continue }
      if (next === 'u' && /^[0-9a-fA-F]{4}$/.test(text.slice(index + 2, index + 6))) { out += text.slice(index, index + 6); index += 5; continue }
      out += '\\\\'
      continue
    }
    const code = char.charCodeAt(0)
    if (code < 0x20) { out += char === '\n' ? '\\n' : char === '\r' ? '\\r' : char === '\t' ? '\\t' : '\\u' + code.toString(16).padStart(4, '0'); continue }
    out += char
  }
  return out
}

export function parseCodexToolResponse(content: string, request: ChatRequest): Message {
  const value = parseJson(content, 'host response')
  if (!object(value) || Object.keys(value).length !== 2 || !Array.isArray(value.toolCalls)
    || typeof value.answer !== 'string' || value.toolCalls.length > MAX_CALLS
    || (value.toolCalls.length > 0 && value.answer.trim())
    || (value.toolCalls.length === 0 && (!value.answer.trim() || request.toolChoice === 'required'))) {
    throw new Error('Invalid Codex host response envelope')
  }
  const available = new Set(request.tools?.map(tool => tool.name))
  const toolCalls = value.toolCalls.map((call: unknown) => {
    if (!object(call) || Object.keys(call).length !== 2 || typeof call.name !== 'string'
      || !available.has(call.name) || typeof call.argumentsJson !== 'string') {
      throw new Error('Invalid Codex host tool request')
    }
    const args = parseJson(call.argumentsJson, 'host tool argument')
    if (!object(args)) throw new Error('Codex host tool arguments must be an object')
    return { id: randomUUID(), name: call.name, arguments: args }
  })
  return { role: 'assistant', content: value.answer, ...(toolCalls.length ? { toolCalls } : {}) }
}
