import { randomUUID } from 'node:crypto'
import type { Readable, Writable } from 'node:stream'
import {
  createAcpInitializeResult,
  createAcpTextContent,
  extractAcpPromptText,
  getAcpSessionId,
  streamDaemonChatEvents,
  type DaemonChatStreamPayload,
  type DaemonClient,
} from '@sepilotd/api-client'
import { JsonRpcFramer, type JsonRpcMessage } from '@sepilotd/api-client/node'
import { openChatStreamWithConnectTimeout } from '../utils/stream-connect.js'

export interface AcpProxyStreams {
  stdin: Readable
  stdout: Writable
}

export interface AcpProxyDeps {
  client: Pick<DaemonClient, 'chatStream'>
}

export interface AcpProxyHandle {
  closed: Promise<void>
  stop(): void
}

type AcpSessionUpdate = Record<string, unknown>

function mapDaemonStreamPayload(payload: DaemonChatStreamPayload): AcpSessionUpdate | null {
  if (!('type' in payload)) return null

  switch (payload.type) {
    case 'text_delta':
      return {
        sessionUpdate: 'agent_message_chunk',
        content: createAcpTextContent(payload.text ?? ''),
      }
    case 'message':
      return {
        sessionUpdate: 'agent_message_chunk',
        content: createAcpTextContent(payload.content ?? ''),
      }
    case 'thinking':
      return {
        sessionUpdate: 'agent_thought_chunk',
        content: createAcpTextContent(payload.content ?? ''),
      }
    case 'tool_call':
      return {
        sessionUpdate: 'tool_call',
        toolCallId: payload.toolCall?.id ?? 'tool',
        title: payload.toolCall?.name || 'tool',
        status: 'pending',
        content: [],
        locations: [],
        rawInput: payload.toolCall?.arguments ?? {},
      }
    case 'tool_result':
      return {
        sessionUpdate: 'tool_call_update',
        toolCallId: payload.toolCallId,
        status: payload.status === 'success' ? 'completed' : 'failed',
        content: [{
          type: 'content',
          content: createAcpTextContent(payload.output ?? ''),
        }],
        rawOutput: { output: payload.output ?? '', status: payload.status },
      }
    case 'error':
      throw new Error(payload.error?.message ?? 'daemon chat stream failed')
    default:
      return null
  }
}

export function runAcpProxy(deps: AcpProxyDeps, streams: AcpProxyStreams): AcpProxyHandle {
  const framer = new JsonRpcFramer()
  let lastThreadId: string | null = null
  let stopped = false

  function writeNotification(message: JsonRpcMessage): void {
    if (stopped) return
    streams.stdout.write(framer.frame(message))
  }

  function emitSessionUpdate(sessionId: string, update: AcpSessionUpdate): void {
    writeNotification({
      jsonrpc: '2.0',
      method: 'session/update',
      params: { sessionId, update },
    })
  }

  async function sendPromptToDaemon(sessionId: string, content: string): Promise<void> {
    const existing = lastThreadId === sessionId ? sessionId : undefined
    const aborter = new AbortController()
    const response = await openChatStreamWithConnectTimeout(
      deps.client.chatStream(content, existing, undefined, { signal: aborter.signal }),
      { abort: (error) => aborter.abort(error) },
    )
    if (!response.ok) {
      throw new Error(`${response.status}: ${await response.text()}`)
    }
    for await (const payload of streamDaemonChatEvents(response)) {
      const update = mapDaemonStreamPayload(payload)
      if (update) {
        emitSessionUpdate(sessionId, update)
      }
    }
  }

  async function dispatch(msg: JsonRpcMessage): Promise<JsonRpcMessage> {
    const base: JsonRpcMessage = { jsonrpc: '2.0', id: msg.id }
    switch (msg.method) {
      case 'initialize':
        return { ...base, result: createAcpInitializeResult() }
      case 'session/new':
      case 'newThread': {
        const threadId = `acp-${randomUUID()}`
        lastThreadId = threadId
        return msg.method === 'session/new'
          ? { ...base, result: { sessionId: threadId } }
          : { ...base, result: { threadId } }
      }
      case 'session/load':
      case 'session/resume': {
        const sessionId = getAcpSessionId(msg.params)
        if (!sessionId) {
          return { ...base, error: { code: -32602, message: 'invalid params: sessionId required' } }
        }
        lastThreadId = sessionId
        return { ...base, result: msg.method === 'session/resume' ? {} : null }
      }
      case 'session/prompt':
      case 'sendMessage': {
        const sessionId = getAcpSessionId(msg.params)
        const content = extractAcpPromptText(msg.params)
        if (!sessionId || !content) {
          return {
            ...base,
            error: {
              code: -32602,
              message: msg.method === 'session/prompt'
                ? 'invalid params: sessionId + prompt required'
                : 'invalid params: threadId + content required',
            },
          }
        }
        try {
          await sendPromptToDaemon(sessionId, content)
          return msg.method === 'session/prompt'
            ? { ...base, result: { stopReason: 'end_turn' } }
            : { ...base, result: { accepted: true } }
        } catch (err) {
          return { ...base, error: { code: -32000, message: err instanceof Error ? err.message : String(err) } }
        }
      }
      case 'session/cancel':
      case 'cancelThread': {
        const sessionId = getAcpSessionId(msg.params)
        if (msg.method === 'session/cancel' && !sessionId) {
          return { ...base, error: { code: -32602, message: 'invalid params: sessionId required' } }
        }
        if (sessionId && lastThreadId === sessionId) {
          lastThreadId = null
        }
        return { ...base, result: { cancelled: true } }
      }
      default:
        return { ...base, error: { code: -32601, message: `method not found: ${String(msg.method)}` } }
    }
  }

  const onData = (chunk: Buffer): void => {
    if (stopped) return
    const messages = framer.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
    for (const msg of messages) {
      const expectsResponse = Object.prototype.hasOwnProperty.call(msg, 'id')
      void dispatch(msg).then((response) => {
        if (stopped || !expectsResponse) return
        streams.stdout.write(framer.frame(response))
      })
    }
  }

  streams.stdin.on('data', onData)

  const closed = new Promise<void>((resolve) => {
    const finish = (): void => {
      stopped = true
      streams.stdin.off('data', onData)
      resolve()
    }
    streams.stdin.once('end', finish)
    streams.stdin.once('close', finish)
  })

  return {
    closed,
    stop() {
      stopped = true
      streams.stdin.off('data', onData)
    },
  }
}
