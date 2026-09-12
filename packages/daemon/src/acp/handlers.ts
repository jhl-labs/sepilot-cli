import {
  createAcpInitializeResult,
  extractAcpPromptText,
  getAcpSessionId,
  type JsonRpcMessage,
} from '@sepilotd/api-client/node'

export interface AcpBridge {
  subscribe: (threadId: string, cb: (event: unknown) => void) => () => void
  send: (threadId: string, content: string) => Promise<void>
  cancel: (threadId: string) => Promise<void>
}

export interface AcpDeps {
  sessions: { create: () => Promise<{ id: string }> }
  bridge: AcpBridge
  notify?: (message: JsonRpcMessage) => void
}

export interface AcpHandlers {
  dispatch(msg: JsonRpcMessage): Promise<JsonRpcMessage>
}

function notifySessionUpdate(
  deps: AcpDeps,
  sessionId: string,
  update: unknown,
): void {
  deps.notify?.({
    jsonrpc: '2.0',
    method: 'session/update',
    params: {
      sessionId,
      update,
    },
  })
}

export function createAcpHandlers(deps: AcpDeps): AcpHandlers {
  async function dispatch(msg: JsonRpcMessage): Promise<JsonRpcMessage> {
    const base: JsonRpcMessage = { jsonrpc: '2.0', id: msg.id }
    switch (msg.method) {
      case 'initialize':
        return {
          ...base,
          result: createAcpInitializeResult(),
        }
      case 'session/new':
      case 'newThread': {
        const s = await deps.sessions.create()
        return msg.method === 'session/new'
          ? { ...base, result: { sessionId: s.id } }
          : { ...base, result: { threadId: s.id } }
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
        const unsubscribe = deps.bridge.subscribe(sessionId, (update) => {
          notifySessionUpdate(deps, sessionId, update)
        })
        try {
          await deps.bridge.send(sessionId, content)
        } finally {
          unsubscribe()
        }
        return msg.method === 'session/prompt'
          ? { ...base, result: { stopReason: 'end_turn' } }
          : { ...base, result: { accepted: true } }
      }
      case 'session/cancel':
      case 'cancelThread': {
        const sessionId = getAcpSessionId(msg.params)
        if (!sessionId) {
          return {
            ...base,
            error: {
              code: -32602,
              message: msg.method === 'session/cancel'
                ? 'invalid params: sessionId required'
                : 'invalid params: threadId required',
            },
          }
        }
        await deps.bridge.cancel(sessionId)
        return { ...base, result: { cancelled: true } }
      }
      default:
        return {
          ...base,
          error: { code: -32601, message: `method not found: ${String(msg.method)}` },
        }
    }
  }
  return { dispatch }
}
