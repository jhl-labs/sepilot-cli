import type { Readable, Writable } from 'node:stream'
import { JsonRpcFramer } from './rpc.js'
import type { JsonRpcMessage } from './rpc.js'
import type { AcpDeps, AcpHandlers } from './handlers.js'
import { createAcpHandlers } from './handlers.js'

export interface AcpServerStreams {
  stdin: Readable
  stdout: Writable
}

export interface AcpServerHandle {
  /** Resolves when stdin emits 'end' or 'close'. */
  closed: Promise<void>
  /** Force close the server (does not kill the streams). */
  stop(): void
}

export function runAcpServer(
  deps: AcpDeps,
  streams: AcpServerStreams,
  handlersOverride?: AcpHandlers,
): AcpServerHandle {
  const framer = new JsonRpcFramer()
  let stopped = false
  const notify = (message: JsonRpcMessage): void => {
    if (stopped) return
    streams.stdout.write(framer.frame(message))
  }
  const handlers = handlersOverride ?? createAcpHandlers({ ...deps, notify })

  const onData = (chunk: Buffer): void => {
    if (stopped) return
    const messages = framer.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
    for (const msg of messages) {
      void dispatch(msg)
    }
  }

  async function dispatch(msg: Parameters<AcpHandlers['dispatch']>[0]): Promise<void> {
    if (stopped) return
    const expectsResponse = Object.hasOwn(msg, 'id')
    try {
      const response = await handlers.dispatch(msg)
      if (stopped || !expectsResponse) return
      streams.stdout.write(framer.frame(response))
    } catch (err) {
      if (!expectsResponse) return
      const message = err instanceof Error ? err.message : String(err)
      streams.stdout.write(framer.frame({
        jsonrpc: '2.0',
        id: msg.id,
        error: { code: -32000, message },
      }))
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
