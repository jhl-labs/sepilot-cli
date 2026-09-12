export interface JsonRpcMessage {
  jsonrpc: '2.0'
  id?: number | string | null
  method?: string
  params?: unknown
  result?: unknown
  error?: { code: number; message: string }
}

export type JsonRpcFraming = 'content-length' | 'ndjson'

export interface JsonRpcCodec {
  push(chunk: Buffer): JsonRpcMessage[]
  frame(msg: JsonRpcMessage): Buffer
}

export class JsonRpcFramer implements JsonRpcCodec {
  private buf = Buffer.alloc(0)

  push(chunk: Buffer): JsonRpcMessage[] {
    this.buf = Buffer.concat([this.buf, chunk])
    const out: JsonRpcMessage[] = []
    for (;;) {
      const headerEnd = this.buf.indexOf('\r\n\r\n')
      if (headerEnd < 0) return out
      const header = this.buf.slice(0, headerEnd).toString('ascii')
      const m = /Content-Length:\s*(\d+)/i.exec(header)
      if (!m) {
        this.buf = this.buf.slice(headerEnd + 4)
        continue
      }
      const len = Number(m[1])
      const start = headerEnd + 4
      if (this.buf.length < start + len) return out
      const body = this.buf.slice(start, start + len).toString('utf8')
      this.buf = this.buf.slice(start + len)
      try {
        out.push(JSON.parse(body) as JsonRpcMessage)
      } catch {
        // Invalid JSON in frame body - skip and continue parsing.
      }
    }
  }

  frame(msg: JsonRpcMessage): Buffer {
    const body = Buffer.from(JSON.stringify(msg), 'utf8')
    return Buffer.concat([
      Buffer.from(`Content-Length: ${body.length}\r\n\r\n`),
      body,
    ])
  }
}

export class NdjsonRpcFramer implements JsonRpcCodec {
  private buf = Buffer.alloc(0)

  push(chunk: Buffer): JsonRpcMessage[] {
    this.buf = Buffer.concat([this.buf, chunk])
    const out: JsonRpcMessage[] = []
    for (;;) {
      const lineEnd = this.buf.indexOf('\n')
      if (lineEnd < 0) return out
      const line = this.buf.slice(0, lineEnd).toString('utf8').trim()
      this.buf = this.buf.slice(lineEnd + 1)
      if (!line) continue
      try {
        out.push(JSON.parse(line) as JsonRpcMessage)
      } catch {
        // Invalid JSON line - skip and continue parsing.
      }
    }
  }

  frame(msg: JsonRpcMessage): Buffer {
    return Buffer.from(`${JSON.stringify(msg)}\n`, 'utf8')
  }
}

export function createJsonRpcFramer(
  framing: JsonRpcFraming = 'content-length',
): JsonRpcCodec {
  return framing === 'ndjson' ? new NdjsonRpcFramer() : new JsonRpcFramer()
}
