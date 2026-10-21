import type { AgentEvent } from '@sepilotd/core'

export async function* streamToolProgress<T>(
  execute: (emit: (event: AgentEvent) => void) => Promise<T>,
): AsyncGenerator<AgentEvent, T> {
  const pending: Array<{ event: AgentEvent; size: number }> = []
  let pendingBytes = 0
  let notify: (() => void) | undefined
  let closed = false
  let settled: { value: T } | { error: unknown } | undefined
  let overflow: Error | undefined
  const wake = () => { notify?.(); notify = undefined }
  const emit = (event: AgentEvent) => {
    if (closed) return
    if (overflow) throw overflow
    const size = Buffer.byteLength(JSON.stringify(event))
    if (pending.length >= 256 || pendingBytes + size > 2 * 1024 * 1024) {
      overflow = new Error('Tool progress buffer exceeded its bounded capacity')
      wake()
      throw overflow
    }
    pending.push({ event, size })
    pendingBytes += size
    wake()
  }
  void Promise.resolve().then(() => execute(emit)).then(
    value => { settled = { value }; wake() },
    error => { settled = { error }; wake() },
  )
  try {
    while (true) {
      while (pending.length) {
        const item = pending.shift()!
        pendingBytes -= item.size
        yield item.event
      }
      if (settled) {
        if (overflow) throw overflow
        if ('error' in settled) throw settled.error
        return settled.value
      }
      await new Promise<void>(resolve => { notify = resolve })
    }
  } finally {
    closed = true
    pending.length = 0
    notify = undefined
  }
}
