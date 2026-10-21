import type { AgentEvent } from '@sepilotd/core'

const MAX_EVENTS_PER_JOB = 2_000
const ENDED_RETENTION_MS = 15 * 60_000
const MAX_JOBS = 200

export interface BackgroundChatEventRecord {
  seq: number
  event: AgentEvent
}

export type BackgroundChatStreamItem =
  | { kind: 'event'; record: BackgroundChatEventRecord }
  | { kind: 'end'; status: string }

interface JobStream {
  events: BackgroundChatEventRecord[]
  nextSeq: number
  firstSeq: number
  listeners: Set<(item: BackgroundChatStreamItem) => void>
  ended?: { status: string; at: number }
}

export class BackgroundChatEventHub {
  private readonly jobs = new Map<string, JobStream>()

  constructor(private readonly now: () => number = Date.now) {}

  private stream(jobId: string): JobStream {
    let stream = this.jobs.get(jobId)
    if (!stream) {
      this.prune()
      stream = { events: [], nextSeq: 1, firstSeq: 1, listeners: new Set() }
      this.jobs.set(jobId, stream)
    }
    return stream
  }

  private prune(): void {
    const now = this.now()
    for (const [jobId, stream] of this.jobs) {
      if (stream.ended && now - stream.ended.at > ENDED_RETENTION_MS && stream.listeners.size === 0) this.jobs.delete(jobId)
    }
    while (this.jobs.size >= MAX_JOBS) {
      const oldestEnded = [...this.jobs].find(([, stream]) => stream.ended && stream.listeners.size === 0)
      if (!oldestEnded) break
      this.jobs.delete(oldestEnded[0])
    }
  }

  push(jobId: string, event: AgentEvent): void {
    const stream = this.stream(jobId)
    if (stream.ended) return
    const record = { seq: stream.nextSeq++, event }
    stream.events.push(record)
    if (stream.events.length > MAX_EVENTS_PER_JOB) {
      stream.events.splice(0, stream.events.length - MAX_EVENTS_PER_JOB)
      stream.firstSeq = stream.events[0]!.seq
    }
    for (const listener of stream.listeners) {
      try { listener({ kind: 'event', record }) } catch {}
    }
  }

  end(jobId: string, status: string): void {
    const stream = this.stream(jobId)
    if (stream.ended) return
    stream.ended = { status, at: this.now() }
    for (const listener of stream.listeners) {
      try { listener({ kind: 'end', status }) } catch {}
    }
  }

  subscribe(jobId: string, afterSeq: number, listener: (item: BackgroundChatStreamItem) => void): {
    replay: BackgroundChatEventRecord[]
    gap: boolean
    ended?: string
    unsubscribe: () => void
  } {
    const stream = this.stream(jobId)
    const replay = stream.events.filter(record => record.seq > afterSeq)
    const gap = afterSeq + 1 < stream.firstSeq
    if (stream.ended) return { replay, gap, ended: stream.ended.status, unsubscribe: () => {} }
    stream.listeners.add(listener)
    return { replay, gap, unsubscribe: () => { stream.listeners.delete(listener) } }
  }

  has(jobId: string): boolean {
    return this.jobs.has(jobId)
  }
}
