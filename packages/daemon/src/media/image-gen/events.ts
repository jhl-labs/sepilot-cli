import { EventEmitter } from 'node:events'
import type { MediaOutputKind } from './adapter.js'

export interface JobEvent {
  id: string
  status: string
  progress: number
  outputs: { id: string; mime: string; kind?: MediaOutputKind }[]
  error: string | null
  createdAt: number
  prompt: string
}

class JobBus extends EventEmitter {}
const bus = new JobBus()

export function publishJob(event: JobEvent): void {
  bus.emit('job', event)
}

export function subscribeJobs(handler: (e: JobEvent) => void): () => void {
  bus.on('job', handler)
  return () => {
    bus.off('job', handler)
  }
}
