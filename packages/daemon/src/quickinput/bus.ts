import { EventEmitter } from 'node:events'

class Bus extends EventEmitter {}
const bus = new Bus()

export function publishQuickInput(text: string): void {
  bus.emit('text', text)
}

export function subscribeQuickInput(
  handler: (text: string) => void,
): () => void {
  bus.on('text', handler)
  return () => {
    bus.off('text', handler)
  }
}
