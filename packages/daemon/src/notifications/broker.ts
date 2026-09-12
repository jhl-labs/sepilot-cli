import { EventEmitter } from 'node:events'
import type { NotificationItem } from './repo.js'

class Bus extends EventEmitter {}
const bus = new Bus()

export function publishNotification(item: NotificationItem): void {
  bus.emit('item', item)
}

export function subscribeNotifications(
  handler: (item: NotificationItem) => void,
): () => void {
  bus.on('item', handler)
  return () => {
    bus.off('item', handler)
  }
}
