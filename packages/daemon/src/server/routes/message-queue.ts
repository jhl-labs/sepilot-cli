import type { FastifyInstance } from 'fastify'
import { bindCapability } from '../capabilities/bind.js'

export interface QueueSnapshotEntry {
  topic: string
  pending: number
  inFlight: number
}

export interface QueueInspector {
  snapshot(): Promise<QueueSnapshotEntry[]>
}

export const emptyQueueInspector: QueueInspector = {
  snapshot: async () => [],
}

export async function registerMessageQueueRoutes(
  app: FastifyInstance,
  inspector: QueueInspector = emptyQueueInspector,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'message-queue',
      version: '1',
      methods: [{ method: 'GET', path: '/message-queue/snapshot' }],
    },
    async (a) => {
      a.get('/message-queue/snapshot', async () => inspector.snapshot())
    },
  )
}
