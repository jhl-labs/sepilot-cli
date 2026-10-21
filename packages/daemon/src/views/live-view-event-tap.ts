import type { AgentEvent, LiveView, LiveViewRecord } from '@sepilotd/core'
import type { LiveViewRegistry, LiveViewRegistryEvent } from './live-view-registry.js'
import type { StillFormat } from '../tools/browser-transport.js'

export interface LiveViewTapOptions {
  views: LiveViewRegistry
  sessionId: string
  stills: { enabled: boolean; maxWidth: number; format?: StillFormat }
  minIntervalMs?: number
  now?: () => number
}

export interface LiveViewTap {
  wrap(events: AsyncIterable<AgentEvent>): AsyncIterable<AgentEvent>
  records(): LiveViewRecord[]
}

const DEFAULT_MIN_INTERVAL_MS = 1000

export const DEFAULT_LIVE_VIEW_STILLS = { enabled: true, maxWidth: 480, format: 'jpeg' } as const

export function createLiveViewTap(options: LiveViewTapOptions): LiveViewTap {
  const now = options.now ?? (() => Date.now())
  const minInterval = options.minIntervalMs ?? DEFAULT_MIN_INTERVAL_MS
  const lastStillAt = new Map<string, number>()
  const finalRecords = new Map<string, LiveViewRecord>()
  let records: LiveViewRecord[] = []

  const record = (view: LiveView, still: LiveViewRecord['still']): LiveViewRecord => ({
    id: view.id,
    kind: view.kind,
    title: view.title,
    ...(view.subtitle ? { subtitle: view.subtitle } : {}),
    still,
  })

  async function stillEvent(view: LiveView, force: boolean): Promise<AgentEvent | undefined> {
    if (!options.stills.enabled) return undefined
    const at = now()
    const last = lastStillAt.get(view.id) ?? Number.NEGATIVE_INFINITY
    if (!force && at - last < minInterval) return undefined
    let still: Awaited<ReturnType<LiveViewRegistry['still']>>
    try {
      still = await options.views.still(view.id, options.stills.maxWidth, options.stills.format)
    } catch {
      return undefined
    }
    if (!still) return undefined
    lastStillAt.set(view.id, at)
    const current = options.views.get(view.id) ?? view
    finalRecords.set(view.id, record(current, still))
    return { type: 'view_still', view: current, still }
  }

  return {
    records: () => records,
    async *wrap(events) {
      const attached = new Set<string>()
      const pending: LiveViewRegistryEvent[] = []
      const stop = options.views.onEvent((event) => {
        const id = event.type === 'detached' ? event.viewId : event.view.id
        const owned =
          event.type === 'detached'
            ? finalRecords.has(id) || lastStillAt.has(id) || attached.has(id)
            : event.view.sessionId === options.sessionId
        if (!owned) return
        if (event.type === 'attached') attached.add(id)
        pending.push(event)
      })
      async function* flush(includeActivity: boolean): AsyncGenerator<AgentEvent> {
        while (pending.length > 0) {
          const change = pending.shift()!
          if (change.type === 'attached') {
            attached.add(change.view.id)
            yield { type: 'view_attached', view: change.view }
          } else if (change.type === 'detached') {
            attached.delete(change.viewId)
            if (!finalRecords.get(change.viewId)?.still.image) finalRecords.delete(change.viewId)
            yield { type: 'view_detached', viewId: change.viewId, reason: change.reason }
          }
        }
        if (!includeActivity) return
        for (const view of options.views.takeDirty(options.sessionId)) {
          const still = await stillEvent(view, false)
          if (still) yield still
        }
      }
      try {
        for (const view of options.views.list(options.sessionId)) {
          attached.add(view.id)
          yield { type: 'view_attached', view }
        }
        for await (const event of events) {
          if (event.type === 'done') {
            yield* flush(false)
            for (const view of options.views.list(options.sessionId)) {
              const still = await stillEvent(view, true)
              if (still) yield still
            }
            records = [...finalRecords.values()]
            yield event
            continue
          }
          yield event
          yield* flush(event.type === 'tool_result')
        }
      } finally {
        stop()
      }
    },
  }
}
