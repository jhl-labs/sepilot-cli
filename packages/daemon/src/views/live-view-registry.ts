import type { LiveView, LiveViewStill } from '@sepilotd/core'
import type { RemoteFrame, StillFormat } from '../tools/browser-transport.js'
import type { ToolRegistry } from '../tools/registry.js'

export interface LiveViewSource {
  readonly view: LiveView
  subscribe(listener: (frame: RemoteFrame) => void): () => void
  latestFrame(): RemoteFrame | undefined
  still(maxWidth: number, format?: StillFormat): Promise<LiveViewStill | undefined>
  onChange(listener: (view: LiveView) => void): () => void
  onClosed(listener: (reason: string) => void): () => void
}

export type LiveViewRegistryEvent =
  | { type: 'attached'; view: LiveView }
  | { type: 'changed'; view: LiveView }
  | { type: 'detached'; viewId: string; reason: string }

interface Entry {
  source: LiveViewSource
  view: LiveView
  listeners: Set<(frame: RemoteFrame) => void>
  stopCapture?: () => void
  detach: () => void
  activitySeq: number
  stillSeq: number
}

export class LiveViewRegistry {
  private entries = new Map<string, Entry>()
  private eventListeners = new Set<(event: LiveViewRegistryEvent) => void>()
  private seq = 0
  constructor(_now: () => number = () => Date.now()) {}

  register(source: LiveViewSource): LiveView {
    const id = source.view.id
    if (this.entries.has(id)) this.unregister(id, 'Re-registered')
    const entry: Entry = {
      source,
      view: { ...source.view },
      listeners: new Set(),
      detach: () => {},
      activitySeq: 0,
      stillSeq: 0,
    }
    const stopChange = source.onChange((view) => {
      entry.view = { ...entry.view, ...view }
      this.emit({ type: 'changed', view: entry.view })
    })
    const stopClosed = source.onClosed((reason) => this.unregister(id, reason))
    entry.detach = () => {
      stopChange()
      stopClosed()
    }
    this.entries.set(id, entry)
    this.emit({ type: 'attached', view: entry.view })
    return entry.view
  }

  unregister(id: string, reason: string): void {
    const entry = this.entries.get(id)
    if (!entry) return
    this.entries.delete(id)
    entry.detach()
    entry.stopCapture?.()
    entry.listeners.clear()
    this.emit({ type: 'detached', viewId: id, reason })
  }

  list(sessionId?: string): LiveView[] {
    const views = [...this.entries.values()].map((entry) => entry.view)
    return sessionId === undefined ? views : views.filter((view) => view.sessionId === sessionId)
  }

  get(id: string): LiveView | undefined {
    return this.entries.get(id)?.view
  }

  subscribeFrames(id: string, listener: (frame: RemoteFrame) => void): () => void {
    const entry = this.entries.get(id)
    if (!entry) throw new Error('View not found')
    entry.listeners.add(listener)
    if (!entry.stopCapture) {
      entry.stopCapture = entry.source.subscribe((frame) => {
        entry.view = { ...entry.view, lastFrameAt: frame.capturedAt }
        for (const subscriber of entry.listeners) subscriber(frame)
      })
    }
    return () => {
      entry.listeners.delete(listener)
      if (entry.listeners.size === 0 && entry.stopCapture) {
        entry.stopCapture()
        entry.stopCapture = undefined
      }
    }
  }

  async still(id: string, maxWidth: number, format: StillFormat = 'jpeg'): Promise<LiveViewStill | undefined> {
    const entry = this.entries.get(id)
    if (!entry) return undefined
    const startedSeq = this.seq
    const still = await entry.source.still(maxWidth, format)
    if (still) entry.stillSeq = Math.max(entry.stillSeq, startedSeq)
    return still
  }

  markActivity(id: string): void {
    const entry = this.entries.get(id)
    if (entry) entry.activitySeq = ++this.seq
  }

  takeDirty(sessionId: string): LiveView[] {
    const dirty: LiveView[] = []
    for (const entry of this.entries.values()) {
      if (entry.view.sessionId !== sessionId) continue
      if (entry.activitySeq <= entry.stillSeq) continue
      entry.stillSeq = entry.activitySeq
      dirty.push(entry.view)
    }
    return dirty
  }

  onEvent(listener: (event: LiveViewRegistryEvent) => void): () => void {
    this.eventListeners.add(listener)
    return () => this.eventListeners.delete(listener)
  }

  close(): void {
    for (const id of [...this.entries.keys()]) this.unregister(id, 'Daemon shutdown')
  }

  private emit(event: LiveViewRegistryEvent) {
    for (const listener of this.eventListeners) listener(event)
  }
}

const registries = new WeakMap<ToolRegistry, LiveViewRegistry>()
export function liveViewRegistry(toolRegistry: ToolRegistry): LiveViewRegistry {
  let registry = registries.get(toolRegistry)
  if (!registry) {
    registry = new LiveViewRegistry()
    registries.set(toolRegistry, registry)
  }
  return registry
}
