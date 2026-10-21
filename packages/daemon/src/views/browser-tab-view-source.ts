import { randomUUID } from 'node:crypto'
import type { LiveView, LiveViewStill } from '@sepilotd/core'
import type { BrowserTransport, RemoteFrame, StillFormat } from '../tools/browser-transport.js'
import type { LiveViewSource } from './live-view-registry.js'

export class BrowserTabViewSource implements LiveViewSource {
  view: LiveView
  private latest: RemoteFrame | undefined
  private changeListeners = new Set<(view: LiveView) => void>()
  private closedListeners = new Set<(reason: string) => void>()

  constructor(
    label: string,
    private readonly transport: BrowserTransport,
    private readonly now: () => number = () => Date.now(),
  ) {
    this.view = {
      id: `view_${randomUUID()}`,
      kind: 'browser_tab',
      title: label,
      subtitle: transport.currentUrl(),
      controller: 'human',
      createdAt: this.now(),
    }
    transport.onClosed((reason) => {
      for (const listener of this.closedListeners) listener(reason)
    })
  }

  subscribe(listener: (frame: RemoteFrame) => void): () => void {
    return this.transport.onFrame((frame) => {
      this.latest = frame
      this.update({ title: frame.title || this.view.title, subtitle: frame.url })
      listener(frame)
    })
  }

  latestFrame(): RemoteFrame | undefined {
    return this.latest
  }

  async still(maxWidth: number, format: StillFormat = 'jpeg'): Promise<LiveViewStill | undefined> {
    const scaled = this.transport.still ? await this.transport.still(maxWidth, format) : undefined
    const frame = scaled ?? this.latest
    if (!frame) return undefined
    return {
      viewId: this.view.id,
      image: frame.image,
      width: frame.width ?? 0,
      height: frame.height ?? 0,
      capturedAt: frame.capturedAt,
    }
  }

  update(patch: Partial<LiveView>): void {
    const changed = (Object.keys(patch) as (keyof LiveView)[]).some((key) => this.view[key] !== patch[key])
    if (!changed) return
    this.view = { ...this.view, ...patch }
    for (const listener of this.changeListeners) listener(this.view)
  }

  noteFrame(frame: RemoteFrame | undefined): void {
    if (frame) this.latest = frame
  }

  onChange(listener: (view: LiveView) => void): () => void {
    this.changeListeners.add(listener)
    return () => this.changeListeners.delete(listener)
  }

  onClosed(listener: (reason: string) => void): () => void {
    this.closedListeners.add(listener)
    return () => this.closedListeners.delete(listener)
  }
}
