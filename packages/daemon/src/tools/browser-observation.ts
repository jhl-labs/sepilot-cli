import { randomUUID } from 'node:crypto'
import type { Page } from 'playwright'
import type { ToolRegistry } from './registry.js'

const TTL_MS = 30_000
const FRESH_MS = 15_000
const MAX_PNG_BYTES = 2 * 1024 * 1024
export interface BrowserObservationView {
  id: string
  sessionId: string
  sourceSessionId?: string
  tool?: string
  state: 'waiting' | 'live' | 'snapshot' | 'paused' | 'stale'
  sequence: number
  capturedAt: string | null
  pageUrl?: string
  pngBase64?: string
}
interface Watch { view: BrowserObservationView; expiresAt: number; png?: Buffer; pageOwner?: symbol }
export interface PageObservation { stop(): Promise<void> }

export class BrowserObservationRegistry {
  private readonly watches = new Map<string, Watch>()
  private readonly pages = new Set<PageObservation>()
  private readonly timer: ReturnType<typeof setInterval>
  private lineage?: (sessionId: string) => Promise<string[]>
  private sequence = 0
  constructor(private readonly now = Date.now) {
    this.timer = setInterval(() => this.expire(), 5000)
    this.timer.unref()
  }
  setLineageResolver(resolve: (sessionId: string) => Promise<string[]>): void { this.lineage = resolve }
  private expire(): void {
    for (const [id, watch] of this.watches) {
      if (watch.expiresAt <= this.now()) { this.watches.delete(id); continue }
      if (watch.view.capturedAt && this.now() - Date.parse(watch.view.capturedAt) > FRESH_MS) {
        watch.png = undefined
        if (watch.view.state === 'live' || watch.view.state === 'snapshot') watch.view.state = 'stale'
      }
    }
  }
  start(sessionId: string): BrowserObservationView {
    this.expire()
    if (this.watches.size >= 16) throw Object.assign(new Error('Too many browser observations'), { statusCode: 429 })
    const view: BrowserObservationView = { id: randomUUID(), sessionId, state: 'waiting', sequence: 0, capturedAt: null }
    this.watches.set(view.id, { view, expiresAt: this.now() + TTL_MS })
    return { ...view }
  }
  read(id: string, after = 0): BrowserObservationView | undefined {
    this.expire()
    const watch = this.watches.get(id)
    if (!watch) return undefined
    watch.expiresAt = this.now() + TTL_MS
    return { ...watch.view, ...(watch.png && watch.view.sequence > after ? { pngBase64: watch.png.toString('base64') } : {}) }
  }
  stop(id: string): void { this.watches.delete(id) }
  async close(): Promise<void> {
    clearInterval(this.timer)
    this.watches.clear()
    await Promise.all([...this.pages].map(page => page.stop()))
    this.lineage = undefined
  }
  async observe(sessionId: string | undefined, page: Page, tool: string): Promise<PageObservation | undefined> {
    this.expire()
    if (!sessionId || !this.watches.size) return undefined
    const lineage = this.lineage ? await this.lineage(sessionId) : [sessionId]
    const targets = () => {
      this.expire()
      return [...this.watches.values()].filter(watch => lineage.includes(watch.view.sessionId))
    }
    if (!targets().length) return undefined
    let stopped = false, timer: ReturnType<typeof setTimeout> | undefined
    let finishing: Promise<void> | undefined
    const pageOwner = Symbol('observed-page')
    let inFlight: Promise<void> = Promise.resolve()
    const capture = async (state: 'live' | 'snapshot') => {
      const watchers = targets()
      if (!watchers.length || page.isClosed()) return
      try {
        const png = await page.screenshot({ type: 'png', fullPage: false, timeout: 500,
          mask: [page.locator('input[type="password"], [data-private], [data-sensitive], input[autocomplete="one-time-code"]')], maskColor: '#64748b' })
        if (png.length > MAX_PNG_BYTES) throw new Error('Observation frame exceeds its byte limit')
        let pageUrl: string | undefined
        try {
          const url = new URL(page.url())
          if (['http:', 'https:'].includes(url.protocol) && !url.username && !url.password) {
            url.search = ''
            if (!/^#[a-zA-Z0-9/_-]*$/.test(url.hash)) url.hash = ''
            pageUrl = url.toString().slice(0, 2000)
          }
        } catch {}
        const sequence = ++this.sequence, capturedAt = new Date(this.now()).toISOString()
        for (const watch of watchers) {
          if (this.watches.get(watch.view.id) !== watch || watch.expiresAt <= this.now()) continue
          watch.png = png
          watch.pageOwner = pageOwner
          watch.view = { id: watch.view.id, sessionId: watch.view.sessionId, sourceSessionId: sessionId,
            tool, sequence, capturedAt, state, ...(pageUrl ? { pageUrl } : {}) }
        }
      } catch {
        for (const watch of watchers) {
          if (!watch.pageOwner || watch.pageOwner === pageOwner) {
            watch.png = undefined
            watch.view.state = 'paused'
            watch.view.sequence = ++this.sequence
            watch.view.capturedAt = new Date(this.now()).toISOString()
          }
        }
      }
    }
    const tick = () => {
      inFlight = capture('live').finally(() => {
        if (!stopped && !page.isClosed() && targets().length) timer = setTimeout(tick, 2000)
      })
      return inFlight
    }
    const handle: PageObservation = {
      stop: () => {
        if (finishing) return finishing
        stopped = true
        clearTimeout(timer)
        finishing = (async () => {
          await inFlight
          if (!page.isClosed()) await capture('snapshot')
          for (const watch of targets()) {
            if (watch.pageOwner === pageOwner && watch.view.state === 'live') {
              watch.view.state = 'snapshot'
              watch.view.sequence = ++this.sequence
            }
          }
          this.pages.delete(handle)
        })()
        return finishing
      },
    }
    this.pages.add(handle)
    page.once('close', () => { void handle.stop().catch(() => {}) })
    await tick()
    return handle
  }
}

const registries = new WeakMap<ToolRegistry, BrowserObservationRegistry>()
export function browserObservations(registry: ToolRegistry): BrowserObservationRegistry {
  let observations = registries.get(registry)
  if (!observations) { observations = new BrowserObservationRegistry(); registries.set(registry, observations) }
  return observations
}
