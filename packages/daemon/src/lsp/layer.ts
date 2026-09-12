import { LspClient, type LspSpawner } from './client.js'
import { discoverLspServers, DEFAULT_LSP_SERVERS, type LspServerSpec } from './registry.js'

export interface LspLayerOptions {
  rootUri: string
  spawner?: LspSpawner
  servers?: readonly LspServerSpec[]
  which?: (bin: string) => Promise<string | null>
  enabled?: boolean
  /**
   * Maximum number of crash-and-respawn cycles before the layer marks
   * a language permanently disabled for the rest of the session.
   * Default 3.
   */
  maxRestartsPerLanguage?: number
  /**
   * Minimum gap (ms) between consecutive starts for the same language.
   * Prevents tight crash-loops from burning CPU. Default 5000.
   */
  minRestartIntervalMs?: number
}

export interface LspLayer {
  get(language: string): LspClient | null
  stopAll(): void
  listLanguages(): string[]
}

interface LanguageState {
  attempts: number
  lastStartAt: number
  disabled: boolean
}

class LazyLspLayer implements LspLayer {
  private readonly clients = new Map<string, LspClient>()
  private readonly languageState = new Map<string, LanguageState>()
  private specs: readonly LspServerSpec[] | null = null
  private readonly maxRestarts: number
  private readonly minRestartIntervalMs: number

  constructor(private readonly options: LspLayerOptions) {
    this.maxRestarts = options.maxRestartsPerLanguage ?? 3
    this.minRestartIntervalMs = options.minRestartIntervalMs ?? 5_000
  }

  async init(): Promise<void> {
    if (this.options.enabled === false) {
      this.specs = []
      return
    }
    const which = this.options.which ?? defaultWhich
    this.specs = await discoverLspServers({ which }, this.options.servers ?? DEFAULT_LSP_SERVERS)
  }

  get(language: string): LspClient | null {
    if (this.specs === null) return null

    const state = this.languageState.get(language)
    if (state?.disabled) return null

    const cached = this.clients.get(language)
    if (cached?.isRunning()) return cached
    // Stale handle (server died) — drop it; we'll respawn below if
    // policy allows.
    if (cached && !cached.isRunning()) {
      this.clients.delete(language)
    }

    const spec = this.specs.find((s) => s.language === language)
    if (!spec) return null

    // Crash-loop guard: refuse to start if the previous start was very
    // recent. Returning null lets the caller fall back gracefully (e.g.
    // postEditAnalysis falls through to the ripgrep heuristic).
    const now = Date.now()
    if (state && now - state.lastStartAt < this.minRestartIntervalMs) {
      return null
    }

    const client = new LspClient(this.options.spawner)
    client.onExit(() => {
      this.handleExit(language)
    })
    try {
      client.start(spec, this.options.rootUri)
    } catch {
      this.handleExit(language)
      return null
    }
    this.clients.set(language, client)
    this.languageState.set(language, {
      attempts: (state?.attempts ?? 0) + 1,
      lastStartAt: now,
      disabled: false,
    })
    return client
  }

  private handleExit(language: string): void {
    this.clients.delete(language)
    const state = this.languageState.get(language)
    if (!state) return
    if (state.attempts >= this.maxRestarts) {
      this.languageState.set(language, { ...state, disabled: true })
    }
  }

  stopAll(): void {
    for (const client of this.clients.values()) {
      try {
        client.stop()
      } catch {
        // Ignore shutdown errors — best-effort cleanup.
      }
    }
    this.clients.clear()
    // Operator-driven shutdown — clear restart bookkeeping so a
    // subsequent get() can respawn freely without tripping the
    // crash-loop guard.
    this.languageState.clear()
  }

  listLanguages(): string[] {
    return this.specs?.map((s) => s.language) ?? []
  }
}

async function defaultWhich(bin: string): Promise<string | null> {
  const { execFile } = await import('node:child_process')
  return new Promise((resolve) => {
    execFile('which', [bin], (err, stdout) => {
      if (err) return resolve(null)
      const path = stdout.toString().trim()
      resolve(path || null)
    })
  })
}

export async function buildLspLayer(options: LspLayerOptions): Promise<LspLayer> {
  const layer = new LazyLspLayer(options)
  await layer.init()
  return layer
}
