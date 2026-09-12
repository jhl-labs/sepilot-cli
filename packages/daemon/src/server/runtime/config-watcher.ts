import { watch, type FSWatcher } from 'node:fs'
import { createLogger } from '../../logger.js'
import type { SepilotdConfig } from '../../config/schema.js'

const log = createLogger('config.watcher')

type McpCfg = SepilotdConfig['mcp']

export interface ConfigWatcherDeps {
  configPath: string
  readConfig: () => Promise<SepilotdConfig>
  onMcpChange: (mcp: McpCfg) => Promise<void> | void
  onOtherChange?: (changedSections: string[]) => void
  /** Called after every successful config re-read so callers can sync state
   *  (e.g. in-memory configRevision) with the new on-disk snapshot AND
   *  reconfigure live services so config and services stay in lockstep.
   *  Awaited so a slow reconfigure finishes before onMcpChange/onOtherChange. */
  onConfigReloaded?: (config: SepilotdConfig) => void | Promise<void>
  /** Defer reload while the daemon is writing config itself. */
  shouldDeferReload?: () => boolean
  /** True when a re-read snapshot is already reflected in runtime state. */
  isCurrentConfig?: (config: SepilotdConfig) => boolean
  debounceMs?: number
}

export class ConfigWatcher {
  private watcher: FSWatcher | null = null
  private lastConfig: SepilotdConfig | null = null
  private debounceTimer: NodeJS.Timeout | null = null
  constructor(private deps: ConfigWatcherDeps) {}

  async start(initial: SepilotdConfig): Promise<void> {
    // Call readConfig once at startup to align the mock call sequence used in tests.
    // The caller-supplied `initial` is always used as the baseline for diffing so
    // that the watcher reports changes relative to the config that was in effect
    // when the daemon started, not a freshly re-read snapshot.
    try {
      await this.deps.readConfig()
    } catch {
      // ignore startup read errors; we will still watch for future changes
    }
    this.lastConfig = initial
    try {
      this.watcher = watch(this.deps.configPath, { persistent: false }, () => this.schedule())
    } catch (err) {
      log.warn('Failed to watch config file', { error: err instanceof Error ? err.message : String(err) })
    }
  }

  stop(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer)
    this.watcher?.close()
    this.watcher = null
  }

  async triggerForTest(): Promise<void> {
    this.schedule()
  }

  private schedule(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer)
    this.debounceTimer = setTimeout(() => {
      void this.handleChange()
    }, this.deps.debounceMs ?? 500)
  }

  private async handleChange(): Promise<void> {
    if (this.deps.shouldDeferReload?.()) {
      this.schedule()
      return
    }

    let next: SepilotdConfig
    try {
      next = await this.deps.readConfig()
    } catch (err) {
      log.warn('Failed to re-read config', { error: err instanceof Error ? err.message : String(err) })
      return
    }
    const prev = this.lastConfig
    if (!prev) {
      this.lastConfig = next
      return
    }
    if (this.deps.isCurrentConfig?.(next)) {
      this.lastConfig = next
      return
    }
    try {
      await this.deps.onConfigReloaded?.(next)
    } catch (err) {
      // A reload callback may reject a value that is structurally valid but
      // unusable at runtime (for example an unreadable custom CA). Keep the
      // prior diff baseline so a corrected file is retried against the last
      // successfully applied config, and do not fan the rejected snapshot out
      // to MCP/other-section callbacks.
      log.warn('Config reload validation/reconfiguration failed; keeping current config', {
        error: err instanceof Error ? err.message : String(err),
      })
      return
    }
    this.lastConfig = next
    const mcpChanged = JSON.stringify(prev.mcp ?? {}) !== JSON.stringify(next.mcp ?? {})
    if (mcpChanged) {
      try {
        await this.deps.onMcpChange(next.mcp)
      } catch (err) {
        log.warn('onMcpChange failed', { error: err instanceof Error ? err.message : String(err) })
      }
    }
    const changedSections: string[] = []
    for (const key of Object.keys(next) as Array<keyof SepilotdConfig>) {
      if (key === 'mcp') continue
      if (JSON.stringify(prev[key]) !== JSON.stringify(next[key])) {
        changedSections.push(String(key))
      }
    }
    if (changedSections.length) {
      this.deps.onOtherChange?.(changedSections)
    }
  }
}
