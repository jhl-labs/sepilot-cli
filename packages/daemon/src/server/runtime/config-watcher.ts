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
  onConfigReloaded?: (config: SepilotdConfig) => void | Promise<void>
  shouldDeferReload?: () => boolean
  isCurrentConfig?: (config: SepilotdConfig) => boolean
  debounceMs?: number
}

export class ConfigWatcher {
  private watcher: FSWatcher | null = null
  private lastConfig: SepilotdConfig | null = null
  private debounceTimer: NodeJS.Timeout | null = null
  constructor(private deps: ConfigWatcherDeps) {}

  async start(initial: SepilotdConfig): Promise<void> {
    try {
      await this.deps.readConfig()
    } catch {
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
