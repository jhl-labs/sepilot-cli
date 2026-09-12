export interface UpdateInfo {
  currentVersion: string
  latestVersion: string
  updateAvailable: boolean
  releaseUrl?: string
  releaseNotes?: string
  publishedAt?: string
}

export interface UpdateConfig {
  channel: 'stable' | 'beta' | 'dev'
  checkIntervalMs?: number
  autoCheck?: boolean
  owner?: string
  repo?: string
}

export class UpdateChecker {
  private config: UpdateConfig
  private currentVersion: string
  private lastCheck: UpdateInfo | null = null
  private timer: ReturnType<typeof setInterval> | null = null
  private stopped = false

  constructor(currentVersion: string, config: UpdateConfig) {
    this.currentVersion = currentVersion
    this.config = config
  }

  async start(): Promise<void> {
    this.stopped = false
    if (!this.config.autoCheck) return
    // Initial check
    await this.check().catch(() => {})
    if (this.stopped) return
    // Periodic checks
    const interval = this.config.checkIntervalMs ?? 24 * 60 * 60 * 1000 // 24h default
    this.timer = setInterval(() => this.check().catch(() => {}), interval)
    this.timer.unref?.()
  }

  async stop(): Promise<void> {
    this.stopped = true
    if (this.timer) { clearInterval(this.timer); this.timer = null }
  }

  async check(): Promise<UpdateInfo> {
    const owner = this.config.owner ?? 'jhl-labs'
    const repo = this.config.repo ?? 'sepilotd'

    try {
      const res = await fetch(`https://api.github.com/repos/${owner}/${repo}/releases/latest`, {
        headers: { 'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'sepilotd' },
      })

      if (!res.ok) {
        this.lastCheck = { currentVersion: this.currentVersion, latestVersion: this.currentVersion, updateAvailable: false }
        return this.lastCheck
      }

      const data = await res.json() as { tag_name: string; html_url: string; body: string; published_at: string }
      const latestVersion = data.tag_name.replace(/^v/, '')
      const updateAvailable = this.compareVersions(latestVersion, this.currentVersion) > 0

      // Filter by channel
      if (this.config.channel === 'stable' && (latestVersion.includes('beta') || latestVersion.includes('dev'))) {
        this.lastCheck = { currentVersion: this.currentVersion, latestVersion: this.currentVersion, updateAvailable: false }
        return this.lastCheck
      }

      this.lastCheck = {
        currentVersion: this.currentVersion,
        latestVersion,
        updateAvailable,
        releaseUrl: data.html_url,
        releaseNotes: data.body?.slice(0, 1000),
        publishedAt: data.published_at,
      }
      return this.lastCheck
    } catch {
      this.lastCheck = { currentVersion: this.currentVersion, latestVersion: this.currentVersion, updateAvailable: false }
      return this.lastCheck
    }
  }

  getLastCheck(): UpdateInfo | null {
    return this.lastCheck
  }

  private compareVersions(a: string, b: string): number {
    const pa = a.split('.').map(Number)
    const pb = b.split('.').map(Number)
    for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
      const na = pa[i] ?? 0
      const nb = pb[i] ?? 0
      if (na > nb) return 1
      if (na < nb) return -1
    }
    return 0
  }
}
