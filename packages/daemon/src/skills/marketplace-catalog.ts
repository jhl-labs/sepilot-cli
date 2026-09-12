import { mkdir, readFile, writeFile, rm } from 'node:fs/promises'
import { join, resolve, sep } from 'node:path'

export interface Marketplace {
  name: string
  url: string
  addedAt: string
  lastSync: string | null
}

const DEFAULT_SEED: Array<Pick<Marketplace, 'name' | 'url'>> = [
  {
    name: 'anthropics',
    url: 'https://github.com/anthropics/claude-plugins-official',
  },
]

export const MARKETPLACE_NAME_PATTERN = /^[a-z0-9][a-z0-9_-]*$/

export function assertMarketplaceName(name: string): string {
  const trimmed = name.trim()
  if (!MARKETPLACE_NAME_PATTERN.test(trimmed)) {
    throw new Error(`unsafe marketplace name: ${JSON.stringify(name)}`)
  }
  return trimmed
}

function containedClonePath(clonesDir: string, name: string): string {
  const safeName = assertMarketplaceName(name)
  const root = resolve(clonesDir)
  const target = resolve(join(root, safeName))
  if (target !== root && !target.startsWith(root + sep)) {
    throw new Error(`marketplace name escapes clones dir: ${name}`)
  }
  return target
}

export class MarketplaceCatalog {
  private filePath: string
  private clonesDir: string
  private entries: Marketplace[] = []

  constructor(private baseDir: string) {
    this.filePath = join(baseDir, 'marketplaces.json')
    this.clonesDir = join(baseDir, 'marketplaces')
  }

  async init(): Promise<void> {
    await mkdir(this.baseDir, { recursive: true })
    try {
      const raw = await readFile(this.filePath, 'utf-8')
      const parsed = JSON.parse(raw) as { marketplaces: Marketplace[] }
      this.entries = parsed.marketplaces ?? []
    } catch {
      this.entries = DEFAULT_SEED.map(m => ({
        ...m,
        addedAt: new Date().toISOString(),
        lastSync: null,
      }))
      await this.persist()
    }
  }

  async list(): Promise<Marketplace[]> {
    return [...this.entries]
  }

  async get(name: string): Promise<Marketplace | null> {
    return this.entries.find(m => m.name.toLowerCase() === name.toLowerCase()) ?? null
  }

  async add(name: string, url: string): Promise<Marketplace> {
    if (this.entries.some(m => m.name.toLowerCase() === name.toLowerCase())) {
      throw new Error(`Marketplace "${name}" already exists`)
    }
    const safeName = assertMarketplaceName(name)
    const entry: Marketplace = {
      name: safeName,
      url,
      addedAt: new Date().toISOString(),
      lastSync: null,
    }
    this.entries.push(entry)
    await this.persist()
    return entry
  }

  async remove(name: string): Promise<boolean> {
    const idx = this.entries.findIndex(m => m.name.toLowerCase() === name.toLowerCase())
    if (idx < 0) return false
    const clonePath = containedClonePath(this.clonesDir, this.entries[idx].name)
    this.entries.splice(idx, 1)
    await this.persist()
    await rm(clonePath, { recursive: true, force: true })
    return true
  }

  clonePath(name: string): string {
    return containedClonePath(this.clonesDir, name)
  }

  async markSynced(name: string): Promise<void> {
    const entry = this.entries.find(m => m.name.toLowerCase() === name.toLowerCase())
    if (!entry) return
    entry.lastSync = new Date().toISOString()
    await this.persist()
  }

  private async persist(): Promise<void> {
    await mkdir(this.baseDir, { recursive: true })
    await writeFile(
      this.filePath,
      JSON.stringify({ marketplaces: this.entries }, null, 2),
      'utf-8',
    )
  }
}
