import { mkdir, readFile, writeFile, rm } from 'node:fs/promises'
import { join } from 'node:path'

export interface McpMarketplace {
  name: string
  url: string
  publisher?: string
  publicKey?: string
  addedAt: string
  lastSync: string | null
}

const DEFAULT_SEED: Array<Pick<McpMarketplace, 'name' | 'url' | 'publisher'>> = [
  {
    name: 'official',
    url: 'https://github.com/modelcontextprotocol/servers',
    publisher: 'modelcontextprotocol',
  },
]

export interface AddMcpMarketplaceOptions {
  publisher?: string
  publicKey?: string
}

export class McpMarketplaceCatalog {
  private filePath: string
  private clonesDir: string
  private entries: McpMarketplace[] = []

  constructor(private baseDir: string) {
    this.filePath = join(baseDir, 'mcp-marketplaces.json')
    this.clonesDir = join(baseDir, 'mcp-marketplaces')
  }

  async init(): Promise<void> {
    await mkdir(this.baseDir, { recursive: true })
    try {
      const raw = await readFile(this.filePath, 'utf-8')
      const parsed = JSON.parse(raw) as { marketplaces: McpMarketplace[] }
      this.entries = parsed.marketplaces ?? []
    } catch {
      this.entries = DEFAULT_SEED.map((m) => ({
        ...m,
        addedAt: new Date().toISOString(),
        lastSync: null,
      }))
      await this.persist()
    }
  }

  async list(): Promise<McpMarketplace[]> {
    return [...this.entries]
  }

  async get(name: string): Promise<McpMarketplace | null> {
    return (
      this.entries.find((m) => m.name.toLowerCase() === name.toLowerCase()) ??
      null
    )
  }

  async add(
    name: string,
    url: string,
    options: AddMcpMarketplaceOptions = {},
  ): Promise<McpMarketplace> {
    if (this.entries.some((m) => m.name.toLowerCase() === name.toLowerCase())) {
      throw new Error(`MCP marketplace "${name}" already exists`)
    }
    const entry: McpMarketplace = {
      name,
      url,
      publisher: options.publisher,
      publicKey: options.publicKey,
      addedAt: new Date().toISOString(),
      lastSync: null,
    }
    this.entries.push(entry)
    await this.persist()
    return entry
  }

  async remove(name: string): Promise<boolean> {
    const idx = this.entries.findIndex(
      (m) => m.name.toLowerCase() === name.toLowerCase(),
    )
    if (idx < 0) return false
    const [removed] = this.entries.splice(idx, 1)
    await this.persist()
    await rm(join(this.clonesDir, removed.name), {
      recursive: true,
      force: true,
    })
    return true
  }

  clonePath(name: string): string {
    return join(this.clonesDir, name)
  }

  async markSynced(name: string): Promise<void> {
    const entry = this.entries.find(
      (m) => m.name.toLowerCase() === name.toLowerCase(),
    )
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
