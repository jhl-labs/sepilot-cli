import { readFile, writeFile } from 'node:fs/promises'
import { parseDocument, stringify } from 'yaml'
import type { McpToolManifest } from './security.js'

export interface McpServerEntry {
  name: string
  transport?: string
  enabled?: boolean
  command?: string
  args?: string[]
  env?: Record<string, string>
  url?: string
  headers?: Record<string, string>
  disabledTools?: string[]
  [key: string]: unknown
}

export interface ConfigDocument {
  version?: number
  device?: {
    id?: string
    name?: string
    role?: string
  }
  mcp: {
    servers: McpServerEntry[]
  }
  [key: string]: unknown
}

export class ConfigWriter {
  constructor(private configPath: string) {}

  async read(): Promise<ConfigDocument> {
    const raw = await readFile(this.configPath, 'utf-8')
    const doc = parseDocument(raw)
    const json = (doc.toJSON() ?? {}) as Partial<ConfigDocument>
    return {
      ...json,
      mcp: {
        servers: json.mcp?.servers ?? [],
      },
    }
  }

  async addMcpServer(entry: Record<string, unknown>): Promise<void> {
    const raw = await readFile(this.configPath, 'utf-8')
    const doc = parseDocument(raw)
    const json = doc.toJSON() as Record<string, unknown>

    const mcp = (json.mcp ?? { servers: [] }) as {
      servers: Array<Record<string, unknown>>
    }
    const servers = mcp.servers ?? []

    const name = entry.name as string
    if (servers.some((s) => s.name === name)) {
      throw new Error(`MCP server "${name}" already exists in config`)
    }

    servers.push(entry)
    mcp.servers = servers
    ;(json as Record<string, unknown>).mcp = mcp

    await writeFile(
      this.configPath,
      stringify(json, { lineWidth: 0 }),
      'utf-8',
    )
  }

  async removeMcpServer(name: string): Promise<void> {
    const raw = await readFile(this.configPath, 'utf-8')
    const doc = parseDocument(raw)
    const json = doc.toJSON() as Record<string, unknown>

    const mcp = (json.mcp ?? { servers: [] }) as {
      servers: Array<Record<string, unknown>>
    }
    mcp.servers = (mcp.servers ?? []).filter((s) => s.name !== name)
    ;(json as Record<string, unknown>).mcp = mcp

    await writeFile(
      this.configPath,
      stringify(json, { lineWidth: 0 }),
      'utf-8',
    )
  }

  async setDisabledTools(serverName: string, tools: string[]): Promise<void> {
    const raw = await readFile(this.configPath, 'utf-8')
    const doc = parseDocument(raw)
    const json = doc.toJSON() as Record<string, unknown>

    const mcp = (json.mcp ?? { servers: [] }) as {
      servers: Array<Record<string, unknown>>
    }
    const server = (mcp.servers ?? []).find((s) => s.name === serverName)
    if (!server) {
      throw new Error(`MCP server "${serverName}" not found in config`)
    }
    server.disabledTools = tools
    ;(json as Record<string, unknown>).mcp = mcp

    await writeFile(
      this.configPath,
      stringify(json, { lineWidth: 0 }),
      'utf-8',
    )
  }

  async setMcpToolManifest(
    serverName: string,
    manifest: McpToolManifest,
  ): Promise<void> {
    const raw = await readFile(this.configPath, 'utf-8')
    const doc = parseDocument(raw)
    const json = doc.toJSON() as Record<string, unknown>

    const mcp = (json.mcp ?? { servers: [] }) as {
      servers: Array<Record<string, unknown>>
    }
    const server = (mcp.servers ?? []).find((s) => s.name === serverName)
    if (!server) {
      throw new Error(`MCP server "${serverName}" not found in config`)
    }
    const existing = server.toolManifest as { digest?: unknown } | undefined
    if (existing?.digest === manifest.digest) return

    server.toolManifest = manifest
    ;(json as Record<string, unknown>).mcp = mcp

    await writeFile(
      this.configPath,
      stringify(json, { lineWidth: 0 }),
      'utf-8',
    )
  }
}
