import { mkdir, readFile, stat } from 'node:fs/promises'
import { join } from 'node:path'
import simpleGit from 'simple-git'
import type {
  McpMarketplaceCatalog,
  McpMarketplace,
} from './marketplace-catalog.js'
import {
  BUILTIN_MCP_MARKETPLACE,
  builtinMcpServerTemplates,
} from './builtin-templates.js'

export interface McpServerTemplate {
  name: string
  description: string
  transport: 'stdio' | 'sse' | 'http'
  command?: string
  args?: string[]
  url?: string
  env?: Record<string, string>
  headers?: Record<string, string>
  tags?: string[]
  homepage?: string
  repository?: string
  publisher?: string
  digest?: string
  signature?: string
  signatureKeyId?: string
  variables?: McpServerTemplateVariable[]
  marketplace: string
  marketplaceUrl?: string
  marketplacePublisher?: string
  marketplacePublicKey?: string
  sourceRef?: string
}

export interface McpServerTemplateVariable {
  name: string
  label?: string
  description?: string
  placeholder?: string
  required?: boolean
  secret?: boolean
}

interface McpServersManifest {
  servers: Array<{
    name: string
    description?: string
    transport: string
    command?: string
    args?: string[]
    url?: string
    headers?: Record<string, string>
    env?: Record<string, string>
    tags?: string[]
    homepage?: string
    repository?: string
    publisher?: string
    digest?: string
    signature?: string
    signatureKeyId?: string
    variables?: McpServerTemplateVariable[]
  }>
}

export interface McpMarketplaceSourceDeps {
  catalog: McpMarketplaceCatalog
}

export class McpMarketplaceSource {
  constructor(private deps: McpMarketplaceSourceDeps) {}

  async search(query: string): Promise<McpServerTemplate[]> {
    const marketplaces = await this.deps.catalog.list()
    const results: McpServerTemplate[] = []
    const lowerQuery = query.toLowerCase()

    for (const mp of marketplaces) {
      try {
        await this.ensureCloned(mp)
      } catch {
        continue
      }
      const sourceRef = await this.currentRef(mp)
      const servers = await this.readManifest(mp)
      for (const server of servers) {
        const searchable = [
          server.name,
          server.description ?? '',
          ...(server.tags ?? []),
        ]
          .join(' ')
          .toLowerCase()
        if (searchable.includes(lowerQuery)) {
          results.push(this.toTemplate(mp, server, sourceRef))
        }
      }
    }

    return [
      ...results,
      ...builtinMcpServerTemplates.filter((template) =>
        templateMatchesQuery(template, lowerQuery),
      ),
    ]
  }

  async get(
    marketplaceName: string | null,
    serverName: string,
  ): Promise<McpServerTemplate | null> {
    if (marketplaceName?.toLowerCase() === BUILTIN_MCP_MARKETPLACE) {
      return getBuiltinTemplate(serverName)
    }

    const marketplaces: McpMarketplace[] = marketplaceName
      ? [await this.deps.catalog.get(marketplaceName)].filter(
          (m): m is McpMarketplace => m !== null,
        )
      : await this.deps.catalog.list()

    for (const mp of marketplaces) {
      try {
        await this.ensureCloned(mp)
      } catch {
        continue
      }
      const sourceRef = await this.currentRef(mp)
      const servers = await this.readManifest(mp)
      const match = servers.find(
        (s) => s.name.toLowerCase() === serverName.toLowerCase(),
      )
      if (match) {
        return this.toTemplate(mp, match, sourceRef)
      }
    }

    return marketplaceName ? null : getBuiltinTemplate(serverName)
  }

  private async ensureCloned(mp: McpMarketplace): Promise<void> {
    const path = this.deps.catalog.clonePath(mp.name)
    try {
      const s = await stat(join(path, '.git'))
      if (s.isDirectory()) {
        try {
          await simpleGit(path).pull()
        } catch {
          // ignore pull failures for offline work
        }
        return
      }
    } catch {
      /* not cloned yet */
    }
    await mkdir(path, { recursive: true })
    await simpleGit().clone(mp.url, path, ['--depth', '1'])
    await this.deps.catalog.markSynced(mp.name)
  }

  private async readManifest(
    mp: McpMarketplace,
  ): Promise<McpServersManifest['servers']> {
    const root = this.deps.catalog.clonePath(mp.name)
    try {
      const raw = await readFile(join(root, 'mcp-servers.json'), 'utf-8')
      const parsed = JSON.parse(raw) as McpServersManifest
      return parsed.servers ?? []
    } catch {
      return []
    }
  }

  private async currentRef(mp: McpMarketplace): Promise<string | undefined> {
    try {
      return await simpleGit(this.deps.catalog.clonePath(mp.name)).revparse(['HEAD'])
    } catch {
      return undefined
    }
  }

  private toTemplate(
    mp: McpMarketplace,
    server: McpServersManifest['servers'][number],
    sourceRef?: string,
  ): McpServerTemplate {
    return {
      name: server.name,
      description: server.description ?? '',
      transport: server.transport as 'stdio' | 'sse' | 'http',
      command: server.command,
      args: server.args,
      url: server.url,
      headers: server.headers,
      env: server.env,
      tags: server.tags ?? [],
      homepage: server.homepage,
      repository: server.repository,
      publisher: server.publisher,
      digest: server.digest,
      signature: server.signature,
      signatureKeyId: server.signatureKeyId,
      variables: server.variables,
      marketplace: mp.name,
      marketplaceUrl: mp.url,
      marketplacePublisher: mp.publisher,
      marketplacePublicKey: mp.publicKey,
      sourceRef,
    }
  }
}

function templateMatchesQuery(
  template: McpServerTemplate,
  lowerQuery: string,
): boolean {
  return [
    template.name,
    template.description,
    ...(template.tags ?? []),
  ]
    .join(' ')
    .toLowerCase()
    .includes(lowerQuery)
}

function getBuiltinTemplate(serverName: string): McpServerTemplate | null {
  return (
    builtinMcpServerTemplates.find(
      (template) => template.name.toLowerCase() === serverName.toLowerCase(),
    ) ?? null
  )
}
