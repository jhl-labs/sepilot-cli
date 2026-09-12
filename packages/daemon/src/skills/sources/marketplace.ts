import { mkdir, mkdtemp, readFile, readdir, rename, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, join, relative, resolve, sep } from 'node:path'
import type { SkillSource, SkillRef, FetchedSkill } from './types.js'
import { parseSkillMd } from '../loader.js'
import {
  SkillFetchError,
  SkillPathTraversalError,
  SkillSourceUrlNotAllowedError,
} from '../errors.js'
import type { MarketplaceCatalog, Marketplace } from '../marketplace-catalog.js'
import type { SkillSourceUrlPolicy } from '../source-url-policy.js'
import {
  checkoutDetachedGitRef,
  validateGitObjectId,
} from './git-ref.js'
import {
  fetchPublicSkillText,
  type PublicSkillFetchRuntime,
} from './public-http.js'
import { prepareSafeGitTransport } from './safe-git.js'

export interface MarketplaceSourceDeps extends PublicSkillFetchRuntime {
  catalog: MarketplaceCatalog
  urlPolicy?: SkillSourceUrlPolicy
}

export interface MarketplaceSkillSearchResult {
  marketplace: string
  source: string
  metadata: import('@sepilotd/core').SkillMetadata
  metadataOnly?: boolean
  sourceKind?: string
}

export interface MarketplaceSkillSearchOptions {
  marketplace?: string | null
  limit?: number
}

interface MarketplacePluginSource {
  source?: string
  url?: string
  path?: string
  ref?: string
  sha?: string
}

interface MarketplacePlugin {
  name?: string
  description?: string
  category?: string
  tags?: unknown
  author?: unknown
  source?: string | MarketplacePluginSource
}

interface MarketplaceJson {
  plugins?: MarketplacePlugin[]
}

function asString(value: unknown): string | undefined {
  if (typeof value === 'string') return value.trim() || undefined
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return undefined
}

function parseTags(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map(asString).filter((item): item is string => Boolean(item))
  }
  const single = asString(value)
  return single ? single.split(/[,\s]+/).filter(Boolean) : []
}

function slugify(input: string): string {
  return input
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function sourceKind(source: MarketplacePlugin['source']): string | undefined {
  if (typeof source === 'string') return 'local'
  return asString(source?.source) ?? (source?.url ? 'url' : undefined)
}

function authorName(author: unknown): string | undefined {
  if (typeof author === 'string') return author.trim() || undefined
  if (author && typeof author === 'object' && 'name' in author) {
    return asString((author as { name?: unknown }).name)
  }
  return undefined
}

function pluginId(plugin: MarketplacePlugin): string {
  return slugify(asString(plugin.name) ?? 'plugin') || 'plugin'
}

function pluginMetadata(plugin: MarketplacePlugin): import('@sepilotd/core').SkillMetadata | null {
  const name = asString(plugin.name)
  if (!name) return null
  const id = pluginId(plugin)
  const category = asString(plugin.category)
  const tags = Array.from(new Set([
    ...parseTags(plugin.tags),
    ...(category ? [category] : []),
  ]))
  return {
    id,
    name,
    version: sourceVersion(plugin.source),
    description: asString(plugin.description) ?? `Marketplace plugin ${name}`,
    author: authorName(plugin.author),
    tags: tags.length ? tags : undefined,
    tools: [],
  }
}

function sourceVersion(source: MarketplacePlugin['source']): string {
  if (typeof source === 'object' && source) {
    const ref = asString(source.ref)
    if (ref) return ref
    const sha = asString(source.sha)
    if (sha) return sha.slice(0, 12)
  }
  return '0.0.0'
}

function matchesPlugin(plugin: MarketplacePlugin, name: string): boolean {
  const target = slugify(name)
  const pluginName = asString(plugin.name)
  if (!pluginName) return false
  return pluginName === name || slugify(pluginName) === target
}

function isHttpsUrl(value: string): boolean {
  return /^https:\/\//i.test(value)
}

function isProbablyLocalPath(value: string): boolean {
  return !/^[a-z][a-z0-9+.-]*:\/\//i.test(value) && !/^git@/i.test(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message ? error.message : String(error)
}

export class MarketplaceSource implements SkillSource {
  constructor(private deps: MarketplaceSourceDeps) {}

  async fetch(ref: SkillRef): Promise<FetchedSkill[]> {
    if (ref.type !== 'marketplace') throw new SkillFetchError(`MarketplaceSource cannot handle ${ref.type}`)

    const marketplaces = await this.resolveMarketplaces(ref.marketplace)

    const skipReasons: string[] = []
    for (const mp of marketplaces) {
      try {
        await this.ensureCloned(mp)
      } catch (err) {
        if (err instanceof SkillSourceUrlNotAllowedError) throw err
        skipReasons.push(`${mp.name}: ${errorMessage(err)}`)
        continue
      }
      const found = await this.findSkillInClone(mp, ref.name)
      if (found.length > 0) return found
    }

    const suffix = skipReasons.length ? ` (skipped: ${skipReasons.join('; ')})` : ''
    throw new SkillFetchError(`Skill "${ref.name}" not found in any marketplace${suffix}`)
  }

  async search(
    query: string,
    options: MarketplaceSkillSearchOptions = {},
  ): Promise<MarketplaceSkillSearchResult[]> {
    const q = query.trim().toLowerCase()
    if (!q) return []

    const limit = Math.max(1, Math.min(options.limit ?? 20, 100))
    const marketplaces = await this.resolveMarketplaces(options.marketplace ?? null)
    const results: MarketplaceSkillSearchResult[] = []
    const seen = new Set<string>()

    for (const mp of marketplaces) {
      try {
        await this.ensureCloned(mp)
      } catch (err) {
        if (err instanceof SkillSourceUrlNotAllowedError) throw err
        continue
      }
      let skills: Array<MarketplaceSkillSearchResult & { content: string }>
      try {
        skills = await this.listSkillsInClone(mp)
      } catch (err) {
        if (err instanceof SkillSourceUrlNotAllowedError) throw err
        continue
      }
      for (const skill of skills) {
        const haystack = [
          skill.metadata.id,
          skill.metadata.name,
          skill.metadata.description,
          ...(skill.metadata.tags ?? []),
        ].join('\n').toLowerCase()
        if (!haystack.includes(q)) continue
        const key = `${skill.marketplace}/${skill.metadata.id}`
        if (seen.has(key)) continue
        seen.add(key)
        results.push(skill)
        if (results.length >= limit) return results
      }
    }
    return results
  }

  private async resolveMarketplaces(name: string | null): Promise<Marketplace[]> {
    if (!name) return this.deps.catalog.list()
    const marketplace = await this.deps.catalog.get(name)
    if (!marketplace) throw new SkillFetchError(`Unknown marketplace: ${name}`)
    return [marketplace]
  }

  private async ensureCloned(mp: Marketplace): Promise<void> {
    const path = this.deps.catalog.clonePath(mp.name)
    const transport = await prepareSafeGitTransport(mp.url, {
      resolveUrl: this.deps.resolveUrl,
      urlPolicy: this.deps.urlPolicy,
    })
    const parentDir = dirname(path)
    await mkdir(parentDir, { recursive: true })
    const stagingDir = await mkdtemp(join(parentDir, `${basename(path)}-sync-`))
    try {
      await transport.createGit().clone(
        transport.source,
        stagingDir,
        ['--depth', '1', '--no-tags'],
      )
      await rm(path, { recursive: true, force: true })
      await rename(stagingDir, path)
      await this.deps.catalog.markSynced(mp.name)
    } catch (error) {
      await rm(stagingDir, { recursive: true, force: true })
      throw error
    }
  }

  private async findSkillInClone(
    mp: Marketplace,
    skillName: string,
  ): Promise<FetchedSkill[]> {
    const root = this.deps.catalog.clonePath(mp.name)
    const manifest = await this.readMarketplaceManifest(root)

    if (manifest?.plugins?.length) {
      const exactPlugins = manifest.plugins.filter((plugin) => matchesPlugin(plugin, skillName))
      for (const plugin of exactPlugins) {
        const skills = await this.fetchPluginSkills(mp, root, plugin)
        if (skills.length > 0) return skills
      }

      for (const plugin of manifest.plugins) {
        if (typeof plugin.source !== 'string') continue
        const skills = await this.fetchLocalPluginSkills(mp, root, plugin)
        for (const skill of skills) {
          const last = skill.metadata.id
          if (
            skill.metadata.name === skillName
            || skill.metadata.id === skillName
            || slugify(skill.metadata.name) === slugify(skillName)
            || last === skillName
          ) {
            return [skill]
          }
        }
      }
      return []
    }

    return this.findSkillInRoots(mp, skillName, this.defaultScanRoots(root))
  }

  private async listSkillsInClone(
    mp: Marketplace,
  ): Promise<Array<MarketplaceSkillSearchResult & { content: string }>> {
    const root = this.deps.catalog.clonePath(mp.name)
    const manifest = await this.readMarketplaceManifest(root)

    const results: Array<MarketplaceSkillSearchResult & { content: string }> = []
    const seen = new Set<string>()
    if (manifest?.plugins?.length) {
      for (const plugin of manifest.plugins) {
        let localSkills: FetchedSkill[] = []
        if (typeof plugin.source === 'string') {
          try {
            localSkills = await this.fetchLocalPluginSkills(mp, root, plugin)
          } catch (err) {
            if (err instanceof SkillSourceUrlNotAllowedError) throw err
          }
        }
        const meta = localSkills.length === 0 ? pluginMetadata(plugin) : null
        if (meta) {
          const key = `${mp.name}/${meta.id}`
          if (!seen.has(key)) {
            seen.add(key)
            results.push({
              marketplace: mp.name,
              source: key,
              metadata: { ...meta, source: { type: 'marketplace', ref: key } },
              content: '',
              metadataOnly: true,
              sourceKind: sourceKind(plugin.source),
            })
          }
        }
        for (const skill of localSkills) {
          const key = `${mp.name}/${skill.metadata.id}`
          if (seen.has(key)) continue
          seen.add(key)
          results.push({
            marketplace: mp.name,
            source: key,
            metadata: skill.metadata,
            content: skill.content,
          })
        }
      }
      return results
    }

    for (const skill of await this.scanSkillRoots(mp, this.defaultScanRoots(root), null)) {
      const key = `${mp.name}/${skill.metadata.id}`
      if (seen.has(key)) continue
      seen.add(key)
      results.push({
        marketplace: mp.name,
        source: key,
        metadata: skill.metadata,
        content: skill.content,
      })
    }
    return results
  }

  private async fetchPluginSkills(
    mp: Marketplace,
    root: string,
    plugin: MarketplacePlugin,
  ): Promise<FetchedSkill[]> {
    if (typeof plugin.source === 'string') {
      return this.fetchLocalPluginSkills(mp, root, plugin)
    }

    const source = plugin.source
    const url = asString(source?.url)
    if (!source || !url) return []

    if (/\.md(?:[?#].*)?$/i.test(url)) {
      return this.fetchDirectSkillUrl(mp, plugin, url)
    }

    const cloneRoot = await this.cloneExternalPluginSource(mp, source)
    try {
      const subRoot = source.path
        ? this.safeJoin(cloneRoot, source.path)
        : cloneRoot
      return this.scanSkillRoots(mp, this.pluginScanRoots(subRoot), pluginId(plugin))
    } finally {
      await rm(cloneRoot, { recursive: true, force: true })
    }
  }

  private async fetchLocalPluginSkills(
    mp: Marketplace,
    root: string,
    plugin: MarketplacePlugin,
  ): Promise<FetchedSkill[]> {
    if (typeof plugin.source !== 'string') return []
    const sourceRoot = this.safeJoin(root, plugin.source)
    return this.scanSkillRoots(mp, this.pluginScanRoots(sourceRoot), null)
  }

  private async fetchDirectSkillUrl(
    mp: Marketplace,
    plugin: MarketplacePlugin,
    url: string,
  ): Promise<FetchedSkill[]> {
    const safeUrl = this.validateExternalPluginUrl(mp, url)
    const res = await fetchPublicSkillText(safeUrl, {
      fetch: this.deps.fetch,
      resolveUrl: this.deps.resolveUrl,
      createDispatcher: this.deps.createDispatcher,
      assertAllowed: (candidate) => {
        this.validateExternalPluginUrl(mp, candidate)
      },
    })
    if (!res.ok) throw new SkillFetchError(`Failed to fetch ${safeUrl}: ${res.status}`)
    const raw = res.text
    const parsed = parseSkillMd(raw, pluginId(plugin))
    return [{
      metadata: parsed.metadata,
      content: parsed.content,
      source: { type: 'marketplace', ref: `${mp.name}/${pluginId(plugin)}` },
      publisher: mp.name,
    }]
  }

  private async cloneExternalPluginSource(
    mp: Marketplace,
    source: MarketplacePluginSource,
  ): Promise<string> {
    const url = this.validateExternalPluginUrl(mp, asString(source.url) ?? '')
    const transport = await prepareSafeGitTransport(url, {
      resolveUrl: this.deps.resolveUrl,
      urlPolicy: this.deps.urlPolicy,
    })
    const targetDir = await mkdtemp(join(tmpdir(), 'sepilotd-ms-'))
    const cloneArgs = ['--depth', '1']
    const ref = asString(source.ref)
    const sha = source.sha
      ? validateGitObjectId(asString(source.sha) ?? '', 'Plugin source SHA')
      : undefined
    if (ref && !sha) cloneArgs.push('--branch', ref)
    try {
      await transport.createGit().clone(transport.source, targetDir, cloneArgs)
      const git = transport.createGit(targetDir)
      if (ref && sha) {
        await checkoutDetachedGitRef(git, ref)
      }
      if (sha) {
        try {
          await checkoutDetachedGitRef(git, sha)
        } catch {
          await git.raw(['fetch', '--depth', '1', '--no-tags', transport.source, sha])
          await checkoutDetachedGitRef(git, sha)
        }
        const head = (await git.revparse(['HEAD'])).trim()
        if (!head.startsWith(sha)) {
          throw new SkillFetchError(`Plugin source SHA mismatch for ${url}`)
        }
      }
      return targetDir
    } catch (error) {
      await rm(targetDir, { recursive: true, force: true })
      if (error instanceof SkillFetchError) throw error
      throw new SkillFetchError(`plugin source fetch failed: ${(error as Error).message}`, error)
    }
  }

  private validateExternalPluginUrl(mp: Marketplace, url: string): string {
    const trimmed = url.trim()
    if (!trimmed) throw new SkillFetchError('Plugin source URL is empty')
    if (isHttpsUrl(trimmed)) {
      this.deps.urlPolicy?.assertAllowed(trimmed, 'marketplace plugin')
      return trimmed
    }
    if (isProbablyLocalPath(trimmed) && isProbablyLocalPath(mp.url)) {
      this.deps.urlPolicy?.assertAllowed(trimmed, 'marketplace plugin')
      return trimmed
    }
    throw new SkillFetchError(`Plugin source must use https:// URL: ${trimmed}`)
  }

  private async findSkillInRoots(
    mp: Marketplace,
    skillName: string,
    roots: string[],
  ): Promise<FetchedSkill[]> {
    const skills = await this.scanSkillRoots(mp, roots, null)
    for (const skill of skills) {
      const last = skill.metadata.id
      if (
        skill.metadata.name === skillName
        || skill.metadata.id === skillName
        || slugify(skill.metadata.name) === slugify(skillName)
        || last === skillName
      ) {
        return [skill]
      }
    }
    return []
  }

  private async scanSkillRoots(
    mp: Marketplace,
    roots: string[],
    sourceName: string | null,
  ): Promise<FetchedSkill[]> {
    const fetched: FetchedSkill[] = []
    const seen = new Set<string>()
    for (const scanRoot of roots) {
      const skillDirs = await this.findSkillDirs(scanRoot)
      for (const dir of skillDirs) {
        const last = dir.split(sep).pop() ?? ''
        let raw: string
        let parsed: ReturnType<typeof parseSkillMd>
        try {
          raw = await readFile(join(dir, 'SKILL.md'), 'utf-8')
          parsed = parseSkillMd(raw, last)
        } catch {
          continue
        }
        if (seen.has(parsed.metadata.id)) continue
        seen.add(parsed.metadata.id)
        fetched.push({
          metadata: parsed.metadata,
          content: parsed.content,
          source: { type: 'marketplace', ref: `${mp.name}/${sourceName ?? parsed.metadata.id}` },
          publisher: mp.name,
        })
      }
    }
    return fetched
  }

  private pluginScanRoots(pluginRoot: string): string[] {
    return [
      this.safeJoin(pluginRoot, 'skills'),
      pluginRoot,
    ]
  }

  private defaultScanRoots(root: string): string[] {
    return [
      this.safeJoin(root, 'skills'),
      root,
    ]
  }

  private async readMarketplaceManifest(root: string): Promise<MarketplaceJson | null> {
    try {
      const raw = await readFile(join(root, '.claude-plugin', 'marketplace.json'), 'utf-8')
      return JSON.parse(raw) as MarketplaceJson
    } catch {
      return null
    }
  }

  private safeJoin(root: string, sub: string): string {
    const rootResolved = resolve(root)
    const resolved = resolve(rootResolved, sub)
    const rel = relative(rootResolved, resolved)
    if (rel === '..' || rel.startsWith(`..${sep}`) || rel.startsWith('../')) {
      throw new SkillPathTraversalError(sub)
    }
    return resolved
  }

  private async findSkillDirs(root: string): Promise<string[]> {
    const out: string[] = []
    const rootResolved = resolve(root)
    const walk = async (dir: string) => {
      let entries
      try {
        entries = await readdir(dir, { withFileTypes: true })
      } catch {
        return
      }
      for (const e of entries) {
        if (e.isSymbolicLink()) continue
        if (e.isDirectory()) {
          const full = join(dir, e.name)
          const rel = relative(rootResolved, resolve(full))
          if (rel === '..' || rel.startsWith(`..${sep}`) || rel.startsWith('../')) continue
          await walk(full)
        } else if (e.isFile() && e.name === 'SKILL.md') {
          out.push(dir)
        }
      }
    }
    await walk(root)
    return out
  }
}
