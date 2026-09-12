import { createHash, verify } from 'node:crypto'
import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { GraphAgentRegistry } from '../agent/graph/registry.js'
import type { GraphAgentInfo } from '../agent/graph/registry.js'
import { createLogger } from '../logger.js'
import type { HookRegistry } from '../hook/registry.js'
import type {
  IChannelFactoryRegistry,
  IProviderFactoryRegistry,
  PluginContext,
  PluginEntry,
  PluginGraphAgent,
  PluginGraphRegistrar,
} from './contracts.js'
import type { FileSkillRegistry } from '../skills/registry.js'
import type { ToolRegistry } from '../tools/registry.js'
import { PluginEventBus } from './event-bus.js'

const log = createLogger('plugins.loader')

export interface PluginManifest {
  name: string
  version: string
  description: string
  main: string
  publisher?: string
  digest?: string
  signature?: string
  signatureKeyId?: string
  tools?: string[]
  hooks?: string[]
  channels?: string[]
  permissions?: PluginPermission[]
}

export interface PluginPermission {
  type: string
  value?: string
  reason?: string
}

export interface LoadedPlugin {
  manifest: PluginManifest
  path: string
  module?: unknown
  entry?: PluginEntry
  status: 'discovered' | 'registered' | 'failed'
  error?: string
  manifestValid?: boolean
  security?: PluginSecurityStatus
}

export interface PluginSecurityStatus {
  digest: string
  verified: boolean
  signature?: {
    keyId: string
    verified: boolean
  }
  warnings?: string[]
}

export interface PluginTrustedSignatureKey {
  id: string
  publicKey: string
}

export interface PluginLoaderOptions {
  strict?: boolean
  trustedSignatureKeys?: PluginTrustedSignatureKey[]
  loadTimeoutMs?: number
}

export interface PluginLoadContext {
  providers: IProviderFactoryRegistry
  channels: IChannelFactoryRegistry
  tools: ToolRegistry
  hooks: HookRegistry
  skills: FileSkillRegistry
  graphs: GraphAgentRegistry
  events?: import('./event-bus.js').PluginEventBus
}

type PluginModule = Partial<PluginEntry> & {
  registerAgents?: PluginGraphRegistrar
  agents?: PluginGraphAgent[]
}

interface ManifestValidationResult {
  manifest?: PluginManifest
  errors: string[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

// Duck-typed view of a registry that supports scoped plugin registration so a
// partially-registered plugin can be unwound. The tool + provider-factory
// registries implement these; other registries (channels/hooks/skills) simply
// skip scoping.
interface ScopedRegistry {
  beginPluginScope(pluginId: string): void
  endPluginScope(): string[]
  unregister(name: string): boolean
}

function asScoped(registry: unknown): ScopedRegistry | null {
  const candidate = registry as Partial<ScopedRegistry> | null
  return candidate
    && typeof candidate.beginPluginScope === 'function'
    && typeof candidate.endPluginScope === 'function'
    && typeof candidate.unregister === 'function'
    ? (candidate as ScopedRegistry)
    : null
}

function beginRegistrationScope(registry: unknown, pluginId: string): void {
  asScoped(registry)?.beginPluginScope(pluginId)
}

function endRegistrationScope(registry: unknown): void {
  asScoped(registry)?.endPluginScope()
}

function unwindRegistrationScope(registry: unknown): void {
  const scoped = asScoped(registry)
  if (!scoped) return
  for (const name of scoped.endPluginScope()) {
    scoped.unregister(name)
  }
}

function readRequiredString(
  input: Record<string, unknown>,
  key: keyof PluginManifest,
  errors: string[],
): string {
  const value = input[key]
  if (typeof value === 'string' && value.trim().length > 0) {
    return value.trim()
  }
  errors.push(`${String(key)} must be a non-empty string`)
  return ''
}

function readOptionalStringArray(
  input: Record<string, unknown>,
  key: 'tools' | 'hooks' | 'channels',
  errors: string[],
): string[] | undefined {
  const value = input[key]
  if (value === undefined) return undefined
  if (!Array.isArray(value)) {
    errors.push(`${key} must be an array of strings`)
    return undefined
  }

  const items: string[] = []
  value.forEach((entry, index) => {
    if (typeof entry === 'string' && entry.trim().length > 0) {
      items.push(entry.trim())
      return
    }
    errors.push(`${key}[${index}] must be a non-empty string`)
  })
  return items
}

function readOptionalString(
  input: Record<string, unknown>,
  key: 'publisher' | 'digest' | 'signature' | 'signatureKeyId',
  errors: string[],
): string | undefined {
  const value = input[key]
  if (value === undefined) return undefined
  if (typeof value === 'string' && value.trim().length > 0) {
    return value.trim()
  }
  errors.push(`${key} must be a non-empty string`)
  return undefined
}

function readOptionalPermissions(
  input: Record<string, unknown>,
  errors: string[],
): PluginPermission[] | undefined {
  const value = input.permissions
  if (value === undefined) return undefined
  if (!Array.isArray(value)) {
    errors.push('permissions must be an array')
    return undefined
  }

  const permissions: PluginPermission[] = []
  value.forEach((entry, index) => {
    if (!isRecord(entry)) {
      errors.push(`permissions[${index}] must be an object`)
      return
    }

    const type = entry.type
    if (typeof type !== 'string' || type.trim().length === 0) {
      errors.push(`permissions[${index}].type must be a non-empty string`)
      return
    }

    const permission: PluginPermission = { type: type.trim() }
    for (const optionalKey of ['value', 'reason'] as const) {
      const optionalValue = entry[optionalKey]
      if (optionalValue === undefined) continue
      if (typeof optionalValue !== 'string' || optionalValue.trim().length === 0) {
        errors.push(`permissions[${index}].${optionalKey} must be a non-empty string`)
        continue
      }
      permission[optionalKey] = optionalValue.trim()
    }
    permissions.push(permission)
  })
  return permissions
}

function isSafeRelativeMainPath(value: string): boolean {
  const normalized = value.replace(/\\/g, '/')
  return (
    !normalized.startsWith('/') &&
    !/^[A-Za-z]:\//.test(normalized) &&
    !normalized.includes('\0') &&
    !normalized.endsWith('/') &&
    normalized !== '.' &&
    !normalized.endsWith('/.') &&
    !normalized.split('/').includes('..')
  )
}

function isMissingFileError(error: unknown): boolean {
  return isRecord(error) && error.code === 'ENOENT'
}

function envFlag(name: string): boolean {
  const value = process.env[name]?.trim().toLowerCase()
  return value === '1' || value === 'true' || value === 'yes'
}

function envPositiveInteger(name: string): number | undefined {
  const value = process.env[name]?.trim()
  if (!value) return undefined
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined
}

function decodeSignature(signature: string): Buffer {
  const value = signature.startsWith('base64:') ? signature.slice('base64:'.length) : signature
  return Buffer.from(value, 'base64')
}

export function validatePluginManifest(input: unknown): ManifestValidationResult {
  const errors: string[] = []
  if (!isRecord(input)) {
    return { errors: ['manifest must be an object'] }
  }

  const name = readRequiredString(input, 'name', errors)
  const version = readRequiredString(input, 'version', errors)
  const description = readRequiredString(input, 'description', errors)
  const main = readRequiredString(input, 'main', errors)
  if (name.includes('/') || name.includes('\\') || name.includes('\0')) {
    errors.push('name must not contain path separators')
  }
  if (main && !isSafeRelativeMainPath(main)) {
    errors.push('main must be a relative file path inside the plugin directory')
  }

  const tools = readOptionalStringArray(input, 'tools', errors)
  const hooks = readOptionalStringArray(input, 'hooks', errors)
  const channels = readOptionalStringArray(input, 'channels', errors)
  const permissions = readOptionalPermissions(input, errors)
  const publisher = readOptionalString(input, 'publisher', errors)
  const digest = readOptionalString(input, 'digest', errors)
  const signature = readOptionalString(input, 'signature', errors)
  const signatureKeyId = readOptionalString(input, 'signatureKeyId', errors)

  if (errors.length > 0) {
    return { errors }
  }

  return {
    errors: [],
    manifest: {
      name,
      version,
      description,
      main,
      ...(publisher ? { publisher } : {}),
      ...(digest ? { digest } : {}),
      ...(signature ? { signature } : {}),
      ...(signatureKeyId ? { signatureKeyId } : {}),
      ...(tools ? { tools } : {}),
      ...(hooks ? { hooks } : {}),
      ...(channels ? { channels } : {}),
      ...(permissions ? { permissions } : {}),
    },
  }
}

class PluginScopedGraphRegistry extends GraphAgentRegistry {
  constructor(private readonly base: GraphAgentRegistry) {
    super()
  }

  override register(info: GraphAgentInfo): void {
    this.base.register({
      ...info,
      source: info.source ?? 'plugin',
    })
  }

  override unregister(id: string): void {
    this.base.unregister(id)
  }

  override get(id: string): GraphAgentInfo | undefined {
    return this.base.get(id)
  }

  override list(): GraphAgentInfo[] {
    return this.base.list()
  }
}

export class PluginLoader {
  private pluginsDir: string
  private plugins = new Map<string, LoadedPlugin>()
  private readonly strict: boolean
  private readonly trustedSignatureKeys = new Map<string, string>()
  private readonly loadTimeoutMs: number

  constructor(pluginsDir: string, options: PluginLoaderOptions = {}) {
    this.pluginsDir = pluginsDir
    this.strict = Boolean(options.strict) || envFlag('SEPILOTD_PLUGINS_STRICT')
    for (const key of options.trustedSignatureKeys ?? []) {
      this.trustedSignatureKeys.set(key.id, key.publicKey)
    }
    this.loadTimeoutMs =
      options.loadTimeoutMs ?? envPositiveInteger('SEPILOTD_PLUGIN_LOAD_TIMEOUT_MS') ?? 10_000
  }

  async discover(): Promise<PluginManifest[]> {
    const manifests: PluginManifest[] = []
    try {
      const entries = await readdir(this.pluginsDir, { withFileTypes: true })
      for (const entry of entries) {
        if (!entry.isDirectory()) continue
        const pluginDir = join(this.pluginsDir, entry.name)
        const manifestPath = join(pluginDir, 'plugin.json')
        let content: string
        try {
          content = await readFile(manifestPath, 'utf-8')
        } catch (error) {
          if (isMissingFileError(error)) {
            continue
          }
          this.recordInvalidManifest(
            entry.name,
            pluginDir,
            `Unable to read plugin.json: ${String(error)}`,
          )
          continue
        }

        let rawManifest: unknown
        try {
          rawManifest = JSON.parse(content)
        } catch (error) {
          this.recordInvalidManifest(
            entry.name,
            pluginDir,
            `Invalid plugin manifest: ${String(error)}`,
          )
          continue
        }

        const result = validatePluginManifest(rawManifest)
        if (result.manifest) {
          const manifest = result.manifest
          manifests.push(manifest)
          this.plugins.set(manifest.name, {
            manifest,
            path: pluginDir,
            status: 'discovered',
            manifestValid: true,
          })
          continue
        }

        this.recordInvalidManifest(
          entry.name,
          pluginDir,
          `Invalid plugin manifest: ${result.errors.join('; ')}`,
        )
      }
    } catch {
      // Plugins dir may not exist.
    }
    return manifests
  }

  private recordInvalidManifest(directoryName: string, pluginDir: string, error: string): void {
    this.plugins.set(`invalid:${directoryName}`, {
      manifest: {
        name: directoryName,
        version: 'unknown',
        description: 'Invalid plugin manifest',
        main: '',
      },
      path: pluginDir,
      status: 'failed',
      error,
      manifestValid: false,
    })
  }

  private createPluginContext(plugin: LoadedPlugin, registries: PluginLoadContext): PluginContext {
    const graphs = new PluginScopedGraphRegistry(registries.graphs)
    return {
      pluginId: plugin.manifest.name,
      pluginDir: plugin.path,
      providers: registries.providers,
      channels: registries.channels,
      tools: registries.tools,
      hooks: registries.hooks,
      skills: registries.skills,
      graphs,
      events: registries.events ?? new PluginEventBus(),
      log: createLogger(`plugin.${plugin.manifest.name}`),
      config: {},
    }
  }

  private resolveGraphRegistrar(
    module: PluginModule,
    candidate: PluginModule,
  ): PluginGraphRegistrar | null {
    const registerAgents = candidate.registerAgents ?? module.registerAgents
    return typeof registerAgents === 'function' ? registerAgents.bind(candidate) : null
  }

  private resolveGraphAgents(module: PluginModule, candidate: PluginModule): PluginGraphAgent[] {
    const agents = candidate.agents ?? module.agents
    return Array.isArray(agents) ? agents : []
  }

  private async registerModuleGraphs(
    module: PluginModule,
    candidate: PluginModule,
    context: PluginContext,
  ): Promise<void> {
    const registerAgents = this.resolveGraphRegistrar(module, candidate)
    if (registerAgents) {
      await registerAgents(context.graphs, context)
    }

    for (const agent of this.resolveGraphAgents(module, candidate)) {
      context.graphs.register(agent)
    }
  }

  private async verifyPluginEntry(
    plugin: LoadedPlugin,
    entryPath: string,
  ): Promise<PluginSecurityStatus> {
    const content = await readFile(entryPath)
    const digest = `sha256:${createHash('sha256').update(content).digest('hex')}`
    const warnings: string[] = []
    let signatureStatus: PluginSecurityStatus['signature']
    if (plugin.manifest.digest && plugin.manifest.digest !== digest) {
      throw new Error(
        `plugin entry digest mismatch: expected ${plugin.manifest.digest}, got ${digest}`,
      )
    }
    if (!plugin.manifest.digest) {
      warnings.push('plugin entry digest is not pinned in plugin.json')
    }
    if (plugin.manifest.signature && !plugin.manifest.digest) {
      warnings.push('plugin signature is present but no digest is pinned')
    }
    if (plugin.manifest.signature) {
      const keyId = plugin.manifest.signatureKeyId
      if (!keyId) {
        throw new Error('plugin signatureKeyId is required when signature is present')
      }
      const publicKey = this.trustedSignatureKeys.get(keyId)
      if (!publicKey) {
        throw new Error(`plugin signature key is not trusted: ${keyId}`)
      }
      const ok = verify(
        null,
        Buffer.from(digest),
        publicKey,
        decodeSignature(plugin.manifest.signature),
      )
      if (!ok) {
        throw new Error(`plugin signature verification failed for key ${keyId}`)
      }
      signatureStatus = { keyId, verified: true }
    } else if (this.strict) {
      throw new Error('plugin signature is required in strict mode')
    } else {
      warnings.push('plugin signature is not pinned in plugin.json')
    }

    return {
      digest,
      verified: plugin.manifest.digest === digest || signatureStatus?.verified === true,
      ...(signatureStatus ? { signature: signatureStatus } : {}),
      ...(warnings.length ? { warnings } : {}),
    }
  }

  private async withTimeout<T>(work: Promise<T>, label: string): Promise<T> {
    if (this.loadTimeoutMs <= 0) return work
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      return await Promise.race([
        work,
        new Promise<T>((_resolve, reject) => {
          timer = setTimeout(() => {
            reject(new Error(`${label} timed out after ${this.loadTimeoutMs}ms`))
          }, this.loadTimeoutMs)
          timer.unref?.()
        }),
      ])
    } finally {
      if (timer) clearTimeout(timer)
    }
  }

  async load(name: string, registries: PluginLoadContext): Promise<LoadedPlugin | null> {
    const plugin = this.plugins.get(name)
    if (!plugin) return null
    if (plugin.manifestValid === false) return null

    try {
      const entryPath = join(plugin.path, plugin.manifest.main)
      plugin.security = await this.verifyPluginEntry(plugin, entryPath)
      const module = await this.withTimeout(
        import(pathToFileURL(entryPath).href),
        `plugin ${name} import`,
      )
      const candidate = (module.default ?? module) as PluginModule
      const graphRegistrar = this.resolveGraphRegistrar(module as PluginModule, candidate)
      const graphAgents = this.resolveGraphAgents(module as PluginModule, candidate)
      if (typeof candidate.register !== 'function' && !graphRegistrar && graphAgents.length === 0) {
        throw new Error(
          'plugin entry must export register(ctx), registerAgents(registry, ctx), or agents[]',
        )
      }

      const entry: PluginEntry = {
        id: typeof candidate.id === 'string' ? candidate.id : plugin.manifest.name,
        name: typeof candidate.name === 'string' ? candidate.name : plugin.manifest.name,
        version:
          typeof candidate.version === 'string' ? candidate.version : plugin.manifest.version,
        register:
          typeof candidate.register === 'function'
            ? candidate.register.bind(candidate)
            : async () => {},
        shutdown:
          typeof candidate.shutdown === 'function' ? candidate.shutdown.bind(candidate) : undefined,
      }

      const context = this.createPluginContext(plugin, registries)
      // Scope this plugin's tool/provider registrations so a failure part-way
      // through registration can be fully unwound (no orphaned tools/factories),
      // and so the registries reject any attempt to shadow a built-in.
      beginRegistrationScope(registries.tools, name)
      beginRegistrationScope(registries.providers, name)
      try {
        await this.withTimeout(Promise.resolve(entry.register(context)), `plugin ${name} register`)
        await this.withTimeout(
          this.registerModuleGraphs(module as PluginModule, candidate, context),
          `plugin ${name} graph registration`,
        )
      } catch (registerError) {
        unwindRegistrationScope(registries.tools)
        unwindRegistrationScope(registries.providers)
        throw registerError
      }
      // Success: close the scope without unwinding.
      endRegistrationScope(registries.tools)
      endRegistrationScope(registries.providers)
      plugin.module = module
      plugin.entry = entry
      plugin.status = 'registered'
      plugin.error = undefined
      return plugin
    } catch (err) {
      plugin.status = 'failed'
      plugin.error = String(err)
      log.error(`Failed to load plugin ${name}`, {
        error: String(err),
      })
      return null
    }
  }

  async loadAll(registries: PluginLoadContext): Promise<LoadedPlugin[]> {
    await this.discover()
    const loaded: LoadedPlugin[] = []
    for (const plugin of this.plugins.values()) {
      if (plugin.manifestValid === false) continue
      const registered = await this.load(plugin.manifest.name, registries)
      if (registered) loaded.push(registered)
    }
    return loaded
  }

  async shutdownAll(): Promise<void> {
    for (const plugin of this.plugins.values()) {
      try {
        await plugin.entry?.shutdown?.()
      } catch (error) {
        log.warn(`Plugin shutdown failed for ${plugin.manifest.name}`, {
          error: String(error),
        })
      }
    }
  }

  list(): LoadedPlugin[] {
    return Array.from(this.plugins.values())
  }

  get(name: string): LoadedPlugin | undefined {
    return this.plugins.get(name)
  }
}
