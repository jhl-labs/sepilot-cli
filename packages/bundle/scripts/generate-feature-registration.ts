import { readFileSync, writeFileSync } from 'node:fs'
import { parse as parseYaml } from 'yaml'
import { validateFeatureSelection } from '../../daemon/src/features/feature-catalog.js'

/**
 * Loads `build.features.yaml` (or an override path) and flattens it into the
 * set of disabled feature ids. A missing file means "all features enabled".
 * `channels:` is a nested map; each `false` entry under it becomes
 * `channels.<name>`. The nested `desktop:` map is a Desktop compile-time
 * contract consumed by Next/Electron builds and is deliberately ignored by
 * daemon binary generation. Other unknown keys are passed through as-is so
 * `validateFeatureSelection` can reject them with a proper error.
 */
export function loadBuildFeatures(path?: string): { disabled: Set<string> } {
  const filePath = path ?? process.env.SEPILOT_BUILD_FEATURES_FILE ?? 'build.features.yaml'
  let raw: string
  try {
    raw = readFileSync(filePath, 'utf8')
  } catch {
    return { disabled: new Set() }
  }
  const doc = parseYaml(raw) as { features?: Record<string, unknown> } | undefined
  const features = doc?.features ?? {}
  const disabled = new Set<string>()
  for (const [key, value] of Object.entries(features)) {
    if (key === 'desktop') continue
    if (key === 'channels') {
      if (value && typeof value === 'object') {
        for (const [channel, enabled] of Object.entries(value as Record<string, unknown>)) {
          if (enabled === false) disabled.add(`channels.${channel}`)
        }
      }
      continue
    }
    if (value === false) disabled.add(key)
  }
  return { disabled }
}

interface ToolEntry {
  featureId: string
  alias: string
  importPath: string
  section: 'tools' | 'postWebTools' | 'delegation'
}

interface ChannelEntry {
  featureId: string
  alias: string
  importPath: string
}

interface RouteEntry {
  featureId: string
  alias: string
  importPath: string
  funcName: string
}

// Order matches the committed stub exactly — see
// packages/daemon/src/generated/feature-registration.ts.
const TOOL_ENTRIES: ToolEntry[] = [
  { featureId: 'lsp', alias: 'lsp', importPath: '../features/entries/lsp.js', section: 'tools' },
  {
    featureId: 'notebook',
    alias: 'notebook',
    importPath: '../features/entries/notebook.js',
    section: 'tools',
  },
  {
    featureId: 'office',
    alias: 'office',
    importPath: '../features/entries/office.js',
    section: 'tools',
  },
  {
    featureId: 'micro-apps',
    alias: 'microApps',
    importPath: '../features/entries/micro-apps.js',
    section: 'tools',
  },
  {
    featureId: 'media',
    alias: 'media',
    importPath: '../features/entries/media.js',
    section: 'tools',
  },
  {
    featureId: 'voice',
    alias: 'voice',
    importPath: '../features/entries/voice.js',
    section: 'tools',
  },
  {
    featureId: 'computer-use',
    alias: 'computerUse',
    importPath: '../features/entries/computer-use.js',
    section: 'tools',
  },
  {
    featureId: 'browser',
    alias: 'browser',
    importPath: '../features/entries/browser.js',
    section: 'tools',
  },
  {
    featureId: 'market',
    alias: 'market',
    importPath: '../features/entries/market.js',
    section: 'postWebTools',
  },
  {
    featureId: 'pages',
    alias: 'pages',
    importPath: '../features/entries/pages.js',
    section: 'postWebTools',
  },
  {
    featureId: 'image-gen',
    alias: 'imageGen',
    importPath: '../features/entries/image-gen.js',
    section: 'postWebTools',
  },
  {
    featureId: 'delegation',
    alias: 'delegation',
    importPath: '../features/entries/delegation.js',
    section: 'delegation',
  },
]

const CHANNEL_ENTRIES: ChannelEntry[] = [
  {
    featureId: 'channels.github-issue',
    alias: 'channelGithubIssue',
    importPath: '../features/entries/channel-github-issue.js',
  },
  {
    featureId: 'channels.telegram',
    alias: 'channelTelegram',
    importPath: '../features/entries/channel-telegram.js',
  },
  {
    featureId: 'channels.slack',
    alias: 'channelSlack',
    importPath: '../features/entries/channel-slack.js',
  },
  {
    featureId: 'channels.discord',
    alias: 'channelDiscord',
    importPath: '../features/entries/channel-discord.js',
  },
  {
    featureId: 'channels.mattermost',
    alias: 'channelMattermost',
    importPath: '../features/entries/channel-mattermost.js',
  },
  {
    featureId: 'channels.webhook',
    alias: 'channelWebhook',
    importPath: '../features/entries/channel-webhook.js',
  },
  {
    featureId: 'channels.webchat',
    alias: 'channelWebchat',
    importPath: '../features/entries/channel-webchat.js',
  },
  {
    featureId: 'channels.whatsapp',
    alias: 'channelWhatsapp',
    importPath: '../features/entries/channel-whatsapp.js',
  },
  {
    featureId: 'channels.teams',
    alias: 'channelTeams',
    importPath: '../features/entries/channel-teams.js',
  },
  {
    featureId: 'channels.line',
    alias: 'channelLine',
    importPath: '../features/entries/channel-line.js',
  },
]

const ROUTE_ENTRIES: RouteEntry[] = [
  {
    featureId: 'browser',
    alias: 'routesBrowser',
    importPath: '../features/entries/routes-browser.js',
    funcName: 'registerBrowserFeatureRoutes',
  },
  {
    featureId: 'voice',
    alias: 'routesVoice',
    importPath: '../features/entries/routes-voice.js',
    funcName: 'registerVoiceFeatureRoutes',
  },
  {
    featureId: 'mcp',
    alias: 'routesMcp',
    importPath: '../features/entries/routes-mcp.js',
    funcName: 'registerMcpFeatureRoutes',
  },
  {
    featureId: 'pages',
    alias: 'routesPages',
    importPath: '../features/entries/routes-pages.js',
    funcName: 'registerPagesFeatureRoutes',
  },
  {
    featureId: 'micro-apps',
    alias: 'routesMicroApps',
    importPath: '../features/entries/routes-micro-apps.js',
    funcName: 'registerMicroAppsFeatureRoutes',
  },
  {
    featureId: 'plugins',
    alias: 'routesPlugins',
    importPath: '../features/entries/routes-plugins.js',
    funcName: 'registerPluginsFeatureRoutes',
  },
  {
    featureId: 'acp',
    alias: 'routesAcp',
    importPath: '../features/entries/routes-acp.js',
    funcName: 'registerAcpFeatureRoutes',
  },
  {
    featureId: 'a2a',
    alias: 'routesA2a',
    importPath: '../features/entries/routes-a2a.js',
    funcName: 'registerA2aFeatureRoutes',
  },
  {
    featureId: 'swarm',
    alias: 'routesSwarm',
    importPath: '../features/entries/routes-swarm.js',
    funcName: 'registerSwarmFeatureRoutes',
  },
]

// Extra entry modules imported only for the second-pass runtime/capability
// wiring below (a2a/acp runtime services, plugin loader, wiki/extensions
// capability routes). delegation + micro-apps reuse the aliases already
// imported by TOOL_ENTRIES.
interface RuntimeEntry {
  featureId: string
  alias: string
  importPath: string
}

const RUNTIME_ENTRIES: RuntimeEntry[] = [
  {
    featureId: 'a2a',
    alias: 'a2aRuntime',
    importPath: '../features/entries/a2a-runtime.js',
  },
  {
    featureId: 'acp',
    alias: 'acpRuntime',
    importPath: '../features/entries/acp-runtime.js',
  },
  {
    featureId: 'plugins',
    alias: 'plugins',
    importPath: '../features/entries/plugins.js',
  },
  {
    featureId: 'wiki',
    alias: 'wiki',
    importPath: '../features/entries/wiki.js',
  },
  {
    featureId: 'extensions',
    alias: 'extensions',
    importPath: '../features/entries/extensions.js',
  },
]

/**
 * Renders the full content of `packages/daemon/src/generated/feature-registration.ts`
 * for the given disabled-feature set. With an empty set, the output is
 * byte-identical to the committed stub (guarded by a regression test).
 */
export function renderFeatureRegistration(disabled: Set<string>): string {
  const errors = validateFeatureSelection(disabled)
  if (errors.length > 0) {
    throw new Error(errors.join('; '))
  }

  const isOn = (featureId: string) => !disabled.has(featureId)

  const lines: string[] = []
  lines.push('// AUTO-GENERATED REGION — the bundle build rewrites this file from')
  lines.push('// build.features.yaml and restores this committed stub afterwards.')
  lines.push('// The stub enables EVERY feature (default full build).')
  lines.push("import type { FastifyInstance } from 'fastify'")
  lines.push("import type { ToolRegistry } from '../tools/registry.js'")
  lines.push('import type {')
  lines.push('  FeatureToolDeps,')
  lines.push('  FeatureRouteDeps,')
  lines.push('  FeatureRuntimeToolDeps,')
  lines.push("} from '../features/types.js'")
  lines.push("import type { ChannelFactoryRegistry } from '../server/runtime/channels.js'")
  lines.push("import type { SepilotdConfig } from '../config/schema.js'")
  lines.push("import type { A2ATaskStore } from '../a2a/server.js'")
  lines.push("import type { ExternalAcpAgentDispatcher } from '../acp/external-agent.js'")
  lines.push("import type { PluginLoader } from '../plugins/loader.js'")
  lines.push(
    "import type { OpenApiComponentOverrides, OpenApiOverrideMap } from '../server/openapi.js'",
  )
  lines.push('')
  for (const entry of TOOL_ENTRIES) {
    if (isOn(entry.featureId)) lines.push(`import * as ${entry.alias} from '${entry.importPath}'`)
  }
  lines.push('')
  for (const entry of CHANNEL_ENTRIES) {
    if (isOn(entry.featureId)) lines.push(`import * as ${entry.alias} from '${entry.importPath}'`)
  }
  lines.push('')
  for (const entry of ROUTE_ENTRIES) {
    if (isOn(entry.featureId)) lines.push(`import * as ${entry.alias} from '${entry.importPath}'`)
  }
  lines.push('')
  for (const entry of RUNTIME_ENTRIES) {
    if (isOn(entry.featureId)) lines.push(`import * as ${entry.alias} from '${entry.importPath}'`)
  }
  lines.push('')

  lines.push(
    'export function registerFeatureTools(registry: ToolRegistry, deps: FeatureToolDeps): void {',
  )
  for (const entry of TOOL_ENTRIES) {
    if (entry.section === 'tools' && isOn(entry.featureId)) {
      lines.push(`  ${entry.alias}.registerTools(registry, deps)`)
    }
  }
  lines.push('}')
  lines.push('')
  lines.push('// tools.ts calls this right after createWebSearchTool/createWebFetchTool so')
  lines.push('// market/pages/image-gen keep their original insertion positions in the')
  lines.push('// registry (ToolRegistry.list() order is LLM-visible).')
  lines.push(
    'export function registerPostWebFeatureTools(registry: ToolRegistry, deps: FeatureToolDeps): void {',
  )
  for (const entry of TOOL_ENTRIES) {
    if (entry.section === 'postWebTools' && isOn(entry.featureId)) {
      lines.push(`  ${entry.alias}.registerTools(registry, deps)`)
    }
  }
  lines.push('}')
  lines.push('')

  const delegationEntry = TOOL_ENTRIES.find((entry) => entry.section === 'delegation')!
  lines.push(
    'export function registerDelegationTools(registry: ToolRegistry, deps: FeatureToolDeps): void {',
  )
  if (isOn(delegationEntry.featureId)) {
    lines.push(`  ${delegationEntry.alias}.registerTools(registry, deps)`)
  } else {
    lines.push('  // delegation feature disabled in this build')
  }
  lines.push('}')
  lines.push('')

  // Second-pass runtime registrations (called from buildRuntime after the
  // storage layer exists). Each is a no-op / null when its feature is off.
  lines.push('export function registerDelegationRuntimeTool(')
  lines.push('  registry: ToolRegistry,')
  lines.push('  deps: FeatureRuntimeToolDeps,')
  lines.push('): void {')
  if (isOn('delegation')) {
    lines.push('  delegation.registerRuntimeTool(registry, deps)')
  } else {
    lines.push('  // delegation feature disabled in this build')
  }
  lines.push('}')
  lines.push('')

  lines.push('export function registerMicroAppsRuntimeTools(')
  lines.push('  registry: ToolRegistry,')
  lines.push('  deps: FeatureRuntimeToolDeps,')
  lines.push('): void {')
  if (isOn('micro-apps')) {
    lines.push('  microApps.registerRuntimeTools(registry, deps)')
  } else {
    lines.push('  // micro-apps feature disabled in this build')
  }
  lines.push('}')
  lines.push('')

  lines.push('export function createA2aRuntime(')
  lines.push('  registry: ToolRegistry,')
  lines.push('  deps: FeatureRuntimeToolDeps,')
  lines.push('): A2ATaskStore | null {')
  if (isOn('a2a')) {
    lines.push('  return a2aRuntime.createA2aRuntime(registry, deps)')
  } else {
    lines.push('  // a2a feature disabled in this build')
    lines.push('  return null')
  }
  lines.push('}')
  lines.push('')

  lines.push('export function createAcpRuntime(')
  lines.push('  registry: ToolRegistry,')
  lines.push('  deps: FeatureRuntimeToolDeps,')
  lines.push('): ExternalAcpAgentDispatcher | null {')
  if (isOn('acp')) {
    lines.push('  return acpRuntime.createAcpRuntime(registry, deps)')
  } else {
    lines.push('  // acp feature disabled in this build')
    lines.push('  return null')
  }
  lines.push('}')
  lines.push('')

  lines.push(
    'export function createPluginLoader(dataDir: string, config?: SepilotdConfig): PluginLoader | null {',
  )
  if (isOn('plugins')) {
    lines.push('  return plugins.createPluginLoader(dataDir, config)')
  } else {
    lines.push('  // plugins feature disabled in this build')
    lines.push('  return null')
  }
  lines.push('}')
  lines.push('')

  // Plugin OpenAPI docs — routed through the manifest so server/app.ts does not
  // statically import the plugin route module (which carries the `List plugins`
  // marker). Empty when plugins are disabled.
  lines.push('export function getPluginsOpenApiComponents(): OpenApiComponentOverrides {')
  if (isOn('plugins')) {
    lines.push('  return plugins.pluginOpenApiComponents')
  } else {
    lines.push('  return {}')
  }
  lines.push('}')
  lines.push('')

  lines.push('export function getPluginsOpenApiOverrides(): OpenApiOverrideMap {')
  if (isOn('plugins')) {
    lines.push('  return plugins.pluginOpenApiOverrides')
  } else {
    lines.push('  return {}')
  }
  lines.push('}')
  lines.push('')

  lines.push(
    'export function registerFeatureChannelFactories(registry: ChannelFactoryRegistry): void {',
  )
  for (const entry of CHANNEL_ENTRIES) {
    if (isOn(entry.featureId)) lines.push(`  ${entry.alias}.registerChannelFactory(registry)`)
  }
  lines.push('}')
  lines.push('')

  lines.push("// NOTE: deviation from the brief's single `registerFeatureRoutes`. app.ts")
  lines.push('// interleaves route-feature registration with core route registration at')
  lines.push('// specific positions, so each route feature is exported individually here')
  lines.push('// to preserve exact registration order at each call site.')
  for (const entry of ROUTE_ENTRIES) {
    lines.push(`export async function ${entry.funcName}(`)
    lines.push('  app: FastifyInstance,')
    lines.push('  deps: FeatureRouteDeps,')
    lines.push('): Promise<void> {')
    if (isOn(entry.featureId)) {
      lines.push(`  await ${entry.alias}.registerRoutes(app, deps)`)
    } else {
      lines.push(`  // ${entry.featureId} route feature disabled in this build`)
    }
    lines.push('}')
    lines.push('')
  }

  // Capability routes registered directly (no prefix) from
  // server/capability-routes.ts. Gated here so wiki/extensions code drops from
  // the bundle when disabled.
  lines.push(
    'export async function registerWikiFeatureRoutes(app: FastifyInstance): Promise<void> {',
  )
  if (isOn('wiki')) {
    lines.push('  await wiki.registerRoutes(app)')
  } else {
    lines.push('  // wiki feature disabled in this build')
  }
  lines.push('}')
  lines.push('')

  lines.push(
    'export async function registerExtensionsFeatureRoutes(app: FastifyInstance): Promise<void> {',
  )
  if (isOn('extensions')) {
    lines.push('  await extensions.registerRoutes(app)')
  } else {
    lines.push('  // extensions feature disabled in this build')
  }
  lines.push('}')
  lines.push('')

  // Drop the trailing blank line after the last function; the file ends with one newline.
  if (lines[lines.length - 1] === '') lines.pop()
  lines.push('')

  return lines.join('\n')
}

/**
 * Loads build.features.yaml (or the given path), renders the generated
 * source, and writes it to `outFile`.
 */
export function generateFeatureRegistration(options: { featuresFile?: string; outFile: string }): {
  disabled: string[]
} {
  const { disabled } = loadBuildFeatures(options.featuresFile)
  const source = renderFeatureRegistration(disabled)
  writeFileSync(options.outFile, source, 'utf8')
  return { disabled: [...disabled] }
}
