import type { FastifyInstance } from 'fastify'
import type { ToolRegistry } from '../tools/registry.js'
import type {
  FeatureToolDeps,
  FeatureRouteDeps,
  FeatureRuntimeToolDeps,
  FeatureEventRoutingHooks,
  FeatureEventRoutingRuntime,
  FeatureEventRoutingStopRuntime,
} from '../features/types.js'
import type { ChannelFactoryRegistry } from '../server/runtime/channels.js'
import type { SepilotdConfig } from '../config/schema.js'
import type { A2ATaskStore } from '../a2a/server.js'
import type { ExternalAgentRuntime } from '../acp/external-agent.js'
import type { PluginLoader } from '../plugins/loader.js'
import type { OpenApiComponentOverrides, OpenApiOverrideMap } from '../server/openapi.js'

import * as lsp from '../features/entries/lsp.js'
import * as notebook from '../features/entries/notebook.js'
import * as office from '../features/entries/office.js'
import * as microApps from '../features/entries/micro-apps.js'
import * as media from '../features/entries/media.js'
import * as voice from '../features/entries/voice.js'
import * as computerUse from '../features/entries/computer-use.js'
import * as browser from '../features/entries/browser.js'
import * as market from '../features/entries/market.js'
import * as pages from '../features/entries/pages.js'
import * as imageGen from '../features/entries/image-gen.js'
import * as delegation from '../features/entries/delegation.js'

import * as channelGithubIssue from '../features/entries/channel-github-issue.js'
import * as channelTelegram from '../features/entries/channel-telegram.js'
import * as channelSlack from '../features/entries/channel-slack.js'
import * as channelDiscord from '../features/entries/channel-discord.js'
import * as channelMattermost from '../features/entries/channel-mattermost.js'
import * as channelWebhook from '../features/entries/channel-webhook.js'
import * as channelWebchat from '../features/entries/channel-webchat.js'
import * as channelWhatsapp from '../features/entries/channel-whatsapp.js'
import * as channelTeams from '../features/entries/channel-teams.js'
import * as channelLine from '../features/entries/channel-line.js'

import * as routesBrowser from '../features/entries/routes-browser.js'
import * as routesVoice from '../features/entries/routes-voice.js'
import * as routesMcp from '../features/entries/routes-mcp.js'
import * as routesPages from '../features/entries/routes-pages.js'
import * as routesMicroApps from '../features/entries/routes-micro-apps.js'
import * as routesPlugins from '../features/entries/routes-plugins.js'
import * as routesAcp from '../features/entries/routes-acp.js'
import * as routesA2a from '../features/entries/routes-a2a.js'
import * as routesSwarm from '../features/entries/routes-swarm.js'
import * as routesEventRouting from '../features/entries/routes-event-routing.js'

import * as a2aRuntime from '../features/entries/a2a-runtime.js'
import * as acpRuntime from '../features/entries/acp-runtime.js'
import * as plugins from '../features/entries/plugins.js'
import * as wiki from '../features/entries/wiki.js'
import * as extensions from '../features/entries/extensions.js'
import * as eventRoutingRuntime from '../features/entries/event-routing-runtime.js'

export function registerFeatureTools(registry: ToolRegistry, deps: FeatureToolDeps): void {
  lsp.registerTools(registry, deps)
  notebook.registerTools(registry, deps)
  office.registerTools(registry, deps)
  microApps.registerTools(registry, deps)
  media.registerTools(registry, deps)
  voice.registerTools(registry, deps)
  computerUse.registerTools(registry, deps)
  browser.registerTools(registry, deps)
}

export function registerPostWebFeatureTools(registry: ToolRegistry, deps: FeatureToolDeps): void {
  market.registerTools(registry, deps)
  pages.registerTools(registry, deps)
  imageGen.registerTools(registry, deps)
}

export function registerDelegationTools(registry: ToolRegistry, deps: FeatureToolDeps): void {
  delegation.registerTools(registry, deps)
}

export function registerDelegationRuntimeTool(
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
): void {
  delegation.registerRuntimeTool(registry, deps)
}

export function registerMicroAppsRuntimeTools(
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
): void {
  microApps.registerRuntimeTools(registry, deps)
}

export function createA2aRuntime(
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
): A2ATaskStore | null {
  return a2aRuntime.createA2aRuntime(registry, deps)
}

export function createAcpRuntime(
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
): ExternalAgentRuntime | null {
  return acpRuntime.createAcpRuntime(registry, deps)
}

export function createPluginLoader(dataDir: string, config?: SepilotdConfig): PluginLoader | null {
  return plugins.createPluginLoader(dataDir, config)
}

export function getPluginsOpenApiComponents(): OpenApiComponentOverrides {
  return plugins.pluginOpenApiComponents
}

export function getPluginsOpenApiOverrides(): OpenApiOverrideMap {
  return plugins.pluginOpenApiOverrides
}

export function registerFeatureChannelFactories(registry: ChannelFactoryRegistry): void {
  channelGithubIssue.registerChannelFactory(registry)
  channelTelegram.registerChannelFactory(registry)
  channelSlack.registerChannelFactory(registry)
  channelDiscord.registerChannelFactory(registry)
  channelMattermost.registerChannelFactory(registry)
  channelWebhook.registerChannelFactory(registry)
  channelWebchat.registerChannelFactory(registry)
  channelWhatsapp.registerChannelFactory(registry)
  channelTeams.registerChannelFactory(registry)
  channelLine.registerChannelFactory(registry)
}

export async function registerBrowserFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesBrowser.registerRoutes(app, deps)
}

export async function registerVoiceFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesVoice.registerRoutes(app, deps)
}

export async function registerMcpFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesMcp.registerRoutes(app, deps)
}

export async function registerPagesFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesPages.registerRoutes(app, deps)
}

export async function registerMicroAppsFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesMicroApps.registerRoutes(app, deps)
}

export async function registerPluginsFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesPlugins.registerRoutes(app, deps)
}

export async function registerAcpFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesAcp.registerRoutes(app, deps)
}

export async function registerA2aFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesA2a.registerRoutes(app, deps)
}

export async function registerSwarmFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesSwarm.registerRoutes(app, deps)
}

export async function registerEventRoutingFeatureRoutes(
  app: FastifyInstance,
  deps: FeatureRouteDeps,
): Promise<void> {
  await routesEventRouting.registerRoutes(app, deps)
}

export async function registerWikiFeatureRoutes(app: FastifyInstance): Promise<void> {
  await wiki.registerRoutes(app)
}

export async function registerExtensionsFeatureRoutes(app: FastifyInstance): Promise<void> {
  await extensions.registerRoutes(app)
}

export function startEventRoutingFeature(runtime: FeatureEventRoutingRuntime): FeatureEventRoutingHooks {
  return eventRoutingRuntime.startEventRouting(runtime)
}

export async function stopEventRoutingFeature(runtime: FeatureEventRoutingStopRuntime): Promise<void> {
  await eventRoutingRuntime.stopEventRouting(runtime)
}

export function installEventRoutingLlmRunner(app: FastifyInstance): void {
  eventRoutingRuntime.installLlmRunner(app)
}
