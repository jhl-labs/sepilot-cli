import type { ChannelSummaryCapabilities } from './capabilities.js'
import '../fastify-types.js'
import type { DiscordChannel } from '../../channels/discord.js'
import type { LINEChannel } from '../../channels/line.js'
import type { MattermostChannel } from '../../channels/mattermost.js'
import type { SlackChannel } from '../../channels/slack.js'
import type { TeamsChannel } from '../../channels/teams.js'
import {
  webhookPublicRouteFromPath,
  type WebhookChannel,
} from '../../channels/webhook.js'
import type { WhatsAppChannel } from '../../channels/whatsapp.js'
import {
  resolveWebhookSecurityPolicyWithSource,
  type ResolvedWebhookSecurityPolicy,
} from './webhook-security-policy.js'

export interface WebhookEndpointSummaryData {
  channelType: string
  route: string
  methods: string[]
  verification: string[]
  verificationReady: boolean
  verificationRequirements: string[]
  verificationMissing: string[]
  securityPolicy: {
    signatureMaxSkewSeconds: number
    verificationUnavailableStatus: 'service_unavailable' | 'not_found'
    source: ResolvedWebhookSecurityPolicy['source']
  }
  status: string
  sourcePath?: string
}

export interface WebhookSecurityChannelSummaryData {
  channelType: string
  endpointCount: number
  verificationReadyCount: number
  verificationNotReadyCount: number
  verificationRequirements: string[]
  verificationMissing: string[]
  methods: string[]
  verification: string[]
  effectivePolicy: WebhookEndpointSummaryData['securityPolicy']
  routes: string[]
}

export interface WebhookSecuritySummaryData {
  totalEndpoints: number
  verificationReadyEndpoints: number
  verificationNotReadyEndpoints: number
  byChannelType: WebhookSecurityChannelSummaryData[]
  unreadySummary: {
    totalEndpoints: number
    byMissingRequirement: Array<{
      requirement: string
      endpointCount: number
      channelTypes: string[]
      routes: string[]
    }>
  }
  unreadyEndpoints: WebhookEndpointSummaryData[]
}

export interface WebhookSecurityFilter {
  channelType?: string
  verificationReady?: boolean
  missingRequirement?: string
}

export interface WebhookSecuritySummaryOptions {
  unreadyOffset?: number
  unreadyLimit?: number
}

function readChannelConfig(channel: unknown): Record<string, unknown> {
  if (!channel || typeof channel !== 'object') {
    return {}
  }
  const config = (channel as { config?: unknown }).config
  return config && typeof config === 'object'
    ? (config as Record<string, unknown>)
    : {}
}

function nonEmptyString(value: unknown): boolean {
  return typeof value === 'string' && value.trim().length > 0
}

function slackVerificationMissing(channel: SlackChannel): string[] {
  return channel.canVerifySignature() ? [] : ['signingSecret']
}

function discordVerificationMissing(channel: DiscordChannel): string[] {
  return channel.canVerifyInteractionSignature() ? [] : ['publicKey']
}

function mattermostVerificationMissing(channel: MattermostChannel): string[] {
  return channel.canVerifyWebhookToken() ? [] : ['webhookToken']
}

function whatsAppVerificationMissing(channel: WhatsAppChannel): string[] {
  const config = readChannelConfig(channel)
  const missing: string[] = []
  if (!channel.canVerifyWebhookChallenge() && !nonEmptyString(config.verifyToken)) {
    missing.push('verifyToken')
  }
  if (!channel.canVerifySignature() && !nonEmptyString(config.appSecret)) {
    missing.push('appSecret')
  }
  return missing
}

function teamsVerificationMissing(channel: TeamsChannel): string[] {
  const config = readChannelConfig(channel)
  const missing: string[] = []
  if (!nonEmptyString(config.appId)) {
    missing.push('appId')
  }
  if (!nonEmptyString(config.appPassword)) {
    missing.push('appPassword')
  }
  return missing
}

function lineVerificationMissing(channel: LINEChannel): string[] {
  return channel.canVerifySignature() ? [] : ['channelSecret']
}

function genericWebhookVerificationMissing(
  endpoint: { secretHeader?: string; secretValue?: string },
): string[] {
  const missing: string[] = []
  if (!nonEmptyString(endpoint.secretHeader)) {
    missing.push('secretHeader')
  }
  if (!nonEmptyString(endpoint.secretValue)) {
    missing.push('secretValue')
  }
  return missing
}

export function listWebhookEndpoints(
  runtime: ChannelSummaryCapabilities,
): WebhookEndpointSummaryData[] {
  const endpoints: WebhookEndpointSummaryData[] = []

  for (const channel of runtime.channels ?? []) {
    switch (channel.type) {
      case 'slack': {
        const slackChannel = channel as SlackChannel
        const securityPolicy = resolveWebhookSecurityPolicyWithSource(
          runtime.config,
          'slack',
        )
        endpoints.push({
          channelType: 'slack',
          route: '/api/v1/webhooks/slack',
          methods: ['POST'],
          verification: ['slack-signature'],
          verificationReady: slackChannel.canVerifySignature(),
          verificationRequirements: ['signingSecret'],
          verificationMissing: slackVerificationMissing(slackChannel),
          securityPolicy: {
            signatureMaxSkewSeconds: securityPolicy.signatureMaxSkewSeconds,
            verificationUnavailableStatus:
              securityPolicy.verificationUnavailableStatus,
            source: securityPolicy.source,
          },
          status: slackChannel.getStatus(),
        })
        break
      }
      case 'discord': {
        const discordChannel = channel as DiscordChannel
        const securityPolicy = resolveWebhookSecurityPolicyWithSource(
          runtime.config,
          'discord',
        )
        endpoints.push({
          channelType: 'discord',
          route: '/api/v1/webhooks/discord',
          methods: ['POST'],
          verification: ['discord-ed25519'],
          verificationReady: discordChannel.canVerifyInteractionSignature(),
          verificationRequirements: ['publicKey'],
          verificationMissing: discordVerificationMissing(discordChannel),
          securityPolicy: {
            signatureMaxSkewSeconds: securityPolicy.signatureMaxSkewSeconds,
            verificationUnavailableStatus:
              securityPolicy.verificationUnavailableStatus,
            source: securityPolicy.source,
          },
          status: discordChannel.getStatus(),
        })
        break
      }
      case 'mattermost': {
        const mattermostChannel = channel as MattermostChannel
        const securityPolicy = resolveWebhookSecurityPolicyWithSource(
          runtime.config,
          'mattermost',
        )
        endpoints.push({
          channelType: 'mattermost',
          route: '/api/v1/webhooks/mattermost',
          methods: ['POST'],
          verification: ['mattermost-token'],
          verificationReady: mattermostChannel.canVerifyWebhookToken(),
          verificationRequirements: ['webhookToken'],
          verificationMissing: mattermostVerificationMissing(mattermostChannel),
          securityPolicy: {
            signatureMaxSkewSeconds: securityPolicy.signatureMaxSkewSeconds,
            verificationUnavailableStatus:
              securityPolicy.verificationUnavailableStatus,
            source: securityPolicy.source,
          },
          status: mattermostChannel.getStatus(),
        })
        break
      }
      case 'whatsapp': {
        const whatsAppChannel = channel as WhatsAppChannel
        const securityPolicy = resolveWebhookSecurityPolicyWithSource(
          runtime.config,
          'whatsapp',
        )
        endpoints.push({
          channelType: 'whatsapp',
          route: '/api/v1/webhooks/whatsapp',
          methods: ['GET', 'POST'],
          verification: ['verify-token', 'meta-signature'],
          verificationReady: whatsAppChannel.isWebhookVerificationReady(),
          verificationRequirements: ['verifyToken', 'appSecret'],
          verificationMissing: whatsAppVerificationMissing(whatsAppChannel),
          securityPolicy: {
            signatureMaxSkewSeconds: securityPolicy.signatureMaxSkewSeconds,
            verificationUnavailableStatus:
              securityPolicy.verificationUnavailableStatus,
            source: securityPolicy.source,
          },
          status: whatsAppChannel.getStatus(),
        })
        break
      }
      case 'teams': {
        const teamsChannel = channel as TeamsChannel
        const securityPolicy = resolveWebhookSecurityPolicyWithSource(
          runtime.config,
          'teams',
        )
        endpoints.push({
          channelType: 'teams',
          route: '/api/v1/webhooks/teams',
          methods: ['POST'],
          verification: ['bot-framework-jwt'],
          verificationReady: teamsChannel.canValidateRequest(),
          verificationRequirements: ['appId', 'appPassword'],
          verificationMissing: teamsVerificationMissing(teamsChannel),
          securityPolicy: {
            signatureMaxSkewSeconds: securityPolicy.signatureMaxSkewSeconds,
            verificationUnavailableStatus:
              securityPolicy.verificationUnavailableStatus,
            source: securityPolicy.source,
          },
          status: teamsChannel.getStatus(),
        })
        break
      }
      case 'line': {
        const lineChannel = channel as LINEChannel
        const securityPolicy = resolveWebhookSecurityPolicyWithSource(
          runtime.config,
          'line',
        )
        endpoints.push({
          channelType: 'line',
          route: '/api/v1/webhooks/line',
          methods: ['POST'],
          verification: ['line-signature'],
          verificationReady: lineChannel.canVerifySignature(),
          verificationRequirements: ['channelSecret'],
          verificationMissing: lineVerificationMissing(lineChannel),
          securityPolicy: {
            signatureMaxSkewSeconds: securityPolicy.signatureMaxSkewSeconds,
            verificationUnavailableStatus:
              securityPolicy.verificationUnavailableStatus,
            source: securityPolicy.source,
          },
          status: lineChannel.getStatus(),
        })
        break
      }
      case 'webhook': {
        const webhookChannel = channel as WebhookChannel
        for (const endpoint of webhookChannel.listConfiguredEndpoints()) {
          const securityPolicy = resolveWebhookSecurityPolicyWithSource(
            runtime.config,
            'webhook',
          )
          const sourceEndpoint = (readChannelConfig(webhookChannel).endpoints as
            | Array<Record<string, unknown>>
            | undefined)
            ?.find(
              (entry) =>
                typeof entry.path === 'string'
                && webhookPublicRouteFromPath(entry.path) === webhookPublicRouteFromPath(endpoint.path),
            )
          const verificationMissing = genericWebhookVerificationMissing({
            secretHeader: sourceEndpoint?.secretHeader as string | undefined,
            secretValue: sourceEndpoint?.secretValue as string | undefined,
          })
          endpoints.push({
            channelType: 'webhook',
            route: webhookPublicRouteFromPath(endpoint.path),
            methods: ['POST'],
            verification: ['header-hmac'],
            verificationReady: endpoint.verificationReady,
            verificationRequirements: ['secretHeader', 'secretValue'],
            verificationMissing,
            securityPolicy: {
              signatureMaxSkewSeconds: securityPolicy.signatureMaxSkewSeconds,
              verificationUnavailableStatus:
                securityPolicy.verificationUnavailableStatus,
              source: securityPolicy.source,
            },
            status: webhookChannel.getStatus(),
            sourcePath: endpoint.path,
          })
        }
        break
      }
    }
  }

  return endpoints.sort((left, right) => left.route.localeCompare(right.route))
}


export function filterWebhookEndpoints(
  endpoints: WebhookEndpointSummaryData[],
  filter?: WebhookSecurityFilter,
): WebhookEndpointSummaryData[] {
  return endpoints.filter((endpoint) => {
    if (filter?.channelType && endpoint.channelType !== filter.channelType) {
      return false
    }
    if (
      filter?.verificationReady !== undefined
      && endpoint.verificationReady !== filter.verificationReady
    ) {
      return false
    }
    if (
      filter?.missingRequirement
      && !endpoint.verificationMissing.includes(filter.missingRequirement)
    ) {
      return false
    }
    return true
  })
}

export function summarizeWebhookEndpoints(
  endpoints: WebhookEndpointSummaryData[],
  options?: WebhookSecuritySummaryOptions,
): WebhookSecuritySummaryData {
  const byChannelType = new Map<string, WebhookSecurityChannelSummaryData>()
  const byMissingRequirement = new Map<string, {
    requirement: string
    endpointCount: number
    channelTypes: Set<string>
    routes: Set<string>
  }>()

  for (const endpoint of endpoints) {
    const existing = byChannelType.get(endpoint.channelType)
    if (existing) {
      existing.endpointCount += 1
      existing.verificationReadyCount += endpoint.verificationReady ? 1 : 0
      existing.verificationNotReadyCount += endpoint.verificationReady ? 0 : 1
      existing.routes.push(endpoint.route)
      for (const method of endpoint.methods) {
        if (!existing.methods.includes(method)) {
          existing.methods.push(method)
        }
      }
      for (const verification of endpoint.verification) {
        if (!existing.verification.includes(verification)) {
          existing.verification.push(verification)
        }
      }
      for (const requirement of endpoint.verificationRequirements) {
        if (!existing.verificationRequirements.includes(requirement)) {
          existing.verificationRequirements.push(requirement)
        }
      }
      for (const requirement of endpoint.verificationMissing) {
        if (!existing.verificationMissing.includes(requirement)) {
          existing.verificationMissing.push(requirement)
        }
      }
      continue
    }

    byChannelType.set(endpoint.channelType, {
      channelType: endpoint.channelType,
      endpointCount: 1,
      verificationReadyCount: endpoint.verificationReady ? 1 : 0,
      verificationNotReadyCount: endpoint.verificationReady ? 0 : 1,
      verificationRequirements: [...endpoint.verificationRequirements],
      verificationMissing: [...endpoint.verificationMissing],
      methods: [...endpoint.methods],
      verification: [...endpoint.verification],
      effectivePolicy: endpoint.securityPolicy,
      routes: [endpoint.route],
    })
  }

  return {
    totalEndpoints: endpoints.length,
    verificationReadyEndpoints: endpoints.filter(
      (endpoint) => endpoint.verificationReady,
    ).length,
    verificationNotReadyEndpoints: endpoints.filter(
      (endpoint) => !endpoint.verificationReady,
    ).length,
    byChannelType: Array.from(byChannelType.values())
      .map((entry) => ({
        ...entry,
        methods: entry.methods.sort(),
        verification: entry.verification.sort(),
        verificationRequirements: entry.verificationRequirements.sort(),
        verificationMissing: entry.verificationMissing.sort(),
        routes: entry.routes.sort(),
      }))
      .sort((left, right) => left.channelType.localeCompare(right.channelType)),
    unreadySummary: buildUnreadySummary(
      endpoints.filter((endpoint) => !endpoint.verificationReady),
      byMissingRequirement,
    ),
    unreadyEndpoints: paginateUnreadyEndpoints(
      endpoints.filter((endpoint) => !endpoint.verificationReady),
      options,
    ),
  }
}

export function summarizeWebhookSecurity(
  runtime: ChannelSummaryCapabilities,
  filter?: WebhookSecurityFilter,
  options?: WebhookSecuritySummaryOptions,
): WebhookSecuritySummaryData {
  return summarizeWebhookEndpoints(
    filterWebhookEndpoints(listWebhookEndpoints(runtime), filter),
    options,
  )
}

function paginateUnreadyEndpoints(
  endpoints: WebhookEndpointSummaryData[],
  options?: WebhookSecuritySummaryOptions,
): WebhookEndpointSummaryData[] {
  const offset = options?.unreadyOffset ?? 0
  const limit = options?.unreadyLimit ?? endpoints.length
  return endpoints.slice(offset, offset + limit)
}

function buildUnreadySummary(
  endpoints: WebhookEndpointSummaryData[],
  byMissingRequirement: Map<string, {
    requirement: string
    endpointCount: number
    channelTypes: Set<string>
    routes: Set<string>
  }>,
) {
  for (const endpoint of endpoints) {
    for (const requirement of endpoint.verificationMissing) {
      const existing = byMissingRequirement.get(requirement)
      if (existing) {
        existing.endpointCount += 1
        existing.channelTypes.add(endpoint.channelType)
        existing.routes.add(endpoint.route)
        continue
      }
      byMissingRequirement.set(requirement, {
        requirement,
        endpointCount: 1,
        channelTypes: new Set([endpoint.channelType]),
        routes: new Set([endpoint.route]),
      })
    }
  }

  return {
    totalEndpoints: endpoints.length,
    byMissingRequirement: Array.from(byMissingRequirement.values())
      .map((entry) => ({
        requirement: entry.requirement,
        endpointCount: entry.endpointCount,
        channelTypes: Array.from(entry.channelTypes).sort(),
        routes: Array.from(entry.routes).sort(),
      }))
      .sort((left, right) => {
        if (right.endpointCount !== left.endpointCount) {
          return right.endpointCount - left.endpointCount
        }
        return left.requirement.localeCompare(right.requirement)
      }),
  }
}
