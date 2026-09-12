import { z } from 'zod'
import { webhookEndpointConfigSchema } from '../channels/webhook-validation.js'
import { channelConfigSchema } from './schema.js'

const nonEmptyStringSchema = z.string().trim().min(1)
const nonEmptyStringArraySchema = z.array(nonEmptyStringSchema)
const channelRateLimitSchema = z.number().int().min(1).max(1000)
const autonomyLevelSchema = z.enum(['readonly', 'accept-edits', 'workspace-write', 'supervised', 'autonomous'])
const discordPublicKeySchema = z.string().regex(/^[0-9a-f]{64}$/i, 'Discord publicKey must be a 64-character hex string')

const webhookChannelConfigUpdateSchema = z.object({
  endpoints: z.array(webhookEndpointConfigSchema).default([]),
  rateLimitPerMinute: channelRateLimitSchema.optional(),
}).passthrough()

const telegramChannelConfigUpdateSchema = z.object({
  botToken: nonEmptyStringSchema.optional(),
  allowedUsers: nonEmptyStringArraySchema.optional(),
  pairingRequired: z.boolean().optional(),
  pairingCodeTtl: z.number().int().min(30).max(3600).optional(),
  rateLimitPerMinute: channelRateLimitSchema.optional(),
}).passthrough()

const slackChannelConfigUpdateSchema = z.object({
  botToken: nonEmptyStringSchema.optional(),
  signingSecret: nonEmptyStringSchema.optional(),
  allowedChannels: nonEmptyStringArraySchema.optional(),
  allowedUsers: nonEmptyStringArraySchema.optional(),
}).passthrough()

const githubIssueChannelConfigUpdateSchema = z.object({
  owner: nonEmptyStringSchema.optional(),
  repo: nonEmptyStringSchema.optional(),
  labels: nonEmptyStringArraySchema.optional(),
  externalTriggerAutonomy: autonomyLevelSchema.optional(),
}).passthrough()

const discordChannelConfigUpdateSchema = z.object({
  botToken: nonEmptyStringSchema.optional(),
  applicationId: nonEmptyStringSchema.optional(),
  publicKey: discordPublicKeySchema.optional(),
  allowedGuilds: nonEmptyStringArraySchema.optional(),
  allowedChannels: nonEmptyStringArraySchema.optional(),
  allowedUsers: nonEmptyStringArraySchema.optional(),
}).passthrough()

const mattermostChannelConfigUpdateSchema = z.object({
  serverUrl: z.string().trim().url().refine(
    (value) => ['http:', 'https:'].includes(new URL(value).protocol),
    'Mattermost serverUrl must use http or https',
  ).optional(),
  botToken: nonEmptyStringSchema.optional(),
  webhookToken: nonEmptyStringSchema.optional(),
  allowedTeams: nonEmptyStringArraySchema.optional(),
  allowedChannels: nonEmptyStringArraySchema.optional(),
  allowedUsers: nonEmptyStringArraySchema.optional(),
}).passthrough()

const whatsappChannelConfigUpdateSchema = z.object({
  phoneNumberId: nonEmptyStringSchema.optional(),
  accessToken: nonEmptyStringSchema.optional(),
  appSecret: nonEmptyStringSchema.optional(),
  verifyToken: nonEmptyStringSchema.optional(),
  allowedNumbers: nonEmptyStringArraySchema.optional(),
}).passthrough()

const teamsChannelConfigUpdateSchema = z.object({
  appId: nonEmptyStringSchema.optional(),
  appPassword: nonEmptyStringSchema.optional(),
  allowedTenants: nonEmptyStringArraySchema.optional(),
  rateLimitPerMinute: channelRateLimitSchema.optional(),
}).passthrough()

const lineChannelConfigUpdateSchema = z.object({
  channelAccessToken: nonEmptyStringSchema.optional(),
  channelSecret: nonEmptyStringSchema.optional(),
  allowedUsers: nonEmptyStringArraySchema.optional(),
  rateLimitPerMinute: channelRateLimitSchema.optional(),
}).passthrough()

interface ChannelConfigUpdateRule {
  schema: z.ZodType<unknown>
  requiredWhenEnabled?: readonly string[]
}

type WebhookChannelConfigUpdate = z.infer<typeof webhookChannelConfigUpdateSchema>

const BUILTIN_CHANNEL_CONFIG_UPDATE_RULES: Record<string, ChannelConfigUpdateRule> = {
  webhook: {
    schema: webhookChannelConfigUpdateSchema,
  },
  telegram: {
    schema: telegramChannelConfigUpdateSchema,
    requiredWhenEnabled: ['botToken'],
  },
  slack: {
    schema: slackChannelConfigUpdateSchema,
    requiredWhenEnabled: ['botToken', 'signingSecret'],
  },
  'github-issue': {
    schema: githubIssueChannelConfigUpdateSchema,
    requiredWhenEnabled: ['owner', 'repo'],
  },
  discord: {
    schema: discordChannelConfigUpdateSchema,
    requiredWhenEnabled: ['botToken', 'applicationId', 'publicKey'],
  },
  mattermost: {
    schema: mattermostChannelConfigUpdateSchema,
    requiredWhenEnabled: ['serverUrl', 'botToken', 'webhookToken'],
  },
  whatsapp: {
    schema: whatsappChannelConfigUpdateSchema,
    requiredWhenEnabled: ['phoneNumberId', 'accessToken', 'verifyToken', 'appSecret'],
  },
  teams: {
    schema: teamsChannelConfigUpdateSchema,
    requiredWhenEnabled: ['appId', 'appPassword'],
  },
  line: {
    schema: lineChannelConfigUpdateSchema,
    requiredWhenEnabled: ['channelAccessToken', 'channelSecret'],
  },
  webchat: {
    schema: z.record(z.unknown()).optional(),
  },
}

function hasNonEmptyString(value: unknown): boolean {
  return typeof value === 'string' && value.trim().length > 0
}

export const configUpdateChannelSchema = channelConfigSchema.superRefine((channel, ctx) => {
  const rule = BUILTIN_CHANNEL_CONFIG_UPDATE_RULES[channel.type]
  if (!rule) {
    return
  }

  const config = channel.config ?? {}
  const parsed = rule.schema.safeParse(config)
  if (!parsed.success) {
    for (const issue of parsed.error.issues) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['config', ...issue.path],
        message: issue.message,
      })
    }
  }

  if (!channel.enabled) {
    return
  }

  if (channel.type === 'webhook' && parsed.success) {
    const endpoints = (parsed.data as WebhookChannelConfigUpdate).endpoints
    if (!endpoints.some((endpoint) => endpoint.enabled !== false)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['config', 'endpoints'],
        message: 'enabled webhook channel requires at least one enabled endpoint',
      })
    }
  }

  for (const field of rule.requiredWhenEnabled ?? []) {
    if (!hasNonEmptyString(config[field])) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['config', field],
        message: `enabled ${channel.type} channel requires config.${field}`,
      })
    }
  }
})

export const configChannelsUpdateSchema = z.array(configUpdateChannelSchema)
