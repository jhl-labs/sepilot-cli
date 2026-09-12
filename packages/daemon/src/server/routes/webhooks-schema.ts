import { z } from 'zod'
import { safeWebhookPathSchema } from './webhooks-path-validation.js'

export const genericWebhookParamsSchema = z.object({
  path: safeWebhookPathSchema,
})

export const genericWebhookWildcardParamsSchema = z.object({
  '*': safeWebhookPathSchema,
}).transform((params) => ({
  path: params['*'],
}))

export const webhookSecurityQuerySchema = z.object({
  channelType: z.string().min(1).optional(),
  verificationReady: z.union([
    z.boolean(),
    z.enum(['true', 'false']),
  ]).transform((value) => value === true || value === 'true').optional(),
  missingRequirement: z.string().min(1).optional(),
  unreadyOffset: z.coerce.number().int().min(0).optional().default(0),
  unreadyLimit: z.coerce.number().int().min(1).max(100).optional().default(20),
})

export const slackUrlVerificationBodySchema = z.object({
  type: z.string().optional(),
  challenge: z.string().optional(),
}).passthrough()

export const whatsAppVerifyQuerySchema = z.object({
  'hub.mode': z.string().optional(),
  'hub.verify_token': z.string().optional(),
  'hub.challenge': z.string().optional(),
})

export const whatsAppWebhookBodySchema = z.object({
  object: z.string().optional(),
  entry: z.array(z.object({
    id: z.string(),
    changes: z.array(z.object({
      field: z.string(),
      value: z.object({
        messaging_product: z.string().optional(),
        contacts: z.array(z.object({
          wa_id: z.string(),
          profile: z.object({
            name: z.string(),
          }).optional(),
        })).optional(),
        messages: z.array(z.object({
          id: z.string(),
          from: z.string(),
          timestamp: z.string(),
          type: z.string(),
          text: z.object({
            body: z.string(),
          }).optional(),
        })).optional(),
      }).partial().optional(),
    })).optional(),
  })).optional(),
})

export const teamsActivitySchema = z.object({
  type: z.string(),
  id: z.string().optional(),
  text: z.string().optional(),
  timestamp: z.string().optional(),
  serviceUrl: z.string().optional(),
  from: z.object({
    id: z.string(),
    name: z.string().optional(),
  }).optional(),
  conversation: z.object({
    id: z.string(),
  }).optional(),
  channelData: z.object({
    tenant: z.object({
      id: z.string(),
    }).optional(),
  }).optional(),
}).passthrough()

export const lineWebhookBodySchema = z.object({
  events: z.array(z.object({
    type: z.string(),
    replyToken: z.string().optional(),
    timestamp: z.number(),
    source: z.object({
      type: z.string(),
      userId: z.string().optional(),
      groupId: z.string().optional(),
      roomId: z.string().optional(),
    }).optional(),
    message: z.object({
      id: z.string(),
      type: z.string(),
      text: z.string().optional(),
    }).optional(),
  })).optional(),
})

export const genericWebhookBodySchema = z.record(z.unknown())

export type SlackUrlVerificationBody = z.infer<typeof slackUrlVerificationBodySchema>
export type WhatsAppVerifyQuery = z.input<typeof whatsAppVerifyQuerySchema>
export type WhatsAppWebhookBody = z.infer<typeof whatsAppWebhookBodySchema>
export type LineWebhookBody = z.infer<typeof lineWebhookBodySchema>
export type GenericWebhookBody = z.infer<typeof genericWebhookBodySchema>

export const webhookSecurityPolicySourceSchema = z.enum([
  'default',
  'global',
  'channel-type',
])

export const webhookSecurityPolicySchema = z.object({
  signatureMaxSkewSeconds: z.number().int().positive(),
  verificationUnavailableStatus: z.enum([
    'service_unavailable',
    'not_found',
  ]),
  source: z.object({
    signatureMaxSkewSeconds: webhookSecurityPolicySourceSchema,
    verificationUnavailableStatus: webhookSecurityPolicySourceSchema,
  }),
})

export const webhookEndpointSummarySchema = z.object({
  channelType: z.string(),
  route: z.string(),
  methods: z.array(z.string()),
  verification: z.array(z.string()),
  verificationReady: z.boolean(),
  verificationRequirements: z.array(z.string()),
  verificationMissing: z.array(z.string()),
  securityPolicy: webhookSecurityPolicySchema,
  status: z.string(),
  sourcePath: z.string().optional(),
})

export const webhookSecurityChannelSummarySchema = z.object({
  channelType: z.string(),
  endpointCount: z.number().int(),
  verificationReadyCount: z.number().int(),
  verificationNotReadyCount: z.number().int(),
  verificationRequirements: z.array(z.string()),
  verificationMissing: z.array(z.string()),
  methods: z.array(z.string()),
  verification: z.array(z.string()),
  effectivePolicy: webhookSecurityPolicySchema,
  routes: z.array(z.string()),
})

export const webhookSecuritySummarySchema = z.object({
  totalEndpoints: z.number().int(),
  verificationReadyEndpoints: z.number().int(),
  verificationNotReadyEndpoints: z.number().int(),
  byChannelType: z.array(webhookSecurityChannelSummarySchema),
  unreadySummary: z.object({
    totalEndpoints: z.number().int(),
    byMissingRequirement: z.array(z.object({
      requirement: z.string(),
      endpointCount: z.number().int(),
      channelTypes: z.array(z.string()),
      routes: z.array(z.string()),
    })),
  }),
  unreadyEndpoints: z.array(webhookEndpointSummarySchema),
})

export const webhookAckResponseSchema = z.object({
  ok: z.literal(true),
})

export const slackChallengeResponseSchema = z.object({
  challenge: z.string().optional(),
})

export const discordWebhookResponseSchema = z.record(z.unknown())

export const mattermostWebhookResponseSchema = z.object({
  response_type: z.literal('ephemeral'),
  text: z.string(),
})
