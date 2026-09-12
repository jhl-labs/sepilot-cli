import { z } from 'zod'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponse,
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { latestWebhookSecurityPolicyChangeSchema } from '../runtime/webhook-security-policy-audit.js'
import {
  discordWebhookResponseSchema,
  genericWebhookParamsSchema,
  mattermostWebhookResponseSchema,
  slackChallengeResponseSchema,
  webhookAckResponseSchema,
  webhookEndpointSummarySchema,
  webhookSecuritySummarySchema,
} from './webhooks-schema.js'

export const webhookOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    WebhookEndpointSummary: webhookEndpointSummarySchema,
    WebhookEndpointListResponse: z.object({
      data: z.array(webhookEndpointSummarySchema),
      meta: z.object({
        latestSecurityPolicyChange:
          latestWebhookSecurityPolicyChangeSchema.nullable(),
      }),
    }),
    WebhookSecuritySummary: webhookSecuritySummarySchema,
    WebhookSecuritySummaryResponse: z.object({
      data: webhookSecuritySummarySchema,
      meta: z.object({
        latestSecurityPolicyChange:
          latestWebhookSecurityPolicyChangeSchema.nullable(),
        filters: z.object({
          channelType: z.string().nullable(),
          verificationReady: z.boolean().nullable(),
          missingRequirement: z.string().nullable(),
        }),
        unreadyPage: z.object({
          offset: z.number().int(),
          limit: z.number().int(),
          total: z.number().int(),
          returned: z.number().int(),
        }),
      }),
    }),
    WebhookAckResponse: webhookAckResponseSchema,
    SlackChallengeResponse: slackChallengeResponseSchema,
    DiscordWebhookResponse: discordWebhookResponseSchema,
    MattermostWebhookResponse: mattermostWebhookResponseSchema,
  },
  parameters: {
    WebhookPathParam: {
      name: 'path',
      in: 'path',
      required: true,
      schema: genericWebhookParamsSchema.shape.path,
    },
  },
})

export const webhookOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/webhooks': {
    get: {
      summary: 'List active inbound webhook endpoints',
      tags: ['Webhooks'],
      responses: { 200: openApiJsonResponseRef('WebhookEndpointListResponse') },
    },
  },
  '/api/v1/webhooks/security': {
    get: {
      summary: 'Get inbound webhook security summary',
      tags: ['Webhooks'],
      parameters: [
        {
          name: 'channelType',
          in: 'query',
          schema: {
            type: 'string',
          },
        },
        {
          name: 'verificationReady',
          in: 'query',
          schema: {
            type: 'boolean',
          },
        },
        {
          name: 'missingRequirement',
          in: 'query',
          schema: {
            type: 'string',
          },
        },
        {
          name: 'unreadyOffset',
          in: 'query',
          schema: {
            type: 'integer',
            minimum: 0,
          },
        },
        {
          name: 'unreadyLimit',
          in: 'query',
          schema: {
            type: 'integer',
            minimum: 1,
            maximum: 100,
          },
        },
      ],
      responses: { 200: openApiJsonResponseRef('WebhookSecuritySummaryResponse') },
    },
  },
  '/api/v1/webhooks/slack': {
    post: {
      summary: 'Slack Events callback',
      tags: ['Webhooks'],
      responses: {
        200: openApiJsonResponse(
          {
            oneOf: [
              openApiSchemaRef('SlackChallengeResponse'),
              openApiSchemaRef('WebhookAckResponse'),
            ],
          },
        ),
        401: { description: 'Unauthorized' },
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
  },
  '/api/v1/webhooks/discord': {
    post: {
      summary: 'Discord interaction callback',
      tags: ['Webhooks'],
      responses: {
        200: openApiJsonResponseRef('DiscordWebhookResponse'),
        401: { description: 'Unauthorized' },
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
  },
  '/api/v1/webhooks/mattermost': {
    post: {
      summary: 'Mattermost slash command or outgoing webhook callback',
      tags: ['Webhooks'],
      responses: {
        200: openApiJsonResponseRef('MattermostWebhookResponse'),
        401: { description: 'Unauthorized' },
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
  },
  '/api/v1/webhooks/whatsapp': {
    get: {
      summary: 'WhatsApp webhook verification',
      tags: ['Webhooks'],
      responses: {
        200: {
          description: 'Challenge response',
          content: {
            'text/plain': {
              schema: {
                type: 'string',
              },
            },
          },
        },
        403: { description: 'Forbidden' },
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
    post: {
      summary: 'WhatsApp webhook callback',
      tags: ['Webhooks'],
      responses: {
        200: openApiJsonResponseRef('WebhookAckResponse'),
        401: { description: 'Unauthorized' },
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
  },
  '/api/v1/webhooks/teams': {
    post: {
      summary: 'Microsoft Teams Bot Framework callback',
      tags: ['Webhooks'],
      responses: {
        200: openApiJsonResponseRef('WebhookAckResponse'),
        401: { description: 'Unauthorized' },
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
  },
  '/api/v1/webhooks/line': {
    post: {
      summary: 'LINE Messaging API callback',
      tags: ['Webhooks'],
      responses: {
        200: openApiJsonResponseRef('WebhookAckResponse'),
        401: { description: 'Unauthorized' },
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
  },
  '/api/v1/webhooks/{path}': {
    post: {
      summary: 'Generic webhook callback',
      tags: ['Webhooks'],
      parameters: [openApiParameterRef('WebhookPathParam')],
      responses: {
        200: openApiJsonResponseRef('WebhookAckResponse'),
        404: { description: 'Not found' },
        503: { description: 'Webhook verification unavailable' },
      },
    },
  },
}
