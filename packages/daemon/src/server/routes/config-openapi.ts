import { openApiComponentsFromZod, openApiSchemaFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiSchema,
  type OpenApiOverrideMap,
} from '../openapi.js'
import {
  channelToggleRequestSchema,
  configChannelPipelineHealthHistoryQuerySchema,
  configChannelPipelineHealthHistoryEntrySchema,
  configChannelPipelineHealthHistoryResponseSchema,
  configChannelPipelineHealthResponseSchema,
  configChannelPipelineHealthUpdateRequestSchema,
  configEnvUpdateRequestSchema,
  configEnvUpdateResponseSchema,
  configProviderInfoSchema,
  configProvidersResponseSchema,
  configProviderValidationRequestSchema,
  configProviderValidationResponseSchema,
  configUpdateRequestSchema,
  configUpdateResponseSchema,
  discordChannelConfigSchema,
  discordChannelResponseSchema,
  discordChannelSummarySchema,
  configWebhookSecurityHealthHistoryEntrySchema,
  configWebhookSecurityHealthHistoryQuerySchema,
  configWebhookSecurityHealthHistoryResponseSchema,
  configWebhookSecurityHealthResponseSchema,
  configWebhookSecurityHealthUpdateRequestSchema,
  configWebhookSecurityPolicyHistoryEntrySchema,
  configWebhookSecurityPolicyHistoryQuerySchema,
  configWebhookSecurityPolicyHistoryResponseSchema,
  configWebhookSecurityPolicyResponseSchema,
  configWebhookSecurityPolicyUpdateRequestSchema,
  configSchema,
  mcpServerSchema,
  outboundWebhookSchema,
  mcpServerNameParamsSchema,
  mcpServerToggleRequestSchema,
  modelCapabilitiesSchema,
  modelInfoSchema,
  outboundWebhookDeadLetterAckAuditRecordSchema,
  outboundWebhookDeadLetterAckHistoryQuerySchema,
  outboundWebhookDeadLetterAckHistoryResponseSchema,
  outboundWebhookDeadLetterAckParamsSchema,
  outboundWebhookDeadLetterAckRequestSchema,
  outboundWebhookDeadLetterListResponseSchema,
  outboundWebhookDeadLetterQuerySchema,
  outboundWebhookDeadLetterResponseSchema,
  outboundWebhookDeadLetterSchema,
  outboundWebhookDeadLetterTimelineEntrySchema,
  outboundWebhookDeadLetterTimelineQuerySchema,
  outboundWebhookDeadLetterTimelineResponseSchema,
  outboundWebhookDeliveryDetailResponseSchema,
  outboundWebhookDeliveryDetailSchema,
  outboundWebhookDeliveryListResponseSchema,
  outboundWebhookDeliveryQuerySchema,
  outboundWebhookDeliveryReplayParamsSchema,
  outboundWebhookDeliveryReplayRequestSchema,
  outboundWebhookDeliveryResponseSchema,
  outboundWebhookDeliverySchema,
  outboundWebhookIdParamsSchema,
  outboundWebhookListResponseSchema,
  outboundWebhookSummarySchema,
  outboundWebhookToggleRequestSchema,
  mattermostChannelConfigSchema,
  mattermostChannelResponseSchema,
  mattermostChannelSummarySchema,
  slackChannelConfigSchema,
  slackChannelResponseSchema,
  slackChannelSummarySchema,
  telegramAllowedUserParamsSchema,
  telegramChannelConfigSchema,
  telegramChannelResponseSchema,
  telegramChannelSummarySchema,
  telegramChannelToggleRequestSchema,
  telegramPairingCodeResponseSchema,
  telegramPairingCodeSchema,
  webhookEndpointIdParamsSchema,
  webhookEndpointListResponseSchema,
  webhookEndpointSchema,
  webhookEndpointSummarySchema,
  webhookEndpointToggleRequestSchema,
} from './config-schema.js'

const configOpenApiZodComponents = openApiComponentsFromZod({
  schemas: {
    ModelCapabilities: modelCapabilitiesSchema,
    ModelInfo: modelInfoSchema,
    ConfigProviderInfo: configProviderInfoSchema,
    ConfigUpdateRequest: configUpdateRequestSchema,
    McpServer: mcpServerSchema,
    OutboundWebhook: outboundWebhookSchema,
    WebhookEndpoint: webhookEndpointSchema,
    TelegramChannel: telegramChannelConfigSchema,
    SlackChannel: slackChannelConfigSchema,
    DiscordChannel: discordChannelConfigSchema,
    MattermostChannel: mattermostChannelConfigSchema,
    McpServerToggleRequest: mcpServerToggleRequestSchema,
    OutboundWebhookToggleRequest: outboundWebhookToggleRequestSchema,
    OutboundWebhookDeliveryReplayRequest: outboundWebhookDeliveryReplayRequestSchema,
    OutboundWebhookDeadLetterAckRequest: outboundWebhookDeadLetterAckRequestSchema,
    WebhookEndpointToggleRequest: webhookEndpointToggleRequestSchema,
    TelegramChannelToggleRequest: telegramChannelToggleRequestSchema,
    ChannelToggleRequest: channelToggleRequestSchema,
    TelegramAllowedUserParams: telegramAllowedUserParamsSchema,
    TelegramPairingCode: telegramPairingCodeSchema,
    OutboundWebhookSummary: outboundWebhookSummarySchema,
    OutboundWebhookDelivery: outboundWebhookDeliverySchema,
    OutboundWebhookDeliveryDetail: outboundWebhookDeliveryDetailSchema,
    OutboundWebhookDeadLetter: outboundWebhookDeadLetterSchema,
    OutboundWebhookDeadLetterAckAuditRecord:
      outboundWebhookDeadLetterAckAuditRecordSchema,
    OutboundWebhookDeadLetterTimelineEntry:
      outboundWebhookDeadLetterTimelineEntrySchema,
    WebhookEndpointSummary: webhookEndpointSummarySchema,
    ConfigProviderValidationRequest: configProviderValidationRequestSchema,
    ConfigProviderValidationResponse: configProviderValidationResponseSchema,
    ConfigEnvUpdateRequest: configEnvUpdateRequestSchema,
    ConfigEnvUpdateResponse: configEnvUpdateResponseSchema,
    ConfigProvidersResponse: configProvidersResponseSchema,
    ConfigUpdateResponse: configUpdateResponseSchema,
    OutboundWebhookListResponse: outboundWebhookListResponseSchema,
    OutboundWebhookDeliveryListResponse: outboundWebhookDeliveryListResponseSchema,
    OutboundWebhookDeliveryDetailResponse:
      outboundWebhookDeliveryDetailResponseSchema,
    OutboundWebhookDeliveryResponse: outboundWebhookDeliveryResponseSchema,
    OutboundWebhookDeadLetterListResponse: outboundWebhookDeadLetterListResponseSchema,
    OutboundWebhookDeadLetterAckHistoryResponse:
      outboundWebhookDeadLetterAckHistoryResponseSchema,
    OutboundWebhookDeadLetterTimelineResponse:
      outboundWebhookDeadLetterTimelineResponseSchema,
    OutboundWebhookDeadLetterResponse: outboundWebhookDeadLetterResponseSchema,
    WebhookEndpointListResponse: webhookEndpointListResponseSchema,
    TelegramChannelSummary: telegramChannelSummarySchema,
    TelegramChannelResponse: telegramChannelResponseSchema,
    SlackChannelSummary: slackChannelSummarySchema,
    SlackChannelResponse: slackChannelResponseSchema,
    DiscordChannelSummary: discordChannelSummarySchema,
    DiscordChannelResponse: discordChannelResponseSchema,
    MattermostChannelSummary: mattermostChannelSummarySchema,
    MattermostChannelResponse: mattermostChannelResponseSchema,
    TelegramPairingCodeResponse: telegramPairingCodeResponseSchema,
    ConfigChannelPipelineHealthUpdateRequest:
      configChannelPipelineHealthUpdateRequestSchema,
    ConfigChannelPipelineHealthResponse:
      configChannelPipelineHealthResponseSchema,
    ConfigChannelPipelineHealthHistoryEntry:
      configChannelPipelineHealthHistoryEntrySchema,
    ConfigChannelPipelineHealthHistoryResponse:
      configChannelPipelineHealthHistoryResponseSchema,
    ConfigWebhookSecurityHealthUpdateRequest:
      configWebhookSecurityHealthUpdateRequestSchema,
    ConfigWebhookSecurityHealthResponse:
      configWebhookSecurityHealthResponseSchema,
    ConfigWebhookSecurityHealthHistoryEntry:
      configWebhookSecurityHealthHistoryEntrySchema,
    ConfigWebhookSecurityHealthHistoryResponse:
      configWebhookSecurityHealthHistoryResponseSchema,
    ConfigWebhookSecurityPolicyUpdateRequest:
      configWebhookSecurityPolicyUpdateRequestSchema,
    ConfigWebhookSecurityPolicyResponse:
      configWebhookSecurityPolicyResponseSchema,
    ConfigWebhookSecurityPolicyHistoryEntry:
      configWebhookSecurityPolicyHistoryEntrySchema,
    ConfigWebhookSecurityPolicyHistoryResponse:
      configWebhookSecurityPolicyHistoryResponseSchema,
  },
  parameters: {
    McpServerNameParam: {
      name: 'name',
      in: 'path',
      required: true,
      schema: mcpServerNameParamsSchema.shape.name,
    },
    OutboundWebhookIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: outboundWebhookIdParamsSchema.shape.id,
    },
    OutboundWebhookDeliveryIdQueryParam: {
      name: 'id',
      in: 'query',
      required: false,
      schema: outboundWebhookDeliveryQuerySchema.shape.id,
    },
    OutboundWebhookDeliveryStatusQueryParam: {
      name: 'status',
      in: 'query',
      required: false,
      schema: outboundWebhookDeliveryQuerySchema.shape.status,
    },
    OutboundWebhookDeliveryCursorQueryParam: {
      name: 'cursor',
      in: 'query',
      required: false,
      schema: outboundWebhookDeliveryQuerySchema.shape.cursor,
    },
    OutboundWebhookDeliveryLimitQueryParam: {
      name: 'limit',
      in: 'query',
      required: false,
      schema: outboundWebhookDeliveryQuerySchema.shape.limit,
    },
    OutboundWebhookDeadLetterStateQueryParam: {
      name: 'state',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterQuerySchema.shape.state,
    },
    OutboundWebhookDeadLetterRootDeliveryIdQueryParam: {
      name: 'rootDeliveryId',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterAckHistoryQuerySchema.shape.rootDeliveryId,
    },
    OutboundWebhookDeadLetterLatestDeliveryIdQueryParam: {
      name: 'latestDeliveryId',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterAckHistoryQuerySchema.shape.latestDeliveryId,
    },
    OutboundWebhookDeadLetterCursorQueryParam: {
      name: 'cursor',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterQuerySchema.shape.cursor,
    },
    OutboundWebhookDeadLetterAckDeviceQueryParam: {
      name: 'device',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterAckHistoryQuerySchema.shape.device,
    },
    OutboundWebhookDeadLetterAckSinceQueryParam: {
      name: 'since',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterAckHistoryQuerySchema.shape.since,
    },
    OutboundWebhookDeadLetterAckCursorQueryParam: {
      name: 'cursor',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterAckHistoryQuerySchema.shape.cursor,
    },
    OutboundWebhookDeadLetterAckLimitQueryParam: {
      name: 'limit',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterAckHistoryQuerySchema.shape.limit,
    },
    OutboundWebhookDeadLetterTimelineCursorQueryParam: {
      name: 'cursor',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterTimelineQuerySchema.shape.cursor,
    },
    OutboundWebhookDeadLetterTimelineLimitQueryParam: {
      name: 'limit',
      in: 'query',
      required: false,
      schema: outboundWebhookDeadLetterTimelineQuerySchema.shape.limit,
    },
    OutboundWebhookDeliveryIdParam: {
      name: 'deliveryId',
      in: 'path',
      required: true,
      schema: outboundWebhookDeliveryReplayParamsSchema.shape.deliveryId,
    },
    OutboundWebhookDeadLetterRootDeliveryIdParam: {
      name: 'rootDeliveryId',
      in: 'path',
      required: true,
      schema: outboundWebhookDeadLetterAckParamsSchema.shape.rootDeliveryId,
    },
    WebhookEndpointIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: webhookEndpointIdParamsSchema.shape.id,
    },
    TelegramAllowedUserIdParam: {
      name: 'userId',
      in: 'path',
      required: true,
      schema: telegramAllowedUserParamsSchema.shape.userId,
    },
  },
})

const configChannelPipelineHealthHistoryEntryOpenApiSchema = configOpenApiZodComponents.schemas?.ConfigChannelPipelineHealthHistoryEntry
const configChannelPipelineHealthHistoryEntryOpenApiProperties: OpenApiSchema =
  configChannelPipelineHealthHistoryEntryOpenApiSchema
    && typeof configChannelPipelineHealthHistoryEntryOpenApiSchema === 'object'
    && 'properties' in configChannelPipelineHealthHistoryEntryOpenApiSchema
    && typeof configChannelPipelineHealthHistoryEntryOpenApiSchema.properties === 'object'
    && configChannelPipelineHealthHistoryEntryOpenApiSchema.properties !== null
    ? configChannelPipelineHealthHistoryEntryOpenApiSchema.properties as OpenApiSchema
    : {}

const configChannelPipelineTimestampSchema: OpenApiSchema =
  typeof configChannelPipelineHealthHistoryEntryOpenApiProperties.timestamp === 'object'
    && configChannelPipelineHealthHistoryEntryOpenApiProperties.timestamp !== null
    ? configChannelPipelineHealthHistoryEntryOpenApiProperties.timestamp as OpenApiSchema
    : {}

export const configOpenApiComponents: OpenApiComponentOverrides = {
  schemas: {
    ...(configOpenApiZodComponents.schemas ?? {}),
    ConfigChannelPipelineHealthHistoryEntry: {
      ...configChannelPipelineHealthHistoryEntryOpenApiSchema,
      properties: {
        ...configChannelPipelineHealthHistoryEntryOpenApiProperties,
        timestamp: {
          ...configChannelPipelineTimestampSchema,
          type: 'string',
          format: 'date-time',
        },
      },
    },
    DaemonConfig: openApiSchemaFromZod(configSchema),
    ConfigResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('DaemonConfig'),
      },
      required: ['data'],
    },
  },
  parameters: {
    ...(configOpenApiZodComponents.parameters ?? {}),
  },
}

export const configOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/config': {
    get: {
      summary: 'Get config',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('ConfigResponse') },
    },
    put: {
      summary: 'Update runtime config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ConfigUpdateRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
  },
  '/api/v1/config/env': {
    put: {
      summary: 'Update daemon-managed env file',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ConfigEnvUpdateRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigEnvUpdateResponse') },
    },
  },
  '/api/v1/config/providers': {
    get: {
      summary: 'List providers',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('ConfigProvidersResponse') },
    },
  },
  '/api/v1/config/providers/validate': {
    post: {
      summary: 'Validate provider configuration',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ConfigProviderValidationRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigProviderValidationResponse'),
      },
    },
  },
  '/api/v1/config/observability/channel-pipeline-health': {
    get: {
      summary: 'Get channel pipeline health policy',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('ConfigChannelPipelineHealthResponse'),
      },
    },
    put: {
      summary: 'Update channel pipeline health policy',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ConfigChannelPipelineHealthUpdateRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigChannelPipelineHealthResponse'),
      },
    },
  },
  '/api/v1/config/observability/channel-pipeline-health/history': {
    get: {
      summary: 'List channel pipeline health policy audit history',
      tags: ['Config'],
      parameters: [
        {
          name: 'route',
          in: 'query',
          schema: openApiSchemaFromZod(
            configChannelPipelineHealthHistoryQuerySchema.shape.route,
          ),
        },
        {
          name: 'limit',
          in: 'query',
          schema: openApiSchemaFromZod(
            configChannelPipelineHealthHistoryQuerySchema.shape.limit,
          ),
        },
        {
          name: 'device',
          in: 'query',
          schema: openApiSchemaFromZod(
            configChannelPipelineHealthHistoryQuerySchema.shape.device,
          ),
        },
        {
          name: 'since',
          in: 'query',
          schema: openApiSchemaFromZod(
            configChannelPipelineHealthHistoryQuerySchema.shape.since,
          ),
        },
        {
          name: 'cursor',
          in: 'query',
          schema: openApiSchemaFromZod(
            configChannelPipelineHealthHistoryQuerySchema.shape.cursor,
          ),
        },
      ],
      responses: {
        200: openApiJsonResponseRef(
          'ConfigChannelPipelineHealthHistoryResponse',
        ),
      },
    },
  },
  '/api/v1/config/observability/webhook-security-health': {
    get: {
      summary: 'Get webhook security health policy',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('ConfigWebhookSecurityHealthResponse'),
      },
    },
    put: {
      summary: 'Update webhook security health policy',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ConfigWebhookSecurityHealthUpdateRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigWebhookSecurityHealthResponse'),
      },
    },
  },
  '/api/v1/config/observability/webhook-security-health/history': {
    get: {
      summary: 'List webhook security health policy audit history',
      tags: ['Config'],
      parameters: [
        {
          name: 'route',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityHealthHistoryQuerySchema.shape.route,
          ),
        },
        {
          name: 'limit',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityHealthHistoryQuerySchema.shape.limit,
          ),
        },
        {
          name: 'device',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityHealthHistoryQuerySchema.shape.device,
          ),
        },
        {
          name: 'since',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityHealthHistoryQuerySchema.shape.since,
          ),
        },
        {
          name: 'cursor',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityHealthHistoryQuerySchema.shape.cursor,
          ),
        },
      ],
      responses: {
        200: openApiJsonResponseRef(
          'ConfigWebhookSecurityHealthHistoryResponse',
        ),
      },
    },
  },
  '/api/v1/config/security/webhooks': {
    get: {
      summary: 'Get webhook security policy',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('ConfigWebhookSecurityPolicyResponse'),
      },
    },
    put: {
      summary: 'Update webhook security policy',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ConfigWebhookSecurityPolicyUpdateRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigWebhookSecurityPolicyResponse'),
      },
    },
  },
  '/api/v1/config/security/webhooks/history': {
    get: {
      summary: 'List webhook security policy audit history',
      tags: ['Config'],
      parameters: [
        {
          name: 'route',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityPolicyHistoryQuerySchema.shape.route,
          ),
        },
        {
          name: 'limit',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityPolicyHistoryQuerySchema.shape.limit,
          ),
        },
        {
          name: 'device',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityPolicyHistoryQuerySchema.shape.device,
          ),
        },
        {
          name: 'since',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityPolicyHistoryQuerySchema.shape.since,
          ),
        },
        {
          name: 'cursor',
          in: 'query',
          schema: openApiSchemaFromZod(
            configWebhookSecurityPolicyHistoryQuerySchema.shape.cursor,
          ),
        },
      ],
      responses: {
        200: openApiJsonResponseRef(
          'ConfigWebhookSecurityPolicyHistoryResponse',
        ),
      },
    },
  },
  '/api/v1/config/mcp/servers': {
    post: {
      summary: 'Add or replace MCP server config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('McpServer'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
  },
  '/api/v1/config/mcp/servers/{name}': {
    delete: {
      summary: 'Remove MCP server config',
      tags: ['Config'],
      parameters: [openApiParameterRef('McpServerNameParam')],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/mcp/servers/{name}/enable': {
    post: {
      summary: 'Enable or disable MCP server config',
      tags: ['Config'],
      parameters: [openApiParameterRef('McpServerNameParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('McpServerToggleRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks': {
    get: {
      summary: 'List outbound webhook config',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('OutboundWebhookListResponse') },
    },
    post: {
      summary: 'Add or replace outbound webhook config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('OutboundWebhook'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/deliveries': {
    get: {
      summary: 'List outbound webhook delivery events',
      tags: ['Config'],
      parameters: [
        openApiParameterRef('OutboundWebhookDeliveryIdQueryParam'),
        openApiParameterRef('OutboundWebhookDeliveryStatusQueryParam'),
        openApiParameterRef('OutboundWebhookDeliveryCursorQueryParam'),
        openApiParameterRef('OutboundWebhookDeliveryLimitQueryParam'),
      ],
      responses: { 200: openApiJsonResponseRef('OutboundWebhookDeliveryListResponse') },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/deliveries/{deliveryId}': {
    get: {
      summary: 'Get outbound webhook delivery',
      tags: ['Config'],
      parameters: [openApiParameterRef('OutboundWebhookDeliveryIdParam')],
      responses: {
        200: openApiJsonResponseRef('OutboundWebhookDeliveryDetailResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/dead-letters': {
    get: {
      summary: 'List unresolved outbound webhook dead letters',
      tags: ['Config'],
      parameters: [
        openApiParameterRef('OutboundWebhookDeliveryIdQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterStateQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterCursorQueryParam'),
        openApiParameterRef('OutboundWebhookDeliveryLimitQueryParam'),
      ],
      responses: { 200: openApiJsonResponseRef('OutboundWebhookDeadLetterListResponse') },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/dead-letters/acknowledgements': {
    get: {
      summary: 'List outbound webhook dead letter acknowledgements',
      tags: ['Config'],
      parameters: [
        openApiParameterRef('OutboundWebhookDeliveryIdQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterRootDeliveryIdQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterLatestDeliveryIdQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterAckDeviceQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterAckSinceQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterAckCursorQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterAckLimitQueryParam'),
      ],
      responses: {
        200: openApiJsonResponseRef(
          'OutboundWebhookDeadLetterAckHistoryResponse',
        ),
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/dead-letters/{rootDeliveryId}': {
    get: {
      summary: 'Get outbound webhook dead letter',
      tags: ['Config'],
      parameters: [
        openApiParameterRef('OutboundWebhookDeadLetterRootDeliveryIdParam'),
      ],
      responses: {
        200: openApiJsonResponseRef('OutboundWebhookDeadLetterResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/dead-letters/{rootDeliveryId}/timeline': {
    get: {
      summary: 'List outbound webhook dead letter timeline',
      tags: ['Config'],
      parameters: [
        openApiParameterRef('OutboundWebhookDeadLetterRootDeliveryIdParam'),
        openApiParameterRef('OutboundWebhookDeadLetterTimelineCursorQueryParam'),
        openApiParameterRef('OutboundWebhookDeadLetterTimelineLimitQueryParam'),
      ],
      responses: {
        200: openApiJsonResponseRef('OutboundWebhookDeadLetterTimelineResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/dead-letters/{rootDeliveryId}/ack': {
    post: {
      summary: 'Acknowledge an outbound webhook dead letter',
      tags: ['Config'],
      parameters: [openApiParameterRef('OutboundWebhookDeadLetterRootDeliveryIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('OutboundWebhookDeadLetterAckRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('OutboundWebhookDeadLetterResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/deliveries/{deliveryId}/replay': {
    post: {
      summary: 'Replay an outbound webhook delivery',
      tags: ['Config'],
      parameters: [openApiParameterRef('OutboundWebhookDeliveryIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('OutboundWebhookDeliveryReplayRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('OutboundWebhookDeliveryResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/{id}': {
    delete: {
      summary: 'Remove outbound webhook config',
      tags: ['Config'],
      parameters: [openApiParameterRef('OutboundWebhookIdParam')],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/hooks/outbound-webhooks/{id}/enable': {
    post: {
      summary: 'Enable or disable outbound webhook config',
      tags: ['Config'],
      parameters: [openApiParameterRef('OutboundWebhookIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('OutboundWebhookToggleRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/webhook/endpoints': {
    get: {
      summary: 'List generic webhook endpoint config',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('WebhookEndpointListResponse') },
    },
    post: {
      summary: 'Add or replace generic webhook endpoint config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('WebhookEndpoint'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
  },
  '/api/v1/config/channels/webhook/endpoints/{id}': {
    delete: {
      summary: 'Remove generic webhook endpoint config',
      tags: ['Config'],
      parameters: [openApiParameterRef('WebhookEndpointIdParam')],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/webhook/endpoints/{id}/enable': {
    post: {
      summary: 'Enable or disable generic webhook endpoint config',
      tags: ['Config'],
      parameters: [openApiParameterRef('WebhookEndpointIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('WebhookEndpointToggleRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/slack': {
    get: {
      summary: 'Get Slack channel config summary',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('SlackChannelResponse') },
    },
    post: {
      summary: 'Add or replace Slack channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SlackChannel'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
    delete: {
      summary: 'Remove Slack channel config',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/slack/enable': {
    post: {
      summary: 'Enable or disable Slack channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ChannelToggleRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/discord': {
    get: {
      summary: 'Get Discord channel config summary',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('DiscordChannelResponse') },
    },
    post: {
      summary: 'Add or replace Discord channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('DiscordChannel'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
    delete: {
      summary: 'Remove Discord channel config',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/discord/enable': {
    post: {
      summary: 'Enable or disable Discord channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ChannelToggleRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/mattermost': {
    get: {
      summary: 'Get Mattermost channel config summary',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('MattermostChannelResponse') },
    },
    post: {
      summary: 'Add or replace Mattermost channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('MattermostChannel'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
    delete: {
      summary: 'Remove Mattermost channel config',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/mattermost/enable': {
    post: {
      summary: 'Enable or disable Mattermost channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ChannelToggleRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/telegram': {
    get: {
      summary: 'Get Telegram channel config summary',
      tags: ['Config'],
      responses: { 200: openApiJsonResponseRef('TelegramChannelResponse') },
    },
    post: {
      summary: 'Add or replace Telegram channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('TelegramChannel'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ConfigUpdateResponse') },
    },
    delete: {
      summary: 'Remove Telegram channel config',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/telegram/enable': {
    post: {
      summary: 'Enable or disable Telegram channel config',
      tags: ['Config'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('TelegramChannelToggleRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/telegram/pairing-code': {
    post: {
      summary: 'Generate a Telegram pairing code',
      tags: ['Config'],
      responses: {
        200: openApiJsonResponseRef('TelegramPairingCodeResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/config/channels/telegram/allowed-users/{userId}': {
    delete: {
      summary: 'Revoke a paired Telegram user',
      tags: ['Config'],
      parameters: [openApiParameterRef('TelegramAllowedUserIdParam')],
      responses: {
        200: openApiJsonResponseRef('ConfigUpdateResponse'),
        404: { description: 'Not found' },
      },
    },
  },
}
