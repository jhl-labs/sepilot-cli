import { openApiComponentsFromZod, openApiSchemaFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import {
  channelCatalogEntrySchema,
  channelPipelineRecentQuerySchema,
  channelPipelineRecentResponseSchema,
  channelPipelineResponseSchema,
  channelsResponseSchema,
  pipelineChannelStatsSchema,
  pipelineFailureSampleSchema,
  pipelineHealthPolicySchema,
  pipelineRecentChannelStatsSchema,
  pipelineRecentStatsSchema,
  pipelineStageStatsSchema,
  pipelineStatsSchema,
  pipelineSummarySchema,
  replaySummarySchema,
  sessionSummarySchema,
} from './channels-schema.js'

const channelOpenApiZodComponents = openApiComponentsFromZod({
  schemas: {
    ChannelCatalogEntry: channelCatalogEntrySchema,
    ChannelPipelineStageStat: pipelineStageStatsSchema,
    ChannelPipelineRecentFailureSample: pipelineFailureSampleSchema,
    ChannelPipelineHealthPolicy: pipelineHealthPolicySchema,
    ChannelPipelineRecentResponse: channelPipelineRecentResponseSchema,
    ChannelPipelineRecentChannelStat: pipelineRecentChannelStatsSchema,
    ChannelPipelineRecentStats: pipelineRecentStatsSchema,
    ChannelPipelineChannelStat: pipelineChannelStatsSchema,
    ChannelPipelineStats: pipelineStatsSchema,
    ChannelPipelineStatsResponse: channelPipelineResponseSchema,
    ChannelPipelineSummary: pipelineSummarySchema,
    ChannelReplaySummary: replaySummarySchema,
    ChannelSessionSummary: sessionSummarySchema,
    ChannelsResponse: channelsResponseSchema,
  },
})

export const channelOpenApiComponents: OpenApiComponentOverrides = {
  schemas: {
    ...(channelOpenApiZodComponents.schemas ?? {}),
  },
}

export const channelOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/channels': {
    get: {
      summary: 'List channel metadata',
      tags: ['Channels'],
      responses: { 200: openApiJsonResponseRef('ChannelsResponse') },
    },
  },
  '/api/v1/channels/pipeline': {
    get: {
      summary: 'Get channel pipeline stats',
      tags: ['Channels'],
      responses: { 200: openApiJsonResponseRef('ChannelPipelineStatsResponse') },
    },
  },
  '/api/v1/channels/pipeline/recent': {
    get: {
      summary: 'Get recent channel pipeline stats',
      tags: ['Channels'],
      parameters: [
        {
          name: 'channelType',
          in: 'query',
          schema: openApiSchemaFromZod(channelPipelineRecentQuerySchema.shape.channelType),
        },
        {
          name: 'outcome',
          in: 'query',
          schema: openApiSchemaFromZod(channelPipelineRecentQuerySchema.shape.outcome),
        },
        {
          name: 'failureOnly',
          in: 'query',
          schema: { type: 'boolean' },
        },
        {
          name: 'sampleOffset',
          in: 'query',
          schema: openApiSchemaFromZod(channelPipelineRecentQuerySchema.shape.sampleOffset),
        },
        {
          name: 'sampleLimit',
          in: 'query',
          schema: openApiSchemaFromZod(channelPipelineRecentQuerySchema.shape.sampleLimit),
        },
      ],
      responses: { 200: openApiJsonResponseRef('ChannelPipelineRecentResponse') },
    },
  },
}
