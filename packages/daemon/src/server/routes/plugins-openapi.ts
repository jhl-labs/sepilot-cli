import { z } from 'zod'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

const pluginManifestSchema = z.object({
  name: z.string(),
  version: z.string(),
  description: z.string(),
  main: z.string(),
  publisher: z.string().optional(),
  digest: z.string().optional(),
  signature: z.string().optional(),
  signatureKeyId: z.string().optional(),
  tools: z.array(z.string()).optional(),
  hooks: z.array(z.string()).optional(),
  channels: z.array(z.string()).optional(),
  permissions: z
    .array(
      z.object({
        type: z.string(),
        value: z.string().optional(),
        reason: z.string().optional(),
      }),
    )
    .optional(),
})

const pluginSummarySchema = pluginManifestSchema.extend({
  status: z.enum(['discovered', 'registered', 'failed']),
  error: z.string().optional(),
  security: z
    .object({
      digest: z.string(),
      verified: z.boolean(),
      warnings: z.array(z.string()).optional(),
    })
    .optional(),
})

const hookHandlerSchema = z.object({
  event: z.string(),
  id: z.string(),
  priority: z.number().int(),
})

const pluginsResponseSchema = z.object({
  data: z.array(pluginSummarySchema),
  meta: z.object({
    providerFactoryTypes: z.array(z.string()),
    channelFactoryTypes: z.array(z.string()),
    hookHandlers: z.array(hookHandlerSchema),
  }),
})

export const pluginOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    PluginManifest: pluginManifestSchema,
    PluginSummary: pluginSummarySchema,
    HookHandler: hookHandlerSchema,
    PluginListResponse: pluginsResponseSchema,
  },
})

export const pluginOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/plugins': {
    get: {
      summary: 'List plugins',
      tags: ['Plugins'],
      responses: { 200: openApiJsonResponseRef('PluginListResponse') },
    },
  },
}
