import { z } from 'zod'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiOverrideMap,
} from '../openapi.js'
import {
  memoryCreateRequestSchema,
  memoryAuditEntrySchema,
  memoryDeleteRequestSchema,
  memoryDocumentChunkSchema,
  memoryDocumentIdParamsSchema,
  memoryDocumentIngestRequestSchema,
  memoryDocumentSchema,
  memoryLifecycleStatusSchema,
  memoryMaintenanceRequestSchema,
  memoryMaintenanceResultSchema,
  memoryEntryIdParamsSchema,
  memoryEntrySchema,
  memoryRecentEntrySchema,
  memorySearchTypeSchema,
  memorySemanticStatusSchema,
  memoryScopeTransferRequestSchema,
  memoryUpdateRequestSchema,
} from './memory-schema.js'

const memoryOpenApiZodComponents = openApiComponentsFromZod({
  schemas: {
    MemoryEntry: memoryEntrySchema,
    MemoryRecentEntry: memoryRecentEntrySchema,
    MemoryDocument: memoryDocumentSchema,
    MemoryDocumentChunk: memoryDocumentChunkSchema,
    MemoryCreateRequest: memoryCreateRequestSchema,
    MemoryDeleteRequest: memoryDeleteRequestSchema,
    MemoryUpdateRequest: memoryUpdateRequestSchema,
    MemoryDocumentIngestRequest: memoryDocumentIngestRequestSchema,
    MemorySemanticStatus: memorySemanticStatusSchema,
    MemoryAuditEntry: memoryAuditEntrySchema,
    MemoryLifecycleStatus: memoryLifecycleStatusSchema,
    MemoryMaintenanceRequest: memoryMaintenanceRequestSchema,
    MemoryMaintenanceResult: memoryMaintenanceResultSchema,
    MemoryScopeTransferRequest: memoryScopeTransferRequestSchema,
  },
  parameters: {
    MemorySearchQueryParam: {
      name: 'query',
      in: 'query',
      required: true,
      schema: z.string(),
    },
    MemorySearchLimitParam: {
      name: 'limit',
      in: 'query',
      schema: z.number().int(),
    },
    MemorySearchTypeParam: {
      name: 'type',
      in: 'query',
      schema: memorySearchTypeSchema,
    },
    MemoryDocumentIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: memoryDocumentIdParamsSchema.shape.id,
    },
    MemoryIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: memoryEntryIdParamsSchema.shape.id,
    },
    MemoryDocumentSearchDocumentIdParam: {
      name: 'documentId',
      in: 'query',
      schema: z.string(),
    },
    MemoryDocumentListQueryParam: {
      name: 'query',
      in: 'query',
      schema: z.string(),
    },
    MemoryAuditMemoryIdParam: {
      name: 'memoryId',
      in: 'query',
      schema: z.string(),
    },
    MemoryLifecycleStaleAfterDaysParam: {
      name: 'staleAfterDays',
      in: 'query',
      schema: z.number().int(),
    },
    MemoryLifecycleLowImportanceParam: {
      name: 'lowImportance',
      in: 'query',
      schema: z.number(),
    },
    MemoryDeleteReasonParam: {
      name: 'reason',
      in: 'query',
      schema: z.string(),
    },
    MemoryDeleteIncludeAllScopesParam: {
      name: 'includeAllScopes',
      in: 'query',
      schema: z.boolean(),
    },
    MemoryScopeParam: {
      name: 'scope',
      in: 'path',
      required: true,
      schema: z.string(),
    },
    MemorySecurityAuditSinceParam: {
      name: 'since',
      in: 'query',
      schema: z.string(),
    },
    MemorySecurityAuditActorParam: {
      name: 'actor',
      in: 'query',
      schema: z.string(),
    },
    MemorySecurityAuditAuthKindParam: {
      name: 'authKind',
      in: 'query',
      schema: z.string(),
    },
    MemorySecurityAuditRouteParam: {
      name: 'route',
      in: 'query',
      schema: z.string(),
    },
  },
})

export const memoryOpenApiComponents = {
  schemas: {
    ...(memoryOpenApiZodComponents.schemas ?? {}),
    MemoryAddResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            id: { type: 'string' },
          },
          required: ['id'],
        },
      },
      required: ['data'],
    },
    MemoryEntryResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('MemoryEntry'),
      },
      required: ['data'],
    },
    MemorySearchResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: openApiSchemaRef('MemoryEntry'),
        },
      },
      required: ['data'],
    },
    MemoryRecentResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: openApiSchemaRef('MemoryRecentEntry'),
        },
      },
      required: ['data'],
    },
    MemoryDocumentResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('MemoryDocument'),
      },
      required: ['data'],
    },
    MemoryDocumentListResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: openApiSchemaRef('MemoryDocument'),
        },
      },
      required: ['data'],
    },
    MemoryDocumentSearchResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: openApiSchemaRef('MemoryDocumentChunk'),
        },
      },
      required: ['data'],
    },
    MemoryReindexResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            started: { type: 'boolean' },
            status: openApiSchemaRef('MemorySemanticStatus'),
          },
          required: ['started', 'status'],
        },
      },
      required: ['data'],
    },
    MemoryAuditResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: openApiSchemaRef('MemoryAuditEntry'),
        },
      },
      required: ['data'],
    },
    MemorySecurityAuditEvent: {
      type: 'object',
      additionalProperties: true,
      properties: {
        timestamp: { type: 'string' },
        event: { type: 'string' },
        device: { type: 'string' },
        route: { type: 'string' },
        method: { type: 'string' },
        actor: { type: 'string' },
        authKind: { type: 'string' },
        scopeTags: { type: 'array', items: { type: 'string' } },
        requested: { type: 'string' },
        reason: { type: 'string' },
        tokenId: { type: 'string' },
        label: { type: 'string' },
        tokenScopes: { type: 'array', items: { type: 'string' } },
      },
      required: ['timestamp', 'event'],
    },
    MemorySecurityAuditResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: openApiSchemaRef('MemorySecurityAuditEvent'),
        },
        meta: {
          type: 'object',
          properties: {
            limit: { type: 'integer' },
            returned: { type: 'integer' },
          },
          required: ['limit', 'returned'],
        },
      },
      required: ['data', 'meta'],
    },
    MemoryLifecycleResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('MemoryLifecycleStatus'),
      },
      required: ['data'],
    },
    MemoryMaintenanceResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('MemoryMaintenanceResult'),
      },
      required: ['data'],
    },
    MemoryScopeTransferResult: {
      type: 'object',
      properties: {
        dryRun: { type: 'boolean' },
        globalSource: { type: 'boolean' },
        fromScope: { type: 'string' },
        toScope: { type: 'string' },
        matchedSemanticMemories: { type: 'integer' },
        wouldRetagSemanticMemories: { type: 'integer' },
        wouldUpdateReminders: { type: 'integer' },
        wouldMoveFileBucket: { anyOf: [{ type: 'string' }, { type: 'null' }] },
        limited: { type: 'boolean' },
        requiresConfirmGlobal: { type: 'boolean' },
        sampleSemanticIds: { type: 'array', items: { type: 'string' } },
        sampleReminderIds: { type: 'array', items: { type: 'string' } },
        retaggedSemanticMemories: { type: 'integer' },
        retaggedReminders: { type: 'integer' },
        fileBucketMoved: { type: 'boolean' },
      },
      required: ['dryRun', 'fromScope', 'toScope'],
    },
    MemoryScopeTransferResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('MemoryScopeTransferResult'),
      },
      required: ['data'],
    },
    MemoryScopesResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            fileScopes: { type: 'array', items: { type: 'string' } },
            semanticScopes: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  scope: { type: 'string' },
                  count: { type: 'integer' },
                },
                required: ['scope', 'count'],
              },
            },
            untaggedSemanticEntries: { type: 'integer' },
            pendingReminders: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  scope: { type: 'string' },
                  count: { type: 'integer' },
                },
                required: ['scope', 'count'],
              },
            },
          },
          required: ['fileScopes', 'semanticScopes', 'untaggedSemanticEntries', 'pendingReminders'],
        },
      },
      required: ['data'],
    },
    MemoryStatusResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('MemorySemanticStatus'),
      },
      required: ['data'],
    },
  },
  parameters: {
    ...(memoryOpenApiZodComponents.parameters ?? {}),
  },
}

export const memoryOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/memory/recent': {
    get: {
      summary: 'List recent memory',
      tags: ['Memory'],
      parameters: [openApiParameterRef('MemorySearchLimitParam')],
      responses: { 200: openApiJsonResponseRef('MemoryRecentResponse') },
    },
  },
  '/api/v1/memory/search': {
    get: {
      summary: 'Search memory',
      tags: ['Memory'],
      parameters: [
        openApiParameterRef('MemorySearchQueryParam'),
        openApiParameterRef('MemorySearchLimitParam'),
        openApiParameterRef('MemorySearchTypeParam'),
      ],
      responses: { 200: openApiJsonResponseRef('MemorySearchResponse') },
    },
  },
  '/api/v1/memory/documents/search': {
    get: {
      summary: 'Search indexed document chunks',
      tags: ['Memory'],
      parameters: [
        openApiParameterRef('MemorySearchQueryParam'),
        openApiParameterRef('MemorySearchLimitParam'),
        openApiParameterRef('MemorySearchTypeParam'),
        openApiParameterRef('MemoryDocumentSearchDocumentIdParam'),
      ],
      responses: { 200: openApiJsonResponseRef('MemoryDocumentSearchResponse') },
    },
  },
  '/api/v1/memory/documents': {
    get: {
      summary: 'List indexed documents',
      tags: ['Memory'],
      parameters: [
        openApiParameterRef('MemoryDocumentListQueryParam'),
        openApiParameterRef('MemorySearchLimitParam'),
      ],
      responses: { 200: openApiJsonResponseRef('MemoryDocumentListResponse') },
    },
    post: {
      summary: 'Index a document for semantic retrieval',
      tags: ['Memory'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('MemoryDocumentIngestRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('MemoryDocumentResponse') },
    },
  },
  '/api/v1/memory/documents/{id}': {
    get: {
      summary: 'Get indexed document metadata',
      tags: ['Memory'],
      parameters: [openApiParameterRef('MemoryDocumentIdParam')],
      responses: {
        200: openApiJsonResponseRef('MemoryDocumentResponse'),
        404: { description: 'Not found' },
      },
    },
    delete: {
      summary: 'Delete indexed document and chunks',
      tags: ['Memory'],
      parameters: [openApiParameterRef('MemoryDocumentIdParam')],
      responses: {
        204: { description: 'Deleted' },
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/memory/status': {
    get: {
      summary: 'Semantic memory status',
      tags: ['Memory'],
      responses: { 200: openApiJsonResponseRef('MemoryStatusResponse') },
    },
  },
  '/api/v1/memory/lifecycle': {
    get: {
      summary: 'Semantic memory lifecycle status',
      tags: ['Memory'],
      parameters: [
        openApiParameterRef('MemoryLifecycleStaleAfterDaysParam'),
        openApiParameterRef('MemoryLifecycleLowImportanceParam'),
      ],
      responses: { 200: openApiJsonResponseRef('MemoryLifecycleResponse') },
    },
  },
  '/api/v1/memory/audit': {
    get: {
      summary: 'List semantic memory audit entries',
      tags: ['Memory'],
      parameters: [
        openApiParameterRef('MemoryAuditMemoryIdParam'),
        openApiParameterRef('MemorySearchLimitParam'),
      ],
      responses: { 200: openApiJsonResponseRef('MemoryAuditResponse') },
    },
  },
  '/api/v1/memory/security-audit': {
    get: {
      summary: 'List denied memory scope bypass audit events',
      tags: ['Memory'],
      parameters: [
        openApiParameterRef('MemorySearchLimitParam'),
        openApiParameterRef('MemorySecurityAuditSinceParam'),
        openApiParameterRef('MemorySecurityAuditActorParam'),
        openApiParameterRef('MemorySecurityAuditAuthKindParam'),
        openApiParameterRef('MemorySecurityAuditRouteParam'),
      ],
      responses: {
        200: openApiJsonResponseRef('MemorySecurityAuditResponse'),
        403: { description: 'Extension tokens cannot read memory security audit history' },
      },
    },
  },
  '/api/v1/memory/scopes': {
    get: {
      summary: 'List active memory scopes',
      tags: ['Memory'],
      responses: { 200: openApiJsonResponseRef('MemoryScopesResponse') },
    },
  },
  '/api/v1/memory/scopes/{scope}/transfer': {
    post: {
      summary: 'Transfer memories from one scope to another',
      tags: ['Memory'],
      parameters: [openApiParameterRef('MemoryScopeParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('MemoryScopeTransferRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('MemoryScopeTransferResponse'),
        400: { description: 'Invalid scope, target, or missing global migration confirmation' },
      },
    },
  },
  '/api/v1/memory/maintenance': {
    post: {
      summary: 'Run semantic memory lifecycle maintenance',
      tags: ['Memory'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('MemoryMaintenanceRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('MemoryMaintenanceResponse') },
    },
  },
  '/api/v1/memory': {
    post: {
      summary: 'Add memory',
      tags: ['Memory'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('MemoryCreateRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('MemoryAddResponse') },
    },
  },
  '/api/v1/memory/{id}': {
    put: {
      summary: 'Update memory',
      tags: ['Memory'],
      parameters: [openApiParameterRef('MemoryIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('MemoryUpdateRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('MemoryEntryResponse'),
        404: { description: 'Not found' },
      },
    },
    delete: {
      summary: 'Delete memory',
      tags: ['Memory'],
      parameters: [
        openApiParameterRef('MemoryIdParam'),
        openApiParameterRef('MemoryDeleteReasonParam'),
        openApiParameterRef('MemoryDeleteIncludeAllScopesParam'),
      ],
      responses: {
        204: { description: 'Deleted' },
        403: { description: 'Scope mismatch or operator scope bypass denied' },
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/memory/delete': {
    post: {
      summary: 'Delete memory by request body',
      description: 'Body form supports ids containing reserved path characters.',
      tags: ['Memory'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('MemoryDeleteRequest'),
          },
        },
      },
      responses: {
        204: { description: 'Deleted' },
        403: { description: 'Scope mismatch or operator scope bypass denied' },
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/memory/reindex': {
    post: {
      summary: 'Rebuild semantic memory index',
      tags: ['Memory'],
      responses: {
        202: openApiJsonResponseRef('MemoryReindexResponse', 'Reindex started'),
        409: { description: 'Semantic memory is not in a reindexable state' },
        503: { description: 'Semantic memory backend unavailable' },
      },
    },
  },
}
