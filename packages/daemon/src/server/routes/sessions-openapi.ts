import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { sessionsOpenApiZodComponents } from './sessions-schema.js'

export const sessionsOpenApiComponents = {
  schemas: {
    ...(sessionsOpenApiZodComponents.schemas ?? {}),
    SessionDetail: {
      allOf: [
        openApiSchemaRef('SessionMeta'),
        {
          type: 'object',
          properties: {
            events: {
              type: 'array',
              items: openApiSchemaRef('SessionEvent'),
            },
            pendingApprovals: {
              type: 'array',
              items: openApiSchemaRef('PendingApproval'),
            },
            pendingQuestions: {
              type: 'array',
              items: openApiSchemaRef('PendingQuestion'),
            },
            traceMetrics: openApiSchemaRef('SessionTraceMetrics'),
            contextEngine: openApiSchemaRef('SessionContextEngine'),
            historyManagement: openApiSchemaRef('SessionHistoryManagement'),
            completionChecklist: openApiSchemaRef('SessionCompletionChecklist'),
            workingMemory: openApiSchemaRef('SessionWorkingMemory'),
            runContract: openApiSchemaRef('SessionRunContract'),
            contractLedger: openApiSchemaRef('SessionContractLedger'),
            evidenceManifest: openApiSchemaRef('SessionEvidenceManifest'),
            evaluationGate: openApiSchemaRef('SessionEvaluationGate'),
            resumableRun: openApiSchemaRef('ResumableRun'),
            resumableRunIssue: openApiSchemaRef('ResumableRunIssue'),
            delegation: openApiSchemaRef('SessionDelegation'),
          },
          required: ['events'],
        },
      ],
    },
    SessionListResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            items: {
              type: 'array',
              items: openApiSchemaRef('SessionMeta'),
            },
            page: { type: 'integer' },
            perPage: { type: 'integer' },
            totalCount: { type: 'integer' },
            hasNextPage: { type: 'boolean' },
          },
          required: ['items', 'page', 'perPage', 'totalCount', 'hasNextPage'],
        },
      },
      required: ['data'],
    },
    SessionCreateResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('SessionMeta'),
      },
      required: ['data'],
    },
    SessionDetailResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('SessionDetail'),
      },
      required: ['data'],
    },
    SessionRunbookResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('SessionRunbook'),
      },
      required: ['data'],
    },
    SessionExportJsonResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            session: openApiSchemaRef('SessionDetail'),
            events: {
              type: 'array',
              items: openApiSchemaRef('SessionEvent'),
            },
            health: openApiSchemaRef('HealthExportSnapshot'),
          },
          required: ['session', 'events'],
        },
      },
      required: ['data'],
    },
    SessionBranchResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            branchId: { type: 'string' },
            sourceId: { type: 'string' },
            copiedEvents: { type: 'integer' },
          },
          required: ['branchId', 'sourceId', 'copiedEvents'],
        },
      },
      required: ['data'],
    },
    SessionManagementOrphanedArtifacts: {
      type: 'object',
      properties: {
        pendingApprovalRequestIds: { type: 'array', items: { type: 'string' } },
        pendingApprovalSessionIds: { type: 'array', items: { type: 'string' } },
        pendingQuestionIds: { type: 'array', items: { type: 'string' } },
        pendingQuestionSessionIds: { type: 'array', items: { type: 'string' } },
        runCheckpointSessionIds: { type: 'array', items: { type: 'string' } },
        approvalCheckpointRequestIds: { type: 'array', items: { type: 'string' } },
        toolExecutionSessionIds: { type: 'array', items: { type: 'string' } },
      },
      required: [
        'pendingApprovalRequestIds',
        'pendingApprovalSessionIds',
        'pendingQuestionIds',
        'pendingQuestionSessionIds',
        'runCheckpointSessionIds',
        'approvalCheckpointRequestIds',
        'toolExecutionSessionIds',
      ],
    },
    SessionManagementHistory: {
      type: 'object',
      properties: {
        totalSessions: { type: 'integer', minimum: 0 },
        compactedSessions: { type: 'integer', minimum: 0 },
        semanticRecallSessions: { type: 'integer', minimum: 0 },
        documentRecallSessions: { type: 'integer', minimum: 0 },
        memorySummarySessions: { type: 'integer', minimum: 0 },
        attentionNeededSessions: { type: 'integer', minimum: 0 },
        historyReadFailures: { type: 'integer', minimum: 0 },
        totalCompactions: { type: 'integer', minimum: 0 },
        totalTokensSaved: { type: 'integer', minimum: 0 },
        totalSemanticContextEvents: { type: 'integer', minimum: 0 },
        totalSemanticContextItems: { type: 'integer', minimum: 0 },
        totalDocumentContextItems: { type: 'integer', minimum: 0 },
        totalMemorySummaryEvents: { type: 'integer', minimum: 0 },
        totalSemanticExtractions: { type: 'integer', minimum: 0 },
        totalRagPromotions: { type: 'integer', minimum: 0 },
        risksByCode: {
          type: 'object',
          additionalProperties: { type: 'integer', minimum: 0 },
        },
        attentionSessionIds: { type: 'array', items: { type: 'string' } },
        failedHistorySessionIds: { type: 'array', items: { type: 'string' } },
        semanticIndex: { type: 'object', additionalProperties: true },
        dreaming: { type: 'object', additionalProperties: true },
        memoryLifecycle: { type: 'object', additionalProperties: true },
      },
      required: [
        'totalSessions',
        'compactedSessions',
        'semanticRecallSessions',
        'documentRecallSessions',
        'memorySummarySessions',
        'attentionNeededSessions',
        'historyReadFailures',
        'totalCompactions',
        'totalTokensSaved',
        'totalSemanticContextEvents',
        'totalSemanticContextItems',
        'totalDocumentContextItems',
        'totalMemorySummaryEvents',
        'totalSemanticExtractions',
        'totalRagPromotions',
        'risksByCode',
        'attentionSessionIds',
        'failedHistorySessionIds',
      ],
    },
    SessionRuntimeCleanupResult: {
      type: 'object',
      properties: {
        pendingApprovals: { type: 'integer', minimum: 0 },
        pendingQuestions: { type: 'integer', minimum: 0 },
        approvalCheckpoints: { type: 'integer', minimum: 0 },
        unavailableApprovalCheckpoints: { type: 'integer', minimum: 0 },
        runCheckpoints: { type: 'integer', minimum: 0 },
        toolExecutions: { type: 'integer', minimum: 0 },
      },
      required: [
        'pendingApprovals',
        'pendingQuestions',
        'approvalCheckpoints',
        'unavailableApprovalCheckpoints',
        'runCheckpoints',
        'toolExecutions',
      ],
    },
    SessionManagementSnapshot: {
      type: 'object',
      properties: {
        totalSessions: { type: 'integer', minimum: 0 },
        byStatus: {
          type: 'object',
          properties: {
            active: { type: 'integer', minimum: 0 },
            completed: { type: 'integer', minimum: 0 },
            abandoned: { type: 'integer', minimum: 0 },
          },
          required: ['active', 'completed', 'abandoned'],
        },
        pendingApprovals: {
          type: 'object',
          properties: {
            total: { type: 'integer', minimum: 0 },
            orphaned: { type: 'integer', minimum: 0 },
          },
          required: ['total', 'orphaned'],
        },
        pendingQuestions: {
          type: 'object',
          properties: {
            total: { type: 'integer', minimum: 0 },
            orphaned: { type: 'integer', minimum: 0 },
          },
          required: ['total', 'orphaned'],
        },
        runCheckpoints: {
          type: 'object',
          properties: {
            total: { type: 'integer', minimum: 0 },
            unavailable: { type: 'integer', minimum: 0 },
            locked: { type: 'integer', minimum: 0 },
            orphaned: { type: 'integer', minimum: 0 },
          },
          required: ['total', 'unavailable', 'locked', 'orphaned'],
        },
        approvalCheckpoints: {
          type: 'object',
          properties: {
            total: { type: 'integer', minimum: 0 },
            unavailable: { type: 'integer', minimum: 0 },
            orphaned: { type: 'integer', minimum: 0 },
          },
          required: ['total', 'unavailable', 'orphaned'],
        },
        toolExecutions: {
          type: 'object',
          properties: {
            total: { type: 'integer', minimum: 0 },
            running: { type: 'integer', minimum: 0 },
            completed: { type: 'integer', minimum: 0 },
            unavailable: { type: 'integer', minimum: 0 },
            orphaned: { type: 'integer', minimum: 0 },
          },
          required: ['total', 'running', 'completed', 'unavailable', 'orphaned'],
        },
        history: openApiSchemaRef('SessionManagementHistory'),
        orphaned: openApiSchemaRef('SessionManagementOrphanedArtifacts'),
      },
      required: [
        'totalSessions',
        'byStatus',
        'pendingApprovals',
        'pendingQuestions',
        'runCheckpoints',
        'approvalCheckpoints',
        'toolExecutions',
        'history',
        'orphaned',
      ],
    },
    SessionManagementResponse: {
      type: 'object',
      properties: {
        data: openApiSchemaRef('SessionManagementSnapshot'),
      },
      required: ['data'],
    },
    SessionManagementCleanupRequest: {
      type: 'object',
      properties: {
        dryRun: { type: 'boolean', default: true },
      },
    },
    SessionManagementCleanupResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            dryRun: { type: 'boolean' },
            candidates: openApiSchemaRef('SessionManagementOrphanedArtifacts'),
            deleted: openApiSchemaRef('SessionRuntimeCleanupResult'),
            before: openApiSchemaRef('SessionManagementSnapshot'),
            after: openApiSchemaRef('SessionManagementSnapshot'),
          },
          required: ['dryRun', 'candidates', 'deleted', 'before', 'after'],
        },
      },
      required: ['data'],
    },
    SessionCompactResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            sessionId: { type: 'string' },
            originalTokens: { type: 'integer' },
            compactedTokens: { type: 'integer' },
            savedTokens: { type: 'integer' },
            summary: { type: 'string' },
            strategy: {
              type: 'string',
              enum: ['preserve_tail', 'summary_only'],
            },
            removedMessageCount: { type: 'integer' },
            preservedMessageCount: { type: 'integer' },
          },
          required: [
            'sessionId',
            'originalTokens',
            'compactedTokens',
            'savedTokens',
            'summary',
          ],
        },
      },
      required: ['data'],
    },
  },
  parameters: {
    ...(sessionsOpenApiZodComponents.parameters ?? {}),
  },
}

export const sessionsOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/sessions': {
    post: {
      summary: 'Create session',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SessionCreateRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('SessionCreateResponse') },
    },
    get: {
      summary: 'List sessions',
      tags: ['Sessions'],
      parameters: [
        openApiParameterRef('SessionsPageParam'),
        openApiParameterRef('SessionsPerPageParam'),
        openApiParameterRef('SessionsQueryParam'),
        openApiParameterRef('SessionsWorkspaceRootParam'),
      ],
      responses: { 200: openApiJsonResponseRef('SessionListResponse') },
    },
  },
  '/api/v1/sessions/watch': {
    get: {
      summary: 'Watch session list updates',
      tags: ['Sessions'],
      parameters: [
        openApiParameterRef('SessionsPageParam'),
        openApiParameterRef('SessionsPerPageParam'),
        openApiParameterRef('SessionsQueryParam'),
        openApiParameterRef('SessionsWorkspaceRootParam'),
      ],
      responses: {
        200: {
          description: 'SSE stream',
          content: { 'text/event-stream': {} },
        },
      },
    },
  },
  '/api/v1/sessions/management': {
    get: {
      summary: 'Inspect session management state',
      tags: ['Sessions'],
      responses: {
        200: openApiJsonResponseRef('SessionManagementResponse'),
        503: { description: 'Runtime not initialized' },
      },
    },
  },
  '/api/v1/sessions/management/cleanup': {
    post: {
      summary: 'Clean orphaned session runtime state',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SessionManagementCleanupRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('SessionManagementCleanupResponse'),
        503: { description: 'Runtime not initialized' },
      },
    },
  },
  '/api/v1/sessions/{id}': {
    get: {
      summary: 'Get session',
      tags: ['Sessions'],
      responses: { 200: openApiJsonResponseRef('SessionDetailResponse'), 404: { description: 'Not found' } },
    },
    patch: {
      summary: 'Update session metadata',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SessionUpdateRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('SessionMeta'),
        400: { description: 'Invalid update payload' },
        404: { description: 'Not found' },
        501: { description: 'Metadata updates unsupported' },
      },
    },
    delete: {
      summary: 'Delete session',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionIdParam')],
      responses: { 204: { description: 'Deleted' }, 404: { description: 'Not found' } },
    },
  },
  '/api/v1/sessions/{id}/export': {
    get: {
      summary: 'Export session',
      tags: ['Sessions'],
      parameters: [
        openApiParameterRef('SessionIdParam'),
        openApiParameterRef('SessionExportFormatParam'),
      ],
      responses: {
        200: {
          description: 'Session export',
          content: {
            'application/json': {
              schema: openApiSchemaRef('SessionExportJsonResponse'),
            },
            'text/markdown': {
              schema: { type: 'string' },
            },
          },
        },
      },
    },
  },
  '/api/v1/sessions/{id}/runbook': {
    get: {
      summary: 'Build a diagnostic runbook for a session',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionIdParam')],
      responses: { 200: openApiJsonResponseRef('SessionRunbookResponse') },
    },
  },
  '/api/v1/sessions/{id}/branch': {
    post: {
      summary: 'Branch session',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SessionBranchRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('SessionBranchResponse') },
    },
  },
  '/api/v1/sessions/{id}/compact': {
    post: {
      summary: 'Compact session context',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionIdParam')],
      responses: { 200: openApiJsonResponseRef('SessionCompactResponse') },
    },
  },
  '/api/v1/sessions/{id}/resume': {
    post: {
      summary: 'Resume session run checkpoint',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SessionResumeRequest'),
          },
        },
      },
      responses: {
        200: {
          description: 'SSE stream',
          content: { 'text/event-stream': {} },
        },
        404: { description: 'Not found' },
        409: { description: 'Resume requires force' },
        503: { description: 'Service unavailable or agent run capacity exhausted' },
      },
    },
  },
  '/api/v1/sessions/{id}/watch': {
    get: {
      summary: 'Watch session updates',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionIdParam')],
      responses: {
        200: {
          description: 'SSE stream',
          content: { 'text/event-stream': {} },
        },
        404: { description: 'Not found' },
      },
    },
  },
}
