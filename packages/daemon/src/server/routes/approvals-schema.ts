import { z } from 'zod'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

export const approvalRequestSchema = z.object({
  requestId: z.string().min(1),
  approved: z.boolean().optional(),
  decision: z.enum(['approved', 'feedback', 'denied']).optional(),
  note: z.string().trim().min(1).optional(),
  approvedBy: z.string().optional(),
  sessionId: z.string().optional(),
  scope: z.enum(['once', 'session', 'always', 'run', 'session-all']).optional(),
  rule: z
    .object({
      tool: z.string().trim().min(1),
      pattern: z.string().trim().min(1),
    })
    .optional(),
}).superRefine((value, ctx) => {
  if (value.approved === undefined && value.decision === undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'Either approved or decision must be provided',
      path: ['approved'],
    })
  }

  if (value.decision && value.approved !== undefined) {
    const canonicalApproved = value.decision === 'approved'
    if (value.approved !== canonicalApproved) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'approved must match decision',
        path: ['approved'],
      })
    }
  }
})

export const approvalResolutionResponseSchema = z.object({
  data: z.object({
    requestId: z.string(),
    decision: z.enum(['approved', 'feedback', 'denied']),
    approved: z.boolean(),
    note: z.string().optional(),
    resolved: z.boolean(),
    state: z.enum(['live', 'stale']),
    rule: z.object({
      tool: z.string(),
      pattern: z.string(),
    }).optional(),
  }),
})

export const rememberedApprovalDecisionSchema = z.object({
  tool: z.string(),
  pattern: z.string(),
  scope: z.enum(['session', 'always']),
  approved: z.boolean(),
  sessionId: z.string().optional(),
  createdAt: z.string(),
  hitCount: z.number().optional(),
  lastHitAt: z.string().optional(),
})

function requireSessionIdForSessionScope(
  value: { scope: 'session' | 'always'; sessionId?: string },
  ctx: z.RefinementCtx,
): void {
  if (value.scope === 'session' && !value.sessionId) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'sessionId is required for session-scoped decisions',
      path: ['sessionId'],
    })
  }
}

export const rememberedApprovalDecisionInputSchema = z.object({
  tool: z.string().trim().min(1),
  pattern: z.string().trim().min(1),
  scope: z.enum(['session', 'always']),
  approved: z.boolean(),
  sessionId: z.string().trim().min(1).optional(),
}).superRefine(requireSessionIdForSessionScope)

export const rememberedApprovalDecisionMatchSchema = z.object({
  tool: z.string().trim().min(1),
  pattern: z.string().trim().min(1),
  scope: z.enum(['session', 'always']),
  sessionId: z.string().trim().min(1).optional(),
}).superRefine(requireSessionIdForSessionScope)

export const rememberedApprovalDecisionUpdateSchema = z.object({
  match: rememberedApprovalDecisionMatchSchema,
  decision: rememberedApprovalDecisionInputSchema,
})

export const rememberedApprovalDecisionDescribeSchema = z.object({
  tool: z.string().trim().min(1),
  input: z.record(z.unknown()).default({}),
})

export const rememberedApprovalDecisionDescribeResponseSchema = z.object({
  data: z.object({
    rule: z.object({
      tool: z.string(),
      pattern: z.string(),
    }),
  }),
})

export const rememberedApprovalListResponseSchema = z.object({
  data: z.object({
    decisions: z.array(rememberedApprovalDecisionSchema),
  }),
})

export const rememberedApprovalMutationResponseSchema = z.object({
  data: z.object({
    decision: rememberedApprovalDecisionSchema,
  }),
})

export const rememberedApprovalDeleteResponseSchema = z.object({
  data: z.object({
    removed: z.boolean(),
  }),
})

export const rememberedApprovalClearResponseSchema = z.object({
  data: z.object({
    cleared: z.boolean(),
  }),
})

export type ApprovalResponseBody = z.infer<typeof approvalRequestSchema>
export type RememberedApprovalDecisionInputBody =
  z.infer<typeof rememberedApprovalDecisionInputSchema>
export type RememberedApprovalDecisionUpdateBody =
  z.infer<typeof rememberedApprovalDecisionUpdateSchema>
export type RememberedApprovalDecisionDescribeBody =
  z.infer<typeof rememberedApprovalDecisionDescribeSchema>

export const approvalOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    ApprovalRequest: approvalRequestSchema,
    ApprovalResolutionResponse: approvalResolutionResponseSchema,
    RememberedApprovalDecision: rememberedApprovalDecisionSchema,
    RememberedApprovalDecisionInput: rememberedApprovalDecisionInputSchema,
    RememberedApprovalDecisionMatch: rememberedApprovalDecisionMatchSchema,
    RememberedApprovalDecisionUpdate: rememberedApprovalDecisionUpdateSchema,
    RememberedApprovalDecisionDescribe: rememberedApprovalDecisionDescribeSchema,
    RememberedApprovalDecisionDescribeResponse:
      rememberedApprovalDecisionDescribeResponseSchema,
    RememberedApprovalListResponse: rememberedApprovalListResponseSchema,
    RememberedApprovalMutationResponse: rememberedApprovalMutationResponseSchema,
    RememberedApprovalDeleteResponse: rememberedApprovalDeleteResponseSchema,
    RememberedApprovalClearResponse: rememberedApprovalClearResponseSchema,
  },
})

export const approvalOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/approvals/respond': {
    post: {
      summary: 'Resolve approval request',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ApprovalRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ApprovalResolutionResponse'),
        400: { description: 'Invalid request' },
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/approvals/resume': {
    post: {
      summary: 'Resume approval checkpoint',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ApprovalRequest'),
          },
        },
      },
      responses: {
        200: {
          description: 'SSE stream',
          content: {
            'text/event-stream': {},
          },
        },
        400: { description: 'Invalid request' },
        404: { description: 'Not found' },
        503: { description: 'Service unavailable or agent run capacity exhausted' },
      },
    },
  },
  '/api/v1/approvals/decisions': {
    get: {
      summary: 'List remembered approval decisions',
      tags: ['Sessions'],
      responses: {
        200: openApiJsonResponseRef('RememberedApprovalListResponse'),
        503: { description: 'Service unavailable' },
      },
    },
    post: {
      summary: 'Create or replace a remembered approval decision',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('RememberedApprovalDecisionInput'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('RememberedApprovalMutationResponse'),
        400: { description: 'Invalid request' },
        503: { description: 'Service unavailable' },
      },
    },
    patch: {
      summary: 'Update a remembered approval decision',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('RememberedApprovalDecisionUpdate'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('RememberedApprovalMutationResponse'),
        400: { description: 'Invalid request' },
        404: { description: 'Not found' },
        503: { description: 'Service unavailable' },
      },
    },
    delete: {
      summary: 'Clear or delete remembered approval decisions',
      tags: ['Sessions'],
      responses: {
        200: {
          description: 'Clear or exact delete result',
          content: {
            'application/json': {
              schema: {
                oneOf: [
                  openApiSchemaRef('RememberedApprovalClearResponse'),
                  openApiSchemaRef('RememberedApprovalDeleteResponse'),
                ],
              },
            },
          },
        },
        400: { description: 'Invalid request' },
        503: { description: 'Service unavailable' },
      },
    },
  },
  '/api/v1/approvals/decisions/describe': {
    post: {
      summary: 'Describe the remembered approval rule for a tool input',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('RememberedApprovalDecisionDescribe'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('RememberedApprovalDecisionDescribeResponse'),
        400: { description: 'Invalid request' },
        503: { description: 'Service unavailable' },
      },
    },
  },
}
