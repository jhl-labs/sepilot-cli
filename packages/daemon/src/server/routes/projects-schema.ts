import { z } from 'zod'
import { safeIdSchema } from '../../utils/safe-id.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

export const projectSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  instructions: z.string(),
  workingDirectory: z.string(),
  sessionIds: z.array(z.string()),
  fileIds: z.array(z.string()),
  createdAt: z.string(),
  updatedAt: z.string(),
})

export const createProjectBodySchema = z.object({
  name: z.string().min(1),
  description: z.string().optional(),
  instructions: z.string().optional(),
  workingDirectory: z.string().optional(),
})

export const updateProjectBodySchema = z.object({
  name: z.string().min(1).optional(),
  description: z.string().optional(),
  instructions: z.string().optional(),
  workingDirectory: z.string().optional(),
  sessionIds: z.array(z.string()).optional(),
  fileIds: z.array(z.string()).optional(),
})

export const addProjectSessionBodySchema = z.object({
  sessionId: z.string().min(1).optional(),
})

export const projectIdParamsSchema = z.object({
  id: safeIdSchema,
})

export const projectListResponseSchema = z.object({
  data: z.array(projectSchema),
})

export const projectResponseSchema = z.object({
  data: projectSchema,
})

export type CreateProjectBody = z.infer<typeof createProjectBodySchema>
export type UpdateProjectBody = z.infer<typeof updateProjectBodySchema>
export type AddProjectSessionBody = z.infer<typeof addProjectSessionBodySchema>
export type ProjectIdParams = z.infer<typeof projectIdParamsSchema>

export const projectOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    Project: projectSchema,
    CreateProjectRequest: createProjectBodySchema,
    UpdateProjectRequest: updateProjectBodySchema,
    AddProjectSessionRequest: addProjectSessionBodySchema,
    ProjectListResponse: projectListResponseSchema,
    ProjectResponse: projectResponseSchema,
  },
  parameters: {
    ProjectIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: projectIdParamsSchema.shape.id,
    },
  },
})

export const projectOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/projects': {
    get: {
      summary: 'List projects',
      tags: ['Projects'],
      responses: { 200: openApiJsonResponseRef('ProjectListResponse') },
    },
    post: {
      summary: 'Create project',
      tags: ['Projects'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('CreateProjectRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('ProjectResponse') },
    },
  },
  '/api/v1/projects/{id}': {
    get: {
      summary: 'Get project',
      tags: ['Projects'],
      parameters: [openApiParameterRef('ProjectIdParam')],
      responses: {
        200: openApiJsonResponseRef('ProjectResponse'),
        404: { description: 'Not found' },
      },
    },
    put: {
      summary: 'Update project',
      tags: ['Projects'],
      parameters: [openApiParameterRef('ProjectIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('UpdateProjectRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ProjectResponse'),
        404: { description: 'Not found' },
      },
    },
    delete: {
      summary: 'Delete project',
      tags: ['Projects'],
      parameters: [openApiParameterRef('ProjectIdParam')],
      responses: { 204: { description: 'Deleted' } },
    },
  },
  '/api/v1/projects/{id}/sessions': {
    post: {
      summary: 'Attach session to project',
      tags: ['Projects'],
      parameters: [openApiParameterRef('ProjectIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('AddProjectSessionRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ProjectResponse'),
        404: { description: 'Not found' },
      },
    },
  },
}
