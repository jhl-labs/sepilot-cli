import { z } from 'zod'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

export const fileIdParamsSchema = z.object({
  id: z.string().min(1),
})

export const uploadedFileMetadataSchema = z.object({
  id: z.string(),
  filename: z.string(),
  mimeType: z.string(),
  size: z.number().int().nonnegative(),
  uploadedAt: z.string(),
})

export const uploadedFileListResponseSchema = z.object({
  data: z.object({
    files: z.array(uploadedFileMetadataSchema),
  }),
})

export const uploadedFileResponseSchema = z.object({
  data: uploadedFileMetadataSchema,
})

export const base64ContentSourceSchema = z.object({
  type: z.literal('base64'),
  mediaType: z.string(),
  data: z.string(),
})

export const urlContentSourceSchema = z.object({
  type: z.literal('url'),
  mediaType: z.string(),
  data: z.string().url(),
})

export const textContentPartSchema = z.object({
  type: z.literal('text'),
  text: z.string(),
})

export const imageContentPartSchema = z.object({
  type: z.literal('image'),
  source: z.union([base64ContentSourceSchema, urlContentSourceSchema]),
})

export const documentContentPartSchema = z.object({
  type: z.literal('document'),
  source: z.union([base64ContentSourceSchema, urlContentSourceSchema]),
})

export const uploadedFileContentPartResponseSchema = z.object({
  data: z.union([
    textContentPartSchema,
    imageContentPartSchema,
    documentContentPartSchema,
  ]),
})

export const uploadedFileIndexRequestSchema = z.object({
  title: z.string().min(1).optional(),
  path: z.string().optional(),
  tags: z.array(z.string()).optional(),
  ocr: z.boolean().optional(),
  ocrLanguages: z.array(z.string().min(1)).min(1).optional(),
  ocrMaxPages: z.number().int().min(1).max(500).optional(),
})

// /files/search query schema. limit is z.coerce.number to absorb the
// stringly-typed query value; route still clamps to [1, 32] inline so
// a regression in zod limits won't widen the cap.
export const filesSearchQuerySchema = z.object({
  q: z.string().optional(),
  cwd: z.string().min(1),
  limit: z.coerce.number().int().min(1).max(32).optional(),
})

export type FileIdParams = z.infer<typeof fileIdParamsSchema>
export type UploadedFileIndexBody = z.infer<typeof uploadedFileIndexRequestSchema>
export type FilesSearchQuery = z.infer<typeof filesSearchQuerySchema>

export const fileOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    UploadedFileMetadata: uploadedFileMetadataSchema,
    UploadedFileListResponse: uploadedFileListResponseSchema,
    UploadedFileResponse: uploadedFileResponseSchema,
    UploadedFileContentPartResponse: uploadedFileContentPartResponseSchema,
    UploadedFileIndexRequest: uploadedFileIndexRequestSchema,
  },
  parameters: {
    FileIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: fileIdParamsSchema.shape.id,
    },
  },
})

export const fileOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/files/upload': {
    post: {
      summary: 'Upload files',
      tags: ['Files'],
      responses: { 200: openApiJsonResponseRef('UploadedFileListResponse') },
    },
  },
  '/api/v1/files/{id}': {
    get: {
      summary: 'Get uploaded file metadata',
      tags: ['Files'],
      parameters: [openApiParameterRef('FileIdParam')],
      responses: {
        200: openApiJsonResponseRef('UploadedFileResponse'),
        404: { description: 'Not found' },
      },
    },
    delete: {
      summary: 'Delete uploaded file',
      tags: ['Files'],
      parameters: [openApiParameterRef('FileIdParam')],
      responses: { 204: { description: 'Deleted' }, 404: { description: 'Not found' } },
    },
  },
  '/api/v1/files/{id}/download': {
    get: {
      summary: 'Download uploaded file bytes',
      tags: ['Files'],
      parameters: [openApiParameterRef('FileIdParam')],
      responses: {
        200: { description: 'File bytes' },
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/files/{id}/content-part': {
    get: {
      summary: 'Resolve uploaded file to content part',
      tags: ['Files'],
      parameters: [openApiParameterRef('FileIdParam')],
      responses: {
        200: openApiJsonResponseRef('UploadedFileContentPartResponse'),
        404: { description: 'Not found' },
      },
    },
  },
  '/api/v1/files/{id}/index': {
    post: {
      summary: 'Index uploaded text, PDF, or OCR-capable image file into semantic memory',
      tags: ['Files'],
      parameters: [openApiParameterRef('FileIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('UploadedFileIndexRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('MemoryDocumentResponse'),
        400: { description: 'File could not be indexed' },
        404: { description: 'Not found' },
      },
    },
  },
}
