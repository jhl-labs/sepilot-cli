import { z } from 'zod'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponse,
  openApiJsonResponseRef,
  openApiParameterRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

export const deviceSchema = z.object({
  id: z.string(),
  name: z.string(),
  role: z.enum(['desktop', 'server', 'edge']),
  status: z.enum(['online', 'offline', 'busy']),
  lastSeen: z.string(),
  capabilities: z.array(z.string()),
  platform: z.string(),
  arch: z.string(),
  trust: z.enum(['local', 'paired']),
  publicKeyFingerprint: z.string().optional(),
  pairedAt: z.string().datetime().optional(),
})

export const devicePairingChallengeRequestSchema = z.object({
  ttlSeconds: z.number().int().min(30).max(3600).optional(),
})

export const devicePairingChallengeSchema = z.object({
  challengeId: z.string(),
  challenge: z.string(),
  payloadToSign: z.string(),
  expiresAt: z.string().datetime(),
})

export const devicePairingChallengeResponseSchema = z.object({
  data: devicePairingChallengeSchema,
})

export const devicePairingCompleteRequestSchema = z.object({
  challengeId: z.string().min(1),
  device: z.object({
    id: z.string().min(1),
    name: z.string().min(1),
    role: z.enum(['desktop', 'server', 'edge']),
  }),
  publicKey: z.string().min(1),
  signature: z.string().min(1),
})

export const deviceMutationResponseSchema = z.object({
  data: z.object({
    id: z.string(),
    revoked: z.boolean().optional(),
  }),
})

export const deviceIdParamsSchema = z.object({
  id: z.string().min(1),
})

export const devicesOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    Device: deviceSchema,
    DeviceListResponse: z.object({
      data: z.array(deviceSchema),
    }),
    DevicePairingChallengeRequest: devicePairingChallengeRequestSchema,
    DevicePairingChallenge: devicePairingChallengeSchema,
    DevicePairingChallengeResponse: devicePairingChallengeResponseSchema,
    DevicePairingCompleteRequest: devicePairingCompleteRequestSchema,
    DeviceMutationResponse: deviceMutationResponseSchema,
  },
  parameters: {
    DeviceIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: deviceIdParamsSchema.shape.id,
    },
  },
})

export const devicesOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/devices': {
    get: {
      summary: 'List devices',
      tags: ['Devices'],
      responses: { 200: openApiJsonResponseRef('DeviceListResponse') },
    },
  },
  '/api/v1/devices/pairing/challenges': {
    post: {
      summary: 'Issue a device pairing challenge',
      tags: ['Devices'],
      requestBody: {
        required: false,
        content: {
          'application/json': {
            schema: {
              $ref: '#/components/schemas/DevicePairingChallengeRequest',
            },
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('DevicePairingChallengeResponse'),
      },
    },
  },
  '/api/v1/devices/pairing/complete': {
    post: {
      summary: 'Complete device pairing with a signed challenge',
      tags: ['Devices'],
      requestBody: {
        required: true,
        content: {
          'application/json': {
            schema: {
              $ref: '#/components/schemas/DevicePairingCompleteRequest',
            },
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('DeviceMutationResponse'),
        400: openApiJsonResponse({
          type: 'object',
          properties: {
            error: {
              type: 'object',
              properties: {
                code: { type: 'string' },
                message: { type: 'string' },
              },
            },
          },
        }),
      },
    },
  },
  '/api/v1/devices/{id}/pairing': {
    delete: {
      summary: 'Revoke a paired device',
      tags: ['Devices'],
      parameters: [openApiParameterRef('DeviceIdParam')],
      responses: {
        200: openApiJsonResponseRef('DeviceMutationResponse'),
        404: openApiJsonResponse({
          type: 'object',
          properties: {
            error: {
              type: 'object',
              properties: {
                code: { type: 'string' },
                message: { type: 'string' },
              },
            },
          },
        }),
      },
    },
  },
}
