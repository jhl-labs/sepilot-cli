import { readFile } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import type { FastifyInstance } from 'fastify'
import type { AuditEvent } from '@sepilotd/core'
import type { z } from 'zod'
import '../fastify-types.js'
import { parseRequestInput } from './utils.js'
import { fingerprintPublicKey } from '../runtime/device-pairing.js'
import {
  devicePairingChallengeRequestSchema,
  devicePairingCompleteRequestSchema,
  deviceIdParamsSchema,
} from './devices-schema.js'

export {
  devicesOpenApiComponents,
  devicesOpenApiOverrides,
} from './devices-schema.js'


export async function devicesRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  app.get('/devices', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const localPublicKey = await loadLocalDevicePublicKey(runtime.dataDir)
    const device = {
      id: runtime.config.device.id,
      name: runtime.config.device.name,
      role: runtime.config.device.role,
      status: 'online',
      lastSeen: new Date().toISOString(),
      capabilities: detectCapabilities(),
      platform: process.platform,
      arch: process.arch,
      trust: 'local' as const,
      ...(localPublicKey
        ? { publicKeyFingerprint: fingerprintPublicKey(localPublicKey) }
        : {}),
    }
    const pairedDevices = (runtime.devicePairingRegistry?.list() ?? [])
      .map((paired) => ({
        id: paired.id,
        name: paired.name,
        role: paired.role,
        status: 'offline' as const,
        lastSeen: paired.lastSeenAt,
        capabilities: [],
        platform: 'unknown',
        arch: 'unknown',
        trust: 'paired' as const,
        publicKeyFingerprint: paired.publicKeyFingerprint,
        pairedAt: paired.pairedAt,
      }))
    return { data: [device, ...pairedDevices] }
  })

  app.post('/devices/pairing/challenges', async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const body = parseRequestInput(
      reply,
      devicePairingChallengeRequestSchema,
      request.body ?? {},
      'Invalid device pairing challenge request',
    )
    if (!body) return reply

    const pairingRegistry = runtime.devicePairingRegistry
    if (!pairingRegistry) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Device pairing registry not initialized',
        },
      })
    }

    const challenge = pairingRegistry.issueChallenge(body.ttlSeconds)
    await runtime.auditLogger.log({
      timestamp: new Date().toISOString(),
      event: 'device.pairing.challenge.issued',
      device: runtime.config.device.name,
      challengeId: challenge.challengeId,
      expiresAt: challenge.expiresAt,
    } satisfies AuditEvent)
    return {
      data: challenge,
    }
  })

  app.post('/devices/pairing/complete', async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const body = parseRequestInput(
      reply,
      devicePairingCompleteRequestSchema,
      request.body,
      'Invalid device pairing completion request',
    )
    if (!body) return reply

    const pairingRegistry = runtime.devicePairingRegistry
    if (!pairingRegistry) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Device pairing registry not initialized',
        },
      })
    }

    const paired = await pairingRegistry.completePairing(body)
    if (!paired) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid, expired, or unverifiable device pairing challenge',
        },
      })
    }

    await runtime.auditLogger.log({
      timestamp: new Date().toISOString(),
      event: 'device.pairing.completed',
      device: runtime.config.device.name,
      pairedDeviceId: paired.id,
      pairedDeviceName: paired.name,
      pairedDeviceRole: paired.role,
      publicKeyFingerprint: paired.publicKeyFingerprint,
    } satisfies AuditEvent)

    return {
      data: {
        id: paired.id,
      },
    }
  })

  app.delete<{ Params: z.infer<typeof deviceIdParamsSchema> }>(
    '/devices/:id/pairing',
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized',
          },
        })
      }

      const params = parseRequestInput(
        reply,
        deviceIdParamsSchema,
        request.params,
        'Invalid device id',
      )
      if (!params) return reply

      if (params.id === runtime.config.device.id) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Local device pairing cannot be revoked',
          },
        })
      }

      const pairingRegistry = runtime.devicePairingRegistry
      if (!pairingRegistry) {
        return reply.status(503).send({
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: 'Device pairing registry not initialized',
          },
        })
      }

      const revoked = await pairingRegistry.revoke(params.id)
      if (!revoked) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `Paired device not found: ${params.id}`,
          },
        })
      }

      await runtime.auditLogger.log({
        timestamp: new Date().toISOString(),
        event: 'device.pairing.revoked',
        device: runtime.config.device.name,
        pairedDeviceId: params.id,
      } satisfies AuditEvent)

      return {
        data: {
          id: params.id,
          revoked: true,
        },
      }
    },
  )
}

async function loadLocalDevicePublicKey(dataDir: string): Promise<string | null> {
  try {
    return await readFile(`${dataDir}/security/device.pub`, 'utf8')
  } catch {
    return null
  }
}

function detectCapabilities(): string[] {
  const caps: string[] = []
  caps.push('terminal')
  caps.push('filesystem')
  // Check for docker
  try {
    execFileSync('docker', ['--version'], { stdio: 'ignore' })
    caps.push('docker')
  } catch { /* docker not installed */ }
  return caps
}
