import { createPublicKey, verify as verifySignature } from 'node:crypto'
import type { RequestAuthContext } from '../auth.js'
import type { DevicePairingRegistry } from './device-pairing.js'

const WS_PAIRED_DEVICE_MAX_SKEW_MS = 5 * 60 * 1000

export interface WsClientIdentity {
  kind: 'master' | 'extension' | 'paired-device' | 'anonymous'
  label: string
  approvedBy: string
  pairedDeviceId?: string
  pairedDeviceName?: string
  pairedDeviceRole?: 'desktop' | 'server' | 'edge'
  extensionTokenId?: string
}

export interface WsClientIdentityResolution {
  ok: boolean
  identity: WsClientIdentity
  errorMessage?: string
}

export function buildWsPairedDevicePayload(
  deviceId: string,
  timestamp: string,
): string {
  return `sepilotd-ws-connect:${deviceId}:${timestamp}`
}

export function resolveWsClientIdentity(
  headers: Record<string, string>,
  authContext: RequestAuthContext | undefined,
  pairingRegistry?: DevicePairingRegistry,
): WsClientIdentityResolution {
  const pairedDeviceId = headers['x-sepilot-device-id']
  const pairedDeviceTimestamp = headers['x-sepilot-device-timestamp']
  const pairedDeviceSignature = headers['x-sepilot-device-signature']
  const hasPairingHeaders = pairedDeviceId != null
    || pairedDeviceTimestamp != null
    || pairedDeviceSignature != null

  if (hasPairingHeaders) {
    if (!pairedDeviceId || !pairedDeviceTimestamp || !pairedDeviceSignature) {
      return {
        ok: false,
        identity: anonymousWsClientIdentity(),
        errorMessage: 'Incomplete paired device headers',
      }
    }

    if (!pairingRegistry) {
      return {
        ok: false,
        identity: anonymousWsClientIdentity(),
        errorMessage: 'Paired device registry unavailable',
      }
    }

    const pairedDevice = pairingRegistry.get(pairedDeviceId)
    if (!pairedDevice) {
      return {
        ok: false,
        identity: anonymousWsClientIdentity(),
        errorMessage: `Unknown paired device: ${pairedDeviceId}`,
      }
    }

    const issuedAtMs = Date.parse(pairedDeviceTimestamp)
    if (!Number.isFinite(issuedAtMs)) {
      return {
        ok: false,
        identity: anonymousWsClientIdentity(),
        errorMessage: 'Invalid paired device timestamp',
      }
    }
    if (Math.abs(Date.now() - issuedAtMs) > WS_PAIRED_DEVICE_MAX_SKEW_MS) {
      return {
        ok: false,
        identity: anonymousWsClientIdentity(),
        errorMessage: 'Paired device timestamp is stale',
      }
    }

    const payload = buildWsPairedDevicePayload(
      pairedDeviceId,
      pairedDeviceTimestamp,
    )
    if (!verifyDetachedSignature(pairedDevice.publicKey, payload, pairedDeviceSignature)) {
      return {
        ok: false,
        identity: anonymousWsClientIdentity(),
        errorMessage: 'Invalid paired device signature',
      }
    }

    return {
      ok: true,
      identity: {
        kind: 'paired-device',
        label: pairedDevice.name,
        approvedBy: `ws:paired-device:${pairedDevice.id}`,
        pairedDeviceId: pairedDevice.id,
        pairedDeviceName: pairedDevice.name,
        pairedDeviceRole: pairedDevice.role,
      },
    }
  }

  if (authContext?.kind === 'master') {
    return {
      ok: true,
      identity: {
        kind: 'master',
        label: 'master',
        approvedBy: 'ws:master',
      },
    }
  }

  if (authContext?.kind === 'extension') {
    return {
      ok: true,
      identity: {
        kind: 'extension',
        label: authContext.label,
        approvedBy: `ws:extension:${authContext.tokenId}`,
        extensionTokenId: authContext.tokenId,
      },
    }
  }

  return {
    ok: true,
    identity: anonymousWsClientIdentity(),
  }
}

function anonymousWsClientIdentity(): WsClientIdentity {
  return {
    kind: 'anonymous',
    label: 'anonymous',
    approvedBy: 'ws:anonymous',
  }
}

function verifyDetachedSignature(
  publicKey: string,
  payload: string,
  signature: string,
): boolean {
  try {
    return verifySignature(
      null,
      Buffer.from(payload, 'utf8'),
      createPublicKey(publicKey),
      Buffer.from(signature, 'base64'),
    )
  } catch {
    return false
  }
}
