import { createHash, createPublicKey, randomBytes, randomUUID, verify as verifySignature } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { z } from 'zod'

const pairedDeviceRecordSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  role: z.enum(['desktop', 'server', 'edge']),
  publicKey: z.string().min(1),
  publicKeyFingerprint: z.string().min(1),
  pairedAt: z.string().datetime(),
  lastSeenAt: z.string().datetime(),
})

const pairingRegistryFileSchema = z.object({
  devices: z.array(pairedDeviceRecordSchema).default([]),
})

interface PendingPairingChallenge {
  challenge: string
  expiresAtMs: number
}

export interface PairingChallenge {
  challengeId: string
  challenge: string
  payloadToSign: string
  expiresAt: string
}

export interface CompletePairingInput {
  challengeId: string
  device: {
    id: string
    name: string
    role: 'desktop' | 'server' | 'edge'
  }
  publicKey: string
  signature: string
}

export type PairedDeviceRecord = z.infer<typeof pairedDeviceRecordSchema>

export function buildDevicePairingPayload(
  challengeId: string,
  challenge: string,
): string {
  return `sepilotd-device-pairing:${challengeId}:${challenge}`
}

export function fingerprintPublicKey(publicKey: string): string {
  return createHash('sha256')
    .update(publicKey, 'utf8')
    .digest('hex')
    .slice(0, 16)
}

export class DevicePairingRegistry {
  private readonly filePath: string
  private readonly defaultChallengeTtlSeconds: number
  private readonly devices = new Map<string, PairedDeviceRecord>()
  private readonly pendingChallenges = new Map<string, PendingPairingChallenge>()

  constructor(
    filePath: string,
    options: { defaultChallengeTtlSeconds?: number } = {},
  ) {
    this.filePath = filePath
    this.defaultChallengeTtlSeconds = options.defaultChallengeTtlSeconds ?? 300
  }

  async init(): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
    try {
      const raw = await readFile(this.filePath, 'utf8')
      const parsed = pairingRegistryFileSchema.parse(JSON.parse(raw))
      this.devices.clear()
      for (const device of parsed.devices) {
        this.devices.set(device.id, device)
      }
    } catch {
      await this.persist()
    }
  }

  list(): PairedDeviceRecord[] {
    return Array.from(this.devices.values()).sort(
      (left, right) =>
        left.name.localeCompare(right.name)
        || left.id.localeCompare(right.id),
    )
  }

  get(deviceId: string): PairedDeviceRecord | null {
    return this.devices.get(deviceId) ?? null
  }

  issueChallenge(ttlSeconds = this.defaultChallengeTtlSeconds): PairingChallenge {
    const challengeId = randomUUID()
    const challenge = randomBytes(24).toString('base64url')
    const expiresAtMs = Date.now() + ttlSeconds * 1000
    this.pendingChallenges.set(challengeId, {
      challenge,
      expiresAtMs,
    })

    return {
      challengeId,
      challenge,
      payloadToSign: buildDevicePairingPayload(challengeId, challenge),
      expiresAt: new Date(expiresAtMs).toISOString(),
    }
  }

  async completePairing(
    input: CompletePairingInput,
  ): Promise<PairedDeviceRecord | null> {
    const pending = this.pendingChallenges.get(input.challengeId)
    if (!pending) {
      return null
    }
    if (pending.expiresAtMs <= Date.now()) {
      this.pendingChallenges.delete(input.challengeId)
      return null
    }

    const payload = buildDevicePairingPayload(input.challengeId, pending.challenge)
    if (!this.verifyPairingSignature(input.publicKey, payload, input.signature)) {
      return null
    }

    const now = new Date().toISOString()
    const existing = this.devices.get(input.device.id)
    const record = pairedDeviceRecordSchema.parse({
      id: input.device.id,
      name: input.device.name,
      role: input.device.role,
      publicKey: input.publicKey,
      publicKeyFingerprint: fingerprintPublicKey(input.publicKey),
      pairedAt: existing?.pairedAt ?? now,
      lastSeenAt: now,
    })

    this.devices.set(record.id, record)
    this.pendingChallenges.delete(input.challengeId)
    await this.persist()
    return record
  }

  async revoke(deviceId: string): Promise<boolean> {
    const removed = this.devices.delete(deviceId)
    if (!removed) {
      return false
    }
    await this.persist()
    return true
  }

  async touch(deviceId: string): Promise<PairedDeviceRecord | null> {
    const record = this.devices.get(deviceId)
    if (!record) {
      return null
    }
    const updated = pairedDeviceRecordSchema.parse({
      ...record,
      lastSeenAt: new Date().toISOString(),
    })
    this.devices.set(updated.id, updated)
    await this.persist()
    return updated
  }

  private verifyPairingSignature(
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

  private async persist(): Promise<void> {
    const payload = pairingRegistryFileSchema.parse({
      devices: this.list(),
    })
    await writeFile(this.filePath, JSON.stringify(payload, null, 2), 'utf8')
  }
}
