import {
  createPrivateKey,
  createPublicKey,
  sign as cryptoSign,
  verify as cryptoVerify,
} from 'node:crypto'

export type DelegationAssertionKind =
  | 'claim'
  | 'claim_release'
  | 'status'
  | 'answer'
  | 'cancel'

export interface DelegationAssertionFields {
  kind: DelegationAssertionKind
  delegationId: string
  deviceId: string
  claimId?: string
  detail?: string
}

const PAYLOAD_PREFIX = 'sepilotd-delegation'

export function buildDelegationAssertionPayload(
  fields: DelegationAssertionFields,
): string {
  return [
    PAYLOAD_PREFIX,
    fields.kind,
    fields.delegationId,
    fields.deviceId,
    fields.claimId ?? '',
    fields.detail ?? '',
  ].join('|')
}

export interface DelegationSigner {
  readonly deviceId: string
  sign(payload: string): string
}

export interface DelegationKeyResolver {
  getPublicKey(deviceId: string): string | null
}

export function createEd25519DelegationSigner(
  deviceId: string,
  privateKeyPem: string,
): DelegationSigner {
  const key = createPrivateKey(privateKeyPem)
  return {
    deviceId,
    sign(payload: string): string {
      return cryptoSign(null, Buffer.from(payload, 'utf8'), key).toString('base64')
    },
  }
}

export function verifyDelegationSignature(
  publicKeyPem: string,
  payload: string,
  signatureB64: string,
): boolean {
  try {
    return cryptoVerify(
      null,
      Buffer.from(payload, 'utf8'),
      createPublicKey(publicKeyPem),
      Buffer.from(signatureB64, 'base64'),
    )
  } catch {
    return false
  }
}

export interface SignedDelegationMarkerFields {
  deviceId?: string
  signature?: string
}

export function verifyDelegationAssertion(
  resolver: DelegationKeyResolver,
  fields: DelegationAssertionFields,
  signature: string | undefined,
): boolean {
  if (!fields.deviceId || !signature) return false
  const publicKey = resolver.getPublicKey(fields.deviceId)
  if (!publicKey) return false
  return verifyDelegationSignature(publicKey, buildDelegationAssertionPayload(fields), signature)
}
