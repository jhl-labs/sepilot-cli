import {
  createPrivateKey,
  createPublicKey,
  sign as cryptoSign,
  verify as cryptoVerify,
} from 'node:crypto'

/**
 * Device-authenticated delegation assertions.
 *
 * Delegation rides over a shared comment transport (a Gateway "delegation"
 * ticket) that every paired device can read and write. Without authentication
 * any device can forge a `[claim]`/`[status]`/`[cancel]` comment addressed with
 * another device's free-text name and hijack or spoof a task (PLAN_021 D1).
 *
 * The fix: a device signs its claim/status assertions with its own ed25519
 * private key (the local `~/.sepilotd/security/device.key`), and consumers
 * verify the signature against the paired device's public key held in the
 * DevicePairingRegistry. A device only holds its own private key, so it can
 * never mint a valid assertion for a device id it does not own.
 */

export type DelegationAssertionKind =
  | 'claim'
  | 'claim_release'
  | 'status'
  | 'answer'
  | 'cancel'

export interface DelegationAssertionFields {
  kind: DelegationAssertionKind
  delegationId: string
  /** Authenticated device id of the writer (the key that signs). */
  deviceId: string
  claimId?: string
  /** Kind-specific discriminator: status value, question id, etc. */
  detail?: string
}

const PAYLOAD_PREFIX = 'sepilotd-delegation'

// Canonical, unambiguous payload. All fields are pipe-joined and empty-string
// filled so a missing claimId can never collide with a present one.
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
  /** Authenticated device id embedded into every assertion this signer writes. */
  readonly deviceId: string
  sign(payload: string): string
}

export interface DelegationKeyResolver {
  /** Paired device public key (spki PEM), or null when the id is unknown. */
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

/**
 * Verify a parsed, signed delegation assertion against the resolver's paired
 * key. Returns false (fail-closed) when the device id/signature is missing, the
 * device is unknown, or the signature does not verify.
 */
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
