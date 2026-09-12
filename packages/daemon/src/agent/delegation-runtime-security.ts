import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { DelegatorSecurityOptions } from './delegator.js'
import {
  createEd25519DelegationSigner,
  type DelegationKeyResolver,
} from './delegation-signing.js'

/**
 * Wire delegation signing/verification from on-disk material:
 *  - the local device ed25519 private key (`<dataDir>/security/device.key`)
 *    signs this device's claim/status assertions;
 *  - the paired-devices registry file (`<dataDir>/security/paired-devices.json`)
 *    resolves peer public keys for verifying inbound assertions.
 *
 * Verification is only ENFORCED when the operator sets
 * SEPILOTD_DELEGATION_REQUIRE_SIGNATURE=1, so a fleet mid-rollout (where a peer
 * does not yet sign) is not cut off. Signing is always attached when a local key
 * exists so peers can begin verifying as soon as they opt in. All reads are
 * best-effort — a missing key/registry degrades to no signer/resolver rather
 * than throwing.
 */
export function buildDelegationSecurityOptions(
  dataDir: string,
  deviceId: string,
  env: NodeJS.ProcessEnv = process.env,
): DelegatorSecurityOptions {
  const securityDir = join(dataDir, 'security')
  const options: DelegatorSecurityOptions = {}

  try {
    const privateKeyPem = readFileSync(join(securityDir, 'device.key'), 'utf8')
    if (privateKeyPem.trim()) {
      options.signer = createEd25519DelegationSigner(deviceId, privateKeyPem)
    }
  } catch {
    // No local device key — this device cannot sign; leave signer unset.
  }

  const registryPath = join(securityDir, 'paired-devices.json')
  const resolver: DelegationKeyResolver = {
    getPublicKey(id: string): string | null {
      try {
        const raw = readFileSync(registryPath, 'utf8')
        const parsed = JSON.parse(raw) as { devices?: Array<{ id?: string; publicKey?: string }> }
        const match = parsed.devices?.find((device) => device.id === id)
        return typeof match?.publicKey === 'string' ? match.publicKey : null
      } catch {
        return null
      }
    },
  }
  options.keyResolver = resolver
  options.requireSignature = env.SEPILOTD_DELEGATION_REQUIRE_SIGNATURE === '1'

  return options
}
