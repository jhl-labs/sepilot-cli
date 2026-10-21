import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { DelegatorSecurityOptions } from './delegator.js'
import {
  createEd25519DelegationSigner,
  type DelegationKeyResolver,
} from './delegation-signing.js'

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
