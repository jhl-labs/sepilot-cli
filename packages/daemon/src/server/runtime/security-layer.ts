import { join } from 'node:path'
import type { SepilotdConfig } from '../../config/schema.js'
import { SecretVault } from '../../security/secret-vault.js'
import { buildExtensionAccessTokenStore } from './extension-tokens.js'
import { buildAuditLogger } from './storage.js'
import { buildChannelAcl, buildEncryption, buildPolicyEngine } from './security.js'
import { resolveAutonomy } from './providers.js'

export interface SecurityLayer {
  policyEngine: ReturnType<typeof buildPolicyEngine>
  autonomy: ReturnType<typeof resolveAutonomy>
  auditLogger: Awaited<ReturnType<typeof buildAuditLogger>>
  extensionTokenStore: Awaited<ReturnType<typeof buildExtensionAccessTokenStore>>
  channelAcl: ReturnType<typeof buildChannelAcl>
  encryption: Awaited<ReturnType<typeof buildEncryption>>
  secretVault: SecretVault
}

/**
 * Build policy, audit, ACL, and at-rest-encryption services. Pure
 * dataDir + config dependencies; runs as early as possible so other
 * assemblers can consume auditLogger and encryption.
 */
export async function assembleSecurityLayer(args: {
  config: SepilotdConfig
  dataDir: string
}): Promise<SecurityLayer> {
  const { config, dataDir } = args

  const policyEngine = buildPolicyEngine(config, dataDir)
  const autonomy = resolveAutonomy(config)
  const auditLogger = await buildAuditLogger(dataDir)
  const extensionTokenStore = await buildExtensionAccessTokenStore(
    dataDir,
    auditLogger,
    config.device.name,
  )
  const channelAcl = buildChannelAcl(config, autonomy)
  const encryption = await buildEncryption(config, dataDir)
  const secretVault = new SecretVault(join(dataDir, 'security'), encryption, {
    allowPlaintextSecrets: config.memory.allowPlaintextSecrets,
  })
  await secretVault.load()

  return {
    policyEngine,
    autonomy,
    auditLogger,
    extensionTokenStore,
    channelAcl,
    encryption,
    secretVault,
  }
}
