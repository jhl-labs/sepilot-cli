import {
  DEFAULT_WEBHOOK_SECURITY_POLICY,
  type SepilotdConfig,
  type WebhookSecurityPolicyConfig,
  type WebhookSecurityPolicyOverrideConfig,
} from '../../config/schema.js'

export type WebhookSecurityPolicyFieldSource =
  | 'default'
  | 'global'
  | 'channel-type'

export interface ResolvedWebhookSecurityPolicy
  extends WebhookSecurityPolicyConfig {
  source: {
    signatureMaxSkewSeconds: WebhookSecurityPolicyFieldSource
    verificationUnavailableStatus: WebhookSecurityPolicyFieldSource
  }
}

export function resolveWebhookSecurityPolicyWithSource(
  config?: Pick<SepilotdConfig, 'security'> | null,
  channelType?: string,
): ResolvedWebhookSecurityPolicy {
  const basePolicy = config?.security?.webhooks
    ?? DEFAULT_WEBHOOK_SECURITY_POLICY
  const byChannelType = (basePolicy.byChannelType ?? {}) as Record<
    string,
    WebhookSecurityPolicyOverrideConfig
  >
  const channelPolicy = channelType
    ? (byChannelType[channelType] ?? {})
    : {}
  const hasChannelSignatureOverride = Object.hasOwn(
    channelPolicy,
    'signatureMaxSkewSeconds',
  )
  const hasChannelVerificationStatusOverride =
    Object.hasOwn(
      channelPolicy,
      'verificationUnavailableStatus',
    )
  const hasGlobalSignatureOverride =
    Object.hasOwn(
      basePolicy,
      'signatureMaxSkewSeconds',
    )
    && basePolicy.signatureMaxSkewSeconds !== undefined
    &&
    basePolicy.signatureMaxSkewSeconds
    !== DEFAULT_WEBHOOK_SECURITY_POLICY.signatureMaxSkewSeconds
  const hasGlobalVerificationStatusOverride =
    Object.hasOwn(
      basePolicy,
      'verificationUnavailableStatus',
    )
    && basePolicy.verificationUnavailableStatus !== undefined
    &&
    basePolicy.verificationUnavailableStatus
    !== DEFAULT_WEBHOOK_SECURITY_POLICY.verificationUnavailableStatus

  return {
    signatureMaxSkewSeconds:
      channelPolicy.signatureMaxSkewSeconds
      ?? basePolicy.signatureMaxSkewSeconds
      ?? DEFAULT_WEBHOOK_SECURITY_POLICY.signatureMaxSkewSeconds,
    verificationUnavailableStatus:
      channelPolicy.verificationUnavailableStatus
      ?? basePolicy.verificationUnavailableStatus
      ?? DEFAULT_WEBHOOK_SECURITY_POLICY.verificationUnavailableStatus,
    byChannelType,
    source: {
      signatureMaxSkewSeconds: hasChannelSignatureOverride
        ? 'channel-type'
        : hasGlobalSignatureOverride
          ? 'global'
          : 'default',
      verificationUnavailableStatus: hasChannelVerificationStatusOverride
        ? 'channel-type'
        : hasGlobalVerificationStatusOverride
          ? 'global'
          : 'default',
    },
  }
}

export function resolveWebhookSecurityPolicy(
  config?: Pick<SepilotdConfig, 'security'> | null,
  channelType?: string,
): WebhookSecurityPolicyConfig {
  const resolved = resolveWebhookSecurityPolicyWithSource(config, channelType)
  return {
    signatureMaxSkewSeconds: resolved.signatureMaxSkewSeconds,
    verificationUnavailableStatus: resolved.verificationUnavailableStatus,
    byChannelType: resolved.byChannelType,
  }
}
