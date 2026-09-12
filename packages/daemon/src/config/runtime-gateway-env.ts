import type { SepilotdConfig } from './schema.js'

export const GATEWAY_RUNTIME_URL_ENV = 'SEPILOTD_GATEWAY_URL'

export class RuntimeGatewayEnvironmentError extends Error {
  readonly code = 'RUNTIME_GATEWAY_ENV_INVALID' as const

  constructor(message: string) {
    super(message)
    this.name = 'RuntimeGatewayEnvironmentError'
  }
}

function parseGatewayOrigin(raw: string): string {
  let parsed: URL
  try {
    parsed = new URL(raw)
  } catch {
    throw new RuntimeGatewayEnvironmentError(
      `${GATEWAY_RUNTIME_URL_ENV} must be an http(s) origin.`,
    )
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new RuntimeGatewayEnvironmentError(
      `${GATEWAY_RUNTIME_URL_ENV} must use http or https.`,
    )
  }
  if (parsed.username || parsed.password) {
    throw new RuntimeGatewayEnvironmentError(
      `${GATEWAY_RUNTIME_URL_ENV} must not contain credentials; use the gateway token file.`,
    )
  }
  if (parsed.pathname !== '/' || parsed.search || parsed.hash) {
    throw new RuntimeGatewayEnvironmentError(
      `${GATEWAY_RUNTIME_URL_ENV} must be an origin without a path, query, or fragment.`,
    )
  }

  return parsed.origin
}

/**
 * Apply an operator-owned gateway endpoint without rewriting config.yaml.
 * Container and service deployments need a routable origin while local
 * profiles retain their persisted loopback default.
 */
export function applyRuntimeGatewayEnvironment(
  config: SepilotdConfig,
  env: Record<string, string | undefined> = process.env,
): SepilotdConfig {
  const raw = env[GATEWAY_RUNTIME_URL_ENV]?.trim()
  if (!raw) return config

  const url = parseGatewayOrigin(raw)
  if (config.gateway.url === url) return config
  return {
    ...config,
    gateway: { ...config.gateway, url },
  }
}
