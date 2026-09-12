import type { SecretVault } from '../security/secret-vault.js'

const SECRET_PATTERN = /\{secret\.([a-zA-Z0-9_.-]+)\}/g

export type SecretResolutionErrorCode = 'SECRET_VAULT_UNAVAILABLE' | 'SECRET_NOT_FOUND'

export class SecretResolutionError extends Error {
  constructor(
    readonly code: SecretResolutionErrorCode,
    readonly secretKey: string,
    message: string,
  ) {
    super(message)
    this.name = 'SecretResolutionError'
  }
}

export function resolveSecretPlaceholders(
  value: string,
  vault: SecretVault | null,
): string {
  SECRET_PATTERN.lastIndex = 0
  if (!SECRET_PATTERN.test(value)) return value
  SECRET_PATTERN.lastIndex = 0

  return value.replace(SECRET_PATTERN, (_match, key) => {
    if (!vault) {
      throw new SecretResolutionError(
        'SECRET_VAULT_UNAVAILABLE',
        key,
        `Secret ${key} cannot be resolved because the secret vault is unavailable`,
      )
    }
    const secret = vault.get(key)
    if (secret === null) {
      throw new SecretResolutionError(
        'SECRET_NOT_FOUND',
        key,
        `Secret ${key} not found`,
      )
    }
    return secret
  })
}
