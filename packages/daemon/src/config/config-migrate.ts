export const CURRENT_CONFIG_VERSION = 1

interface ConfigMigration {
  from: number
  to: number
  description: string
  up(config: Record<string, unknown>): Record<string, unknown>
}

const MIGRATIONS: ConfigMigration[] = []

export function migrateConfig(raw: unknown): unknown {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return raw
  }
  const config: Record<string, unknown> = { ...(raw as Record<string, unknown>) }

  const rawVersion = config.version
  let version: number
  if (rawVersion === undefined || rawVersion === null) {
    version = 1
  } else if (typeof rawVersion === 'number' && Number.isInteger(rawVersion)) {
    version = rawVersion
  } else {
    throw new Error(
      `Config "version" must be an integer, got ${JSON.stringify(rawVersion)}.`,
    )
  }

  if (version > CURRENT_CONFIG_VERSION) {
    throw new Error(
      `Config version ${version} is newer than this daemon supports (max ${CURRENT_CONFIG_VERSION}). `
      + 'This config was written by a newer sepilotd — upgrade the daemon or restore a compatible config instead of letting it load as an empty config.',
    )
  }

  let current = config
  current.version = version
  while (version < CURRENT_CONFIG_VERSION) {
    const migration = MIGRATIONS.find((m) => m.from === version)
    if (!migration) {
      throw new Error(
        `No config migration registered from version ${version} toward ${CURRENT_CONFIG_VERSION}.`,
      )
    }
    current = migration.up(current)
    current.version = migration.to
    version = migration.to
  }
  return current
}
