/**
 * Config version migration.
 *
 * `configSchema` pins `version` to the current literal, so before PLAN_012 a
 * schema bump would make every older config file fail validation and (via the
 * fail-loud loader) refuse to start — there was no path to carry a v(N-1) file
 * forward. `migrateConfig` runs before `configSchema.parse`: it back-fills
 * missing fields and bumps the version step by step up to the current one, and
 * guards against a config written by a *newer* daemon (downgrade) with a clear
 * error instead of a confusing schema failure.
 *
 * When you bump `CURRENT_CONFIG_VERSION`, add a migration entry describing the
 * v(N-1) -> vN transformation (back-fill new required fields with their
 * defaults, rename/move moved fields). Keep each `up` pure and total.
 */

export const CURRENT_CONFIG_VERSION = 1

interface ConfigMigration {
  from: number
  to: number
  description: string
  up(config: Record<string, unknown>): Record<string, unknown>
}

// No migrations yet — v1 is the current (and earliest) schema. Future bumps add
// entries here, e.g. { from: 1, to: 2, description: '...', up(c) { ... } }.
const MIGRATIONS: ConfigMigration[] = []

/**
 * Normalize a raw parsed config object to the current schema version. Non-object
 * input is returned untouched so the caller's schema parse produces the normal
 * validation error. Throws on a non-integer version, an unknown upgrade step, or
 * a version newer than this daemon supports.
 */
export function migrateConfig(raw: unknown): unknown {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return raw
  }
  const config: Record<string, unknown> = { ...(raw as Record<string, unknown>) }

  const rawVersion = config.version
  let version: number
  if (rawVersion === undefined || rawVersion === null) {
    // A pre-versioned config predates the version marker. Adopt the earliest
    // known version so the migration chain can carry it forward.
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
