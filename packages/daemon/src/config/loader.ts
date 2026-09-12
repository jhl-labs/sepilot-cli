import YAML from 'yaml'
import { configSchema, type SepilotdConfig } from './schema.js'
import { migrateConfig } from './config-migrate.js'

const ENV_VAR_RE = /\$\{([^}]+)\}/g
const WHOLE_ENV_VAR_RE = /^\$\{([^}]+)\}$/

/**
 * Text-level `${VAR}` substitution. Retained for compatibility; unresolved
 * vars collapse to an empty string. Prefer `parseConfig`, which substitutes
 * at the parsed-object level (safe for secrets containing YAML metacharacters)
 * and drops whole-value unresolved fields instead of injecting ''.
 */
export function substituteEnvVars(
  text: string,
  env: Record<string, string | undefined> = process.env,
): string {
  return text.replace(ENV_VAR_RE, (_, varName) => env[varName] ?? '')
}

interface UnresolvedEnvRef {
  path: string
  name: string
}

/**
 * Recursively substitute `${VAR}` references inside a parsed config value.
 *
 * - A string that is exactly one unresolved `${VAR}` is dropped (returns
 *   undefined) so an optional URL/field becomes *unset* rather than an empty
 *   string that fails `.url()` and throws the whole config into degraded mode.
 * - A string with inline `${VAR}` mixed with other text substitutes '' for an
 *   unresolved var (can't drop a partial value) and records a diagnostic.
 * - Object keys / array elements whose value resolves to undefined are omitted.
 */
function deepSubstituteEnv(
  value: unknown,
  env: Record<string, string | undefined>,
  unresolved: UnresolvedEnvRef[],
  path: string,
): unknown {
  if (typeof value === 'string') {
    const whole = value.match(WHOLE_ENV_VAR_RE)
    if (whole) {
      const name = whole[1]!
      const resolved = env[name]
      if (resolved === undefined) {
        unresolved.push({ path: path || '(root)', name })
        return undefined
      }
      return resolved
    }
    return value.replace(ENV_VAR_RE, (_, name: string) => {
      const resolved = env[name]
      if (resolved === undefined) {
        unresolved.push({ path: path || '(root)', name })
        return ''
      }
      return resolved
    })
  }
  if (Array.isArray(value)) {
    const out: unknown[] = []
    value.forEach((item, index) => {
      const substituted = deepSubstituteEnv(item, env, unresolved, `${path}[${index}]`)
      if (substituted !== undefined) out.push(substituted)
    })
    return out
  }
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      const nextPath = path ? `${path}.${key}` : key
      const substituted = deepSubstituteEnv(entry, env, unresolved, nextPath)
      if (substituted !== undefined) out[key] = substituted
    }
    return out
  }
  return value
}

export function parseConfig(
  yamlText: string,
  env: Record<string, string | undefined> = process.env,
): SepilotdConfig {
  const raw = YAML.parse(yamlText)
  const unresolved: UnresolvedEnvRef[] = []
  const substituted = deepSubstituteEnv(raw, env, unresolved, '')
  // Version-migrate before schema validation so a config written by an older
  // daemon (or with a missing version marker) is carried forward instead of
  // failing the pinned `version` literal. Rethrows the migration's own clear
  // error for a downgrade (config from a newer daemon).
  const migrated = migrateConfig(substituted)
  try {
    return configSchema.parse(migrated)
  } catch (error) {
    if (unresolved.length > 0) {
      // Turn a confusing downstream schema error (e.g. `.url()` on a blank
      // string) into a targeted per-field diagnostic naming the missing env
      // var and the config path that referenced it.
      const details = unresolved
        .map((ref) => `  - ${ref.path}: unresolved \${${ref.name}} (set ${ref.name} in the environment or daemon-managed env file)`)
        .join('\n')
      const underlying = error instanceof Error ? error.message : String(error)
      throw new Error(
        `Config load failed because of unresolved environment variables:\n${details}\n\nUnderlying validation error:\n${underlying}`,
      )
    }
    throw error
  }
}
