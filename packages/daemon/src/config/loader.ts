import YAML from 'yaml'
import { configSchema, type SepilotdConfig } from './schema.js'
import { migrateConfig } from './config-migrate.js'

const ENV_VAR_RE = /\$\{([^}]+)\}/g
const WHOLE_ENV_VAR_RE = /^\$\{([^}]+)\}$/

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
  const migrated = migrateConfig(substituted)
  try {
    return configSchema.parse(migrated)
  } catch (error) {
    if (unresolved.length > 0) {
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
