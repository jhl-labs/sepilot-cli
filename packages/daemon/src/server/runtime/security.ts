import { AutonomyLevel, type PolicyConfig, type PolicyRule, type ToolPolicyMode } from '@sepilotd/core'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import { randomBytes } from 'node:crypto'
import { dirname, isAbsolute, resolve } from 'node:path'
import YAML from 'yaml'
import type { SepilotdConfig } from '../../config/schema.js'
import { ChannelAcl } from '../../security/channel-acl.js'
import { EncryptionManager } from '../../security/encryption.js'
import {
  PolicyEngine,
  createDefaultPolicy,
  withOperatorValidationAllowPatterns,
} from '../../security/policy-engine.js'

const POLICY_MODES = new Set<ToolPolicyMode>(['autonomous', 'supervised', 'ask', 'blocked'])

type LoadedPolicyRule = Partial<PolicyRule>

interface LoadedPolicyConfig {
  version?: number
  defaults?: Partial<PolicyConfig['defaults']>
  tools?: Record<string, LoadedPolicyRule>
  elevated?: Record<string, LoadedPolicyRule>
}

function asRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`)
  }
  return value as Record<string, unknown>
}

function optionalStringArray(value: unknown, label: string): string[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value) || !value.every((entry) => typeof entry === 'string')) {
    throw new Error(`${label} must be an array of strings`)
  }
  return value
}

function optionalPositiveInteger(value: unknown, label: string): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${label} must be a positive integer`)
  }
  return value as number
}

function optionalPolicyMode(value: unknown, label: string): ToolPolicyMode | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || !POLICY_MODES.has(value as ToolPolicyMode)) {
    throw new Error(`${label} must be one of autonomous, supervised, ask, blocked`)
  }
  return value as ToolPolicyMode
}

function normalizeRule(raw: unknown, label: string): LoadedPolicyRule {
  const value = asRecord(raw, label)
  return {
    mode: optionalPolicyMode(value.mode, `${label}.mode`),
    deny_patterns: optionalStringArray(value.deny_patterns, `${label}.deny_patterns`),
    deny_paths: optionalStringArray(value.deny_paths, `${label}.deny_paths`),
    deny_urls: optionalStringArray(value.deny_urls, `${label}.deny_urls`),
    deny_executables: optionalStringArray(value.deny_executables, `${label}.deny_executables`),
    allow_patterns: optionalStringArray(value.allow_patterns, `${label}.allow_patterns`),
    allow_paths: optionalStringArray(value.allow_paths, `${label}.allow_paths`),
    allow_urls: optionalStringArray(value.allow_urls, `${label}.allow_urls`),
    allow_hosts: optionalStringArray(value.allow_hosts, `${label}.allow_hosts`),
    max_timeout_ms: optionalPositiveInteger(value.max_timeout_ms, `${label}.max_timeout_ms`),
    max_output_bytes: optionalPositiveInteger(value.max_output_bytes, `${label}.max_output_bytes`),
  }
}

function normalizePolicyConfig(raw: unknown, source: string): LoadedPolicyConfig {
  const value = asRecord(raw, `policy ${source}`)
  const version =
    value.version === undefined
      ? undefined
      : optionalPositiveInteger(value.version, 'policy.version')
  const defaults =
    value.defaults === undefined ? undefined : asRecord(value.defaults, 'policy.defaults')
  const tools =
    value.tools === undefined
      ? {}
      : Object.fromEntries(
          Object.entries(asRecord(value.tools, 'policy.tools')).map(([tool, rule]) => [
            tool,
            normalizeRule(rule, `policy.tools.${tool}`),
          ]),
        )
  const elevated =
    value.elevated === undefined
      ? undefined
      : Object.fromEntries(
          Object.entries(asRecord(value.elevated, 'policy.elevated')).map(([tool, rule]) => [
            tool,
            normalizeRule(rule, `policy.elevated.${tool}`),
          ]),
        )

  let normalizedDefaults: LoadedPolicyConfig['defaults'] | undefined
  if (defaults) {
    const unmatched = defaults.unmatched_policy
    if (unmatched !== undefined && unmatched !== 'allow' && unmatched !== 'deny') {
      throw new Error('policy.defaults.unmatched_policy must be allow or deny')
    }
    const honors = defaults.autonomous_honors_supervised_rules
    if (honors !== undefined && typeof honors !== 'boolean') {
      throw new Error('policy.defaults.autonomous_honors_supervised_rules must be a boolean')
    }
    normalizedDefaults = {
      mode: optionalPolicyMode(defaults.mode, 'policy.defaults.mode'),
      unmatched_policy: unmatched as 'allow' | 'deny' | undefined,
      autonomous_honors_supervised_rules: honors,
      max_timeout_ms: optionalPositiveInteger(
        defaults.max_timeout_ms,
        'policy.defaults.max_timeout_ms',
      ),
      max_output_bytes: optionalPositiveInteger(
        defaults.max_output_bytes,
        'policy.defaults.max_output_bytes',
      ),
    }
  }

  return {
    version,
    defaults: normalizedDefaults,
    tools,
    elevated,
  }
}

function mergeUnique(
  base: string[] | undefined,
  override: string[] | undefined,
): string[] | undefined {
  if (!base && !override) return undefined
  return Array.from(new Set([...(base ?? []), ...(override ?? [])]))
}

function mergePolicyRule(
  base: Partial<PolicyRule> | undefined,
  override: LoadedPolicyRule,
  defaultMode: ToolPolicyMode,
): PolicyRule {
  return {
    ...base,
    ...override,
    mode: override.mode ?? base?.mode ?? defaultMode,
    deny_patterns: mergeUnique(base?.deny_patterns, override.deny_patterns),
    deny_paths: mergeUnique(base?.deny_paths, override.deny_paths),
    deny_urls: mergeUnique(base?.deny_urls, override.deny_urls),
    deny_executables: mergeUnique(base?.deny_executables, override.deny_executables),
    allow_patterns: override.allow_patterns ?? base?.allow_patterns,
    allow_paths: override.allow_paths ?? base?.allow_paths,
    allow_urls: override.allow_urls ?? base?.allow_urls,
    allow_hosts: override.allow_hosts ?? base?.allow_hosts,
    max_timeout_ms: override.max_timeout_ms ?? base?.max_timeout_ms,
    max_output_bytes: override.max_output_bytes ?? base?.max_output_bytes,
  }
}

function mergePolicyConfig(base: PolicyConfig, override: LoadedPolicyConfig): PolicyConfig {
  const defaults = {
    mode: override.defaults?.mode ?? base.defaults.mode,
    unmatched_policy: override.defaults?.unmatched_policy ?? base.defaults.unmatched_policy,
    autonomous_honors_supervised_rules:
      override.defaults?.autonomous_honors_supervised_rules
      ?? base.defaults.autonomous_honors_supervised_rules
      ?? false,
    max_timeout_ms: override.defaults?.max_timeout_ms ?? base.defaults.max_timeout_ms,
    max_output_bytes: override.defaults?.max_output_bytes ?? base.defaults.max_output_bytes,
  }
  const tools: PolicyConfig['tools'] = { ...base.tools }
  for (const [tool, rule] of Object.entries(override.tools ?? {})) {
    tools[tool] = mergePolicyRule(tools[tool], rule, defaults.mode)
  }

  const elevated: PolicyConfig['elevated'] = override.elevated
    ? Object.fromEntries(
        Object.entries(override.elevated).map(([tool, rule]) => [
          tool,
          mergePolicyRule(base.elevated?.[tool], rule, defaults.mode),
        ]),
      )
    : base.elevated

  return {
    version: override.version ?? base.version,
    defaults,
    tools,
    elevated,
  }
}

function isNodeError(error: unknown, code: string): boolean {
  return (error as NodeJS.ErrnoException | undefined)?.code === code
}

function isEnoent(error: unknown): boolean {
  return isNodeError(error, 'ENOENT')
}

/**
 * Write the built-in default policy to `policyPath` and return it. Writing is
 * best-effort: if the directory is read-only the daemon still boots on the
 * in-memory default rather than refusing to start.
 */
function materializeDefaultPolicyFile(policyPath: string): PolicyConfig {
  const policy = createDefaultPolicy()
  try {
    mkdirSync(dirname(policyPath), { recursive: true, mode: 0o700 })
    writeFileSync(policyPath, YAML.stringify(policy), { mode: 0o600, flag: 'wx' })
  } catch {
    // Already created by a racing boot, or the directory is not writable.
  }
  return policy
}

export function resolveToolPolicyPath(config: SepilotdConfig, dataDir: string): string {
  const policyPath = config.security.toolPolicy || 'policies.yaml'
  return isAbsolute(policyPath) ? policyPath : resolve(dataDir, 'security', policyPath)
}

export function loadPolicyConfig(config: SepilotdConfig, dataDir: string): PolicyConfig {
  const policyPath = resolveToolPolicyPath(config, dataDir)
  let rawText: string
  try {
    rawText = readFileSync(policyPath, 'utf8')
  } catch (error) {
    // A data dir that has never been initialised simply has no policy file
    // yet. That is a provisioning gap, not an operator error: materialize the
    // built-in default policy so first boot succeeds without manual file
    // surgery. Every other read failure (permissions, IO, a directory in the
    // way) still fails loudly — silently defaulting there would quietly widen
    // the tool policy the operator believes is in force.
    if (!isEnoent(error)) {
      throw new Error(
        `Could not load tool policy at ${policyPath}: ${
          error instanceof Error ? error.message : String(error)
        }`,
        { cause: error },
      )
    }
    return materializeDefaultPolicyFile(policyPath)
  }

  let rawPolicy: unknown
  try {
    rawPolicy = YAML.parse(rawText)
  } catch (error) {
    throw new Error(
      `Could not parse tool policy at ${policyPath}: ${
        error instanceof Error ? error.message : String(error)
      }`,
      { cause: error },
    )
  }

  return mergePolicyConfig(createDefaultPolicy(), normalizePolicyConfig(rawPolicy, policyPath))
}

export function buildPolicyEngine(config: SepilotdConfig, dataDir: string): PolicyEngine {
  return new PolicyEngine(withOperatorValidationAllowPatterns(loadPolicyConfig(config, dataDir)))
}

export function buildChannelAcl(config: SepilotdConfig, autonomy: AutonomyLevel): ChannelAcl {
  const channelAcl = new ChannelAcl()

  const defaultAclModes: Record<string, 'open' | 'allowlist' | 'pairing'> = {
    'github-issue': 'open',
    webhook: 'open',
    webchat: 'open',
    telegram: 'pairing',
    slack: 'allowlist',
    discord: 'allowlist',
    mattermost: 'allowlist',
    whatsapp: 'allowlist',
    teams: 'allowlist',
    line: 'allowlist',
  }

  for (const channel of config.channels) {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    const rawAllowed = channelConfig?.allowedUsers
    const allowedUsers = Array.isArray(rawAllowed)
      ? rawAllowed.filter((user): user is string => typeof user === 'string')
      : []
    const externalTriggerAutonomy =
      channel.type === 'github-issue'
        ? parseAutonomyLevel(channelConfig?.externalTriggerAutonomy, AutonomyLevel.Supervised)
        : autonomy
    channelAcl.configure({
      channelType: channel.type,
      mode: defaultAclModes[channel.type] ?? 'allowlist',
      allowedUsers,
      maxAutonomy: externalTriggerAutonomy,
    })
  }

  return channelAcl
}

function parseAutonomyLevel(value: unknown, fallback: AutonomyLevel): AutonomyLevel {
  return Object.values(AutonomyLevel).includes(value as AutonomyLevel)
    ? (value as AutonomyLevel)
    : fallback
}

export async function buildEncryption(
  config: SepilotdConfig,
  dataDir: string,
): Promise<EncryptionManager> {
  const encryption = new EncryptionManager()

  if (!config.memory.encryption) return encryption

  // The operator explicitly opted in to memory encryption. The
  // previous implementation swallowed every loadKey failure and
  // returned an EncryptionManager with `key=null`, which means
  // every downstream `encrypt`/`encryptFile` call would throw at
  // use time AND, worse, any plaintext fall-back path in callers
  // would silently persist sensitive memory state to disk in
  // plaintext. Refuse to start instead — the operator can either
  // generate the key (`sepilotd init` or manual openssl rand 32)
  // or set `memory.encryption=false`.
  const keyPath = `${dataDir}/security/data.key`
  // First boot into an empty data dir: generate the key instead of refusing to
  // start. Only a *missing* key is auto-provisioned — an existing but invalid
  // key still fails loudly, because regenerating it would permanently destroy
  // access to memory that was already encrypted under the previous key.
  try {
    await mkdir(dirname(keyPath), { recursive: true, mode: 0o700 })
    await writeFile(keyPath, randomBytes(32), { mode: 0o600, flag: 'wx' })
  } catch (err) {
    if (!isNodeError(err, 'EEXIST')) {
      throw new Error(
        `memory.encryption is enabled but the key at ${keyPath} could not be created: ${
          err instanceof Error ? err.message : String(err)
        }`,
        { cause: err },
      )
    }
  }
  try {
    await encryption.loadKey(keyPath)
  } catch (err) {
    throw new Error(
      `memory.encryption is enabled but the key at ${keyPath} could not be loaded: ${
        err instanceof Error ? err.message : String(err)
      }`,
      { cause: err },
    )
  }

  return encryption
}
