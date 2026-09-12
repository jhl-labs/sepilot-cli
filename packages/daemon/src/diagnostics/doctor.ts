import { execFileSync } from 'node:child_process'
import { access, readFile, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { parseConfig } from '../config/loader.js'
import type { SepilotdConfig } from '../config/schema.js'
import type { ChannelSummaryCapabilities } from '../server/runtime/capabilities.js'
import type { AuditChainVerificationResult } from '../security/audit-logger.js'
import { summarizeWebhookSecurity } from '../server/runtime/webhook-endpoints.js'

export type DoctorCheckStatus = 'PASS' | 'WARN' | 'FAIL'

export interface DoctorCheckResult {
  name: string
  category: string
  status: DoctorCheckStatus
  message: string
  recommendation?: string
}

export interface DoctorReportSummary {
  score: number
  grade: 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'Critical'
  warnings: number
  errors: number
  generatedAt: string
}

export interface DoctorReport {
  data: DoctorCheckResult[]
  summary: DoctorReportSummary
}

interface PolicySnapshot {
  defaults?: {
    unmatched_policy?: string
  }
}

interface PluginSummary {
  manifest?: {
    name?: string
  }
  status?: string
  error?: string
}

export interface DoctorRuntime extends ChannelSummaryCapabilities {
  dataDir?: string
  auditLogger?: {
    verifyChain?: () => Promise<AuditChainVerificationResult>
  }
  policyEngine?: {
    describe(): PolicySnapshot
  }
  pluginLoader?: {
    list(): PluginSummary[]
  }
}

export interface BuildDoctorReportOptions {
  dataDir?: string
  runtime?: DoctorRuntime
  authToken?: string | null
  platform?: NodeJS.Platform
}

type SandboxSecurityConfig = {
  sandbox?: 'local' | 'docker' | 'bubblewrap'
  sandboxHostFileTools?: 'deny' | 'allow'
  sandboxDocker?: {
    networkMode?: 'none' | 'bridge'
    mountMode?: 'rw' | 'ro'
    readOnlyRootfs?: boolean
    noNewPrivileges?: boolean
    capDrop?: string[]
  }
  sandboxBubblewrap?: {
    networkMode?: 'none' | 'host'
    readOnlyWorkspace?: boolean
    processIsolation?: 'namespace' | 'container-boundary'
    bwrapPath?: string
    resourceLimits?: {
      pidsMax?: number
      memoryMaxBytes?: number
      tmpfsSizeBytes?: number
    }
  }
}

export async function buildDoctorReport(
  options: BuildDoctorReportOptions = {},
): Promise<DoctorReport> {
  const dataDir = options.dataDir ?? options.runtime?.dataDir ?? join(homedir(), '.sepilotd')
  const configPath = join(dataDir, 'config.yaml')
  const securityDir = join(dataDir, 'security')
  const platform = options.platform ?? process.platform
  const results: DoctorCheckResult[] = []
  const effectiveConfig =
    options.runtime?.config
    ?? await readDoctorConfig(configPath)

  results.push(
    await checkDirPerms(dataDir, 'data_dir_permissions', '~/.sepilotd/', 0o700, platform),
  )
  results.push(
    await checkDirPerms(securityDir, 'security_dir_permissions', 'security/', 0o700, platform),
  )
  results.push(
    await checkFilePerms(
      join(securityDir, 'daemon.token'),
      'daemon_token_permissions',
      'daemon.token',
      0o600,
      platform,
    ),
  )
  results.push(
    await checkFilePerms(
      join(securityDir, 'gateway.token'),
      'gateway_token_permissions',
      'gateway.token',
      0o600,
      platform,
    ),
  )
  results.push(
    await checkFilePerms(
      join(securityDir, 'device.key'),
      'device_key_permissions',
      'device.key',
      0o600,
      platform,
    ),
  )
  results.push(
    await checkFilePerms(
      join(securityDir, 'data.key'),
      'data_key_permissions',
      'data.key',
      0o600,
      platform,
    ),
  )
  results.push(
    await checkFilePerms(
      configPath,
      'config_permissions',
      'config.yaml',
      0o600,
      platform,
    ),
  )
  results.push(
    await checkFilePerms(join(dataDir, '.env'), 'env_permissions', '.env', 0o600, platform),
  )
  results.push(await checkNoPlaintextSecrets(configPath))
  results.push(
    await checkFileExists(join(securityDir, 'policies.yaml'), 'policies_file', 'policies.yaml'),
  )
  if (effectiveConfig) {
    const sandboxChecks = checkSandboxConfiguration(effectiveConfig)
    results.push(...sandboxChecks)
    const runtimeCheck = checkSandboxRuntime(effectiveConfig.security)
    if (runtimeCheck) results.push(runtimeCheck)
  } else {
    // Preserve the installation hint when the config is absent or invalid and
    // there is no selected runtime whose availability can be checked.
    results.push(checkDocker())
  }
  results.push(await checkFileExists(join(securityDir, 'audit.log'), 'audit_log_file', 'audit.log'))

  if (options.runtime) {
    results.push(checkRuntimeAuth(options.authToken))
    results.push(checkDaemonBind(options.runtime.config))
    results.push(checkAuditLogConfig(options.runtime.config))
    results.push(checkToolPolicy(options.runtime.policyEngine))
    results.push(checkPluginLoad(options.runtime.pluginLoader))
    results.push(checkWebhookVerification(options.runtime))
    if (options.runtime.auditLogger) {
      results.push(await checkAuditLogIntegrity(options.runtime.auditLogger))
    }
  }

  return {
    data: results,
    summary: {
      ...summarizeDoctorChecks(results),
      generatedAt: new Date().toISOString(),
    },
  }
}

async function readDoctorConfig(path: string): Promise<SepilotdConfig | undefined> {
  try {
    return parseConfig(await readFile(path, 'utf-8'))
  } catch {
    return undefined
  }
}

/**
 * Explain the effective sandbox boundary without claiming that a configured
 * backend covers tools which still execute in the daemon process.
 */
export function checkSandboxConfiguration(config: {
  security?: SandboxSecurityConfig
}): DoctorCheckResult[] {
  const security = config.security
  const sandbox = security?.sandbox ?? 'local'
  const hostFileTools = security?.sandboxHostFileTools ?? 'deny'
  const checks: DoctorCheckResult[] = []

  if (sandbox === 'local') {
    checks.push({
      name: 'sandbox_execution_boundary',
      category: 'sandbox',
      status: 'WARN',
      message: 'terminal.run executes directly on the host (security.sandbox=local)',
      recommendation:
        'Set security.sandbox to docker or bubblewrap for filesystem, process, and network isolation.',
    })
    checks.push({
      name: 'sandbox_file_tool_boundary',
      category: 'sandbox',
      status: 'PASS',
      message: 'Built-in host file tools match the explicitly local execution profile',
    })
    return checks
  }

  checks.push({
    name: 'sandbox_execution_boundary',
    category: 'sandbox',
    status: 'PASS',
    message: `terminal.run uses the fail-closed ${sandbox} isolation backend`,
  })

  if (hostFileTools === 'allow') {
    checks.push({
      name: 'sandbox_file_tool_boundary',
      category: 'sandbox',
      status: 'WARN',
      message: `Built-in host file tools are enabled and bypass the ${sandbox} filesystem namespace`,
      recommendation:
        'Set security.sandboxHostFileTools to deny; use terminal.run inside the mounted workspace for file access.',
    })
  } else {
    checks.push({
      name: 'sandbox_file_tool_boundary',
      category: 'sandbox',
      status: 'PASS',
      message: `Built-in fs.* and apply_patch tools are disabled under ${sandbox}`,
    })
  }

  checks.push(
    sandbox === 'docker'
      ? checkDockerHardening(security?.sandboxDocker)
      : checkBubblewrapHardening(security?.sandboxBubblewrap),
  )
  return checks
}

function checkDockerHardening(
  config: SandboxSecurityConfig['sandboxDocker'],
): DoctorCheckResult {
  const weakened: string[] = []
  if ((config?.networkMode ?? 'none') !== 'none') weakened.push('networkMode is bridge')
  if ((config?.readOnlyRootfs ?? true) !== true) weakened.push('root filesystem is writable')
  if ((config?.noNewPrivileges ?? true) !== true) weakened.push('no-new-privileges is disabled')
  if (!(config?.capDrop ?? ['ALL']).some((capability) => capability.toUpperCase() === 'ALL')) {
    weakened.push('capDrop does not include ALL')
  }

  if (weakened.length > 0) {
    return {
      name: 'sandbox_hardening',
      category: 'sandbox',
      status: 'WARN',
      message: `Docker sandbox hardening is weakened: ${weakened.join('; ')}`,
      recommendation:
        'Use networkMode=none, readOnlyRootfs=true, noNewPrivileges=true, and capDrop=[ALL] unless a reviewed workload requires otherwise.',
    }
  }

  return {
    name: 'sandbox_hardening',
    category: 'sandbox',
    status: 'PASS',
    message:
      `Docker sandbox drops all capabilities, blocks network, uses a read-only rootfs, and prevents privilege escalation (workspace mount ${config?.mountMode ?? 'rw'})`,
  }
}

function checkBubblewrapHardening(
  config: SandboxSecurityConfig['sandboxBubblewrap'],
): DoctorCheckResult {
  const weakened: string[] = []
  if ((config?.networkMode ?? 'none') !== 'none') weakened.push('host networking is enabled')
  if ((config?.processIsolation ?? 'namespace') === 'container-boundary') {
    weakened.push('PID isolation is delegated to an outer container boundary')
  }
  const limits = config?.resourceLimits
  if (limits?.pidsMax === 0) weakened.push('process limit is disabled')
  if (limits?.memoryMaxBytes === 0) weakened.push('memory limit is disabled')
  if (limits?.tmpfsSizeBytes === 0) weakened.push('temporary-filesystem limit is disabled')

  if (weakened.length > 0) {
    return {
      name: 'sandbox_hardening',
      category: 'sandbox',
      status: 'WARN',
      message: `Bubblewrap sandbox hardening is weakened: ${weakened.join('; ')}`,
      recommendation:
        'Use networkMode=none, processIsolation=namespace, and leave resource limits enabled unless a reviewed outer container provides the omitted boundary.',
    }
  }

  return {
    name: 'sandbox_hardening',
    category: 'sandbox',
    status: 'PASS',
    message:
      `Bubblewrap isolates host paths and networking with resource limits (workspace ${config?.readOnlyWorkspace ? 'read-only' : 'read-write'})`,
  }
}

function checkSandboxRuntime(
  security: SandboxSecurityConfig | undefined,
): DoctorCheckResult | undefined {
  const sandbox = security?.sandbox ?? 'local'
  if (sandbox === 'local') return undefined

  const executable =
    sandbox === 'docker'
      ? 'docker'
      : security?.sandboxBubblewrap?.bwrapPath ?? 'bwrap'
  const args =
    sandbox === 'docker'
      ? ['version', '--format', '{{.Server.Version}}']
      : ['--version']

  try {
    execFileSync(executable, args, { stdio: 'ignore', timeout: 5_000 })
    return {
      name: 'sandbox_runtime',
      category: 'sandbox',
      status: 'PASS',
      message: `${sandbox} sandbox runtime is available`,
    }
  } catch {
    return {
      name: 'sandbox_runtime',
      category: 'sandbox',
      status: 'FAIL',
      message: `${sandbox} sandbox is configured but its runtime is unavailable`,
      recommendation:
        sandbox === 'docker'
          ? 'Install and start Docker, then rerun doctor before relying on terminal tools.'
          : `Install bubblewrap or correct security.sandboxBubblewrap.bwrapPath (${executable}).`,
    }
  }
}

export function summarizeDoctorChecks(
  results: DoctorCheckResult[],
): Omit<DoctorReportSummary, 'generatedAt'> {
  let score = 10
  let warnings = 0
  let errors = 0

  for (const result of results) {
    if (result.status === 'FAIL') {
      score -= 2
      errors += 1
    }
    if (result.status === 'WARN') {
      score -= 0.5
      warnings += 1
    }
  }

  const boundedScore = Math.max(0, Math.round(score * 10) / 10)
  return {
    score: boundedScore,
    grade: gradeForScore(boundedScore),
    warnings,
    errors,
  }
}

async function checkDirPerms(
  path: string,
  name: string,
  label: string,
  expected: number,
  platform: NodeJS.Platform,
): Promise<DoctorCheckResult> {
  return checkPathPerms(path, name, label, expected, platform)
}

async function checkFilePerms(
  path: string,
  name: string,
  label: string,
  expected: number,
  platform: NodeJS.Platform,
): Promise<DoctorCheckResult> {
  return checkPathPerms(path, name, label, expected, platform)
}

async function checkPathPerms(
  path: string,
  name: string,
  label: string,
  expected: number,
  platform: NodeJS.Platform,
): Promise<DoctorCheckResult> {
  try {
    const s = await stat(path)
    if (platform === 'win32') {
      return {
        name,
        category: 'filesystem',
        status: 'PASS',
        message: `Windows ACLs are used for ${label}; POSIX permission bits are not evaluated`,
      }
    }
    const mode = s.mode & 0o777
    if (mode <= expected) {
      return {
        name,
        category: 'filesystem',
        status: 'PASS',
        message: `File permissions: ${label} (${mode.toString(8)})`,
      }
    }
    return {
      name,
      category: 'filesystem',
      status: 'FAIL',
      message: `${label} permissions too open: ${mode.toString(8)} (expected ${expected.toString(8)})`,
      recommendation: `Restrict ${label} to ${expected.toString(8)} or tighter.`,
    }
  } catch {
    return {
      name,
      category: 'filesystem',
      status: 'WARN',
      message: `${label} not found`,
      recommendation: 'Run sepilotd init or start the daemon once to create runtime state.',
    }
  }
}

async function checkFileExists(
  path: string,
  name: string,
  label: string,
): Promise<DoctorCheckResult> {
  try {
    await access(path)
    return {
      name,
      category: 'config',
      status: 'PASS',
      message: `${label} exists`,
    }
  } catch {
    return {
      name,
      category: 'config',
      status: 'WARN',
      message: `${label} not found`,
      recommendation: `Create ${label} if this daemon should enforce that capability.`,
    }
  }
}

async function checkNoPlaintextSecrets(configPath: string): Promise<DoctorCheckResult> {
  try {
    const content = await readFile(configPath, 'utf-8')
    const secretPatterns = [
      /sk-[a-zA-Z0-9]{20,}/,
      /sk-ant-[a-zA-Z0-9]{20,}/,
      /ghp_[a-zA-Z0-9]{36}/,
      /ghs_[a-zA-Z0-9]{36}/,
    ]
    for (const pattern of secretPatterns) {
      if (pattern.test(content)) {
        return {
          name: 'plaintext_secrets',
          category: 'config',
          status: 'FAIL',
          message: 'Plaintext secrets found in config.yaml',
          recommendation: 'Use ${ENV_VAR} references or the daemon secret store instead.',
        }
      }
    }
    return {
      name: 'plaintext_secrets',
      category: 'config',
      status: 'PASS',
      message: 'No plaintext secrets in config',
    }
  } catch {
    return {
      name: 'plaintext_secrets',
      category: 'config',
      status: 'WARN',
      message: 'config.yaml not found',
      recommendation: 'Create config.yaml with sepilotd init before relying on this daemon.',
    }
  }
}

function checkDocker(): DoctorCheckResult {
  try {
    execFileSync('docker', ['--version'], { stdio: 'ignore' })
    return {
      name: 'docker',
      category: 'sandbox',
      status: 'PASS',
      message: 'Docker installed',
    }
  } catch {
    return {
      name: 'docker',
      category: 'sandbox',
      status: 'WARN',
      message: 'Docker not installed; Docker sandbox unavailable',
      recommendation: 'Install Docker if tools should run in Docker isolation.',
    }
  }
}

function checkRuntimeAuth(authToken: string | null | undefined): DoctorCheckResult {
  if (authToken && authToken.trim().length > 0) {
    return {
      name: 'runtime_auth_token',
      category: 'auth',
      status: 'PASS',
      message: 'Daemon HTTP auth token is enabled',
    }
  }

  return {
    name: 'runtime_auth_token',
    category: 'auth',
    status: 'FAIL',
    message: 'Daemon HTTP auth token is disabled',
    recommendation: 'Restore security/daemon.token before exposing the daemon API.',
  }
}

function checkDaemonBind(config: Partial<SepilotdConfig>): DoctorCheckResult {
  const host = config.daemon?.host?.trim()
  if (!host) {
    return {
      name: 'daemon_bind',
      category: 'network',
      status: 'WARN',
      message: 'Daemon bind host is not configured',
      recommendation: 'Bind to 127.0.0.1 unless remote access is explicitly required.',
    }
  }

  if (isLoopbackHost(host)) {
    return {
      name: 'daemon_bind',
      category: 'network',
      status: 'PASS',
      message: `Daemon binds to loopback host ${host}`,
    }
  }

  return {
    name: 'daemon_bind',
    category: 'network',
    status: 'WARN',
    message: `Daemon binds to non-loopback host ${host}`,
    recommendation:
      'Require bearer auth and put remote access behind a trusted tunnel or reverse proxy.',
  }
}

function checkAuditLogConfig(config: Partial<SepilotdConfig>): DoctorCheckResult {
  if (config.security?.auditLog === false) {
    return {
      name: 'audit_log_config',
      category: 'audit',
      status: 'FAIL',
      message: 'Security audit logging is disabled',
      recommendation: 'Set security.auditLog to true for operator-visible security trails.',
    }
  }

  return {
    name: 'audit_log_config',
    category: 'audit',
    status: 'PASS',
    message: 'Security audit logging is enabled',
  }
}

async function checkAuditLogIntegrity(
  auditLogger: NonNullable<DoctorRuntime['auditLogger']>,
): Promise<DoctorCheckResult> {
  if (!auditLogger.verifyChain) {
    return {
      name: 'audit_log_integrity',
      category: 'audit',
      status: 'WARN',
      message: 'Audit logger does not expose hash-chain verification',
      recommendation: 'Restart the daemon with the JSONL audit logger enabled.',
    }
  }

  try {
    const result = await auditLogger.verifyChain()
    if (result.ok) {
      return {
        name: 'audit_log_integrity',
        category: 'audit',
        status: 'PASS',
        message: 'Audit hash-chain verification passed',
      }
    }
    return {
      name: 'audit_log_integrity',
      category: 'audit',
      status: 'FAIL',
      message: `Audit hash-chain verification failed at line ${result.firstBreakLine ?? 'unknown'}`,
      recommendation: 'Preserve the audit log for forensic review and rotate to a fresh log.',
    }
  } catch (error) {
    return {
      name: 'audit_log_integrity',
      category: 'audit',
      status: 'WARN',
      message: `Audit hash-chain verification could not run: ${String(error)}`,
      recommendation: 'Check audit log readability and retry doctor.',
    }
  }
}

function checkToolPolicy(policyEngine: DoctorRuntime['policyEngine']): DoctorCheckResult {
  if (!policyEngine) {
    return {
      name: 'tool_policy_default',
      category: 'policy',
      status: 'WARN',
      message: 'Tool policy engine is not initialized',
      recommendation: 'Initialize the policy engine so tool fallback behavior is explicit.',
    }
  }

  const unmatchedPolicy = policyEngine.describe().defaults?.unmatched_policy
  if (unmatchedPolicy === 'deny') {
    return {
      name: 'tool_policy_default',
      category: 'policy',
      status: 'PASS',
      message: 'Unmatched tool policy defaults to deny',
    }
  }

  return {
    name: 'tool_policy_default',
    category: 'policy',
    status: 'WARN',
    message: `Unmatched tool policy defaults to ${unmatchedPolicy ?? 'unknown'}`,
    recommendation: 'Set defaults.unmatched_policy to deny for stricter tool governance.',
  }
}

function checkPluginLoad(pluginLoader: DoctorRuntime['pluginLoader']): DoctorCheckResult {
  if (!pluginLoader) {
    return {
      name: 'plugin_load',
      category: 'plugins',
      status: 'WARN',
      message: 'Plugin loader is not initialized',
      recommendation: 'Initialize the plugin loader before relying on plugin capabilities.',
    }
  }

  const failed = pluginLoader.list().filter((plugin) => plugin.status === 'failed')
  if (failed.length === 0) {
    return {
      name: 'plugin_load',
      category: 'plugins',
      status: 'PASS',
      message: 'No failed plugins',
    }
  }

  const names = failed
    .map((plugin) => plugin.manifest?.name ?? 'unknown')
    .slice(0, 5)
    .join(', ')
  return {
    name: 'plugin_load',
    category: 'plugins',
    status: 'WARN',
    message: `${failed.length} plugin(s) failed to load: ${names}`,
    recommendation: 'Fix or remove failed plugins so extension state is deterministic.',
  }
}

function checkWebhookVerification(runtime: DoctorRuntime): DoctorCheckResult {
  try {
    const summary = summarizeWebhookSecurity(runtime, undefined, {
      unreadyLimit: 5,
    })
    if (summary.totalEndpoints === 0) {
      return {
        name: 'webhook_verification',
        category: 'webhooks',
        status: 'PASS',
        message: 'No inbound webhook endpoints configured',
      }
    }
    if (summary.verificationNotReadyEndpoints === 0) {
      return {
        name: 'webhook_verification',
        category: 'webhooks',
        status: 'PASS',
        message: `All ${summary.totalEndpoints} webhook endpoint(s) have verification material`,
      }
    }

    const missing = summary.unreadySummary.byMissingRequirement
      .slice(0, 3)
      .map((entry) => `${entry.requirement} (${entry.endpointCount})`)
      .join(', ')
    return {
      name: 'webhook_verification',
      category: 'webhooks',
      status: 'FAIL',
      message: `${summary.verificationNotReadyEndpoints}/${summary.totalEndpoints} webhook endpoint(s) lack verification material`,
      recommendation: missing
        ? `Configure missing webhook secrets: ${missing}.`
        : 'Configure webhook verification secrets for every inbound channel.',
    }
  } catch (error) {
    return {
      name: 'webhook_verification',
      category: 'webhooks',
      status: 'WARN',
      message: `Webhook verification could not be evaluated: ${String(error)}`,
      recommendation: 'Check channel configuration and retry doctor.',
    }
  }
}

function gradeForScore(score: number): DoctorReportSummary['grade'] {
  if (score >= 9) return 'Excellent'
  if (score >= 7) return 'Good'
  if (score >= 5) return 'Fair'
  if (score >= 3) return 'Poor'
  return 'Critical'
}

function isLoopbackHost(host: string): boolean {
  return host === '127.0.0.1' || host === 'localhost' || host === '::1' || host === '[::1]'
}
