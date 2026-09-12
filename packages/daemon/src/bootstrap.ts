import { access, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { generateKeyPairSync, randomBytes, randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import YAML from 'yaml'
import { createApp } from './server/app.js'
import { loadManagedEnvFile } from './config/env-file.js'
import { parseConfig } from './config/loader.js'
import { applyRuntimeChannelEnvironment } from './config/runtime-channel-env.js'
import { applyRuntimeGatewayEnvironment } from './config/runtime-gateway-env.js'
import { buildRuntime, type RuntimeServices } from './server/runtime.js'
import {
  configSchema,
  hasLegacyUnsafeProxyMigration,
  loggingSchema,
  type LoggingConfig,
  type SepilotdConfig,
  DEFAULT_DEVICE_NAME,
} from './config/schema.js'
import {
  assertDaemonAuthPolicy,
  createFileDaemonAuthTokenResolver,
  daemonAuthTokenPath,
  ensureDaemonAuthToken,
} from './server/auth.js'
import { LifecycleManager, resolveShutdownDeadlineMs } from './lifecycle/manager.js'
import { PidManager, formatPidConflictMessage } from './lifecycle/pid.js'
import { resolveIdleShutdownMs, startIdleReaper } from './server/runtime/idle-reaper.js'
import { validateConfig } from './config/validator.js'
import { createLogger, setLogFile } from './logger.js'
import { setAgentTraceLogFile } from './observability/agent-trace.js'
import { applyLoggingConfig } from './observability/apply-logging-config.js'
import { isNodeFsError } from './utils/fs-error.js'
import { createDefaultPolicy } from './security/policy-engine.js'
import {
  closeProviderHttpDispatcher,
  configureProviderHttpTimeout,
  configureProviderNetworkBlocked,
  CUSTOM_CA_UNAVAILABLE_DEGRADED_REASON,
  LEGACY_PROXY_UNSAFE_DEGRADED_REASON,
  NETWORK_PROXY_UNAVAILABLE_DEGRADED_REASON,
  ProviderNetworkConfigurationError,
} from './providers/http-timeout.js'
import { configureDaemonStorage } from './storage/home.js'
import { prepareDaemonStorage } from './storage/layout.js'

const log = createLogger('main')
const DEFAULT_DAEMON_HOST = '127.0.0.1'
const DEFAULT_DAEMON_PORT = 17600
const BOOTSTRAP_DIRS = ['sessions', 'memory', 'skills', 'security', 'cache', 'logs'] as const
const CONFIG_DEGRADED_ENV = 'SEPILOTD_CONFIG_DEGRADED_OK'

function configureStartupProviderNetwork(network: SepilotdConfig['network']): void {
  if (hasLegacyUnsafeProxyMigration(network)) {
    log.warn(
      'Unsafe legacy proxy settings were migrated; provider network egress is blocked until Settings saves an explicit policy',
      { code: LEGACY_PROXY_UNSAFE_DEGRADED_REASON },
    )
    configureProviderNetworkBlocked(
      network,
      LEGACY_PROXY_UNSAFE_DEGRADED_REASON,
    )
    return
  }

  try {
    configureProviderHttpTimeout(network)
  } catch (error) {
    if (!(error instanceof ProviderNetworkConfigurationError)) {
      throw error
    }

    const degradedReason = error.code === 'NETWORK_CA_INVALID'
      ? CUSTOM_CA_UNAVAILABLE_DEGRADED_REASON
      : NETWORK_PROXY_UNAVAILABLE_DEGRADED_REASON
    // A persisted CA path or environment proxy can become invalid between
    // launches. Keep only the daemon control plane reachable so Settings can
    // repair it. Direct fallback could bypass an enterprise egress boundary.
    log.warn(
      'Provider network policy is unavailable; provider egress is blocked until Settings repairs it',
      { code: error.code, degradedReason },
    )
    configureProviderNetworkBlocked(
      network,
      degradedReason,
    )
  }
}

export interface StartDaemonRuntimeOptions {
  dataDir?: string
  port?: number
  host?: string
  autoApproveCliFlag?: boolean
  setupSignalHandlers?: boolean
  exitOnShutdown?: boolean
}

export interface DaemonRuntimeHandle {
  app: FastifyInstance
  runtime: RuntimeServices
  dataDir: string
  host: string
  port: number
  shutdown(reason?: string): Promise<void>
}

export interface StartupConfigLoadResult {
  config: SepilotdConfig
  configLoadFailed: boolean
  configLoadError?: string
}

export function resolveDaemonDataDir(dataDir = process.env.SEPILOTD_DATA_DIR): string {
  return dataDir ?? join(homedir(), '.sepilotd')
}

// Validate a numeric SEPILOTD_* env knob against a range, warning (not
// silently defaulting) when the value is out of range or non-numeric so an
// operator typo like SEPILOTD_PORT=999999 is visible instead of quietly
// falling back to the default.
export function resolveNumericEnv(
  name: string,
  raw: string | undefined,
  opts: { min: number; max: number; fallback: number },
): number {
  if (raw === undefined || raw.trim() === '') return opts.fallback
  const parsed = Number(raw)
  if (!Number.isInteger(parsed) || parsed < opts.min || parsed > opts.max) {
    log.warn('Ignoring out-of-range environment knob; using default', {
      name,
      value: raw,
      min: opts.min,
      max: opts.max,
      fallback: opts.fallback,
    })
    return opts.fallback
  }
  return parsed
}

function resolveStartupPort(options: StartDaemonRuntimeOptions): number {
  if (options.port !== undefined) return options.port
  return resolveNumericEnv('SEPILOTD_PORT', process.env.SEPILOTD_PORT, {
    min: 1,
    max: 65_535,
    fallback: DEFAULT_DAEMON_PORT,
  })
}

function resolveStartupHost(options: StartDaemonRuntimeOptions): string {
  return options.host ?? (process.env.SEPILOTD_HOST?.trim() || DEFAULT_DAEMON_HOST)
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

async function writeFileIfMissing(
  path: string,
  data: string | Buffer,
  mode: number,
): Promise<void> {
  try {
    await writeFile(path, data, { mode, flag: 'wx' })
  } catch (error) {
    if (!isNodeFsError(error, 'EEXIST')) throw error
  }
}

async function ensureDefaultDaemonDataDir(
  dataDir: string,
  options: { configPath: string; port: number; host: string },
): Promise<void> {
  for (const dir of BOOTSTRAP_DIRS) {
    await mkdir(join(dataDir, dir), {
      recursive: true,
      mode: dir === 'security' ? 0o700 : 0o755,
    })
  }

  const securityDir = join(dataDir, 'security')
  const deviceKeyPath = join(securityDir, 'device.key')
  const devicePubPath = join(securityDir, 'device.pub')
  const hasDeviceKey = await pathExists(deviceKeyPath)
  const hasDevicePub = await pathExists(devicePubPath)
  if (!hasDeviceKey && !hasDevicePub) {
    const { publicKey, privateKey } = generateKeyPairSync('ed25519', {
      publicKeyEncoding: { type: 'spki', format: 'pem' },
      privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
    })
    await writeFileIfMissing(deviceKeyPath, privateKey, 0o600)
    await writeFileIfMissing(devicePubPath, publicKey, 0o644)
  }

  await writeFileIfMissing(join(securityDir, 'data.key'), randomBytes(32), 0o600)
  await writeFileIfMissing(
    join(securityDir, 'daemon.token'),
    randomBytes(32).toString('hex'),
    0o600,
  )
  await writeFileIfMissing(
    join(securityDir, 'gateway.token'),
    randomBytes(32).toString('hex'),
    0o600,
  )
  await writeFileIfMissing(
    join(securityDir, 'policies.yaml'),
    YAML.stringify(createDefaultPolicy()),
    0o644,
  )

  await writeFileIfMissing(
    options.configPath,
    YAML.stringify({
      version: 1,
      // Host-neutral: deriving the device name from the OS user or hostname
      // would leak host identity into a config artifact.
      device: { id: randomUUID(), name: DEFAULT_DEVICE_NAME, role: 'desktop' },
      daemon: {
        port: options.port,
        host: options.host,
        resumeArtifactRetentionDays: 30,
      },
      gateway: { url: 'http://127.0.0.1:17610' },
      providers: [],
      agent: {
        autonomy: 'supervised',
        thinkingLevel: 'medium',
        capabilities: { hostSystemInfo: true },
      },
      channels: [],
      memory: { encryption: true, vectorBackend: 'auto' },
      security: { toolPolicy: 'policies.yaml', auditLog: true, sandbox: 'local' },
      observability: { telemetry: false },
    }),
    0o600,
  )
}

function isConfigDegradedBootAllowed(): boolean {
  return process.env[CONFIG_DEGRADED_ENV] === '1'
}

function degradedConfig(): SepilotdConfig {
  return configSchema.parse({
    version: 1,
    device: { id: 'default', name: 'default', role: 'desktop' },
    providers: [],
  })
}

/**
 * A config.yaml without a `device:` block gets a freshly generated identity
 * from the schema default. Persist it so the identity is stable across
 * restarts instead of changing on every boot. Best-effort: a read-only config
 * file must not block startup, it only means the id is re-generated next time.
 */
async function persistGeneratedDeviceIdentity(
  configPath: string,
  yamlText: string,
  config: SepilotdConfig,
): Promise<void> {
  let raw: unknown
  try {
    raw = YAML.parse(yamlText)
  } catch {
    return
  }
  if (raw && typeof raw === 'object' && 'device' in (raw as Record<string, unknown>)) return
  try {
    const doc = YAML.parseDocument(yamlText)
    doc.set('device', config.device)
    await writeFile(configPath, doc.toString(), { mode: 0o600 })
    log.info('Device identity provisioned', { path: configPath })
  } catch (error) {
    log.warn('Could not persist generated device identity', {
      path: configPath,
      error: error instanceof Error ? error.message : String(error),
    })
  }
}

export async function loadStartupConfig(
  dataDir: string,
  configPath: string,
  options: StartDaemonRuntimeOptions = {},
): Promise<StartupConfigLoadResult> {
  await loadManagedEnvFile(dataDir)

  let config: SepilotdConfig
  let configLoadFailed = false
  let configLoadError: string | undefined

  try {
    const yamlText = await readFile(configPath, 'utf-8')
    config = parseConfig(yamlText)
    await persistGeneratedDeviceIdentity(configPath, yamlText, config)
    log.info('Config loaded', { path: configPath })
  } catch (error) {
    if (isNodeFsError(error, 'ENOENT')) {
      await ensureDefaultDaemonDataDir(dataDir, {
        configPath,
        port: resolveStartupPort(options),
        host: resolveStartupHost(options),
      })
      const yamlText = await readFile(configPath, 'utf-8')
      config = parseConfig(yamlText)
      log.info('Default config created', { path: configPath })
    } else {
      configLoadFailed = true
      configLoadError = String(error)
      log.error('Config could not be parsed', { path: configPath, error: configLoadError })
      if (!isConfigDegradedBootAllowed()) {
        throw new Error(
          `Refusing to start with an unparseable config at ${configPath}. `
          + `Fix it or set ${CONFIG_DEGRADED_ENV}=1 to boot read-only. `
          + `Cause: ${configLoadError}`,
        )
      }
      log.warn(`${CONFIG_DEGRADED_ENV}=1 - booting read-only with empty config`)
      config = degradedConfig()
    }
  }

  // Cluster/systemd operators can supply a complete Mattermost channel from
  // runtime secrets without ever writing those credentials to config.yaml.
  // A degraded config boot stays control-plane-only and deliberately skips
  // outbound channel activation.
  if (!configLoadFailed) {
    config = applyRuntimeGatewayEnvironment(
      applyRuntimeChannelEnvironment(config),
    )
  }

  const validation = validateConfig(config)
  for (const w of validation.warnings) log.warn(w)
  for (const e of validation.errors) log.error(e)
  if (!validation.valid) {
    const validationError = validation.errors.join(' ')
    log.error('Config validation failed. Fix errors above and restart.')
    if (!isConfigDegradedBootAllowed()) {
      throw new Error(
        `Refusing to start with an invalid config at ${configPath}. `
        + `Fix it or set ${CONFIG_DEGRADED_ENV}=1 to boot read-only. `
        + `Cause: ${validationError}`,
      )
    }
    configLoadFailed = true
    configLoadError = validationError
    log.warn(`${CONFIG_DEGRADED_ENV}=1 - continuing read-only with invalid config`)
  }

  return { config, configLoadFailed, configLoadError }
}

export async function startDaemonRuntime(
  options: StartDaemonRuntimeOptions = {},
): Promise<DaemonRuntimeHandle> {
  const dataDir = resolveDaemonDataDir(options.dataDir)
  const configPath = join(dataDir, 'config.yaml')
  const profileExisted = await pathExists(configPath)

  const { config, configLoadFailed, configLoadError } = await loadStartupConfig(
    dataDir,
    configPath,
    options,
  )

  // Env override allows container/systemd deployments to change bind
  // without mutating ~/.sepilotd/config.yaml. config.yaml is still
  // the primary source — env only wins when set.
  const envPort = Number(process.env.SEPILOTD_PORT)
  const port =
    options.port ?? (Number.isFinite(envPort) && envPort > 0 ? envPort : config.daemon.port)
  const host = options.host ?? (process.env.SEPILOTD_HOST?.trim() || config.daemon.host)

  // Single instance check. Allow a short retry window so a freshly-spawned
  // daemon can wait out a predecessor that is still completing graceful
  // shutdown — common when a watcher (tsup --watch) restarts the process.
  const pidAcquireRetries = (() => {
    const raw = Number(process.env.SEPILOTD_PID_ACQUIRE_RETRIES)
    if (Number.isFinite(raw) && raw >= 0) return Math.floor(raw)
    // The default must outlast a predecessor's graceful-shutdown deadline,
    // otherwise an immediate restart races a daemon that is still draining and
    // fails with "already running". Keep one extra second of slack.
    return Math.ceil((resolveShutdownDeadlineMs() + 1_000) / 500)
  })()
  const pidPath = join(dataDir, 'sepilotd.pid')
  const pidManager = new PidManager(pidPath)
  if (!(await pidManager.acquire({ retries: pidAcquireRetries, retryDelayMs: 500 }))) {
    throw new Error(formatPidConflictMessage(pidPath, await pidManager.readHolder()))
  }

  const releasePid = async () => {
    await pidManager.release()
  }

  let releaseStorage: (() => void) | undefined

  try {
    // Domain repositories historically ignored dataDir and resolved through
    // SEPILOTD_HOME. Prepare a consistent snapshot before any of them opens,
    // then install a process-scoped canonical root for embedded and standalone
    // boot paths alike. An explicit SEPILOTD_HOME remains an operator-owned
    // compatibility override and bypasses automatic migration.
    const storage = await prepareDaemonStorage({
      dataDir,
      profileExisted,
    })
    releaseStorage = configureDaemonStorage({
      dataDir: storage.dataDir,
      domainHome: storage.domainHome,
    })
    log.info('Storage layout ready', {
      mode: storage.mode,
      migratedEntries: storage.migratedEntries.length,
      rollbackAvailable: Boolean(storage.rollbackRoot),
    })

    // loadStartupConfig has already loaded the daemon-managed .env and parsed
    // canonical network config. Configure here (rather than in main.ts) so
    // standalone and desktop-embedded boot paths apply the same policy.
    configureStartupProviderNetwork(config.network)

    // Load auth token
    // Guarantee a presentable credential exists before auth is wired up: a
    // fresh data dir must never boot into a state where every request is
    // rejected and no token exists for the operator to present.
    const authToken = await ensureDaemonAuthToken(dataDir)
    const authTokenResolver = authToken
      ? createFileDaemonAuthTokenResolver(daemonAuthTokenPath(dataDir), { initialToken: authToken })
      : undefined
    assertDaemonAuthPolicy({
      host,
      tokenConfigured: Boolean(authToken),
      allowUnauthenticatedExternal:
        process.env.SEPILOTD_ALLOW_UNAUTHENTICATED_EXTERNAL === '1' ||
        process.env.SEPILOTD_ALLOW_UNAUTHENTICATED_EXTERNAL === 'true',
    })

    // Setup file logging — config drives rotation; env vars stay as
    // emergency overrides for headless deployments without a Settings UI.
    // Falls back to schema defaults when the loaded config is partial (mocked
    // configs in tests, or future migrations that haven't filled this in yet).
    const loggingConfig: LoggingConfig = config.logging ?? loggingSchema.parse({})
    await setLogFile(join(dataDir, 'logs', 'daemon.log'), loggingConfig.app)
    await setAgentTraceLogFile(
      join(dataDir, 'logs', 'agent-trace.jsonl'),
      loggingConfig.trace,
    )
    applyLoggingConfig(loggingConfig, { envLogLevel: process.env.SEPILOTD_LOG_LEVEL })

    const runtime = await buildRuntime(config, dataDir, configPath, {
      autoApproveCliFlag: options.autoApproveCliFlag ?? false,
      configLoadFailed,
      configLoadError,
    })
    log.info('Providers registered', {
      providers: runtime.providerRegistry.list().map((p) => p.id),
    })
    log.info('Tools registered', { tools: runtime.toolRegistry.list().map((t) => t.name) })

    // Start server
    const app = await createApp({ port, host, runtime, authToken, authTokenResolver })

    // Setup graceful shutdown
    const lifecycle = new LifecycleManager(app, runtime, {
      exitAfterShutdown: options.exitOnShutdown ?? true,
      afterShutdown: async () => {
        await closeProviderHttpDispatcher()
        releaseStorage?.()
        await releasePid()
      },
    })
    // Expose the same shutdown path to the HTTP /system/shutdown endpoint so
    // remote clients (desktop tray "Shutdown daemon", `sepilot daemon stop`
    // when the PID file is gone, idle reaper) trigger the identical teardown
    // sequence without having to fake a process signal.
    const shutdownController = {
      shutdown: (reason: string) => lifecycle.shutdown(reason),
    }
    app.decorate('shutdownController', shutdownController)

    try {
      await app.listen({ port, host })
      log.info('Server started', { host, port })
    } catch (err) {
      await releasePid()
      log.error('Failed to start server', { error: String(err) })
      throw err
    }

    if (options.setupSignalHandlers ?? true) {
      lifecycle.setupSignalHandlers()
      // Route crashes through the same graceful teardown as signals so a
      // fatal error still drains HTTP, checkpoints/closes DBs, and flushes
      // swarm state instead of Node's abrupt default exit(1).
      lifecycle.setupCrashHandlers()
    }

    // Optional self-shutdown: when SEPILOTD_IDLE_SHUTDOWN_MS is set, the daemon
    // tears itself down once both the WS connection count is zero AND no HTTP
    // traffic has hit it for the configured window. Default 0 (disabled), so
    // nothing changes for users that prefer a long-lived daemon.
    const idleMs = resolveIdleShutdownMs()
    if (idleMs > 0 && app.connectionRegistry) {
      // A connectionless daemon can still be doing work: an active/queued
      // agent run, a scheduled job, or a swarm run. The reaper must observe
      // all of these before shutting down, or it kills background work.
      const isBusy = (): boolean => {
        const runStats = runtime.runLimiter?.getStats()
        if (runStats && (runStats.active > 0 || runStats.queued > 0)) return true
        if ((runtime.schedulerEngine?.activeRunCount ?? 0) > 0) return true
        if ((runtime.swarmRunRegistry?.list().length ?? 0) > 0) return true
        return false
      }
      startIdleReaper(app.connectionRegistry, shutdownController, { idleMs, isBusy })
      log.info('Idle reaper armed', { idleMs })
    }

    // Embedded desktop mode normally shuts down through the returned handle.
    // For standalone mode, LifecycleManager's signal handlers run the same
    // releasePid hook via afterShutdown before exiting. This last-resort hook
    // must be synchronous: on 'exit' the event loop is stopped, so an async
    // unlink would never run and would leave a stale PID file behind.
    process.on('exit', () => {
      pidManager.releaseSync()
    })

    return {
      app,
      runtime,
      dataDir,
      host,
      port,
      shutdown: (reason = 'embedded shutdown') => lifecycle.shutdown(reason),
    }
  } catch (error) {
    await closeProviderHttpDispatcher()
    releaseStorage?.()
    await releasePid()
    throw error
  }
}
