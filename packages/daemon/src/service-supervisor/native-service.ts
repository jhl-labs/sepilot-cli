import { execFile } from 'node:child_process'
import {
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { sepilotdHome } from '../storage/home.js'
import {
  hasServiceLogOutput,
  normalizeServiceLogFollowMs,
  normalizeServiceLogPollIntervalMs,
  normalizeServiceLogRotation,
  readServiceLogFile,
  rotateServiceLogFile,
  sleep,
  type ServiceLogRotationOptions,
} from './logs.js'

const execFileAsync = promisify(execFile)

export type NativeServiceProvider = 'systemd-user' | 'launchd-user' | 'windows-service'
export type NativeServiceRestart = 'no' | 'on-failure' | 'always'

export interface NativeServiceInstallInput {
  id: string
  name?: string
  description?: string
  provider?: NativeServiceProvider
  executable: string
  args?: string[]
  cwd?: string
  env?: Record<string, string>
  restart?: NativeServiceRestart
  enable?: boolean
  start?: boolean
  reload?: boolean
}

export interface NativeServiceControlInput {
  id: string
  start?: boolean
  stop?: boolean
  reload?: boolean
}

export interface NativeServiceSnapshot {
  id: string
  name: string
  provider: NativeServiceProvider
  unitName: string
  unitPath: string
  installed: boolean
  enabled: boolean | null
  running: boolean | null
  enableState?: string
  activeState?: string
  executable?: string
  args: string[]
  cwd?: string
  envKeys: string[]
  restart: NativeServiceRestart
  logs: {
    stdout: string
    stderr: string
  }
  createdAt?: string
  updatedAt?: string
}

export interface NativeServiceLogChunk {
  service: NativeServiceSnapshot
  stdout: string
  stderr: string
  nextStdoutOffset: number
  nextStderrOffset: number
}

interface NativeServiceRecord {
  id: string
  name: string
  provider: NativeServiceProvider
  unitName: string
  unitPath: string
  executable: string
  args: string[]
  cwd?: string
  envKeys: string[]
  restart: NativeServiceRestart
  logs: {
    stdout: string
    stderr: string
  }
  createdAt: string
  updatedAt: string
}

export interface NativeServiceManagerOptions {
  rootDir?: string
  systemdUserDir?: string
  systemctlPath?: string
  launchdUserDir?: string
  launchctlPath?: string
  launchdDomain?: string
  powershellPath?: string
  platform?: NodeJS.Platform
  logRotation?: ServiceLogRotationOptions
}

const SERVICE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/

function defaultRootDir(): string {
  const dataDir = process.env.SEPILOTD_HOME?.trim()
    || process.env.SEPILOTD_DATA_DIR?.trim()
    || sepilotdHome()
    || join(homedir(), '.sepilotd')
  return join(dataDir, 'native-services')
}

function defaultSystemdUserDir(): string {
  return join(homedir(), '.config', 'systemd', 'user')
}

function defaultLaunchdUserDir(): string {
  return join(homedir(), 'Library', 'LaunchAgents')
}

function defaultLaunchdDomain(): string {
  const rawUid = typeof process.getuid === 'function'
    ? process.getuid()
    : Number(process.env.UID ?? 0)
  const uid = Number.isFinite(rawUid) ? rawUid : 0
  return `gui/${uid}`
}

function nowIso(): string {
  return new Date().toISOString()
}

function normalizeServiceId(id: string): string {
  const candidate = id.trim()
  if (!SERVICE_ID_PATTERN.test(candidate)) {
    throw new Error('native service id must match /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/')
  }
  return candidate
}

function normalizeArgs(raw: unknown): string[] {
  return Array.isArray(raw)
    ? raw.filter((item): item is string => typeof item === 'string')
    : []
}

function normalizeEnv(raw: unknown): Record<string, string> | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const entries = Object.entries(raw)
    .filter((entry): entry is [string, string] => (
      /^[A-Za-z_][A-Za-z0-9_]*$/.test(entry[0])
      && typeof entry[1] === 'string'
    ))
  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

function normalizeRestart(raw: NativeServiceRestart | undefined): NativeServiceRestart {
  return raw === 'always' || raw === 'on-failure' || raw === 'no' ? raw : 'no'
}

function isKnownProvider(provider: unknown): provider is NativeServiceProvider {
  return provider === 'systemd-user' || provider === 'launchd-user' || provider === 'windows-service'
}

function systemdQuote(value: string): string {
  return `"${value
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/%/g, '%%')}"`
}

function renderSystemdUserUnit(input: {
  id: string
  description: string
  executable: string
  args: string[]
  cwd?: string
  env?: Record<string, string>
  restart: NativeServiceRestart
  stdoutPath: string
  stderrPath: string
}): string {
  const lines = [
    '[Unit]',
    `Description=${input.description.replace(/%/g, '%%')}`,
    'After=network.target',
    '',
    '[Service]',
    'Type=simple',
    `ExecStart=${[input.executable, ...input.args].map(systemdQuote).join(' ')}`,
    `Restart=${input.restart}`,
    `Environment=${systemdQuote(`SEPILOTD_NATIVE_SERVICE_ID=${input.id}`)}`,
  ]
  if (input.cwd) lines.push(`WorkingDirectory=${systemdQuote(input.cwd)}`)
  for (const [key, value] of Object.entries(input.env ?? {}).sort(([left], [right]) => left.localeCompare(right))) {
    lines.push(`Environment=${systemdQuote(`${key}=${value}`)}`)
  }
  lines.push(
    `StandardOutput=append:${input.stdoutPath.replace(/%/g, '%%')}`,
    `StandardError=append:${input.stderrPath.replace(/%/g, '%%')}`,
    '',
    '[Install]',
    'WantedBy=default.target',
    '',
  )
  return lines.join('\n')
}

function plistEscape(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

function plistString(value: string): string {
  return `<string>${plistEscape(value)}</string>`
}

function powershellQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

function powershellArray(values: string[]): string {
  return values.length > 0
    ? `@(${values.map(powershellQuote).join(', ')})`
    : '@()'
}

function windowsCommandQuote(value: string): string {
  if (!/[\s"]/.test(value)) return value
  return `"${value
    .replace(/(\\*)"/g, '$1$1\\"')
    .replace(/\\+$/g, '$&$&')}"`
}

function windowsCommandLine(executable: string, args: string[]): string {
  return [executable, ...args].map(windowsCommandQuote).join(' ')
}

function renderLaunchdUserPlist(input: {
  label: string
  executable: string
  args: string[]
  cwd?: string
  env?: Record<string, string>
  restart: NativeServiceRestart
  stdoutPath: string
  stderrPath: string
}): string {
  const lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    '<dict>',
    '  <key>Label</key>',
    `  ${plistString(input.label)}`,
    '  <key>ProgramArguments</key>',
    '  <array>',
    ...[input.executable, ...input.args].map((item) => `    ${plistString(item)}`),
    '  </array>',
    '  <key>RunAtLoad</key>',
    '  <true/>',
  ]
  if (input.cwd) {
    lines.push('  <key>WorkingDirectory</key>', `  ${plistString(input.cwd)}`)
  }
  const envEntries = Object.entries(input.env ?? {}).sort(([left], [right]) => left.localeCompare(right))
  if (envEntries.length > 0) {
    lines.push('  <key>EnvironmentVariables</key>', '  <dict>')
    for (const [key, value] of envEntries) {
      lines.push(`    <key>${plistEscape(key)}</key>`, `    ${plistString(value)}`)
    }
    lines.push('  </dict>')
  }
  if (input.restart === 'always') {
    lines.push('  <key>KeepAlive</key>', '  <true/>')
  } else if (input.restart === 'on-failure') {
    lines.push(
      '  <key>KeepAlive</key>',
      '  <dict>',
      '    <key>SuccessfulExit</key>',
      '    <false/>',
      '  </dict>',
    )
  }
  lines.push(
    '  <key>StandardOutPath</key>',
    `  ${plistString(input.stdoutPath)}`,
    '  <key>StandardErrorPath</key>',
    `  ${plistString(input.stderrPath)}`,
    '</dict>',
    '</plist>',
    '',
  )
  return lines.join('\n')
}

function getExecOutput(error: unknown): string {
  const value = error as { stdout?: unknown; stderr?: unknown; message?: unknown }
  if (typeof value.stdout === 'string' && value.stdout.trim()) return value.stdout.trim()
  if (typeof value.stderr === 'string' && value.stderr.trim()) return value.stderr.trim()
  return typeof value.message === 'string' ? value.message : String(error)
}

function snapshotEnabled(state: string | undefined): boolean | null {
  if (!state) return null
  if (state === 'enabled') return true
  if (state === 'disabled') return false
  if (state === 'manual') return false
  if (state === 'unknown') return null
  return false
}

function snapshotRunning(state: string | undefined): boolean | null {
  if (!state) return null
  if (state === 'active' || state === 'running') return true
  if (state === 'inactive' || state === 'stopped' || state === 'unloaded') return false
  if (state === 'unknown') return null
  return false
}

function recordToSnapshot(
  record: NativeServiceRecord,
  input: {
    installed: boolean
    enableState?: string
    activeState?: string
  },
): NativeServiceSnapshot {
  return {
    id: record.id,
    name: record.name,
    provider: record.provider,
    unitName: record.unitName,
    unitPath: record.unitPath,
    installed: input.installed,
    enabled: snapshotEnabled(input.enableState),
    running: snapshotRunning(input.activeState),
    ...(input.enableState ? { enableState: input.enableState } : {}),
    ...(input.activeState ? { activeState: input.activeState } : {}),
    executable: record.executable,
    args: [...record.args],
    ...(record.cwd ? { cwd: record.cwd } : {}),
    envKeys: [...record.envKeys].sort(),
    restart: record.restart,
    logs: { ...record.logs },
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
  }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export class NativeServiceManager {
  private readonly rootDir: string
  private readonly recordsPath: string
  private readonly systemdUserDir: string
  private readonly systemctlPath: string
  private readonly launchdUserDir: string
  private readonly launchctlPath: string
  private readonly launchdDomain: string
  private readonly powershellPath: string
  private readonly platform: NodeJS.Platform
  private readonly logRotation: Required<ServiceLogRotationOptions>
  private readonly records = new Map<string, NativeServiceRecord>()
  private loaded = false

  constructor(options: NativeServiceManagerOptions = {}) {
    this.rootDir = options.rootDir ?? defaultRootDir()
    this.recordsPath = join(this.rootDir, 'native-services.json')
    this.systemdUserDir = options.systemdUserDir ?? defaultSystemdUserDir()
    this.systemctlPath = options.systemctlPath ?? 'systemctl'
    this.launchdUserDir = options.launchdUserDir ?? defaultLaunchdUserDir()
    this.launchctlPath = options.launchctlPath ?? 'launchctl'
    this.launchdDomain = options.launchdDomain ?? defaultLaunchdDomain()
    this.powershellPath = options.powershellPath ?? 'powershell.exe'
    this.platform = options.platform ?? process.platform
    this.logRotation = normalizeServiceLogRotation(options.logRotation)
  }

  async install(input: NativeServiceInstallInput): Promise<NativeServiceSnapshot> {
    await this.ensureLoaded()
    const provider = input.provider ?? this.defaultProvider()
    this.assertSupportedProvider(provider)
    const id = normalizeServiceId(input.id)
    const executable = input.executable.trim()
    if (!executable) throw new Error('executable is required')
    const args = normalizeArgs(input.args)
    const cwd = typeof input.cwd === 'string' && input.cwd.trim() ? input.cwd.trim() : undefined
    const env = normalizeEnv(input.env)
    const now = nowIso()
    const existing = this.records.get(id)
    const serviceDir = join(this.rootDir, id)
    await mkdir(serviceDir, { recursive: true, mode: 0o700 })
    const unitName = this.unitName(provider, id)
    const unitPath = this.unitPath(provider, unitName, id)
    await mkdir(this.definitionDir(provider, serviceDir), { recursive: true, mode: 0o700 })
    const logs = {
      stdout: join(serviceDir, 'stdout.log'),
      stderr: join(serviceDir, 'stderr.log'),
    }
    const record: NativeServiceRecord = {
      id,
      name: input.name?.trim() || id,
      provider,
      unitName,
      unitPath,
      executable,
      args,
      ...(cwd ? { cwd } : {}),
      envKeys: Object.keys(env ?? {}).sort(),
      restart: normalizeRestart(input.restart),
      logs,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    }
    const unit = provider === 'systemd-user'
      ? renderSystemdUserUnit({
          id,
          description: input.description?.trim() || record.name,
          executable,
          args,
          ...(cwd ? { cwd } : {}),
          ...(env ? { env } : {}),
          restart: record.restart,
          stdoutPath: logs.stdout,
          stderrPath: logs.stderr,
        })
      : provider === 'launchd-user'
        ? renderLaunchdUserPlist({
          label: unitName,
          executable,
          args,
          ...(cwd ? { cwd } : {}),
          ...(env ? { env } : {}),
          restart: record.restart,
          stdoutPath: logs.stdout,
          stderrPath: logs.stderr,
        })
        : JSON.stringify({
            id,
            provider,
            serviceName: unitName,
            executable,
            args,
            ...(cwd ? { cwd } : {}),
            envKeys: Object.keys(env ?? {}).sort(),
            restart: record.restart,
          }, null, 2)
    await writeFile(unitPath, unit, { mode: 0o600 })
    this.records.set(id, record)
    await this.save()

    if (provider === 'systemd-user') {
      if (input.reload !== false) await this.daemonReload()
      if (input.enable) await this.runSystemctl(['enable', unitName])
      if (input.start) await this.runSystemctl(['start', unitName])
    } else if (provider === 'windows-service') {
      await this.installWindowsService(record, {
        description: input.description?.trim() || record.name,
        env,
      })
      if (input.enable) await this.enable({ id, start: input.start })
      else if (input.start) await this.startWindowsService(record)
    } else {
      if (input.enable) await this.launchdBootstrap(record)
      if (input.enable) await this.runLaunchctl(['enable', this.launchdTarget(unitName)])
      if (input.start) {
        if (!input.enable) await this.launchdBootstrap(record).catch(() => {})
        await this.runLaunchctl(['kickstart', '-k', this.launchdTarget(unitName)])
      }
    }
    return this.status(id)
  }

  async status(id: string): Promise<NativeServiceSnapshot> {
    await this.ensureLoaded()
    const record = await this.recordOrPlaceholder(id)
    if (record.provider === 'windows-service') {
      const state = await this.readWindowsServiceState(record.unitName)
      return recordToSnapshot(record, {
        installed: state.installed,
        enableState: state.enableState,
        activeState: state.activeState,
      })
    }
    const installed = await stat(record.unitPath).then(() => true).catch(() => false)
    if (!installed) return recordToSnapshot(record, { installed: false })
    if (record.provider === 'launchd-user') {
      const enableState = await this.readLaunchdEnableState(record.unitName)
      const activeState = await this.readLaunchdActiveState(record.unitName)
      return recordToSnapshot(record, { installed, enableState, activeState })
    }
    const enableState = await this.readSystemctlState(['is-enabled', record.unitName])
    const activeState = await this.readSystemctlState(['is-active', record.unitName])
    return recordToSnapshot(record, { installed, enableState, activeState })
  }

  async logs(input: {
    id: string
    stdoutOffset?: number
    stderrOffset?: number
    limitBytes?: number
    tailBytes?: number
    followMs?: number
    pollIntervalMs?: number
  }): Promise<NativeServiceLogChunk> {
    const record = await this.requireRecord(input.id)
    const followMs = normalizeServiceLogFollowMs(input.followMs)
    const pollIntervalMs = normalizeServiceLogPollIntervalMs(input.pollIntervalMs)
    let chunk = await this.readLogsOnce(record, input)
    if (followMs <= 0 || hasServiceLogOutput(chunk)) {
      return chunk
    }

    const deadline = Date.now() + followMs
    const followInput = {
      ...input,
      tailBytes: undefined,
      stdoutOffset: chunk.nextStdoutOffset,
      stderrOffset: chunk.nextStderrOffset,
    }
    while (Date.now() < deadline) {
      await sleep(Math.min(pollIntervalMs, Math.max(0, deadline - Date.now())))
      chunk = await this.readLogsOnce(record, followInput)
      if (hasServiceLogOutput(chunk)) {
        return chunk
      }
      followInput.stdoutOffset = chunk.nextStdoutOffset
      followInput.stderrOffset = chunk.nextStderrOffset
    }

    return chunk
  }

  private async readLogsOnce(record: NativeServiceRecord, input: {
    stdoutOffset?: number
    stderrOffset?: number
    limitBytes?: number
    tailBytes?: number
  }): Promise<NativeServiceLogChunk> {
    const stdout = await readServiceLogFile(record.logs.stdout, {
      offset: input.stdoutOffset,
      limitBytes: input.limitBytes,
      tailBytes: input.tailBytes,
    })
    const stderr = await readServiceLogFile(record.logs.stderr, {
      offset: input.stderrOffset,
      limitBytes: input.limitBytes,
      tailBytes: input.tailBytes,
    })
    await this.rotateRecordLogs(record)
    return {
      service: await this.status(record.id),
      stdout: stdout.text,
      stderr: stderr.text,
      nextStdoutOffset: stdout.nextOffset,
      nextStderrOffset: stderr.nextOffset,
    }
  }

  private async rotateRecordLogs(record: NativeServiceRecord): Promise<void> {
    await Promise.all([
      rotateServiceLogFile(record.logs.stdout, this.logRotation),
      rotateServiceLogFile(record.logs.stderr, this.logRotation),
    ])
  }

  async enable(input: NativeServiceControlInput): Promise<NativeServiceSnapshot> {
    const record = await this.requireRecord(input.id)
    if (record.provider === 'windows-service') {
      await this.setWindowsServiceStartup(record.unitName, 'Automatic')
      if (input.start) await this.startWindowsService(record)
      return this.status(record.id)
    }
    if (record.provider === 'launchd-user') {
      await this.launchdBootstrap(record)
      await this.runLaunchctl(['enable', this.launchdTarget(record.unitName)])
      if (input.start) await this.runLaunchctl(['kickstart', '-k', this.launchdTarget(record.unitName)])
      return this.status(record.id)
    }
    if (input.reload !== false) await this.daemonReload()
    await this.runSystemctl(['enable', record.unitName])
    if (input.start) await this.runSystemctl(['start', record.unitName])
    return this.status(record.id)
  }

  async disable(input: NativeServiceControlInput): Promise<NativeServiceSnapshot> {
    const record = await this.requireRecord(input.id)
    if (record.provider === 'windows-service') {
      if (input.stop) await this.stopWindowsService(record).catch(() => {})
      await this.setWindowsServiceStartup(record.unitName, 'Disabled')
      return this.status(record.id)
    }
    if (record.provider === 'launchd-user') {
      if (input.stop) await this.runLaunchctl(['bootout', this.launchdTarget(record.unitName)]).catch(() => {})
      await this.runLaunchctl(['disable', this.launchdTarget(record.unitName)])
      return this.status(record.id)
    }
    if (input.stop) await this.runSystemctl(['stop', record.unitName]).catch(() => {})
    await this.runSystemctl(['disable', record.unitName])
    if (input.reload !== false) await this.daemonReload()
    return this.status(record.id)
  }

  async uninstall(input: NativeServiceControlInput): Promise<{ removed: NativeServiceSnapshot }> {
    const record = await this.requireRecord(input.id)
    const before = await this.status(record.id)
    if (record.provider === 'windows-service') {
      if (input.stop !== false) await this.stopWindowsService(record).catch(() => {})
      await this.runWindowsPowerShell('uninstall', { serviceName: record.unitName }, [
        `$name = ${powershellQuote(record.unitName)}`,
        '& sc.exe delete $name | Out-Null',
      ])
    } else if (record.provider === 'launchd-user') {
      if (input.stop !== false) await this.runLaunchctl(['bootout', this.launchdTarget(record.unitName)]).catch(() => {})
      await this.runLaunchctl(['disable', this.launchdTarget(record.unitName)]).catch(() => {})
    } else {
      if (input.stop !== false) await this.runSystemctl(['stop', record.unitName]).catch(() => {})
      await this.runSystemctl(['disable', record.unitName]).catch(() => {})
    }
    await rm(record.unitPath, { force: true })
    this.records.delete(record.id)
    await this.save()
    if (record.provider === 'systemd-user' && input.reload !== false) await this.daemonReload()
    return { removed: before }
  }

  private defaultProvider(): NativeServiceProvider {
    if (this.platform === 'win32') return 'windows-service'
    if (this.platform === 'darwin') return 'launchd-user'
    return 'systemd-user'
  }

  private assertSupportedProvider(provider: NativeServiceProvider): void {
    if (!isKnownProvider(provider)) {
      throw new Error(`unsupported native service provider '${provider}'`)
    }
    if (provider === 'systemd-user' && this.platform !== 'linux') {
      throw new Error(`native service provider '${provider}' is only supported on linux`)
    }
    if (provider === 'launchd-user' && this.platform !== 'darwin') {
      throw new Error(`native service provider '${provider}' is only supported on macOS`)
    }
    if (provider === 'windows-service' && this.platform !== 'win32') {
      throw new Error(`native service provider '${provider}' is only supported on Windows`)
    }
  }

  private unitName(provider: NativeServiceProvider, id: string): string {
    if (provider === 'systemd-user') return `sepilotd-${id}.service`
    if (provider === 'launchd-user') return `com.sepilotd.${id}`
    return `sepilotd-${id}`
  }

  private unitPath(provider: NativeServiceProvider, unitName: string, id: string): string {
    if (provider === 'systemd-user') return join(this.systemdUserDir, unitName)
    if (provider === 'launchd-user') return join(this.launchdUserDir, `${unitName}.plist`)
    return join(this.rootDir, id, 'windows-service.json')
  }

  private definitionDir(provider: NativeServiceProvider, serviceDir: string): string {
    if (provider === 'systemd-user') return this.systemdUserDir
    if (provider === 'launchd-user') return this.launchdUserDir
    return serviceDir
  }

  private async runSystemctl(args: string[]): Promise<{ stdout: string; stderr: string }> {
    return execFileAsync(this.systemctlPath, ['--user', ...args], {
      windowsHide: true,
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
    })
  }

  private async readSystemctlState(args: string[]): Promise<string> {
    try {
      const { stdout } = await this.runSystemctl(args)
      return stdout.trim() || 'unknown'
    } catch (error) {
      const output = getExecOutput(error)
      return output.trim() || 'unknown'
    }
  }

  private async daemonReload(): Promise<void> {
    await this.runSystemctl(['daemon-reload'])
  }

  private launchdTarget(label: string): string {
    return `${this.launchdDomain}/${label}`
  }

  private async runLaunchctl(args: string[]): Promise<{ stdout: string; stderr: string }> {
    return execFileAsync(this.launchctlPath, args, {
      windowsHide: true,
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
    })
  }

  private async launchdBootstrap(record: NativeServiceRecord): Promise<void> {
    try {
      await this.runLaunchctl(['bootstrap', this.launchdDomain, record.unitPath])
    } catch (error) {
      const output = getExecOutput(error)
      if (!/already|exists|in progress/i.test(output)) throw error
    }
  }

  private async readLaunchdEnableState(label: string): Promise<string> {
    try {
      const { stdout } = await this.runLaunchctl(['print-disabled', this.launchdDomain])
      const match = stdout.match(new RegExp(`"${escapeRegExp(label)}"\\s*=>\\s*(true|false)`))
      if (!match) return 'enabled'
      return match[1] === 'true' ? 'disabled' : 'enabled'
    } catch (error) {
      const output = getExecOutput(error)
      return output.trim() || 'unknown'
    }
  }

  private async readLaunchdActiveState(label: string): Promise<string> {
    try {
      const { stdout } = await this.runLaunchctl(['print', this.launchdTarget(label)])
      if (/state\s*=\s*running/i.test(stdout) || /\bpid\s*=\s*\d+/i.test(stdout)) return 'running'
      const state = stdout.match(/state\s*=\s*([A-Za-z0-9_-]+)/i)?.[1]
      return state?.toLowerCase() || 'loaded'
    } catch (error) {
      const output = getExecOutput(error)
      if (/could not find|not found|no such/i.test(output)) return 'unloaded'
      return output.trim() || 'unknown'
    }
  }

  private async runWindowsPowerShell(
    operation: string,
    metadata: Record<string, unknown>,
    scriptLines: string[],
  ): Promise<{ stdout: string; stderr: string }> {
    const script = [
      `# sepilotd:native-service ${operation} ${JSON.stringify(metadata)}`,
      ...scriptLines,
    ].join('\n')
    return execFileAsync(this.powershellPath, [
      '-NoProfile',
      '-NonInteractive',
      '-ExecutionPolicy',
      'Bypass',
      '-Command',
      script,
    ], {
      windowsHide: true,
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
    })
  }

  private async installWindowsService(
    record: NativeServiceRecord,
    input: {
      description: string
      env?: Record<string, string>
    },
  ): Promise<void> {
    const binaryPath = windowsCommandLine(record.executable, record.args)
    const envValues = Object.entries(input.env ?? {})
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => `${key}=${value}`)
    const restartAction = record.restart === 'no' ? '' : 'restart/5000'
    await this.runWindowsPowerShell('install', {
      serviceName: record.unitName,
      binaryPath,
      restart: record.restart,
      envKeys: Object.keys(input.env ?? {}).sort(),
    }, [
      `$name = ${powershellQuote(record.unitName)}`,
      `$binaryPath = ${powershellQuote(binaryPath)}`,
      `$displayName = ${powershellQuote(record.name)}`,
      `$description = ${powershellQuote(input.description)}`,
      '$existing = Get-Service -Name $name -ErrorAction SilentlyContinue',
      'if ($null -eq $existing) {',
      '  New-Service -Name $name -BinaryPathName $binaryPath -DisplayName $displayName -StartupType Manual | Out-Null',
      '} else {',
      '  & sc.exe config $name binPath= $binaryPath DisplayName= $displayName start= demand | Out-Null',
      '}',
      'if ($description) { & sc.exe description $name $description | Out-Null }',
      `$envValues = ${powershellArray(envValues)}`,
      '$serviceKey = "HKLM:\\SYSTEM\\CurrentControlSet\\Services\\$name"',
      'if ($envValues.Count -gt 0) {',
      '  New-Item -Path $serviceKey -Force | Out-Null',
      '  New-ItemProperty -Path $serviceKey -Name Environment -PropertyType MultiString -Value $envValues -Force | Out-Null',
      '} else {',
      '  Remove-ItemProperty -Path $serviceKey -Name Environment -ErrorAction SilentlyContinue',
      '}',
      `& sc.exe failure $name reset= 86400 actions= ${powershellQuote(restartAction)} | Out-Null`,
    ])
  }

  private async readWindowsServiceState(serviceName: string): Promise<{
    installed: boolean
    enableState?: string
    activeState?: string
  }> {
    try {
      const { stdout } = await this.runWindowsPowerShell('status', { serviceName }, [
        `$name = ${powershellQuote(serviceName)}`,
        '$service = Get-CimInstance Win32_Service -Filter "Name = \'$name\'"',
        'if ($null -eq $service) { exit 3 }',
        '[PSCustomObject]@{',
        '  Status = $service.State;',
        '  StartType = $service.StartMode',
        '} | ConvertTo-Json -Compress',
      ])
      const parsed = JSON.parse(stdout) as { Status?: string; StartType?: string }
      const startType = parsed.StartType?.toLowerCase()
      return {
        installed: true,
        enableState: startType === 'auto' || startType === 'automatic'
          ? 'enabled'
          : startType === 'disabled'
            ? 'disabled'
            : 'manual',
        activeState: parsed.Status?.toLowerCase() === 'running' ? 'running' : 'stopped',
      }
    } catch {
      return { installed: false }
    }
  }

  private async setWindowsServiceStartup(serviceName: string, startupType: 'Automatic' | 'Disabled'): Promise<void> {
    await this.runWindowsPowerShell('set-startup', { serviceName, startupType }, [
      `$name = ${powershellQuote(serviceName)}`,
      `Set-Service -Name $name -StartupType ${startupType}`,
    ])
  }

  private async startWindowsService(record: NativeServiceRecord): Promise<void> {
    await this.runWindowsPowerShell('start', { serviceName: record.unitName }, [
      `$name = ${powershellQuote(record.unitName)}`,
      'Start-Service -Name $name',
    ])
  }

  private async stopWindowsService(record: NativeServiceRecord): Promise<void> {
    await this.runWindowsPowerShell('stop', { serviceName: record.unitName }, [
      `$name = ${powershellQuote(record.unitName)}`,
      'Stop-Service -Name $name -ErrorAction SilentlyContinue',
    ])
  }

  private async requireRecord(id: string): Promise<NativeServiceRecord> {
    await this.ensureLoaded()
    const cleanId = normalizeServiceId(id)
    const record = this.records.get(cleanId)
    if (!record) throw new Error(`native service '${cleanId}' not found`)
    this.assertSupportedProvider(record.provider)
    return record
  }

  private async recordOrPlaceholder(id: string): Promise<NativeServiceRecord> {
    await this.ensureLoaded()
    const cleanId = normalizeServiceId(id)
    const record = this.records.get(cleanId)
    if (record) {
      this.assertSupportedProvider(record.provider)
      return record
    }
    const now = nowIso()
    const provider = this.defaultProvider()
    const unitName = this.unitName(provider, cleanId)
    return {
      id: cleanId,
      name: cleanId,
      provider,
      unitName,
      unitPath: this.unitPath(provider, unitName, cleanId),
      executable: '',
      args: [],
      envKeys: [],
      restart: 'no',
      logs: {
        stdout: join(this.rootDir, cleanId, 'stdout.log'),
        stderr: join(this.rootDir, cleanId, 'stderr.log'),
      },
      createdAt: now,
      updatedAt: now,
    }
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return
    await mkdir(this.rootDir, { recursive: true, mode: 0o700 })
    try {
      const raw = await readFile(this.recordsPath, 'utf-8')
      const parsed = JSON.parse(raw) as NativeServiceRecord[]
      for (const record of parsed) {
        if (record?.id && isKnownProvider(record.provider)) {
          this.records.set(record.id, {
            ...record,
            args: normalizeArgs(record.args),
            envKeys: normalizeArgs(record.envKeys).sort(),
            restart: normalizeRestart(record.restart),
          })
        }
      }
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code
      if (code !== 'ENOENT') throw error
    }
    this.loaded = true
  }

  private async save(): Promise<void> {
    await mkdir(this.rootDir, { recursive: true, mode: 0o700 })
    const tmp = `${this.recordsPath}.${process.pid}.tmp`
    await writeFile(tmp, `${JSON.stringify([...this.records.values()], null, 2)}\n`, { mode: 0o600 })
    await rename(tmp, this.recordsPath)
  }
}
