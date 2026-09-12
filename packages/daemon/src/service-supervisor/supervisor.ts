import { execFile, spawn, type SpawnOptions } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { connect } from 'node:net'
import {
  appendFile,
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises'
import { closeSync, openSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, join } from 'node:path'
import { promisify } from 'node:util'
import { sepilotdHome } from '../storage/home.js'
import {
  MAX_SERVICE_LOG_READ_BYTES,
  hasServiceLogOutput,
  normalizeServiceLogRotation,
  normalizeServiceLogFollowMs,
  normalizeServiceLogPollIntervalMs,
  readServiceLogFile,
  readServiceLogText,
  rotateServiceLogFile,
  sleep,
  type ServiceLogRotationOptions,
} from './logs.js'

const execFileAsync = promisify(execFile)

export type ServiceBackend = 'process' | 'container'
export type ServiceStatus = 'running' | 'stopping' | 'stopped' | 'exited' | 'failed' | 'unknown'
export type RestartPolicyMode = 'never' | 'on-failure' | 'always'
export type ServiceHealthStatus = 'unknown' | 'starting' | 'healthy' | 'unhealthy'
export type ServiceHealthCheck =
  | { type: 'process'; intervalMs?: number; timeoutMs?: number; graceMs?: number }
  | { type: 'http'; url: string; expectedStatus?: number; intervalMs?: number; timeoutMs?: number; graceMs?: number }
  | { type: 'tcp'; host: string; port: number; intervalMs?: number; timeoutMs?: number; graceMs?: number }

export interface ServiceRestartPolicy {
  mode: RestartPolicyMode
  maxRestarts?: number
  backoffMs?: number
}

export interface ProcessIdentity {
  platform: NodeJS.Platform
  startToken?: string
  commandLine?: string
}

export interface ServiceContainerPort {
  containerPort: number
  hostPort?: number
  protocol?: 'tcp' | 'udp'
}

export interface ServiceContainerVolume {
  source: string
  target: string
  readonly?: boolean
}

export interface ServiceContainerSpec {
  image: string
  runtime?: string
  name?: string
  command?: string[]
  ports?: ServiceContainerPort[]
  volumes?: ServiceContainerVolume[]
}

export interface ServiceSpec {
  id?: string
  name?: string
  backend?: ServiceBackend
  executable?: string
  args?: string[]
  cwd?: string
  env?: Record<string, string>
  container?: ServiceContainerSpec
  image?: string
  containerRuntime?: string
  containerName?: string
  command?: string[]
  ports?: ServiceContainerPort[]
  volumes?: ServiceContainerVolume[]
  restart?: ServiceRestartPolicy
  health?: ServiceHealthCheck
}

export interface ServiceHealthState {
  status: ServiceHealthStatus
  checkedAt?: string
  message?: string
  consecutiveFailures: number
  nextCheckAt?: string
}

export interface PersistedServiceRecord {
  id: string
  name: string
  backend: ServiceBackend
  executable?: string
  args: string[]
  cwd?: string
  env?: Record<string, string>
  container?: ServiceContainerSpec
  containerId?: string
  restart: ServiceRestartPolicy
  health?: ServiceHealthCheck
  healthState: ServiceHealthState
  pid: number | null
  identity?: ProcessIdentity
  status: ServiceStatus
  createdAt: string
  updatedAt: string
  startedAt?: string
  stoppedAt?: string
  exitedAt?: string
  exitCode?: number | null
  signal?: NodeJS.Signals | null
  lastError?: string
  nextRestartAt?: string
  restartCount: number
  logs: {
    stdout: string
    stderr: string
    events: string
  }
}

export interface ServiceSnapshot {
  id: string
  name: string
  backend: ServiceBackend
  executable?: string
  args: string[]
  cwd?: string
  envKeys: string[]
  container?: ServiceContainerSpec
  containerId?: string
  restart: ServiceRestartPolicy
  health?: ServiceHealthCheck
  healthState: ServiceHealthState
  pid: number | null
  status: ServiceStatus
  createdAt: string
  updatedAt: string
  startedAt?: string
  stoppedAt?: string
  exitedAt?: string
  exitCode?: number | null
  signal?: NodeJS.Signals | null
  lastError?: string
  nextRestartAt?: string
  restartCount: number
  logs: {
    stdout: string
    stderr: string
    events: string
  }
}

export interface ServiceLogChunk {
  service: ServiceSnapshot
  stdout: string
  stderr: string
  nextStdoutOffset: number
  nextStderrOffset: number
}

export interface ServiceSupervisorOptions {
  rootDir?: string
  autoStartMonitor?: boolean
  monitorIntervalMs?: number
  containerRuntime?: string
  logRotation?: ServiceLogRotationOptions
}

export interface ServiceSupervisorShutdownOptions {
  stopServices?: boolean
  signal?: NodeJS.Signals
  timeoutMs?: number
}

const DEFAULT_RESTART: ServiceRestartPolicy = { mode: 'never' }
const DEFAULT_HEALTH_INTERVAL_MS = 10_000
const DEFAULT_HEALTH_TIMEOUT_MS = 5_000
const SERVICE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/

function defaultSupervisorRoot(): string {
  const dataDir = process.env.SEPILOTD_HOME?.trim()
    || process.env.SEPILOTD_DATA_DIR?.trim()
    || sepilotdHome()
    || join(homedir(), '.sepilotd')
  return join(dataDir, 'services')
}

function nowIso(): string {
  return new Date().toISOString()
}

function normalizeArgs(raw: unknown): string[] {
  return Array.isArray(raw)
    ? raw.filter((item): item is string => typeof item === 'string')
    : []
}

function normalizeEnv(raw: unknown): Record<string, string> | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const entries = Object.entries(raw)
    .filter((entry): entry is [string, string] => typeof entry[1] === 'string')
  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

function normalizeContainerPorts(raw: unknown): ServiceContainerPort[] | undefined {
  if (!Array.isArray(raw)) return undefined
  const ports = raw.flatMap((item): ServiceContainerPort[] => {
    if (!item || typeof item !== 'object') return []
    const value = item as Record<string, unknown>
    const containerPort = value.containerPort
    if (typeof containerPort !== 'number' || !Number.isFinite(containerPort)) return []
    const normalized: ServiceContainerPort = {
      containerPort: Math.max(1, Math.min(65_535, Math.trunc(containerPort))),
    }
    if (typeof value.hostPort === 'number' && Number.isFinite(value.hostPort)) {
      normalized.hostPort = Math.max(1, Math.min(65_535, Math.trunc(value.hostPort)))
    }
    if (value.protocol === 'udp') {
      normalized.protocol = 'udp'
    } else if (value.protocol === 'tcp') {
      normalized.protocol = 'tcp'
    }
    return [normalized]
  })
  return ports.length > 0 ? ports : undefined
}

function normalizeContainerVolumes(raw: unknown): ServiceContainerVolume[] | undefined {
  if (!Array.isArray(raw)) return undefined
  const volumes = raw.flatMap((item): ServiceContainerVolume[] => {
    if (!item || typeof item !== 'object') return []
    const value = item as Record<string, unknown>
    const source = typeof value.source === 'string' ? value.source.trim() : ''
    const target = typeof value.target === 'string' ? value.target.trim() : ''
    if (!source || !target) return []
    return [{
      source,
      target,
      ...(value.readonly === true ? { readonly: true } : {}),
    }]
  })
  return volumes.length > 0 ? volumes : undefined
}

function normalizeContainer(input: ServiceSpec): ServiceContainerSpec {
  const raw = input.container
  const image = (raw?.image ?? input.image ?? '').trim()
  if (!image) throw new Error('container image is required')

  const runtime = (raw?.runtime ?? input.containerRuntime ?? '').trim()
  const name = (raw?.name ?? input.containerName ?? '').trim()
  const command = normalizeArgs(raw?.command ?? input.command)
  const ports = normalizeContainerPorts(raw?.ports ?? input.ports)
  const volumes = normalizeContainerVolumes(raw?.volumes ?? input.volumes)

  return {
    image,
    ...(runtime ? { runtime } : {}),
    ...(name ? { name } : {}),
    ...(command.length > 0 ? { command } : {}),
    ...(ports ? { ports } : {}),
    ...(volumes ? { volumes } : {}),
  }
}

function normalizeRestart(raw: ServiceRestartPolicy | undefined): ServiceRestartPolicy {
  if (!raw || !['never', 'on-failure', 'always'].includes(raw.mode)) {
    return { ...DEFAULT_RESTART }
  }
  return {
    mode: raw.mode,
    ...(typeof raw.maxRestarts === 'number' && Number.isFinite(raw.maxRestarts)
      ? { maxRestarts: Math.max(0, Math.trunc(raw.maxRestarts)) }
      : {}),
    ...(typeof raw.backoffMs === 'number' && Number.isFinite(raw.backoffMs)
      ? { backoffMs: Math.max(0, Math.trunc(raw.backoffMs)) }
      : {}),
  }
}

function normalizeHealth(raw: ServiceHealthCheck | undefined): ServiceHealthCheck | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const intervalMs = typeof raw.intervalMs === 'number' && Number.isFinite(raw.intervalMs)
    ? Math.max(250, Math.trunc(raw.intervalMs))
    : undefined
  const timeoutMs = typeof raw.timeoutMs === 'number' && Number.isFinite(raw.timeoutMs)
    ? Math.max(100, Math.trunc(raw.timeoutMs))
    : undefined
  const graceMs = typeof raw.graceMs === 'number' && Number.isFinite(raw.graceMs)
    ? Math.max(0, Math.trunc(raw.graceMs))
    : undefined
  if (raw.type === 'process') {
    return {
      type: 'process',
      ...(intervalMs ? { intervalMs } : {}),
      ...(timeoutMs ? { timeoutMs } : {}),
      ...(graceMs !== undefined ? { graceMs } : {}),
    }
  }
  if (raw.type === 'http' && typeof raw.url === 'string' && raw.url.trim()) {
    const expectedStatus = typeof raw.expectedStatus === 'number' && Number.isFinite(raw.expectedStatus)
      ? Math.trunc(raw.expectedStatus)
      : undefined
    return {
      type: 'http',
      url: raw.url.trim(),
      ...(expectedStatus && expectedStatus >= 100 && expectedStatus <= 599
        ? { expectedStatus }
        : {}),
      ...(intervalMs ? { intervalMs } : {}),
      ...(timeoutMs ? { timeoutMs } : {}),
      ...(graceMs !== undefined ? { graceMs } : {}),
    }
  }
  if (raw.type === 'tcp' && typeof raw.host === 'string' && raw.host.trim() && Number.isFinite(raw.port)) {
    return {
      type: 'tcp',
      host: raw.host.trim(),
      port: Math.max(1, Math.min(65_535, Math.trunc(raw.port))),
      ...(intervalMs ? { intervalMs } : {}),
      ...(timeoutMs ? { timeoutMs } : {}),
      ...(graceMs !== undefined ? { graceMs } : {}),
    }
  }
  return undefined
}

function initialHealthState(): ServiceHealthState {
  return { status: 'unknown', consecutiveFailures: 0 }
}

function invalidateHealth(record: PersistedServiceRecord, message: string): boolean {
  if (record.status === 'running') return false
  const nextState: ServiceHealthState = {
    status: 'unknown',
    message,
    consecutiveFailures: 0,
  }
  const changed = record.healthState.status !== nextState.status
    || record.healthState.message !== nextState.message
    || record.healthState.consecutiveFailures !== nextState.consecutiveFailures
    || record.healthState.checkedAt !== undefined
    || record.healthState.nextCheckAt !== undefined
  record.healthState = nextState
  return changed
}

function normalizeServiceId(id: string | undefined, fallback: string): string {
  const candidate = (id ?? fallback).trim()
  if (!SERVICE_ID_PATTERN.test(candidate)) {
    throw new Error('service id must match /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/')
  }
  return candidate
}

function toSnapshot(record: PersistedServiceRecord): ServiceSnapshot {
  return {
    id: record.id,
    name: record.name,
    backend: record.backend,
    ...(record.executable ? { executable: record.executable } : {}),
    args: [...record.args],
    ...(record.cwd ? { cwd: record.cwd } : {}),
    envKeys: Object.keys(record.env ?? {}).sort(),
    ...(record.container ? { container: { ...record.container } } : {}),
    ...(record.containerId ? { containerId: record.containerId } : {}),
    restart: { ...record.restart },
    ...(record.health ? { health: { ...record.health } } : {}),
    healthState: { ...record.healthState },
    pid: record.pid,
    status: record.status,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    ...(record.startedAt ? { startedAt: record.startedAt } : {}),
    ...(record.stoppedAt ? { stoppedAt: record.stoppedAt } : {}),
    ...(record.exitedAt ? { exitedAt: record.exitedAt } : {}),
    ...(record.exitCode !== undefined ? { exitCode: record.exitCode } : {}),
    ...(record.signal !== undefined ? { signal: record.signal } : {}),
    ...(record.lastError ? { lastError: record.lastError } : {}),
    ...(record.nextRestartAt ? { nextRestartAt: record.nextRestartAt } : {}),
    restartCount: record.restartCount,
    logs: { ...record.logs },
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    return code === 'EPERM'
  }
}

async function readLinuxIdentity(pid: number): Promise<ProcessIdentity | null> {
  try {
    const statText = await readFile(`/proc/${pid}/stat`, 'utf-8')
    const endComm = statText.lastIndexOf(')')
    const fields = statText.slice(endComm + 2).trim().split(/\s+/)
    const startTimeTicks = fields[19]
    const cmdline = await readFile(`/proc/${pid}/cmdline`, 'utf-8')
      .then((value) => value.split('\0').filter(Boolean).join(' '))
      .catch(() => undefined)
    return {
      platform: process.platform,
      ...(startTimeTicks ? { startToken: startTimeTicks } : {}),
      ...(cmdline ? { commandLine: cmdline } : {}),
    }
  } catch {
    return null
  }
}

async function readWindowsIdentity(pid: number): Promise<ProcessIdentity | null> {
  try {
    const script = [
      `$p = Get-CimInstance Win32_Process -Filter "ProcessId=${pid}"`,
      'if ($null -eq $p) { exit 3 }',
      '$p | Select-Object -First 1 ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress',
    ].join('; ')
    const { stdout } = await execFileAsync('powershell.exe', ['-NoProfile', '-Command', script], {
      windowsHide: true,
      timeout: 5_000,
      maxBuffer: 1024 * 1024,
    })
    const parsed = JSON.parse(stdout) as { CreationDate?: string; CommandLine?: string }
    return {
      platform: process.platform,
      ...(parsed.CreationDate ? { startToken: parsed.CreationDate } : {}),
      ...(parsed.CommandLine ? { commandLine: parsed.CommandLine } : {}),
    }
  } catch {
    return null
  }
}

async function readPosixIdentity(pid: number): Promise<ProcessIdentity | null> {
  try {
    const { stdout } = await execFileAsync('ps', ['-p', String(pid), '-o', 'lstart=', '-o', 'command='], {
      timeout: 5_000,
      maxBuffer: 1024 * 1024,
    })
    const text = stdout.trim()
    if (!text) return null
    const startToken = text.slice(0, 24).trim() || undefined
    const commandLine = text.slice(24).trim() || undefined
    return {
      platform: process.platform,
      ...(startToken ? { startToken } : {}),
      ...(commandLine ? { commandLine } : {}),
    }
  } catch {
    return null
  }
}

async function readProcessIdentity(pid: number): Promise<ProcessIdentity | null> {
  if (process.platform === 'linux') return readLinuxIdentity(pid)
  if (process.platform === 'win32') return readWindowsIdentity(pid)
  return readPosixIdentity(pid)
}

async function matchesRecordedIdentity(pid: number, identity: ProcessIdentity | undefined): Promise<boolean> {
  if (!isProcessAlive(pid)) return false
  if (!identity?.startToken) return true
  const current = await readProcessIdentity(pid)
  if (!current?.startToken) return true
  return current.startToken === identity.startToken
}

async function terminatePid(pid: number, signal: NodeJS.Signals): Promise<void> {
  if (process.platform === 'win32') {
    const args = ['/PID', String(pid), '/T']
    if (signal === 'SIGKILL') args.push('/F')
    await execFileAsync('taskkill.exe', args, { windowsHide: true }).catch((error) => {
      if (!isProcessAlive(pid)) return
      throw error
    })
    return
  }

  try {
    process.kill(-pid, signal)
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    if (code === 'ESRCH') {
      process.kill(pid, signal)
      return
    }
    try {
      process.kill(pid, signal)
    } catch {
      throw error
    }
  }
}

async function waitForExit(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + Math.max(0, timeoutMs)
  while (Date.now() <= deadline) {
    if (!isProcessAlive(pid)) return true
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  return !isProcessAlive(pid)
}

async function tcpConnect(input: { host: string; port: number; timeoutMs: number }): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const socket = connect({ host: input.host, port: input.port })
    const timer = setTimeout(() => {
      socket.destroy()
      reject(new Error(`tcp health check timed out after ${input.timeoutMs}ms`))
    }, input.timeoutMs)
    socket.once('connect', () => {
      clearTimeout(timer)
      socket.end()
      resolve()
    })
    socket.once('error', (error) => {
      clearTimeout(timer)
      reject(error)
    })
  })
}

export class ServiceSupervisor {
  private readonly rootDir: string
  private readonly recordsPath: string
  private readonly monitorIntervalMs: number
  private readonly defaultContainerRuntime: string
  private readonly logRotation: Required<ServiceLogRotationOptions>
  private readonly records = new Map<string, PersistedServiceRecord>()
  private readonly restartInFlight = new WeakMap<PersistedServiceRecord, Promise<void>>()
  private readonly restartOperations = new Set<Promise<void>>()
  private readonly lifecycleOperations = new Set<Promise<void>>()
  private loaded = false
  private monitorTimer: NodeJS.Timeout | null = null
  private shuttingDown = false
  private saveInFlight: Promise<void> = Promise.resolve()

  constructor(options: ServiceSupervisorOptions = {}) {
    this.rootDir = options.rootDir ?? defaultSupervisorRoot()
    this.recordsPath = join(this.rootDir, 'services.json')
    this.monitorIntervalMs = Math.max(1_000, Math.trunc(options.monitorIntervalMs ?? 5_000))
    this.defaultContainerRuntime = options.containerRuntime?.trim()
      || process.env.SEPILOTD_CONTAINER_RUNTIME?.trim()
      || 'docker'
    this.logRotation = normalizeServiceLogRotation(options.logRotation)
    if (options.autoStartMonitor !== false) {
      this.startMonitor()
    }
  }

  startMonitor(): void {
    if (this.shuttingDown) throw new Error('service supervisor is shut down')
    if (this.monitorTimer) return
    this.monitorTimer = setInterval(() => {
      void this.sweep().catch(() => {})
    }, this.monitorIntervalMs)
    this.monitorTimer.unref?.()
  }

  stopMonitor(): void {
    if (!this.monitorTimer) return
    clearInterval(this.monitorTimer)
    this.monitorTimer = null
  }

  async shutdown(options: ServiceSupervisorShutdownOptions = {}): Promise<ServiceSnapshot[]> {
    this.shuttingDown = true
    this.stopMonitor()

    const failures = await this.settleLifecycleOperations()
    await this.ensureLoaded()
    if (options.stopServices) {
      for (const record of this.records.values()) {
        try {
          const stopped = await this.stop({
            id: record.id,
            signal: options.signal ?? 'SIGTERM',
            timeoutMs: options.timeoutMs ?? 5_000,
          })
          if (stopped.status === 'stopping') {
            failures.push(new Error(`service '${record.id}' did not stop during supervisor shutdown`))
          }
        } catch (error) {
          failures.push(error)
        }
      }
      failures.push(...await this.settleLifecycleOperations())
    }
    if (failures.length > 0) {
      throw new AggregateError(
        failures,
        options.stopServices
          ? 'failed to stop all managed services during supervisor shutdown'
          : 'failed to settle managed service lifecycle during supervisor shutdown',
      )
    }

    return [...this.records.values()].map(toSnapshot)
  }

  async sweep(): Promise<ServiceSnapshot[]> {
    await this.ensureLoaded()
    await this.reconcileAll({ checkHealth: true, applyRestartPolicy: true })
    return [...this.records.values()].map(toSnapshot)
  }

  async start(input: ServiceSpec): Promise<ServiceSnapshot> {
    if (this.shuttingDown) throw new Error('service supervisor is shut down')
    await this.ensureLoaded()
    const backend = input.backend ?? 'process'
    if (backend !== 'process' && backend !== 'container') {
      throw new Error(`unsupported service backend '${backend}'`)
    }
    const executable = typeof input.executable === 'string' ? input.executable.trim() : ''
    const container = backend === 'container' ? normalizeContainer(input) : undefined
    if (backend === 'process' && !executable) throw new Error('executable is required')

    const id = normalizeServiceId(input.id, randomUUID())
    if (this.records.has(id)) {
      throw new Error(`service '${id}' already exists; use service.restart or service.remove`)
    }

    const serviceDir = join(this.rootDir, id)
    await mkdir(serviceDir, { recursive: true, mode: 0o700 })
    const logs = {
      stdout: join(serviceDir, 'stdout.log'),
      stderr: join(serviceDir, 'stderr.log'),
      events: join(serviceDir, 'events.jsonl'),
    }
    const now = nowIso()
    const env = normalizeEnv(input.env)
    const health = normalizeHealth(input.health)
    const record: PersistedServiceRecord = {
      id,
      name: input.name?.trim() || (backend === 'process' ? basename(executable) : container?.image ?? id),
      backend,
      ...(backend === 'process' ? { executable } : {}),
      args: backend === 'process' ? normalizeArgs(input.args) : [],
      ...(typeof input.cwd === 'string' && input.cwd.trim() ? { cwd: input.cwd } : {}),
      ...(env ? { env } : {}),
      ...(container ? { container } : {}),
      restart: normalizeRestart(input.restart),
      ...(health ? { health } : {}),
      healthState: initialHealthState(),
      pid: null,
      status: 'running',
      createdAt: now,
      updatedAt: now,
      startedAt: now,
      restartCount: 0,
      logs,
    }

    await this.spawnRecord(record, false)
    this.records.set(record.id, record)
    await this.save()
    await this.appendEvent(record, 'started', { pid: record.pid, containerId: record.containerId })
    return toSnapshot(record)
  }

  async list(): Promise<ServiceSnapshot[]> {
    await this.ensureLoaded()
    await this.reconcileAll({ checkHealth: true, applyRestartPolicy: true })
    return [...this.records.values()].map(toSnapshot)
  }

  async status(id: string): Promise<ServiceSnapshot> {
    const record = await this.requireRecord(id)
    await this.reconcile(record, { checkHealth: true, applyRestartPolicy: true })
    return toSnapshot(record)
  }

  async healthcheck(id: string): Promise<ServiceSnapshot> {
    const record = await this.requireRecord(id)
    await this.reconcile(record, { checkHealth: true, forceHealth: true, applyRestartPolicy: true })
    return toSnapshot(record)
  }

  async logs(input: {
    id: string
    stdoutOffset?: number
    stderrOffset?: number
    limitBytes?: number
    tailBytes?: number
    followMs?: number
    pollIntervalMs?: number
  }): Promise<ServiceLogChunk> {
    const record = await this.requireRecord(input.id)
    await this.reconcile(record, { checkHealth: false, applyRestartPolicy: true })
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
      await this.reconcile(record, { checkHealth: false, applyRestartPolicy: true })
      chunk = await this.readLogsOnce(record, followInput)
      if (hasServiceLogOutput(chunk)) {
        return chunk
      }
      followInput.stdoutOffset = chunk.nextStdoutOffset
      followInput.stderrOffset = chunk.nextStderrOffset
    }

    return chunk
  }

  private async readLogsOnce(inputRecord: PersistedServiceRecord, input: {
    stdoutOffset?: number
    stderrOffset?: number
    limitBytes?: number
    tailBytes?: number
  }): Promise<ServiceLogChunk> {
    const record = inputRecord
    if (record.backend === 'container') {
      return this.readContainerLogs(record, {
        stdoutOffset: input.stdoutOffset,
        stderrOffset: input.stderrOffset,
        limitBytes: input.limitBytes,
        tailBytes: input.tailBytes,
      })
    }
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
      service: toSnapshot(record),
      stdout: stdout.text,
      stderr: stderr.text,
      nextStdoutOffset: stdout.nextOffset,
      nextStderrOffset: stderr.nextOffset,
    }
  }

  async stop(input: {
    id: string
    signal?: NodeJS.Signals
    timeoutMs?: number
  }): Promise<ServiceSnapshot> {
    const record = await this.requireRecord(input.id)
    await this.reconcile(record, { checkHealth: false, applyRestartPolicy: false })
    if ((record.backend === 'process' && !record.pid) || (record.backend === 'container' && !record.containerId)) {
      return toSnapshot(record)
    }
    // Durable status can lag a detached process or container generation. An
    // explicit stop is authoritative when the recorded target is still live,
    // but recordIsRunning retains the PID-identity guard against PID reuse.
    if (!await this.recordIsRunning(record)) return toSnapshot(record)

    const signal = input.signal ?? 'SIGTERM'
    record.status = 'stopping'
    invalidateHealth(record, 'service is stopping')
    record.updatedAt = nowIso()
    await this.save()
    await this.appendEvent(record, 'stopping', {
      pid: record.pid,
      containerId: record.containerId,
      signal,
    })

    const stopped = record.backend === 'container'
      ? await this.stopContainer(record, input.timeoutMs ?? 5_000)
      : !!record.pid && await terminatePid(record.pid, signal).then(
        () => waitForExit(record.pid!, input.timeoutMs ?? 5_000),
      )

    if (stopped) {
      record.status = 'stopped'
      invalidateHealth(record, 'service is stopped')
      record.stoppedAt = nowIso()
      record.updatedAt = record.stoppedAt
      record.signal = signal
      await this.appendEvent(record, 'stopped', {
        pid: record.pid,
        containerId: record.containerId,
        signal,
      })
    } else {
      record.status = 'stopping'
      record.updatedAt = nowIso()
      record.lastError = record.backend === 'container'
        ? `container ${record.containerId ?? 'unknown'} did not stop within timeout`
        : `process ${record.pid} did not exit within timeout`
      await this.appendEvent(record, 'stop-timeout', {
        pid: record.pid,
        containerId: record.containerId,
        signal,
      })
    }
    await this.save()
    return toSnapshot(record)
  }

  async restart(id: string): Promise<ServiceSnapshot> {
    if (this.shuttingDown) throw new Error('service supervisor is shut down')
    const record = await this.requireRecord(id)
    const stopped = await this.stop({ id, signal: 'SIGTERM', timeoutMs: 5_000 })
    if (stopped.status === 'stopping') {
      throw new Error(`service '${id}' did not stop cleanly; refusing to start a duplicate process`)
    }
    record.restartCount += 1
    await this.spawnRecord(record, true)
    await this.save()
    await this.appendEvent(record, 'restarted', { pid: record.pid, containerId: record.containerId })
    return toSnapshot(record)
  }

  async remove(input: { id: string; force?: boolean; deleteLogs?: boolean }): Promise<{ removed: ServiceSnapshot }> {
    const record = await this.requireRecord(input.id)
    await this.reconcile(record, { checkHealth: false, applyRestartPolicy: false })
    if (record.status === 'running' || record.status === 'stopping') {
      if (!input.force) throw new Error(`service '${record.id}' is running; stop it or pass force`)
      const stopped = await this.stop({ id: record.id, signal: 'SIGTERM', timeoutMs: 5_000 })
      if (stopped.status === 'stopping') {
        throw new Error(`service '${record.id}' did not stop cleanly; refusing to remove the record`)
      }
    }
    const snapshot = toSnapshot(record)
    if (record.backend === 'container') {
      await this.removeContainer(record)
    }
    this.records.delete(record.id)
    await this.save()
    await this.appendEvent(record, 'removed', { deleteLogs: input.deleteLogs === true })
    if (input.deleteLogs) await rm(join(this.rootDir, record.id), { recursive: true, force: true })
    return { removed: snapshot }
  }

  private async spawnRecord(record: PersistedServiceRecord, preserveCreatedAt: boolean): Promise<void> {
    if (record.backend === 'container') {
      await this.startContainerRecord(record, preserveCreatedAt)
      return
    }
    await this.startProcessRecord(record, preserveCreatedAt)
  }

  private async startProcessRecord(record: PersistedServiceRecord, preserveCreatedAt: boolean): Promise<void> {
    const executable = record.executable
    if (!executable) throw new Error('executable is required')
    const stdoutFd = openSync(record.logs.stdout, 'a', 0o600)
    const stderrFd = openSync(record.logs.stderr, 'a', 0o600)
    try {
      const spawnOptions: SpawnOptions = {
        cwd: record.cwd,
        env: {
          ...process.env,
          ...(record.env ?? {}),
          SEPILOTD_SERVICE_ID: record.id,
        },
        detached: true,
        stdio: ['ignore', stdoutFd, stderrFd],
        windowsHide: true,
      }
      const child = spawn(executable, record.args, spawnOptions)
      if (!child.pid) throw new Error('failed to spawn service process')
      child.unref()
      const childPid = child.pid
      record.pid = childPid
      record.identity = await readProcessIdentity(childPid) ?? undefined
      record.status = 'running'
      record.startedAt = nowIso()
      record.updatedAt = record.startedAt
      record.stoppedAt = undefined
      record.exitedAt = undefined
      record.exitCode = null
      record.signal = null
      record.lastError = undefined
      record.nextRestartAt = undefined
      record.healthState = initialHealthState()
      if (!preserveCreatedAt) record.createdAt = record.startedAt

      child.once('error', (error: Error) => {
        this.trackLifecycleOperation(
          this.markExited(record.id, childPid, 'failed', { lastError: error.message }),
        )
      })
      child.once('exit', (code: number | null, signal: NodeJS.Signals | null) => {
        this.trackLifecycleOperation(
          this.markExited(record.id, childPid, 'exited', { exitCode: code, signal }),
        )
      })
    } finally {
      closeSync(stdoutFd)
      closeSync(stderrFd)
    }
  }

  private containerRuntime(record: PersistedServiceRecord): string {
    return record.container?.runtime?.trim() || this.defaultContainerRuntime
  }

  private async runContainerCommand(
    record: PersistedServiceRecord,
    args: string[],
    options: { env?: Record<string, string> } = {},
  ): Promise<{ stdout: string; stderr: string }> {
    return execFileAsync(this.containerRuntime(record), args, {
      env: {
        ...process.env,
        ...(options.env ?? {}),
      },
      windowsHide: true,
      timeout: 30_000,
      maxBuffer: MAX_SERVICE_LOG_READ_BYTES * 2,
    })
  }

  private containerName(record: PersistedServiceRecord): string {
    return record.container?.name?.trim() || `sepilotd-${record.id}`
  }

  private async startContainerRecord(record: PersistedServiceRecord, preserveCreatedAt: boolean): Promise<void> {
    const spec = record.container
    if (!spec?.image) throw new Error('container image is required')
    if (record.containerId) {
      await this.removeContainer(record).catch(() => {})
    }

    const args = [
      'run',
      '--detach',
      '--name',
      this.containerName(record),
      '--label',
      `sepilotd.service.id=${record.id}`,
    ]
    const commandEnv: Record<string, string> = {}
    for (const [key, value] of Object.entries(record.env ?? {}).sort(([left], [right]) => left.localeCompare(right))) {
      args.push('--env', key)
      commandEnv[key] = value
    }
    for (const port of spec.ports ?? []) {
      const protocol = port.protocol ?? 'tcp'
      const publish = port.hostPort
        ? `${port.hostPort}:${port.containerPort}/${protocol}`
        : `${port.containerPort}/${protocol}`
      args.push('--publish', publish)
    }
    for (const volume of spec.volumes ?? []) {
      args.push('--volume', `${volume.source}:${volume.target}${volume.readonly ? ':ro' : ''}`)
    }
    args.push(spec.image, ...(spec.command ?? []))

    const { stdout } = await this.runContainerCommand(record, args, { env: commandEnv })
    const containerId = stdout.trim().split(/\s+/)[0]
    if (!containerId) throw new Error('container runtime did not return a container id')

    record.containerId = containerId
    record.pid = null
    record.identity = undefined
    record.status = 'running'
    record.startedAt = nowIso()
    record.updatedAt = record.startedAt
    record.stoppedAt = undefined
    record.exitedAt = undefined
    record.exitCode = null
    record.signal = null
    record.lastError = undefined
    record.nextRestartAt = undefined
    record.healthState = initialHealthState()
    if (!preserveCreatedAt) record.createdAt = record.startedAt
  }

  private async inspectContainer(record: PersistedServiceRecord): Promise<{
    running: boolean
    exitCode: number | null
    status?: string
  } | null> {
    if (!record.containerId) return null
    try {
      const { stdout } = await this.runContainerCommand(record, ['inspect', record.containerId])
      const parsed = JSON.parse(stdout) as unknown
      const item = Array.isArray(parsed) ? parsed[0] : parsed
      if (!item || typeof item !== 'object') return null
      const state = (item as { State?: unknown }).State
      if (!state || typeof state !== 'object') return null
      const values = state as { Running?: unknown; ExitCode?: unknown; Status?: unknown }
      return {
        running: values.Running === true,
        exitCode: typeof values.ExitCode === 'number' && Number.isFinite(values.ExitCode)
          ? values.ExitCode
          : null,
        ...(typeof values.Status === 'string' ? { status: values.Status } : {}),
      }
    } catch {
      return null
    }
  }

  private async stopContainer(record: PersistedServiceRecord, timeoutMs: number): Promise<boolean> {
    if (!record.containerId) return true
    const seconds = Math.max(1, Math.ceil(timeoutMs / 1_000))
    try {
      await this.runContainerCommand(record, ['stop', '--time', String(seconds), record.containerId])
      return true
    } catch {
      const inspected = await this.inspectContainer(record)
      return inspected?.running !== true
    }
  }

  private async removeContainer(record: PersistedServiceRecord): Promise<void> {
    if (!record.containerId) return
    await this.runContainerCommand(record, ['rm', '--force', record.containerId]).catch(() => {})
    record.containerId = undefined
  }

  private async readContainerLogs(
    record: PersistedServiceRecord,
    input: {
      stdoutOffset?: number
      stderrOffset?: number
      limitBytes?: number
      tailBytes?: number
    },
  ): Promise<ServiceLogChunk> {
    if (!record.containerId) {
      return {
        service: toSnapshot(record),
        stdout: '',
        stderr: '',
        nextStdoutOffset: 0,
        nextStderrOffset: 0,
      }
    }
    const result = await this.runContainerCommand(record, ['logs', record.containerId]).catch((error) => ({
      stdout: '',
      stderr: error instanceof Error ? error.message : String(error),
    }))
    const stdout = readServiceLogText(result.stdout, {
      offset: input.stdoutOffset,
      limitBytes: input.limitBytes,
      tailBytes: input.tailBytes,
    })
    const stderr = readServiceLogText(result.stderr, {
      offset: input.stderrOffset,
      limitBytes: input.limitBytes,
      tailBytes: input.tailBytes,
    })
    return {
      service: toSnapshot(record),
      stdout: stdout.text,
      stderr: stderr.text,
      nextStdoutOffset: stdout.nextOffset,
      nextStderrOffset: stderr.nextOffset,
    }
  }

  private async recordIsRunning(record: PersistedServiceRecord): Promise<boolean> {
    if (record.backend === 'container') {
      return (await this.inspectContainer(record))?.running === true
    }
    return !!record.pid && await matchesRecordedIdentity(record.pid, record.identity)
  }

  private async markExited(
    id: string,
    pid: number,
    status: 'exited' | 'failed',
    details: { exitCode?: number | null; signal?: NodeJS.Signals | null; lastError?: string },
  ): Promise<void> {
    await this.ensureLoaded()
    const record = this.records.get(id)
    if (!record) return
    if (record.pid !== pid) return
    if (record.status !== 'running' && record.status !== 'stopping') return
    const wasStopping = record.status === 'stopping'
    record.status = status
    invalidateHealth(record, status === 'failed' ? 'service failed' : 'service exited')
    record.exitedAt = nowIso()
    record.updatedAt = record.exitedAt
    record.exitCode = details.exitCode ?? record.exitCode ?? null
    record.signal = details.signal ?? record.signal ?? null
    if (details.lastError) record.lastError = details.lastError
    await this.save()
    await this.appendEvent(record, status, {
      pid,
      exitCode: record.exitCode,
      signal: record.signal,
      lastError: record.lastError,
    })
    if (!wasStopping) {
      await this.applyRestartPolicy(record, {
        reason: details.lastError ?? `process exited with code ${record.exitCode ?? 'unknown'}`,
        exitCode: record.exitCode,
        forceFailure: status === 'failed',
      })
    }
  }

  private async reconcileAll(options: {
    checkHealth?: boolean
    forceHealth?: boolean
    applyRestartPolicy?: boolean
  } = {}): Promise<void> {
    for (const record of this.records.values()) {
      await this.reconcile(record, options)
    }
  }

  private async reconcile(record: PersistedServiceRecord, options: {
    checkHealth?: boolean
    forceHealth?: boolean
    applyRestartPolicy?: boolean
  } = {}): Promise<void> {
    if ((record.status === 'exited' || record.status === 'failed') && options.applyRestartPolicy) {
      await this.applyRestartPolicy(record, {
        reason: record.lastError ?? 'service is not running',
        exitCode: record.exitCode ?? null,
        forceFailure: record.status === 'failed',
      })
      return
    }

    if (record.status !== 'running' && record.status !== 'stopping') return

    if (record.backend === 'container') {
      if (!record.containerId) return
      const inspected = await this.inspectContainer(record)
      if (inspected?.running) {
        if (record.status === 'stopping') return
        record.status = 'running'
        if (options.checkHealth) {
          await this.checkHealth(record, { force: options.forceHealth === true })
        }
        return
      }
      if (record.status === 'stopping') {
        record.status = 'stopped'
        invalidateHealth(record, 'service is stopped')
        record.stoppedAt = nowIso()
        record.updatedAt = record.stoppedAt
        record.exitCode = inspected?.exitCode ?? record.exitCode ?? null
        await this.save()
        await this.appendEvent(record, 'reconciled-stopped', { containerId: record.containerId })
        return
      }
      record.status = 'exited'
      invalidateHealth(record, 'service exited')
      record.exitedAt = nowIso()
      record.updatedAt = record.exitedAt
      record.exitCode = inspected?.exitCode ?? record.exitCode ?? null
      record.lastError = inspected
        ? `container exited with status ${inspected.status ?? 'unknown'}`
        : 'container is no longer inspectable'
      await this.save()
      await this.appendEvent(record, 'reconciled-exited', {
        containerId: record.containerId,
        exitCode: record.exitCode,
        reason: record.lastError,
      })
      if (options.applyRestartPolicy) {
        await this.applyRestartPolicy(record, {
          reason: record.lastError,
          exitCode: record.exitCode ?? null,
          forceFailure: inspected === null,
        })
      }
      return
    }

    if (!record.pid) return
    const matches = await matchesRecordedIdentity(record.pid, record.identity)
    if (matches) {
      if (record.status === 'stopping') return
      record.status = 'running'
      if (options.checkHealth) {
        await this.checkHealth(record, { force: options.forceHealth === true })
      }
      return
    }
    if (record.status === 'stopping') {
      record.status = 'stopped'
      invalidateHealth(record, 'service is stopped')
      record.stoppedAt = nowIso()
      record.updatedAt = record.stoppedAt
      await this.save()
      await this.appendEvent(record, 'reconciled-stopped', { pid: record.pid })
      return
    }
    record.status = 'exited'
    invalidateHealth(record, 'service exited')
    record.exitedAt = nowIso()
    record.updatedAt = record.exitedAt
    record.lastError = record.identity?.startToken
      ? 'recorded pid is no longer alive or no longer matches recorded process identity'
      : 'recorded pid is no longer alive'
    await this.save()
    await this.appendEvent(record, 'reconciled-exited', { pid: record.pid, reason: record.lastError })
    if (options.applyRestartPolicy) {
      await this.applyRestartPolicy(record, {
        reason: record.lastError,
        exitCode: record.exitCode ?? null,
        forceFailure: true,
      })
    }
  }

  private async checkHealth(record: PersistedServiceRecord, options: { force?: boolean } = {}): Promise<void> {
    const check = record.health
    if (!check || record.status !== 'running') return
    const now = Date.now()
    const intervalMs = check.intervalMs ?? DEFAULT_HEALTH_INTERVAL_MS
    const timeoutMs = check.timeoutMs ?? DEFAULT_HEALTH_TIMEOUT_MS
    const nextCheckAt = record.healthState.nextCheckAt ? Date.parse(record.healthState.nextCheckAt) : 0
    if (!options.force && nextCheckAt > now) return

    const startedAt = record.startedAt ? Date.parse(record.startedAt) : now
    const graceMs = check.graceMs ?? 0
    if (graceMs > 0 && now - startedAt < graceMs) {
      record.healthState = {
        ...record.healthState,
        status: 'starting',
        checkedAt: nowIso(),
        message: 'within health check grace period',
        nextCheckAt: new Date(Math.min(startedAt + graceMs, now + intervalMs)).toISOString(),
      }
      await this.save()
      return
    }

    const previousStatus = record.healthState.status
    try {
      if (check.type === 'process') {
        if (!await this.recordIsRunning(record)) {
          throw new Error(record.backend === 'container'
            ? 'container is not running'
            : 'process is not alive or no longer matches recorded identity')
        }
      } else if (check.type === 'http') {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), timeoutMs)
        try {
          const response = await fetch(check.url, { signal: controller.signal })
          const expected = check.expectedStatus
          const ok = expected !== undefined
            ? response.status === expected
            : response.status >= 200 && response.status < 500
          if (!ok) throw new Error(`HTTP ${response.status}`)
        } finally {
          clearTimeout(timer)
        }
      } else if (check.type === 'tcp') {
        await tcpConnect({ host: check.host, port: check.port, timeoutMs })
      }

      record.healthState = {
        status: 'healthy',
        checkedAt: nowIso(),
        message: 'ok',
        consecutiveFailures: 0,
        nextCheckAt: new Date(now + intervalMs).toISOString(),
      }
      await this.save()
      if (previousStatus !== 'healthy') {
        await this.appendEvent(record, 'health-healthy', { check: check.type })
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      record.healthState = {
        status: 'unhealthy',
        checkedAt: nowIso(),
        message,
        consecutiveFailures: record.healthState.consecutiveFailures + 1,
        nextCheckAt: new Date(now + intervalMs).toISOString(),
      }
      record.lastError = `health check failed: ${message}`
      record.updatedAt = nowIso()
      await this.save()
      await this.appendEvent(record, 'health-unhealthy', {
        check: check.type,
        failures: record.healthState.consecutiveFailures,
        message,
      })
      await this.applyRestartPolicy(record, {
        reason: record.lastError,
        exitCode: null,
        forceFailure: true,
        stopRunningProcess: true,
      })
    }
  }

  private shouldRestart(record: PersistedServiceRecord, input: {
    exitCode: number | null
    forceFailure?: boolean
    stopRunningProcess?: boolean
  }): boolean {
    if (record.restart.mode === 'never') return false
    if (record.status === 'stopped' || record.status === 'stopping') return false
    // Exit observers and reconciliation can race. Once either path has
    // already replaced the failed process, a late observation must not apply
    // the old exit policy to the new running generation. Health-triggered
    // restarts are the only valid policy transition from a running state.
    if (record.status === 'running' && input.stopRunningProcess !== true) return false
    if (record.restart.mode === 'always') return true
    if (input.forceFailure) return true
    return input.exitCode !== 0
  }

  private async applyRestartPolicy(record: PersistedServiceRecord, input: {
    reason: string
    exitCode: number | null
    forceFailure?: boolean
    stopRunningProcess?: boolean
  }): Promise<void> {
    if (this.shuttingDown) return
    const inFlight = this.restartInFlight.get(record)
    if (inFlight) {
      await inFlight
      return
    }

    const operation = this.applyRestartPolicyOnce(record, input)
    this.restartInFlight.set(record, operation)
    this.restartOperations.add(operation)
    try {
      await operation
    } finally {
      if (this.restartInFlight.get(record) === operation) {
        this.restartInFlight.delete(record)
      }
      this.restartOperations.delete(operation)
    }
  }

  private trackLifecycleOperation(operation: Promise<void>): void {
    this.lifecycleOperations.add(operation)
    void operation.then(
      () => this.lifecycleOperations.delete(operation),
      () => this.lifecycleOperations.delete(operation),
    )
  }

  private async settleLifecycleOperations(): Promise<unknown[]> {
    const failures: unknown[] = []
    while (this.lifecycleOperations.size > 0 || this.restartOperations.size > 0) {
      const operations = new Set([
        ...this.lifecycleOperations,
        ...this.restartOperations,
      ])
      const results = await Promise.allSettled([...operations])
      failures.push(...results.flatMap((result) => result.status === 'rejected' ? [result.reason] : []))
    }
    return failures
  }

  private async applyRestartPolicyOnce(record: PersistedServiceRecord, input: {
    reason: string
    exitCode: number | null
    forceFailure?: boolean
    stopRunningProcess?: boolean
  }): Promise<void> {
    if (!this.shouldRestart(record, input)) return

    const maxRestarts = record.restart.maxRestarts
    if (typeof maxRestarts === 'number' && record.restartCount >= maxRestarts) {
      record.status = 'failed'
      record.nextRestartAt = undefined
      record.lastError = `restart limit reached (${maxRestarts}); last failure: ${input.reason}`
      invalidateHealth(record, record.lastError)
      record.updatedAt = nowIso()
      await this.save()
      await this.appendEvent(record, 'restart-limit-reached', {
        maxRestarts,
        reason: input.reason,
      })
      return
    }

    const now = Date.now()
    const dueAt = record.nextRestartAt ? Date.parse(record.nextRestartAt) : 0
    const backoffMs = record.restart.backoffMs ?? 0
    if (!record.nextRestartAt && backoffMs > 0) {
      record.nextRestartAt = new Date(now + backoffMs).toISOString()
      invalidateHealth(record, `restart scheduled for ${record.nextRestartAt}`)
      record.updatedAt = nowIso()
      await this.save()
      await this.appendEvent(record, 'restart-scheduled', {
        reason: input.reason,
        nextRestartAt: record.nextRestartAt,
      })
      return
    }
    if (dueAt > now) {
      const changed = invalidateHealth(record, `restart scheduled for ${record.nextRestartAt}`)
      if (changed) {
        record.updatedAt = nowIso()
        await this.save()
      }
      return
    }

    if (input.stopRunningProcess && record.backend === 'container' && record.containerId) {
      const stopped = await this.stopContainer(record, 5_000)
      if (!stopped) {
        record.status = 'failed'
        record.lastError = `health restart could not stop container ${record.containerId}`
        invalidateHealth(record, record.lastError)
        record.updatedAt = nowIso()
        await this.save()
        await this.appendEvent(record, 'restart-stop-timeout', { containerId: record.containerId })
        return
      }
    } else if (input.stopRunningProcess && record.pid && isProcessAlive(record.pid)) {
      await terminatePid(record.pid, 'SIGTERM')
      const stopped = await waitForExit(record.pid, 5_000)
      if (!stopped) {
        record.status = 'failed'
        record.lastError = `health restart could not stop process ${record.pid}`
        invalidateHealth(record, record.lastError)
        record.updatedAt = nowIso()
        await this.save()
        await this.appendEvent(record, 'restart-stop-timeout', { pid: record.pid })
        return
      }
    }

    record.restartCount += 1
    record.nextRestartAt = undefined
    await this.spawnRecord(record, true)
    await this.save()
    await this.appendEvent(record, 'auto-restarted', {
      pid: record.pid,
      containerId: record.containerId,
      reason: input.reason,
      restartCount: record.restartCount,
    })
  }

  private async requireRecord(id: string): Promise<PersistedServiceRecord> {
    await this.ensureLoaded()
    const cleanId = id.trim()
    const record = this.records.get(cleanId)
    if (!record) throw new Error(`service '${cleanId}' not found`)
    return record
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return
    await mkdir(this.rootDir, { recursive: true, mode: 0o700 })
    try {
      const raw = await readFile(this.recordsPath, 'utf-8')
      const parsed = JSON.parse(raw) as PersistedServiceRecord[]
      for (const record of parsed) {
        if (record?.id && (record.backend === 'process' || record.backend === 'container')) {
          const health = normalizeHealth(record.health)
          let container: ServiceContainerSpec | undefined
          if (record.backend === 'container') {
            try {
              container = normalizeContainer({ container: record.container })
            } catch {
              continue
            }
          }
          this.records.set(record.id, {
            ...record,
            restart: normalizeRestart(record.restart),
            ...(health ? { health } : { health: undefined }),
            healthState: record.healthState ?? initialHealthState(),
            args: normalizeArgs(record.args),
            ...(container ? { container } : { container: undefined }),
            restartCount: typeof record.restartCount === 'number' ? record.restartCount : 0,
          })
        }
      }
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code
      if (code !== 'ENOENT') throw error
    }
    this.loaded = true
  }

  private save(): Promise<void> {
    const operation = this.saveInFlight.catch(() => {}).then(() => this.saveNow())
    this.saveInFlight = operation
    return operation
  }

  private async saveNow(): Promise<void> {
    await mkdir(this.rootDir, { recursive: true, mode: 0o700 })
    const tmp = `${this.recordsPath}.${process.pid}.${randomUUID()}.tmp`
    const payload = JSON.stringify([...this.records.values()], null, 2)
    try {
      await writeFile(tmp, `${payload}\n`, { mode: 0o600 })
      await rename(tmp, this.recordsPath)
    } finally {
      await rm(tmp, { force: true }).catch(() => {})
    }
  }

  private async appendEvent(
    record: PersistedServiceRecord,
    type: string,
    details: Record<string, unknown>,
  ): Promise<void> {
    await mkdir(join(this.rootDir, record.id), { recursive: true, mode: 0o700 })
    await appendFile(
      record.logs.events,
      `${JSON.stringify({ timestamp: nowIso(), serviceId: record.id, type, ...details })}\n`,
      { mode: 0o600 },
    )
    await this.rotateRecordLogs(record)
  }

  private async rotateRecordLogs(record: PersistedServiceRecord): Promise<void> {
    await Promise.all([
      rotateServiceLogFile(record.logs.stdout, this.logRotation),
      rotateServiceLogFile(record.logs.stderr, this.logRotation),
      rotateServiceLogFile(record.logs.events, this.logRotation),
    ])
  }
}

export const __testables = {
  defaultSupervisorRoot,
  initialHealthState,
  isProcessAlive,
  matchesRecordedIdentity,
  normalizeArgs,
  normalizeContainer,
  normalizeContainerPorts,
  normalizeContainerVolumes,
  normalizeEnv,
  normalizeHealth,
  normalizeRestart,
  normalizeServiceId,
  readLinuxIdentity,
  readPosixIdentity,
  readProcessIdentity,
  readWindowsIdentity,
  tcpConnect,
  terminatePid,
  toSnapshot,
  waitForExit,
}
