import chalk from 'chalk'
import type {
  DaemonNativeServiceControlInput,
  DaemonNativeServiceInstallInput,
  DaemonNativeServiceLogChunk,
  DaemonNativeServiceSnapshot,
  DaemonServiceHealthCheck,
  DaemonServiceLogChunk,
  DaemonServiceLogsOptions,
  DaemonServiceRemoveInput,
  DaemonServiceRestartMode,
  DaemonServiceSnapshot,
  DaemonServiceStartInput,
  DaemonServiceStopInput,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface ServiceBaseOptions {
  url?: string
}

export interface ServiceStartOptions extends ServiceBaseOptions {
  name?: string
  backend?: string
  executable?: string
  arg?: string[]
  cwd?: string
  env?: string[]
  image?: string
  runtime?: string
  containerName?: string
  command?: string[]
  port?: string[]
  volume?: string[]
  restart?: string
  maxRestarts?: string
  backoffMs?: string
  healthProcess?: boolean
  healthHttp?: string
  healthTcp?: string
  healthIntervalMs?: string
  healthTimeoutMs?: string
  healthGraceMs?: string
  expectedStatus?: string
}

export interface ServiceLogsCommandOptions extends ServiceBaseOptions {
  stdoutOffset?: string
  stderrOffset?: string
  limitBytes?: string
  tailBytes?: string
  followMs?: string
  pollIntervalMs?: string
}

export interface ServiceStopCommandOptions extends ServiceBaseOptions {
  signal?: string
  timeoutMs?: string
}

export interface ServiceRemoveCommandOptions extends ServiceBaseOptions {
  force?: boolean
  deleteLogs?: boolean
}

export interface NativeServiceInstallOptions extends ServiceBaseOptions {
  name?: string
  description?: string
  provider?: string
  executable: string
  arg?: string[]
  cwd?: string
  env?: string[]
  restart?: string
  enable?: boolean
  start?: boolean
  skipReload?: boolean
}

export interface NativeServiceControlOptions extends ServiceBaseOptions {
  start?: boolean
  stop?: boolean
  skipReload?: boolean
}

function stringArray(value: string[] | string | undefined): string[] {
  if (Array.isArray(value)) return value
  return value ? [value] : []
}

function parseIntegerOption(
  raw: string | undefined,
  name: string,
  options: { min?: number } = {},
): number | undefined {
  if (raw === undefined || raw === '') return undefined
  const value = Number(raw)
  if (!Number.isInteger(value) || value < (options.min ?? 0)) {
    throw new Error(`${name} must be an integer >= ${options.min ?? 0}`)
  }
  return value
}

function parseEnv(entries: string[] | string | undefined): Record<string, string> | undefined {
  const pairs = stringArray(entries)
  if (pairs.length === 0) return undefined
  const env: Record<string, string> = {}
  for (const pair of pairs) {
    const index = pair.indexOf('=')
    const key = index >= 0 ? pair.slice(0, index).trim() : ''
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      throw new Error(`Invalid --env entry "${pair}". Expected KEY=VALUE.`)
    }
    env[key] = pair.slice(index + 1)
  }
  return env
}

function parseRestart(
  rawMode: string | undefined,
  maxRestarts: string | undefined,
  backoffMs: string | undefined,
): DaemonServiceStartInput['restart'] | undefined {
  const mode = (rawMode ?? (maxRestarts || backoffMs ? 'on-failure' : undefined)) as
    | DaemonServiceRestartMode
    | undefined
  if (!mode) return undefined
  if (!['never', 'on-failure', 'always'].includes(mode)) {
    throw new Error('--restart must be one of: never, on-failure, always')
  }
  return {
    mode,
    maxRestarts: parseIntegerOption(maxRestarts, '--max-restarts'),
    backoffMs: parseIntegerOption(backoffMs, '--backoff-ms'),
  }
}

function parsePorts(entries: string[] | string | undefined): DaemonServiceStartInput['ports'] {
  const ports = stringArray(entries).map((entry) => {
    const [mapping, rawProtocol = 'tcp'] = entry.split('/')
    if (rawProtocol !== 'tcp' && rawProtocol !== 'udp') {
      throw new Error(`Invalid --port protocol in "${entry}". Expected tcp or udp.`)
    }
    const protocol: 'tcp' | 'udp' = rawProtocol
    const parts = (mapping ?? '').split(':')
    if (parts.length < 1 || parts.length > 2) {
      throw new Error(`Invalid --port entry "${entry}". Expected HOST:CONTAINER or CONTAINER.`)
    }
    const hostPort = parts.length === 2 ? Number(parts[0]) : undefined
    const containerPort = Number(parts[parts.length - 1])
    if (
      !Number.isInteger(containerPort) ||
      containerPort <= 0 ||
      (hostPort !== undefined && (!Number.isInteger(hostPort) || hostPort <= 0))
    ) {
      throw new Error(`Invalid --port entry "${entry}". Ports must be positive integers.`)
    }
    return {
      containerPort,
      ...(hostPort !== undefined ? { hostPort } : {}),
      ...(protocol === 'udp' ? { protocol } : {}),
    }
  })
  return ports.length > 0 ? ports : undefined
}

function parseVolumes(entries: string[] | string | undefined): DaemonServiceStartInput['volumes'] {
  const volumes = stringArray(entries).map((entry) => {
    const [source, target, mode] = entry.split(':')
    if (!source || !target || (mode && !['ro', 'readonly', 'rw'].includes(mode))) {
      throw new Error(`Invalid --volume entry "${entry}". Expected SOURCE:TARGET[:ro].`)
    }
    return {
      source,
      target,
      ...(mode === 'ro' || mode === 'readonly' ? { readonly: true } : {}),
    }
  })
  return volumes.length > 0 ? volumes : undefined
}

function parseHealth(options: ServiceStartOptions): DaemonServiceHealthCheck | undefined {
  const selected = [
    options.healthProcess ? 'process' : undefined,
    options.healthHttp ? 'http' : undefined,
    options.healthTcp ? 'tcp' : undefined,
  ].filter(Boolean)
  if (selected.length > 1) {
    throw new Error('Use only one health check option: --health-process, --health-http, or --health-tcp')
  }
  const base = {
    intervalMs: parseIntegerOption(options.healthIntervalMs, '--health-interval-ms'),
    timeoutMs: parseIntegerOption(options.healthTimeoutMs, '--health-timeout-ms'),
    graceMs: parseIntegerOption(options.healthGraceMs, '--health-grace-ms'),
  }
  if (options.healthProcess) return { type: 'process', ...base }
  if (options.healthHttp) {
    return {
      type: 'http',
      url: options.healthHttp,
      expectedStatus: parseIntegerOption(options.expectedStatus, '--expected-status'),
      ...base,
    }
  }
  if (options.healthTcp) {
    const index = options.healthTcp.lastIndexOf(':')
    const host = index > 0 ? options.healthTcp.slice(0, index) : ''
    const port = index > 0 ? Number(options.healthTcp.slice(index + 1)) : Number.NaN
    if (!host || !Number.isInteger(port) || port <= 0) {
      throw new Error('--health-tcp must be HOST:PORT')
    }
    return { type: 'tcp', host, port, ...base }
  }
  return undefined
}

function parseLogsOptions(options: ServiceLogsCommandOptions): DaemonServiceLogsOptions {
  return {
    stdoutOffset: parseIntegerOption(options.stdoutOffset, '--stdout-offset'),
    stderrOffset: parseIntegerOption(options.stderrOffset, '--stderr-offset'),
    limitBytes: parseIntegerOption(options.limitBytes, '--limit-bytes'),
    tailBytes: parseIntegerOption(options.tailBytes, '--tail-bytes'),
    followMs: parseIntegerOption(options.followMs, '--follow-ms'),
    pollIntervalMs: parseIntegerOption(options.pollIntervalMs, '--poll-interval-ms'),
  }
}

function buildStartInput(id: string | undefined, options: ServiceStartOptions): DaemonServiceStartInput {
  const backend = options.backend ?? (options.image ? 'container' : 'process')
  if (backend !== 'process' && backend !== 'container') {
    throw new Error('--backend must be process or container')
  }
  if (backend === 'process' && !options.executable) {
    throw new Error('--executable is required for process services')
  }
  if (backend === 'container' && !options.image) {
    throw new Error('--image is required for container services')
  }
  return {
    id,
    name: options.name,
    backend,
    executable: options.executable,
    args: stringArray(options.arg),
    cwd: options.cwd,
    env: parseEnv(options.env),
    image: options.image,
    containerRuntime: options.runtime,
    containerName: options.containerName,
    command: stringArray(options.command),
    ports: parsePorts(options.port),
    volumes: parseVolumes(options.volume),
    restart: parseRestart(options.restart, options.maxRestarts, options.backoffMs),
    health: parseHealth(options),
  }
}

function buildNativeControlInput(options: NativeServiceControlOptions): DaemonNativeServiceControlInput {
  return {
    start: options.start,
    stop: options.stop,
    reload: options.skipReload ? false : undefined,
  }
}

function statusColor(status: string): string {
  if (status === 'running' || status === 'healthy') return chalk.green(status)
  if (status === 'failed' || status === 'unhealthy' || status === 'exited') return chalk.red(status)
  if (status === 'stopping' || status === 'starting') return chalk.yellow(status)
  return chalk.gray(status)
}

function formatServiceRow(service: DaemonServiceSnapshot): string {
  return [
    service.id.padEnd(24),
    service.backend.padEnd(9),
    statusColor(service.status).padEnd(18),
    statusColor(service.healthState.status).padEnd(18),
    service.name,
  ].join(' ')
}

function formatServices(services: DaemonServiceSnapshot[]): string {
  if (services.length === 0) return 'No services.'
  return [
    `${'ID'.padEnd(24)} ${'BACKEND'.padEnd(9)} ${'STATUS'.padEnd(18)} ${'HEALTH'.padEnd(18)} NAME`,
    ...services.map(formatServiceRow),
  ].join('\n')
}

function formatService(service: DaemonServiceSnapshot): string {
  const command =
    service.backend === 'container'
      ? [service.container?.image ?? service.containerId ?? service.executable, ...service.args]
      : [service.executable, ...service.args]
  return [
    `${chalk.bold(service.id)} (${service.name})`,
    `backend: ${service.backend}`,
    `status: ${statusColor(service.status)}`,
    `health: ${statusColor(service.healthState.status)}${
      service.healthState.message ? ` - ${service.healthState.message}` : ''
    }`,
    `pid: ${service.pid ?? '-'}`,
    `restart: ${service.restart.mode} count=${service.restartCount}`,
    `next restart: ${service.nextRestartAt ?? '-'}`,
    `command: ${command.filter(Boolean).join(' ') || '-'}`,
    `logs: ${service.logs.stdout} ${service.logs.stderr}`,
  ].join('\n')
}

function formatLogs(chunk: DaemonServiceLogChunk | DaemonNativeServiceLogChunk): string {
  const parts: string[] = []
  if (chunk.stdout) parts.push(chunk.stdout.replace(/\s+$/, ''))
  if (chunk.stderr) parts.push(chalk.red(chunk.stderr.replace(/\s+$/, '')))
  return parts.length > 0 ? parts.join('\n') : 'No new log output.'
}

function formatNativeService(service: DaemonNativeServiceSnapshot): string {
  return [
    `${chalk.bold(service.id)} (${service.name})`,
    `provider: ${service.provider}`,
    `unit: ${service.unitName}`,
    `installed: ${service.installed}`,
    `enabled: ${service.enabled ?? 'unknown'}${service.enableState ? ` (${service.enableState})` : ''}`,
    `running: ${service.running ?? 'unknown'}${service.activeState ? ` (${service.activeState})` : ''}`,
    `restart: ${service.restart}`,
    `unit path: ${service.unitPath}`,
  ].join('\n')
}

export async function serviceListCommand(options: ServiceBaseOptions = {}): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.listServices(), formatServices)
}

export async function serviceStartCommand(
  id: string | undefined,
  options: ServiceStartOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.startService(buildStartInput(id, options)), formatService)
}

export async function serviceStatusCommand(id: string, options: ServiceBaseOptions): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.serviceStatus(id), formatService)
}

export async function serviceLogsCommand(
  id: string,
  options: ServiceLogsCommandOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.serviceLogs(id, parseLogsOptions(options)), formatLogs)
}

export async function serviceHealthcheckCommand(
  id: string,
  options: ServiceBaseOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.serviceHealthcheck(id), formatService)
}

export async function serviceStopCommand(
  id: string,
  options: ServiceStopCommandOptions,
): Promise<void> {
  const input: DaemonServiceStopInput = {
    signal: options.signal,
    timeoutMs: parseIntegerOption(options.timeoutMs, '--timeout-ms'),
  }
  const client = new DaemonClient(options.url)
  output(await client.stopService(id, input), formatService)
}

export async function serviceRestartCommand(id: string, options: ServiceBaseOptions): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.restartService(id), formatService)
}

export async function serviceRemoveCommand(
  id: string,
  options: ServiceRemoveCommandOptions,
): Promise<void> {
  const input: DaemonServiceRemoveInput = {
    force: options.force,
    deleteLogs: options.deleteLogs,
  }
  const client = new DaemonClient(options.url)
  output(await client.removeService(id, input), (result) =>
    `Removed ${chalk.bold(result.removed.id)} (${result.removed.name})`,
  )
}

export async function nativeServiceInstallCommand(
  id: string,
  options: NativeServiceInstallOptions,
): Promise<void> {
  const restart = options.restart
  if (restart && !['no', 'on-failure', 'always'].includes(restart)) {
    throw new Error('--restart must be one of: no, on-failure, always')
  }
  if (options.provider && !['systemd-user', 'launchd-user'].includes(options.provider)) {
    throw new Error('--provider must be one of: systemd-user, launchd-user')
  }
  const input: DaemonNativeServiceInstallInput = {
    id,
    name: options.name,
    description: options.description,
    provider: options.provider as DaemonNativeServiceInstallInput['provider'],
    executable: options.executable,
    args: stringArray(options.arg),
    cwd: options.cwd,
    env: parseEnv(options.env),
    restart: restart as DaemonNativeServiceInstallInput['restart'],
    enable: options.enable,
    start: options.start,
    reload: options.skipReload ? false : undefined,
  }
  const client = new DaemonClient(options.url)
  output(await client.installNativeService(input), formatNativeService)
}

export async function nativeServiceStatusCommand(
  id: string,
  options: ServiceBaseOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.nativeServiceStatus(id), formatNativeService)
}

export async function nativeServiceLogsCommand(
  id: string,
  options: ServiceLogsCommandOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.nativeServiceLogs(id, parseLogsOptions(options)), formatLogs)
}

export async function nativeServiceEnableCommand(
  id: string,
  options: NativeServiceControlOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.enableNativeService(id, buildNativeControlInput(options)), formatNativeService)
}

export async function nativeServiceDisableCommand(
  id: string,
  options: NativeServiceControlOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.disableNativeService(id, buildNativeControlInput(options)), formatNativeService)
}

export async function nativeServiceUninstallCommand(
  id: string,
  options: NativeServiceControlOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  output(await client.uninstallNativeService(id, buildNativeControlInput(options)), (result) =>
    `Uninstalled ${chalk.bold(result.removed.id)} (${result.removed.name})`,
  )
}

export const __testables = {
  buildStartInput,
  parseLogsOptions,
  parseEnv,
  parsePorts,
  parseVolumes,
  parseHealth,
  formatService,
  formatServices,
  formatLogs,
  formatNativeService,
}
