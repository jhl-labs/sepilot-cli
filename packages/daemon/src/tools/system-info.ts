import { execFile } from 'node:child_process'
import { statfs } from 'node:fs/promises'
import { homedir } from 'node:os'
import os from 'node:os'
import { isAbsolute, resolve } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'
import { promisify } from 'node:util'
import { getAbortError, isAbortError, throwIfAborted } from '../abort.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const execFileAsync = promisify(execFile)

export interface SystemInfoSectionCpu {
  model: string
  logicalCores: number
  speedMHzAverage: number
  usagePercent: number
}

export interface SystemInfoSectionMemory {
  totalBytes: number
  freeBytes: number
  usedBytes: number
  usagePercent: number
}

export interface SystemInfoSectionStorage {
  path: string
  totalBytes: number
  freeBytes: number
  usedBytes: number
  usagePercent: number
}

export interface SystemInfoGpuDevice {
  name: string
  memoryTotalMiB: number | null
  memoryUsedMiB: number | null
  memoryFreeMiB: number | null
  utilizationGpuPercent: number | null
  temperatureC: number | null
  powerDrawW: number | null
  powerLimitW: number | null
}

export interface SystemInfoSectionGpu {
  available: boolean
  reason?: string
  devices: SystemInfoGpuDevice[]
}

export interface SystemInfoSnapshot {
  collectedAt: string
  host: {
    hostname: string
    platform: NodeJS.Platform
    release: string
    arch: string
    uptimeSeconds: number
    loadAverage: number[]
  }
  cpu: SystemInfoSectionCpu
  memory: SystemInfoSectionMemory
  storage: SystemInfoSectionStorage
  gpu?: SystemInfoSectionGpu
}

export interface SystemInfoSeriesSummary {
  cpuUsagePercent: { min: number; max: number; average: number }
  memoryUsagePercent: { min: number; max: number; average: number }
  memoryUsedBytes: { min: number; max: number; average: number }
}

export interface SystemInfoSeriesSample {
  collectedAt: string
  uptimeSeconds: number
  loadAverage: number[]
  cpuUsagePercent: number
  memory: Pick<SystemInfoSectionMemory, 'freeBytes' | 'usedBytes' | 'usagePercent'>
  storage: Pick<SystemInfoSectionStorage, 'freeBytes' | 'usedBytes' | 'usagePercent'>
  gpu?: SystemInfoSectionGpu
}

export interface SystemInfoSeries {
  startedAt: string
  completedAt: string
  sampleCount: number
  sampleIntervalMs: number
  observationDurationMs: number
  /** Full first snapshot supplies constant host/capacity metadata once. */
  baseline: SystemInfoSnapshot
  /** Dynamic values only, so long observations do not repeat host metadata. */
  samples: SystemInfoSeriesSample[]
  summary: SystemInfoSeriesSummary
}

const DEFAULT_SAMPLE_INTERVAL_MS = 1_000
const MIN_SAMPLE_INTERVAL_MS = 250
const MAX_SAMPLE_INTERVAL_MS = 60_000
const MAX_SAMPLE_COUNT = 60
const MAX_OBSERVATION_DURATION_MS = 5 * 60_000

function round(value: number, digits = 1): number {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

function getCpuTotals(cpus: ReturnType<typeof os.cpus>): { idle: number; total: number } {
  return cpus.reduce((acc, cpu) => {
    const total = Object.values(cpu.times).reduce((sum, value) => sum + value, 0)
    acc.idle += cpu.times.idle
    acc.total += total
    return acc
  }, { idle: 0, total: 0 })
}

async function sampleCpuUsagePercent(delayMs = 160, signal?: AbortSignal): Promise<number> {
  const before = getCpuTotals(os.cpus())
  await sleep(delayMs, undefined, signal ? { signal } : undefined)
  const after = getCpuTotals(os.cpus())
  const idleDelta = after.idle - before.idle
  const totalDelta = after.total - before.total
  if (totalDelta <= 0) {
    return 0
  }

  return round((1 - idleDelta / totalDelta) * 100, 1)
}

function finiteIntegerInRange(
  raw: unknown,
  fallback: number,
  min: number,
  max: number,
): number {
  if (typeof raw !== 'number' || !Number.isFinite(raw)) return fallback
  return Math.max(min, Math.min(max, Math.trunc(raw)))
}

function metricSummary(values: number[]): { min: number; max: number; average: number } {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const average = values.reduce((sum, value) => sum + value, 0) / values.length
  return {
    min: round(min, 1),
    max: round(max, 1),
    average: round(average, 1),
  }
}

function normalizeStoragePath(rawPath: unknown): string {
  if (typeof rawPath !== 'string' || rawPath.trim().length === 0) {
    return '/'
  }

  if (rawPath.startsWith('~/')) {
    return resolve(homedir(), rawPath.slice(2))
  }

  return isAbsolute(rawPath)
    ? rawPath
    : resolve(process.cwd(), rawPath)
}

function parseGpuNumber(value: string | undefined): number | null {
  if (!value) {
    return null
  }

  const normalized = value.trim()
  if (!normalized || normalized === '[N/A]' || normalized === 'N/A') {
    return null
  }

  const parsed = Number(normalized)
  return Number.isFinite(parsed) ? parsed : null
}

async function collectGpuSection(timeoutMs = 3_000): Promise<SystemInfoSectionGpu> {
  try {
    const { stdout } = await execFileAsync('nvidia-smi', [
      '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit',
      '--format=csv,noheader,nounits',
    ], {
      timeout: timeoutMs,
      maxBuffer: 1024 * 1024,
    })

    const devices = stdout
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const parts = line.split(',').map((part) => part.trim())
        return {
          name: parts[0] ?? 'Unknown GPU',
          memoryTotalMiB: parseGpuNumber(parts[1]),
          memoryUsedMiB: parseGpuNumber(parts[2]),
          memoryFreeMiB: parseGpuNumber(parts[3]),
          utilizationGpuPercent: parseGpuNumber(parts[4]),
          temperatureC: parseGpuNumber(parts[5]),
          powerDrawW: parseGpuNumber(parts[6]),
          powerLimitW: parseGpuNumber(parts[7]),
        } satisfies SystemInfoGpuDevice
      })

    return {
      available: devices.length > 0,
      reason: devices.length > 0 ? undefined : 'nvidia-smi returned no GPU rows',
      devices,
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return {
      available: false,
      reason: message,
      devices: [],
    }
  }
}

async function collectSystemInfoSnapshot(
  input: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<SystemInfoSnapshot> {
  throwIfAborted(signal, 'System information sampling aborted')
  const cpus = os.cpus()
  const logicalCores = cpus.length
  const speedMHzAverage = logicalCores > 0
    ? round(cpus.reduce((sum, cpu) => sum + cpu.speed, 0) / logicalCores, 0)
    : 0
  const cpuUsagePercent = await sampleCpuUsagePercent(160, signal)
  throwIfAborted(signal, 'System information sampling aborted')
  const totalMemory = os.totalmem()
  const freeMemory = os.freemem()
  const usedMemory = Math.max(0, totalMemory - freeMemory)
  const storagePath = normalizeStoragePath(input.storagePath)
  const storageStats = await statfs(storagePath)
  const blockSize = Number(storageStats.bsize)
  const totalStorage = blockSize * Number(storageStats.blocks)
  const freeStorage = blockSize * Number(storageStats.bavail)
  const usedStorage = Math.max(0, totalStorage - freeStorage)

  const snapshot: SystemInfoSnapshot = {
    collectedAt: new Date().toISOString(),
    host: {
      hostname: os.hostname(),
      platform: os.platform(),
      release: os.release(),
      arch: os.arch(),
      uptimeSeconds: os.uptime(),
      loadAverage: os.loadavg().map((value) => round(value, 2)),
    },
    cpu: {
      model: cpus[0]?.model ?? 'Unknown CPU',
      logicalCores,
      speedMHzAverage,
      usagePercent: cpuUsagePercent,
    },
    memory: {
      totalBytes: totalMemory,
      freeBytes: freeMemory,
      usedBytes: usedMemory,
      usagePercent: totalMemory > 0 ? round((usedMemory / totalMemory) * 100, 1) : 0,
    },
    storage: {
      path: storagePath,
      totalBytes: totalStorage,
      freeBytes: freeStorage,
      usedBytes: usedStorage,
      usagePercent: totalStorage > 0 ? round((usedStorage / totalStorage) * 100, 1) : 0,
    },
  }

  if (input.includeGpu === true) {
    snapshot.gpu = await collectGpuSection()
  }
  return snapshot
}

function summarizeSystemInfoSeries(samples: SystemInfoSnapshot[]): SystemInfoSeriesSummary {
  return {
    cpuUsagePercent: metricSummary(samples.map((sample) => sample.cpu.usagePercent)),
    memoryUsagePercent: metricSummary(samples.map((sample) => sample.memory.usagePercent)),
    memoryUsedBytes: metricSummary(samples.map((sample) => sample.memory.usedBytes)),
  }
}

function toSystemInfoSeriesSample(snapshot: SystemInfoSnapshot): SystemInfoSeriesSample {
  return {
    collectedAt: snapshot.collectedAt,
    uptimeSeconds: snapshot.host.uptimeSeconds,
    loadAverage: snapshot.host.loadAverage,
    cpuUsagePercent: snapshot.cpu.usagePercent,
    memory: {
      freeBytes: snapshot.memory.freeBytes,
      usedBytes: snapshot.memory.usedBytes,
      usagePercent: snapshot.memory.usagePercent,
    },
    storage: {
      freeBytes: snapshot.storage.freeBytes,
      usedBytes: snapshot.storage.usedBytes,
      usagePercent: snapshot.storage.usagePercent,
    },
    ...(snapshot.gpu ? { gpu: snapshot.gpu } : {}),
  }
}

export function createSystemInfoTool(): ToolDefinitionRuntime {
  return {
    name: 'system.info',
    description: 'Collect current host OS CPU, memory, storage, uptime, and optional GPU information. For a bounded observation window, set sampleCount and sampleIntervalMs; the tool waits asynchronously, returns every sample plus min/max/average summaries, and is preferred over shell sleep loops or interactive monitors that need a TTY. Host-only: do not use for Kubernetes, pod, container, cluster, node, GitHub Actions runner, or orchestrated workload resource usage. Use only when the user wants current host values or a report of those values; do not use for code/program/script/dashboard/monitor implementation requests.',
    resumeSafety: 'replay-safe',
    // Read-only OS introspection. Safe to run alongside other parallel-safe
    // tools — pulls from /proc, statfs, and an optional nvidia-smi probe;
    // none of these mutate host state.
    scheduling: { mode: 'parallel-safe', resource: 'system-info' },
    inputSchema: {
      type: 'object',
      properties: {
        includeGpu: {
          type: 'boolean',
          description: 'Include GPU details when available',
        },
        storagePath: {
          type: 'string',
          description: 'Filesystem path to inspect for storage usage (default: /)',
        },
        sampleCount: {
          type: 'number',
          description: 'Number of snapshots. Defaults to 1; maximum 60. Values above 1 create a bounded time series.',
        },
        sampleIntervalMs: {
          type: 'number',
          description: 'Observation interval in milliseconds when sampleCount is above 1. Defaults to 1000; range 250..60000. Samples start immediately, so the observation window is approximately (sampleCount - 1) × sampleIntervalMs. Series output stores full host metadata once in baseline and compact dynamic values in samples.',
        },
      },
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        const sampleCount = finiteIntegerInRange(input.sampleCount, 1, 1, MAX_SAMPLE_COUNT)
        if (sampleCount === 1) {
          const snapshot = await collectSystemInfoSnapshot(input, context?.signal)
          return {
            output: JSON.stringify(snapshot, null, 2),
            status: 'success',
            durationMs: Date.now() - start,
          }
        }

        const sampleIntervalMs = finiteIntegerInRange(
          input.sampleIntervalMs,
          DEFAULT_SAMPLE_INTERVAL_MS,
          MIN_SAMPLE_INTERVAL_MS,
          MAX_SAMPLE_INTERVAL_MS,
        )
        const requestedDurationMs = (sampleCount - 1) * sampleIntervalMs
        if (requestedDurationMs > MAX_OBSERVATION_DURATION_MS) {
          return {
            output: `Requested observation window ${requestedDurationMs}ms exceeds the ${MAX_OBSERVATION_DURATION_MS}ms limit. Reduce sampleCount or sampleIntervalMs.`,
            status: 'error',
            code: 'OBSERVATION_WINDOW_TOO_LONG_PERMANENT',
            durationMs: Date.now() - start,
          }
        }

        const observationStartedAtMs = Date.now()
        const startedAt = new Date(observationStartedAtMs).toISOString()
        const samples: SystemInfoSnapshot[] = []
        for (let index = 0; index < sampleCount; index += 1) {
          if (index > 0) {
            const nextSampleAt = observationStartedAtMs + (index * sampleIntervalMs)
            await sleep(
              Math.max(0, nextSampleAt - Date.now()),
              undefined,
              context?.signal ? { signal: context.signal } : undefined,
            )
          }
          samples.push(await collectSystemInfoSnapshot(input, context?.signal))
        }

        const series: SystemInfoSeries = {
          startedAt,
          completedAt: new Date().toISOString(),
          sampleCount,
          sampleIntervalMs,
          observationDurationMs: Date.now() - start,
          baseline: samples[0]!,
          samples: samples.map(toSystemInfoSeriesSample),
          summary: summarizeSystemInfoSeries(samples),
        }
        return {
          output: JSON.stringify(series, null, 2),
          status: 'success',
          durationMs: Date.now() - start,
          metadata: {
            observation: {
              kind: 'host-system-time-series',
              sampleCount,
              sampleIntervalMs,
            },
          },
        }
      } catch (error) {
        if (isAbortError(error) || context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'System information sampling aborted')
        }
        return {
          output: error instanceof Error ? error.message : String(error),
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
    },
  }
}
